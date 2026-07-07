import os
import json
import re
import requests
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm

from pyruqo1.utils.logger import get_logger, progress_bar


REASONING_END = "\n\n</think>"


class DatasetGenerator:
    """Двухэтапная генерация датасета через API llama.cpp (1 сервер или мульти-сервер).

    Использует suffix-based reasoning control: вставляем
    "assistant": {"content": "\n\n</think>"} с add_generation_prompt=false,
    чтобы модель пропускала reasoning и сразу выдавала content.
    """

    # --- Этап 1: генерация вопроса ---
    DEFAULT_QUESTION_SYSTEM_PROMPT = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи и "
        "придумать к нему сложный аналитический вопрос (1-2 предложения). "
        "Ответь СТРОГО в формате JSON с ключом 'prompt'. Все ответы должны быть на русском языке."
    )

    MATH_QUESTION_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
        "научной статьи с формулами в формате LaTeX. Выбери из текста ключевое математическое "
        "уравнение или теоретический вывод и сформулируй сложную задачу, требующую доказать, "
        "вывести или решить это уравнение. Ответь СТРОГО в формате JSON с ключом 'prompt'. "
        "Для всех математических символов и формул используй СТРОГИЙ синтаксис LaTeX. "
        "Все ответы должны быть на русском языке."
    )

    # --- Этап 2: генерация ответа ---
    DEFAULT_ANSWER_SYSTEM_PROMPT = (
        "Ты — ведущий научный методолог. Перед тобой фрагмент научной статьи и "
        "аналитический вопрос к нему. Детально распиши логику рассуждения и дай ответ. "
        "Используй формат: <Thought> Ваши размышления </Thought> <output> Ваш ответ </output>. "
        "Все ответы должны быть на русском языке."
    )

    MATH_ANSWER_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
        "научной статьи с формулами в формате LaTeX и математическая задача. Подробно решите "
        "эту задачу, пошагово расписав математическую логику решения, промежуточные преобразования "
        "и законы. В финальном ответе запишите структурированный результат и конечную формулу. "
        "Для всех математических символов и формул используйте СТРОГИЙ синтаксис LaTeX. "
        "Используйте формат: <Thought> Ваши размышления </Thought> <output> Ваш ответ </output>. "
        "Все ответы должны быть на русском языке."
    )

    def __init__(
        self,
        servers: List[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2500,
        save_interval: int = 20,
        timeout: int = 300,
    ):
        self.servers = servers or ["http://localhost:8079/v1/chat/completions"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.save_interval = save_interval
        self.timeout = timeout
        self.logger = get_logger()
        self._server_index = 0

    def _get_next_server(self) -> str:
        server = self.servers[self._server_index % len(self.servers)]
        self._server_index += 1
        return server

    # ==================== Этап 1: генерация вопроса ====================

    def generate_from_chunks(
        self,
        chunks: List[str],
        output_file: str,
        mode: str = "simple",
    ) -> List[Dict]:
        """Двухэтапная генерация датасета.

        Этап 1: генерация вопросов (suffix-based reasoning control)
        Этап 2: генерация ответов на вопросы (suffix-based reasoning control)
        """
        question_system_prompt = (
            self.DEFAULT_QUESTION_SYSTEM_PROMPT
            if mode == "simple"
            else self.MATH_QUESTION_SYSTEM_PROMPT
        )
        answer_system_prompt = (
            self.DEFAULT_ANSWER_SYSTEM_PROMPT
            if mode == "simple"
            else self.MATH_ANSWER_SYSTEM_PROMPT
        )
        user_prompt_template = (
            "Фрагмент научной публикации с LaTeX-формулами:"
            if mode == "math"
            else "Фрагмент научной публикации:"
        )

        self.logger.info(
            f"Генерация: {len(chunks)} чанков, режим={mode}, серверов={len(self.servers)}"
        )

        # Этап 1: генерация вопросов
        questions = self._generate_questions(
            chunks, question_system_prompt, output_file, user_prompt_template
        )
        self.logger.info(
            f"Этап 1 завершён. Сгенерировано {len(questions)} вопросов из {len(chunks)} чанков."
        )

        # Этап 2: генерация ответов
        dataset_rows = self._generate_answers(
            chunks, questions, answer_system_prompt, output_file, user_prompt_template
        )

        # Удаляем временный файл вопросов
        tmp_file = output_file + ".tmp_questions"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

        self.logger.info(
            f"Генерация завершена. Сохранено {len(dataset_rows)} строк в {output_file}"
        )
        return dataset_rows

    def _generate_questions(
        self,
        chunks: List[str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> Dict[int, str]:
        """Этап 1: генерация вопросов для всех чанков."""
        if len(self.servers) > 1:
            return self._generate_questions_multi_server(
                chunks, system_prompt, output_file, user_prompt_template
            )
        else:
            return self._generate_questions_single_server(
                chunks, system_prompt, output_file, user_prompt_template
            )

    def _generate_questions_single_server(
        self,
        chunks: List[str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> Dict[int, str]:
        questions = {}

        for i, chunk in enumerate(tqdm(chunks, desc="Этап 1: генерация вопросов")):
            user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\""
            question = self._query_question(
                server_url=self._get_next_server(),
                chunk=chunk,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if question:
                questions[i] = question

            if len(questions) % self.save_interval == 0:
                self._save_questions_to_tmp(output_file, questions)

        return questions

    def _generate_questions_multi_server(
        self,
        chunks: List[str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> Dict[int, str]:
        questions = {}
        server_queue = Queue()
        for server in self.servers:
            server_queue.put(server)

        def worker(idx, chunk):
            server = server_queue.get()
            try:
                user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\""
                return idx, self._query_question(
                    server, chunk, system_prompt, user_prompt
                )
            finally:
                server_queue.put(server)

        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {
                executor.submit(worker, i, chunk): i
                for i, chunk in enumerate(chunks)
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Этап 1: генерация вопросов"
            ):
                idx = futures[future]
                try:
                    question = future.result(timeout=self.timeout)
                    if question:
                        questions[idx] = question

                        if len(questions) % self.save_interval == 0:
                            self._save_questions_to_tmp(output_file, questions)
                except Exception as e:
                    self.logger.warning(f"Ошибка этапа 1 для чанка {idx}: {e}")

        return questions

    def _query_question(self, server_url: str, chunk: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Запрос к серверу для генерации вопроса (Этап 1).

        Использует suffix-based reasoning control:
        вставляем assistant message с "\n\n</think>" и add_generation_prompt=false,
        чтобы модель пропускала reasoning и сразу выдавала JSON в content.
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": REASONING_END},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "add_generation_prompt": False,
        }

        try:
            response = requests.post(
                server_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                result_json = response.json()
                choice = result_json["choices"][0]["message"]
                return self._parse_question_response(choice)
            else:
                self.logger.warning(
                    f"Сервер {server_url} вернул код {response.status_code}: {response.text}"
                )
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url} (этап 1): {e}")

        return None

    def _parse_question_response(self, choice: dict) -> Optional[str]:
        """Извлекает вопрос из JSON-ответа модели (Этап 1).

        Убирает suffix "
</think>

" из начала и markdown-обёртку ```json ... ```.
        Обрабатывает обрезанный JSON (добавляет скобки если нужно).
        """
        content_str = choice.get("content", "").strip()
        if not content_str:
            return None

        # Убираем suffix-тег
        content_str = re.sub(r"^</think>\s*", "", content_str)

        # Убираем markdown-обёртку
        content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
        content_str = re.sub(r"\s*```$", "", content_str)

        # Убираем trailing кавычку/скобку если ответ обрезан
        content_str = content_str.rstrip('",')

        try:
            data = json.loads(content_str)
            return data.get("prompt", "").strip()
        except json.JSONDecodeError:
            # Попробуем извлечь строку из обрезанного JSON
            match = re.search(r'"prompt"\s*:\s*"(.+)', content_str, re.DOTALL)
            if match:
                return match.group(1).strip()
            return None
        except AttributeError:
            return None

    def _save_questions_to_tmp(self, output_file: str, questions: Dict[int, str]) -> None:
        """Сохраняет промежуточные вопросы в файл (для защиты от сбоев)."""
        tmp_file = output_file + ".tmp_questions"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False)

    # ==================== Этап 2: генерация ответа ====================

    def _generate_answers(
        self,
        chunks: List[str],
        questions: Dict[int, str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> List[Dict]:
        """Этап 2: генерация ответов для чанков с успешными вопросами."""
        if len(self.servers) > 1:
            return self._generate_answers_multi_server(
                chunks, questions, system_prompt, output_file, user_prompt_template
            )
        else:
            return self._generate_answers_single_server(
                chunks, questions, system_prompt, output_file, user_prompt_template
            )

    def _generate_answers_single_server(
        self,
        chunks: List[str],
        questions: Dict[int, str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> List[Dict]:
        dataset_rows = []

        for i, chunk in enumerate(tqdm(chunks, desc="Этап 2: генерация ответов")):
            if i not in questions:
                continue

            question = questions[i]
            user_prompt = (
                f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"\n\n"
                f"Вопрос: {question}"
            )
            row = self._generate_answer_row(
                server_url=self._get_next_server(),
                chunk=chunk,
                question=question,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if row:
                dataset_rows.append(row)

            if len(dataset_rows) % self.save_interval == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

        return dataset_rows

    def _generate_answers_multi_server(
        self,
        chunks: List[str],
        questions: Dict[int, str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> List[Dict]:
        dataset_rows = []
        server_queue = Queue()
        for server in self.servers:
            server_queue.put(server)

        def worker(idx, chunk):
            if idx not in questions:
                return idx, None

            server = server_queue.get()
            try:
                question = questions[idx]
                user_prompt = (
                    f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"\n\n"
                    f"Вопрос: {question}"
                )
                return idx, self._query_answer(
                    server, chunk, question, system_prompt, user_prompt
                )
            finally:
                server_queue.put(server)

        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {
                executor.submit(worker, i, chunk): i
                for i, chunk in enumerate(chunks)
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Этап 2: генерация ответов"
            ):
                idx = futures[future]
                try:
                    result = future.result(timeout=self.timeout)
                    _, row = result
                    if row:
                        dataset_rows.append(row)

                        if len(dataset_rows) % self.save_interval == 0:
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    self.logger.warning(f"Ошибка этапа 2 для чанка {idx}: {e}")

        return dataset_rows

    def _query_answer(
        self, server_url: str, chunk: str, question: str, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """Запрос к серверу для генерации ответа (Этап 2).

        Использует suffix-based reasoning control:
        вставляем assistant message с "\n\n</think>" и add_generation_prompt=false,
        чтобы модель пропускала reasoning и сразу выдавала ответ с тегами <Thought>...</Thought> <output>...</output>.
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": REASONING_END},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "add_generation_prompt": False,
        }

        try:
            response = requests.post(
                server_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                result_json = response.json()
                choice = result_json["choices"][0]["message"]
                return self._parse_answer_response(choice)
            else:
                self.logger.warning(
                    f"Сервер {server_url} вернул код {response.status_code}: {response.text}"
                )
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url} (этап 2): {e}")

        return None

    def _parse_answer_response(self, choice: dict) -> Optional[str]:
        """Извлекает полный ответ модели (Этап 2).

        Возвращает response строку в формате:
        <Thought> ... </Thought> <output> ... </output>
        """
        content = choice.get("content", "").strip()
        if not content:
            return None
        # Убираем suffix-тег если модель его добавила
        content = re.sub(r"^</think>\s*", "", content)
        return content or None

    def _generate_answer_row(
        self,
        server_url: str,
        chunk: str,
        question: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[Dict]:
        """Собирает строку датасета в формате HuggingFace Dataset_of_Russian_thinking."""
        full_response = self._query_answer(
            server_url, chunk, question, system_prompt, user_prompt
        )

        if full_response:
            prompt_with_context = question
            return {
                "system": system_prompt,
                "prompt": prompt_with_context,
                "response": full_response,
            }
        return None
