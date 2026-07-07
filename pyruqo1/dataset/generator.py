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

class DatasetGenerator:
    """Двухэтапная генерация датасета через API llama.cpp (1 сервер или мульти-сервер).
    
    Этап 1: генерация коротких вопросов (без рассуждений).
    Этап 2: генерация развернутых ответов в формате CoT (с рассуждениями),
    длина которых зависит от целевого размера контекста.
    """
    
    # --- Базовые инструкции для Этапа 1 (Вопросы всегда короткие, без мыслей) ---
    DEFAULT_QUESTION_SYSTEM_PROMPT = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи и "
        "придумать к нему ОДИН короткий, емкий аналитический вопрос (1-2 предложения). "
        "Вопрос должен быть сформулирован максимально лаконично, чтобы вместе со следующим "
        "за ним ответом гарантированно поместиться в лимит контекста.\n"
        "CRITICAL: Do not internalize thoughts. Do not use reasoning. "
        "Do NOT output <think> tags. Provide the final exact JSON immediately."
    )
    
    MATH_QUESTION_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
        "научной статьи с формулами в формате LaTeX. Выбери из текста ключевое математическое "
        "уравнение или теоретический вывод и сформулируй сложную, но лаконичную задачу (1-2 предложения), "
        "требующую доказать, вывести или решить это уравнение.\n"
        "Для всех математических символов и формул используй СТРОГИЙ синтаксис LaTeX. "
        "CRITICAL: Do not internalize thoughts. Do not use reasoning. "
        "Do NOT output <think> tags. Provide the final exact JSON immediately."
    )

    def __init__(
        self,
        servers: List[str] = None,
        context_size: int = 2048,
        temperature: float = 0.2,
        max_tokens: int = 2500,
        save_interval: int = 20,
        timeout: int = 300,
    ):
        self.servers = servers or ["http://localhost:8079/v1/chat/completions"]
        self.context_size = context_size  # Влияет на размер генерируемого ответа
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

    def generate_from_chunks(
        self,
        chunks: List[str],
        output_file: str,
        mode: str = "simple",
    ) -> List[Dict]:
        """Двухэтапная генерация датасета."""
        
        # Настройка системного промпта для Вопросов
        question_system_prompt = (
            self.DEFAULT_QUESTION_SYSTEM_PROMPT
            if mode == "simple"
            else self.MATH_QUESTION_SYSTEM_PROMPT
        )
        
        # Настройка системного промпта для Ответов (динамическая длина)
        if self.context_size == 2048:
            length_instruction = (
                "Пиши рассуждения лаконично, без избыточных повторений. "
                "Ограничь скрытый блок <Thought> максимум 1-2 короткими абзацами. "
                "Итоговый ответ в блоке <output> должен быть кратким и содержательным. "
                "Весь твой ответ обязан быть не длиннее 500-600 токенов."
            )
        else:  # 8192
            length_instruction = (
                "Детально и глубоко распиши пошаговую логику рассуждения в блоке <Thought>. "
                "Проведи полную декомпозицию вопроса, сопоставь все факты. "
                "Дай развернутый, академический ответ в блоке <output>."
            )

        if mode == "math":
            answer_system_prompt = (
                "ВНИМАНИЕ: Все рассуждения и весь ответ должны быть СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование английского языка запрещено.\n"
                "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
                "научной статьи с формулами в формате LaTeX и математическая задача. Подробно решите "
                "эту задачу, пошагово расписав математическую логику решения, промежуточные преобразования "
                "и законы. В финальном ответе запишите структурированный результат и конечную формулу. "
                "Для всех математических символов и формул используйте СТРОГИЙ синтаксис LaTeX.\n"
                f"Используйте формат: <Thought> Ваши рассуждения </Thought> <output> Ваш ответ </output>.\n"
                f"ТРЕБОВАНИЕ К РАЗМЕРУ: {length_instruction}\n"
                "ОБЯЗАТЕЛЬНОЕ УСЛОВИЕ: Пиши логику мышления и выводы исключительно на русском языке!"
            )
        else:
            answer_system_prompt = (
                "ВНИМАНИЕ: Все рассуждения и весь ответ должны быть СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование английского языка запрещено.\n"
                "Ты — ведущий научный методолог. Перед тобой фрагмент научной статьи и "
                "аналитический вопрос к нему. Детально распиши логику рассуждения и дай ответ.\n"
                f"Используй формат: <Thought> Ваши размышления </Thought> <output> Ваш ответ </output>.\n"
                f"ТРЕБОВАНИЕ К РАЗМЕРУ: {length_instruction}\n"
                "ОБЯЗАТЕЛЬНОЕ УСЛОВИЕ: Пиши логику мышления и выводы исключительно на русском языке!"
            )
        user_prompt_template = (
            "Фрагмент научной публикации с LaTeX-формулами:"
            if mode == "math"
            else "Фрагмент научной публикации:"
        )
        
        self.logger.info(
            f"Генерация: {len(chunks)} чанков, лимит контекста={self.context_size}, режим={mode}, серверов={len(self.servers)}"
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
                return idx, self._query_question(server, system_prompt, user_prompt)
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

    def _query_question(self, server_url: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Запрос к серверу для быстрой генерации вопроса в JSON-формате."""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nВыдай ответ СТРОГО в формате JSON с ключом 'prompt': {{\"prompt\": \"...\"}}"}
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            
            # --- УНИВЕРСАЛЬНОЕ ОТКЛЮЧЕНИЕ REASONING ДЛЯ ВСЕХ ВЕРСИЙ LLAMA.CPP ---
            "reasoning_budget": 0,       # Для самых свежих сборок (официальный параметр)
            "thinking_budget_tokens": 0, # Для промежуточных версий
            
            "response_format": {"type": "json_object"}
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
        """Извлекает и очищает сгенерированный вопрос из JSON-ответа."""
        content_str = choice.get("content", "").strip()
        if not content_str:
            return None
            
        content_str = re.sub(r"^(?:json)?\s*", "", content_str)
        content_str = re.sub(r"\s*$", "", content_str)
        content_str = content_str.rstrip('",')
        
        try:
            data = json.loads(content_str)
            return data.get("prompt", "").strip()
        except json.JSONDecodeError:
            # Исправлено регулярное выражение: [^"]+ вместо (^")+ для корректного поиска строки внутри кавычек
            match = re.search(r'"prompt"\s*:\s*"([^"]+)"', content_str, re.DOTALL)
            if match:
                return match.group(1).strip()
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
            question = questions[i]  # Исправлено: заменено с круглых скобок на квадратные
            user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"\n\nВопрос: {question}"
            
            row = self._generate_answer_row(
                server_url=self._get_next_server(),
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
                question = questions[idx]  # Исправлено: заменено с круглых скобок на квадратные
                user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"\n\nВопрос: {question}"
                row = self._generate_answer_row(server, question, system_prompt, user_prompt)
                return idx, row
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
                idx = futures[future]  # Исправлено: заменено с круглых скобок на квадратные
                try:
                    res = future.result(timeout=self.timeout)
                    if res and len(res) == 2:
                        _, row = res
                        if row:
                            dataset_rows.append(row)
                    if len(dataset_rows) % self.save_interval == 0:
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    self.logger.warning(f"Ошибка этапа 2 для чанка {idx}: {e}")
        return dataset_rows

    def _query_answer(self, server_url: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Запрос к серверу для генерации ответа (Этап 2)."""
        target_max_tokens = 900 if self.context_size == 2048 else 4000
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.6,
            "max_tokens": target_max_tokens,
            # Здесь МЫСЛИ НУЖНЫ, поэтому бюджет не ограничиваем
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
                message_data = result_json["choices"][0]["message"]
                
                # Достаем отдельно мысли и отдельно ответ
                reasoning = message_data.get("reasoning_content", "").strip()
                final_output = message_data.get("content", "").strip()
                
                # Если сервер вернул мысли в отдельном поле, склеиваем их для вашего датасета
                if reasoning:
                    return f"<Thought>\n{reasoning}\n</Thought>\n<output>\n{final_output}\n</output>"
                
                # Если сервер вернул всё в одном поле (старый формат)
                return final_output
            else:
                self.logger.warning(
                    f"Сервер {server_url} вернул код {response.status_code}: {response.text}"
                )
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url} (этап 2): {e}")
        return None

    def _needs_translation(self, text: str, threshold: float = 0.3) -> bool:
        """Проверяет, превышает ли процент английских букв заданный порог."""
        if not text:
            return False
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        eng_letters = sum(1 for c in letters if c.lower() in 'abcdefghijklmnopqrstuvwxyz')
        return (eng_letters / len(letters)) > threshold

    def _translate_block(self, server_url: str, text: str) -> str:
        """Синхронно переводит текст на русский язык, бережно сохраняя Markdown-структуру."""
        if not text.strip() or not self._needs_translation(text):
            return text

        system_prompt = (
            "Ты — профессиональный ИИ-переводчик научных публикаций и логов рассуждений (Chain-of-Thought). "
            "Переведи предоставленный текст на русский язык. "
            "КРИТИЧЕСКИ ВАЖНО: сохраняй структуру Markdown, списки, жирный текст (например, **Analyze Input:**), "
            "маркеры пунктов (1., 2., -) и формулы LaTeX в исходном виде. Переводи только сам текст описания. "
            "Выведи ТОЛЬКО чистый перевод, без каких-либо твоих вводных слов и комментариев.\n"
            "CRITICAL: Do not internalize thoughts. Do not use reasoning. "
            "Do NOT output <think> tags. Provide the final translation immediately."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Переведи этот научно-аналитический текст на русский язык, строго сохраняя разметку:\n\n{text}"}
            ],
            "temperature": 0.1,  # Минимальная температура для точности перевода
            "max_tokens": 3500,
            "reasoning_budget": 0,
            "thinking_budget_tokens": 0
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
                return result_json["choices"]["message"]["content"].strip()
        except Exception as e:
            self.logger.warning(f"Ошибка перевода блока на сервере {server_url}: {e}")
        return text

    def _process_translation_pipeline(self, server_url: str, full_response: str) -> str:
        """Разбирает ответ по тегам, переводит англоязычный CoT и output, сохраняя структуру тегов."""
        # Паттерны для поиска блоков с учетом регистра и возможных пробелов
        thought_match = re.search(r"<Thought>(.*?)</Thought>", full_response, re.DOTALL | re.IGNORECASE)
        output_match = re.search(r"<output>(.*?)</output>", full_response, re.DOTALL | re.IGNORECASE)

        if thought_match and output_match:
            thought_text = thought_match.group(1).strip()
            output_text = output_match.group(1).strip()

            # Отправляем на перевод только содержательную часть блоков
            translated_thought = self._translate_block(server_url, thought_text)
            translated_output = self._translate_block(server_url, output_text)

            # Собираем красивую XML-подобную структуру обратно
            return f"<Thought>\n{translated_thought}\n</Thought>\n<output>\n{translated_output}\n</output>"
        
        # Запасной вариант, если модель выдала текст без тегов
        if self._needs_translation(full_response):
            return self._translate_block(server_url, full_response)
            
        return full_response

    def _generate_answer_row(
        self,
        server_url: str,
        question: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[Dict]:
        """Собирает финальную строку для датасета и переводит её при необходимости."""
        full_response = self._query_answer(server_url, system_prompt, user_prompt)
        if full_response:
            # Очистка вопроса
            if isinstance(question, (list, tuple)):
                clean_question = question[-1]
            else:
                clean_question = question
            clean_question = str(clean_question).strip().strip('"').strip("'")
            
            # --- ЭТАП 3: ПОСТ-ОБРАБОТКА И ПЕРЕВОД ---
            # Проверяем и при необходимости переводим внутренности блоков на русский язык
            final_response = self._process_translation_pipeline(server_url, full_response)
            
            return {
                "system": system_prompt,
                "prompt": clean_question,
                "response": final_response,
            }
        return None