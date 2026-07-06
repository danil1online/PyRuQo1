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
    """Генерация датасета через API llama.cpp (1 сервер или мульти-сервер).

    Параметры по умолчанию соответствуют настройкам оригинальных скриптов,
    которые работают с reasoning-моделями (Qwen 35B, o1-LoRA и т.п.):
    - temperature=0.2 (низкая температура для строгости)
    - max_tokens=2500 (запас токенов на формулы и рассуждения)
    - timeout=300 (5 минут на сложный чанк)
    - response_format УБРАН — ломает reasoning-модели
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи, "
        "придумать сложный аналитический вопрос к нему, детально расписать логику "
        "рассуждения и выдать ответ. Ты должен вернуть результат СТРОГО в формате JSON "
        "со следующими ключами: 'prompt', 'thought', 'response'. "
        "Все ответы и рассуждения должны быть на русском языке."
    )

    MATH_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент научной статьи "
        "с формулами в формате LaTeX. Выбери из текста ключевое математическое уравнение или теоретический вывод. "
        "Сформулируй сложную задачу, требующую доказать, вывести или решить это уравнение. "
        "Сначала выдай сформулированную задачу, а затем напиши подробное итоговое решение. "
        "Для всех математических символов и формул используй СТРОГИЙ синтаксис LaTeX. "
        "Ты должен вернуть результат СТРОГО в формате JSON со следующими ключами: 'prompt', 'thought', 'response'. "
        "Все ответы и рассуждения должны быть на русском языке."
    )

    SYSTEM_TEMPLATE = (
        "Вы — ИИ-ассистент. Форматируйте ваши ответы следующим образом: "
        "<Thought> Ваши размышления (понимание, логика) </Thought> <output> Ваш ответ </output>"
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

    def _parse_response(self, choice: dict, chunk: str) -> Optional[Dict]:
        """Извлекает prompt/thought/response из ответа модели.

        Работает с reasoning-моделями llama.cpp, которые отдают:
        - reasoning_content — нативные мысли + ответ (Qwen3.6-35B-A3B-UD)
        - content — финальный ответ (некоторые модели)

        Для Qwen3.6-35B-A3B-UD: content="" (пустой), всё в reasoning_content.
        """
        thought_text = choice.get("reasoning_content", "").strip()

        # Страховка: если сервер отдал мысли в другом поле
        if not thought_text:
            thought_text = choice.get("data", {}).get("reasoning_content", "").strip()

        full_response_text = choice.get("content", "").strip()

        # Формируем prompt с фрагментом статьи (как в оригинальных скриптах)
        prompt_with_context = f"На основе фрагмента статьи решите аналитическую задачу: {chunk[:150]}..."

        # Если content пустой (как у Qwen3.6-35B-A3B-UD), используем reasoning_content как ответ
        if not full_response_text and thought_text:
            return {
                "prompt": prompt_with_context,
                "thought": thought_text,
                "response": full_response_text,
            }
        elif full_response_text and thought_text:
            return {
                "prompt": prompt_with_context,
                "thought": thought_text,
                "response": full_response_text,
            }
        elif full_response_text:
            return {
                "prompt": prompt_with_context,
                "thought": "Анализ предоставленного контекста.",
                "response": full_response_text,
            }
        return None

    def _query_server(self, server_url: str, chunk: str, system_prompt: str, user_prompt: str = None) -> Optional[Dict]:
        if user_prompt is None:
            user_prompt = f"Фрагмент научной публикации:\n\"\"\"\n{chunk}\n\"\"\""

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # response_format УБРАН — он ломает reasoning-модели
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
                return self._parse_response(choice, chunk)
            else:
                self.logger.warning(f"Сервер {server_url} вернул код {response.status_code}: {response.text}")
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url}: {e}")

        return None

    def _generate_row(self, chunk: str, system_prompt: str, user_prompt: str = None) -> Optional[Dict]:
        llm_data = self._query_server(self._get_next_server(), chunk, system_prompt, user_prompt)

        if llm_data and all(k in llm_data for k in ["prompt", "thought", "response"]):
            return {
                "system": self.SYSTEM_TEMPLATE,
                "prompt": llm_data["prompt"],
                "response": f"<Thought>\n{llm_data['thought']}\n</Thought>\n<output>\n{llm_data['response']}\n</output>",
            }
        return None

    def generate_from_chunks(
        self,
        chunks: List[str],
        output_file: str,
        mode: str = "simple",
    ) -> List[Dict]:
        system_prompt = self.DEFAULT_SYSTEM_PROMPT if mode == "simple" else self.MATH_SYSTEM_PROMPT
        user_prompt_template = (
            "Фрагмент научной публикации с LaTeX-формулами:"
            if mode == "math"
            else "Фрагмент научной публикации:"
        )

        self.logger.info(f"Генерация: {len(chunks)} чанков, режим={mode}, серверов={len(self.servers)}")

        dataset_rows = []

        if len(self.servers) > 1:
            dataset_rows = self._generate_multi_server(chunks, system_prompt, output_file, user_prompt_template)
        else:
            dataset_rows = self._generate_single_server(chunks, system_prompt, output_file, user_prompt_template)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Генерация завершена. Сохранено {len(dataset_rows)} строк в {output_file}")
        return dataset_rows

    def _generate_single_server(self, chunks: List[str], system_prompt: str, output_file: str = None, user_prompt_template: str = "Фрагмент научной публикации:") -> List[Dict]:
        dataset_rows = []

        for i, chunk in enumerate(tqdm(chunks, desc="Генерация (1 сервер)")):
            user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\""
            row = self._generate_row(chunk, system_prompt, user_prompt)
            if row:
                dataset_rows.append(row)

            if output_file and len(dataset_rows) % self.save_interval == 0:
                # Автосохранение каждые save_interval строк
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

        return dataset_rows

    def _generate_multi_server(self, chunks: List[str], system_prompt: str, output_file: str = None, user_prompt_template: str = "Фрагмент научной публикации:") -> List[Dict]:
        dataset_rows = []
        server_queue = Queue()
        for server in self.servers:
            server_queue.put(server)

        def worker(chunk):
            server = server_queue.get()
            try:
                user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\""
                return self._query_server(server, chunk, system_prompt, user_prompt)
            finally:
                server_queue.put(server)

        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {executor.submit(worker, chunk): chunk for chunk in chunks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Генерация (мульти-сервер)"):
                chunk = futures[future]
                try:
                    llm_data = future.result(timeout=self.timeout)
                    if llm_data and all(k in llm_data for k in ["prompt", "thought", "response"]):
                        row = {
                            "system": self.SYSTEM_TEMPLATE,
                            "prompt": llm_data["prompt"],
                            "response": f"<Thought>\n{llm_data['thought']}\n</Thought>\n<output>\n{llm_data['response']}\n</output>",
                        }
                        dataset_rows.append(row)

                        if len(dataset_rows) % self.save_interval == 0:
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки чанка: {e}")

        return dataset_rows
