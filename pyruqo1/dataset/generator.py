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
        "рассуждения и выдать ответ."
    )

    MATH_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент научной статьи "
        "с формулами в формате LaTeX. Выбери из текста ключевое математическое уравнение или теоретический вывод. "
        "Сформулируй сложную задачу, требующую доказать, вывести или решить это уравнение. "
        "Сначала выдай сформулированную задачу, а затем напиши подробное итоговое решение. "
        "Для всех математических символов и формул используй СТРОГИЙ синтаксис LaTeX."
    )

    SYSTEM_TEMPLATE = (
        "You are an AI assistant. Format your answers as follows: "
        "<Thought> Your thoughts (understanding, reasoning) </Thought> <output> Your answer </output>"
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

    def _parse_response(self, choice: dict) -> Optional[Dict]:
        """Извлекает prompt/thought/response из ответа модели.

        Работает с reasoning-моделями, которые отдают:
        - reasoning_content — нативные мысли модели (llama.cpp)
        - content — финальный текстовый ответ
        """
        full_response_text = choice.get("content", "").strip()

        thought_text = choice.get("reasoning_content", "").strip()

        # Страховка: если сервер отдал мысли в другом поле
        if not thought_text:
            thought_text = choice.get("data", {}).get("reasoning_content", "").strip()

        if full_response_text and thought_text:
            return {
                "prompt": f"На основе фрагмента статьи решите аналитическую задачу: ...",
                "thought": thought_text,
                "response": full_response_text,
            }
        elif full_response_text:
            return {
                "prompt": f"Решите аналитическую задачу на основе текста: ...",
                "thought": "Анализ предоставленного контекста.",
                "response": full_response_text,
            }
        return None

    def _query_server(self, server_url: str, chunk: str, system_prompt: str) -> Optional[Dict]:
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
                return self._parse_response(choice)
            else:
                self.logger.warning(f"Сервер {server_url} вернул код {response.status_code}: {response.text}")
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url}: {e}")

        return None

    def _generate_row(self, chunk: str, system_prompt: str) -> Optional[Dict]:
        llm_data = self._query_server(self._get_next_server(), chunk, system_prompt)

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

        self.logger.info(f"Генерация: {len(chunks)} чанков, режим={mode}, серверов={len(self.servers)}")

        dataset_rows = []

        if len(self.servers) > 1:
            dataset_rows = self._generate_multi_server(chunks, system_prompt)
        else:
            dataset_rows = self._generate_single_server(chunks, system_prompt)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Генерация завершена. Сохранено {len(dataset_rows)} строк в {output_file}")
        return dataset_rows

    def _generate_single_server(self, chunks: List[str], system_prompt: str) -> List[Dict]:
        dataset_rows = []

        for i, chunk in enumerate(tqdm(chunks, desc="Генерация (1 сервер)")):
            row = self._generate_row(chunk, system_prompt)
            if row:
                dataset_rows.append(row)

            if len(dataset_rows) % self.save_interval == 0:
                # Автосохранение каждые save_interval строк
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

        return dataset_rows

    def _generate_multi_server(self, chunks: List[str], system_prompt: str) -> List[Dict]:
        dataset_rows = []
        server_queue = Queue()
        for server in self.servers:
            server_queue.put(server)

        def worker(chunk):
            server = server_queue.get()
            try:
                return self._query_server(server, chunk, system_prompt)
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
