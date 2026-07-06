import os
import json
import re
import requests
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm

from npi.utils.logger import get_logger, progress_bar


class DatasetGenerator:
    """Генерация датасета через API llama.cpp (1 сервер или мульти-сервер)."""

    DEFAULT_SYSTEM_PROMPT = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи, "
        "придумать сложный аналитический вопрос к нему, детально расписать логику "
        "рассуждения и выдать ответ. Ты должен вернуть результат СТРОГО в формате JSON "
        "со следующими ключами: 'prompt', 'thought', 'response'."
    )

    MATH_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент научной статьи "
        "с формулами в формате LaTeX. Выбери из текста ключевое математическое уравнение или теоретический вывод. "
        "Сформулируй сложную задачу, требующую доказать, вывести или решить это уравнение. "
        "В поле 'thought' пошагово распиши математическую логику решения, промежуточные преобразования и законы. "
        "В поле 'response' запиши финальный структурированный ответ и конечную формулу. "
        "И в вопросе, и в рассуждениях, и в ответе используй СТРОГИЙ синтаксис LaTeX для математических символов. "
        "Выведи результат СТРОГО в формате JSON с ключами: 'prompt', 'thought', 'response'."
    )

    SYSTEM_TEMPLATE = (
        "You are an AI assistant. Format your answers as follows: "
        "<Thought> Your thoughts (understanding, reasoning) </Thought> <output> Your answer </output>"
    )

    def __init__(
        self,
        servers: List[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
        save_interval: int = 10,
        timeout: int = 120,
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

    def _query_server(self, server_url: str, chunk: str, system_prompt: str) -> Optional[Dict]:
        user_prompt = f"Фрагмент научной публикации:\n\"\"\"\n{chunk}\n\"\"\""

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
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
                content_str = result_json["choices"][0]["message"]["content"]
                return json.loads(content_str)
            else:
                self.logger.warning(f"Сервер {server_url} вернул код {response.status_code}")
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
        output_file = None

        for i, chunk in enumerate(tqdm(chunks, desc="Генерация (1 сервер)")):
            row = self._generate_row(chunk, system_prompt)
            if row:
                dataset_rows.append(row)

            if len(dataset_rows) % self.save_interval == 0:
                output_file = dataset_rows[-self.save_interval:] if len(dataset_rows) >= self.save_interval else dataset_rows
                # save partial is handled in generate_from_chunks

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
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки чанка: {e}")

        return dataset_rows
