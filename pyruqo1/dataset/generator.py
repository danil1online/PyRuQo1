import os
import json
import re
import requests
import time
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm
from pyruqo1.utils.logger import get_logger, progress_bar


class DatasetGenerator:
    """Двухэтапная генерация датасета с поддержкой Мульти-серверов (llama.cpp) и GigaChat API.
    
    По умолчанию (если servers=None) используется GigaChat от Сбера.
    """
    
    # --- Базовые инструкции для Этапа 1 (Вопросы всегда короткие, без мыслей) ---
    DEFAULT_QUESTION_SYSTEM_PROMPT = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи и "
        "придумать к нему ОДИН короткий, емкий аналитический вопрос (1-2 предложения). "
        "Вопрос должен быть сформулирован максимально лаконично.\n"
        "Отвечай строго текстом вопроса, без вводных слов, кавычек и форматирования JSON."
    )
    
    MATH_QUESTION_SYSTEM_PROMPT = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
        "научной статьи с формулами в формате LaTeX. Выбери из текста ключевое математическое "
        "уравнение или теоретический вывод и сформулируй сложную, но лаконичную задачу (1-2 предложения), "
        "требующую доказать, вывести или решить это уравнение.\n"
        "Для всех математических символов и формул используй СТРОГИЙ синтаксис LaTeX."
    )
 
    def __init__(
        self,
        servers: Optional[List[str]] = None,
        context_size: int = 2048,
        temperature: float = 0.2,
        max_tokens: int = 2500,
        save_interval: int = 20,
        timeout: int = 300,
        gigachat_model: str = "GigaChat"
    ):
        self.context_size = context_size 
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.save_interval = save_interval
        self.timeout = timeout
        self.logger = get_logger()
        self._server_index = 0
        self.gigachat_model = gigachat_model
        self.gigachat_client = None

        if servers is None:
            self.servers = ["gigachat"]
        else:
            self.servers = servers

        # Режим GigaChat
        if "gigachat" in self.servers:
            # ЛЕНИВЫЙ ИМПОРТ: Проверяем наличие библиотеки в виртуальном окружении
            try:
                from gigachat import GigaChat
                from gigachat.models import Chat
            except ImportError:
                raise ImportError(
                    "Выбран режим GigaChat, но библиотека не установлена в текущем окружении. "
                    "Выполните: pip install gigachat"
                )
            
            credentials = os.getenv("GIGACHAT_CREDENTIALS")
            if not credentials:
                print("\n" + "="*60)
                print("🔑 КЛЮЧ АВТОРИЗАЦИИ GIGACHAT НЕ НАЙДЕН!")
                print("Пожалуйста, введите ваш GigaChat Authorization Key (Client Secret):")
                credentials = input("> ").strip()
                print("="*60 + "\n")
                if not credentials:
                    raise ValueError("Критическая ошибка: GigaChat Authorization Key не может быть пустым.")
                os.environ["GIGACHAT_CREDENTIALS"] = credentials

            # Инициализируем клиент без проверки SSL
            self.gigachat_client = GigaChat(credentials=credentials, verify_ssl_certs=False, timeout=self.timeout)
            self.logger.info(f"DatasetGenerator инициализирован в режиме GigaChat ({self.gigachat_model})")
        else:
            self.logger.info(f"DatasetGenerator инициализирован в режиме кастомных серверов: {self.servers}")

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
        """Главный управляющий метод: координирует двухэтапный процесс генерации."""
        
        # 1. ТИПОВЫЕ СИСТЕМНЫЕ ПРОМПТЫ ДЛЯ СОХРАНЕНИЯ В ДАТАСЕТ (Короткие и чистые)
        if mode == "math":
            dataset_system_prompt = (
                "Ты — профессор высшей математики и теоретической физики. Перед тобой сложная "
                "научно-теоретическая задача. Пошагово распиши математическую логику решения, "
                "используя синтаксис LaTeX, и дай развернутый академический ответ."
            )
        else:
            dataset_system_prompt = (
                "Ты — ведущий научный методолог. Перед тобой сложный аналитический вопрос. "
                "Детально распиши логику рассуждения и дай содержательный, академический ответ."
            )

        # Настройка системного промпта для Вопросов (Этап 1)
        question_system_prompt = (
            self.DEFAULT_QUESTION_SYSTEM_PROMPT if mode == "simple" else self.MATH_QUESTION_SYSTEM_PROMPT
        )
        
        # Настройка ограничений длины для API
        if self.context_size == 2048:
            length_instruction = "Пиши рассуждения лаконично, ограничь скрытый блок мыслей максимум 2-3 абзацами. Конечный ответ сделай кратким."
        else:
            length_instruction = "Проведи глубокую декомпозицию вопроса, сопоставь все факты. Разверни подробную цепочку шагов."

        # СТРОГОЕ ТРЕБОВАНИЕ К ФОРМАТУ И ПОВЕДЕНИЮ РАССУЖДЕНИЙ
        format_instruction = (
            "Ты ОБЯЗАН структурировать ответ строго с помощью XML-тегов. "
            "Сначала открой тег начала мысли <Thought>, детально распиши логику и закрой тегом конца мысли </Thought>. "
            "Затем открой тег начала вывода <output>, запиши туда итоговый академический ответ и закрой тегом </output>.\n"
            "КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К СТИЛЮ РАССУЖДЕНИЙ:\n"
            "Строий свои размышления в блоке <Thought> так, будто ты отвечаешь из своей фундаментальной памяти и широких экспертных знаний. "
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать фразы: 'согласно предоставленному фрагменту', 'автор статьи указывает', 'в тексте подчеркивается', 'на основе статьи' и любые их аналоги. "
            "Представь, что никакого фрагмента перед тобой нет — пиши рассуждения от первого лица как независимый эксперт, аргументируя логические тезисы.\n"
            "КРИТИЧЕСКОЕ ПРАВИЛО ФОРМАТА: Внутри текста рассуждений никогда не дублируй и не цитируй названия самих этих тегов "
            "в кавычках или в виде примеров, пиши сразу суть анализа!"
        )

        # 2. РАБОЧИЕ ДЛИННЫЕ ПРОМПТЫ ДЛЯ ОТПРАВКИ В API GIGACHAT
        if mode == "math":
            api_system_prompt = (
                "ВНИМАНИЕ: Все рассуждения и весь ответ должны быть СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование английского языка запрещено.\n"
                "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
                "научной статьи с формулами в формате LaTeX и математическая задача. Подробно решите "
                "эту задачу, пошагово расписав математическую логику решения, промежуточные преобразования и законы.\n"
                f"{format_instruction}\n"
                f"ТРЕБОВАНИЕ К РАЗМЕРУ: {length_instruction}"
            )
        else:
            api_system_prompt = (
                "ВНИМАНИЕ: Все рассуждения и весь ответ должны быть СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование английского языка запрещено.\n"
                "Ты — ведущий научный методолог. Перед тобой фрагмент научной статьи и аналитический вопрос к нему. "
                "Детально распиши логику рассуждения и дай ответ.\n"
                f"{format_instruction}\n"
                f"ТРЕБОВАНИЕ К РАЗМЕРУ: {length_instruction}"
            )

        user_prompt_template = "Фрагмент научной публикации с LaTeX-формулами:" if mode == "math" else "Фрагмент научной публикации:"
        
        self.logger.info(
            f"Генерация: {len(chunks)} чанков, лимит контекста={self.context_size}, режим={mode}, серверов={len(self.servers)}"
        )
        
        # Этап 1: генерация вопросов
        questions = self._generate_questions(chunks, question_system_prompt, output_file, user_prompt_template)
        self.logger.info(f"Этап 1 завершён. Сгенерировано {len(questions)} вопросов.")

        # Этап 2: генерация ответов. ПЕРЕДАЕМ И РАБОЧИЙ ПРОМПТ ДЛЯ API, И ЧИСТЫЙ ПРОМПТ ДЛЯ СОХРАНЕНИЯ В JSON
        dataset_rows = self._generate_answers(
            chunks, questions, api_system_prompt, dataset_system_prompt, output_file, user_prompt_template
        )

        # Удаляем временный файл вопросов
        tmp_file = output_file + ".tmp_questions"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"Генерация завершена. Сохранено {len(dataset_rows)} строк в {output_file}")
        return dataset_rows

    def _generate_questions(self, chunks: List[str], system_prompt: str, output_file: str, user_prompt_template: str) -> Dict[int, str]:
        """Точка входа для Этапа 1. Проверяет, работаем ли мы с GigaChat."""
        if "gigachat" not in self.servers:
            return self._generate_questions_servers(chunks, system_prompt, output_file, user_prompt_template)
        
        # Режим GigaChat (прямая многопоточность)
        questions = {}
        with ThreadPoolExecutor(max_workers=2) as executor: # Снизили количество воркеров до 2
            futures = {
                executor.submit(self._query_gigachat, system_prompt, f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"", max_tokens=400): i
                for i, chunk in enumerate(chunks)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Этап 1: генерация вопросов (GigaChat)"):
                idx = futures[future]
                try:
                    question = future.result()
                    if question:
                        questions[idx] = question.strip().strip('"').strip("'")
                    if len(questions) % self.save_interval == 0:
                        self._save_questions_to_tmp(output_file, questions)
                except Exception as e:
                    self.logger.warning(f"Ошибка этапа 1 для чанка {idx}: {e}")
        return questions

    def _generate_questions_servers(
        self,
        chunks: List[str],
        system_prompt: str,
        output_file: str,
        user_prompt_template: str,
    ) -> Dict[int, str]:
        """Генерация коротких вопросов через пул кастомных инференс-серверов."""
        questions = {}
        server_queue = Queue()
        for server in self.servers:
            server_queue.put(server)

        def worker(idx, chunk):
            server = server_queue.get()
            try:
                user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\""
                return idx, self._query_server_question(server, system_prompt, user_prompt)
            finally:
                server_queue.put(server)

        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {executor.submit(worker, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Этап 1: генерация вопросов (Серверы)"):
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

    def generate_from_chunks(
        self,
        chunks: List[str],
        output_file: str,
        mode: str = "simple",
    ) -> List[Dict]:
        """Главный управляющий метод: координирует двухэтапный процесс генерации."""
        # Настройка системного промпта для Вопросов
        question_system_prompt = (
            self.DEFAULT_QUESTION_SYSTEM_PROMPT if mode == "simple" else self.MATH_QUESTION_SYSTEM_PROMPT
        )
        
        # Настройка системного промпта для Ответов (динамическая длина)
        if self.context_size == 2048:
            length_instruction = "Пиши рассуждения лаконично, ограничь скрытый блок мыслей максимум 2-3 абзацами. Конечный ответ сделай кратким."
        else: # 8192
            length_instruction = "Проведи глубокую декомпозицию вопроса, сопоставь все факты. Разверни подробную цепочку шагов."

        # СТРОГОЕ ТРЕБОВАНИЕ К ФОРМАТУ БЕЗ ПОВТОРЕНИЯ НАЗВАНИЙ ТЕГОВ ТЕКСТОМ
        format_instruction = (
            "Ты ОБЯЗАН структурировать ответ строго с помощью XML-тегов. "
            "Сначала открой тег начала мысли <Thought>, детально распиши логику и закрой тегом конца мысли </Thought>. "
            "Затем открой тег начала вывода <output>, запиши туда итоговый академический ответ и закрой тегом </output>.\n"
            "КРИТИЧЕСКОЕ ПРАВИЛО: Внутри текста рассуждений никогда не дублируй и не цитируй названия самих этих тегов "
            "в кавычках или в виде примеров, пиши сразу суть анализа!"
        )

        # Формируем системные промпты под требования русской локализации и архитектуры тегов
        if mode == "math":
            answer_system_prompt = (
                "ВНИМАНИЕ: Все рассуждения и весь ответ должны быть СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование английского языка запрещено.\n"
                "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент "
                "научной статьи с формулами в формате LaTeX и математическая задача. Подробно решите "
                "эту задачу, пошагово расписав математическую логику решения, промежуточные преобразования и законы.\n"
                f"{format_instruction}\n"
                f"ТРЕБОВАНИЕ К РАЗМЕРУ: {length_instruction}"
            )
        else:
            answer_system_prompt = (
                "ВНИМАНИЕ: Все рассуждения и весь ответ должны быть СТРОГО НА РУССКОМ ЯЗЫКЕ. Использование английского языка запрещено.\n"
                "Ты — ведущий научный методолог. Перед тобой фрагмент научной статьи и аналитический вопрос к нему. "
                "Детально распиши логику рассуждения и дай ответ.\n"
                f"{format_instruction}\n"
                f"ТРЕБОВАНИЕ К РАЗМЕРУ: {length_instruction}"
            )

        user_prompt_template = "Фрагмент научной публикации с LaTeX-формулами:" if mode == "math" else "Фрагмент научной публикации:"
        
        self.logger.info(
            f"Генерация: {len(chunks)} чанков, лимит контекста={self.context_size}, режим={mode}, серверов={len(self.servers)}"
        )
        
        # Этап 1: генерация вопросов
        questions = self._generate_questions(chunks, question_system_prompt, output_file, user_prompt_template)
        self.logger.info(f"Этап 1 завершён. Сгенерировано {len(questions)} вопросов.")

        # Этап 2: генерация ответов
        dataset_rows = self._generate_answers(chunks, questions, answer_system_prompt, output_file, user_prompt_template)

        # Удаляем временный файл вопросов
        tmp_file = output_file + ".tmp_questions"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"Генерация завершена. Сохранено {len(dataset_rows)} строк в {output_file}")
        return dataset_rows

    def _generate_answers(
        self, 
        chunks: List[str], 
        questions: Dict[int, str], 
        api_system_prompt: str, 
        dataset_system_prompt: str, # Принимаем чистый промпт
        output_file: str, 
        user_prompt_template: str
    ) -> List[Dict]:
        """Точка входа для Этапа 2. Маршрутизирует генерацию CoT-ответов."""
        if "gigachat" not in self.servers:
            # Если работаем с кастомными серверами, прокидываем логику туда (при необходимости)
            return self._generate_answers_servers(chunks, questions, api_system_prompt, output_file, user_prompt_template)

        # Режим GigaChat для ответов
        dataset_rows = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(
                    self._query_gigachat, 
                    api_system_prompt, # Модели отправляем сложный длинный промпт с инструкциями
                    f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"\n\nВопрос: {questions[i]}", 
                    max_tokens=self.max_tokens
                ): i
                for i, chunk in enumerate(chunks) if i in questions
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Этап 2: CoT-ответов (GigaChat)"):
                idx = futures[future]
                try:
                    full_response = future.result()
                    if full_response:
                        # В ДАТАСЕТ сохраняем ЧИСТЫЙ, короткий системный промпт
                        dataset_rows.append({
                            "system": dataset_system_prompt, 
                            "prompt": questions[idx],
                            "response": full_response,
                        })
                        if len(dataset_rows) % self.save_interval == 0:
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    self.logger.warning(f"Ошибка этапа 2 для чанка {idx}: {e}")
        return dataset_rows

    def _generate_answers_servers(
        self, 
        chunks: List[str], 
        questions: Dict[int, str], 
        system_prompt: str, 
        output_file: str, 
        user_prompt_template: str
    ) -> List[Dict]:
        """Генерация CoT-ответов через пул ваших локальных инференс-серверов."""
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
                user_prompt = f"{user_prompt_template}\n\"\"\"\n{chunk}\n\"\"\"\n\nВопрос: {question}"
                row = self._generate_answer_row_server(server, question, system_prompt, user_prompt)
                return idx, row
            finally:
                server_queue.put(server)

        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {executor.submit(worker, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Этап 2: генерация ответов (Серверы)"):
                idx = futures[future]
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

    def _query_gigachat(self, system_prompt: str, user_prompt: str, max_tokens: int) -> Optional[str]:
        """Прямой синхронный запрос к API GigaChat с защитой от 429 и универсальным парсингом словаря."""
        import time
        
        payload = {
            "model": self.gigachat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(5):
            try:
                time.sleep(0.5) # Пауза между запросами
                
                res = self.gigachat_client.chat(payload)
                
                # --- УНИВЕРСАЛЬНЫЙ И БЕЗОПАСНЫЙ ПАРСИНГ ОТВЕТА СБЕРА ---
                if res and hasattr(res, "choices") and res.choices:
                    # Извлекаем первый элемент из списка choices
                    first_choice = res.choices[0]
                    
                    # Проверяем, вернулся ли словарь (dict) или объект библиотеки
                    if isinstance(first_choice, dict):
                        message = first_choice.get("message", {})
                        if isinstance(message, dict):
                            return message.get("content", "").strip()
                    else:
                        # Если это объект класса, берем атрибуты напрямую
                        if hasattr(first_choice, "message"):
                            return first_choice.message.content.strip()
                            
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    sleep_time = (attempt + 1) * 3
                    self.logger.warning(f"GigaChat Rate Limit (429). Повтор через {sleep_time} сек...")
                    time.sleep(sleep_time)
                    continue
                else:
                    self.logger.warning(f"Ошибка вызова GigaChat API: {e}")
                    break
        return None

    def _query_server_question(self, server_url: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Запрос короткого вопроса к вашему инференс-серверу (Этап 1)."""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nВыдай ответ СТРОГО в формате JSON с ключом 'prompt': {{\"prompt\": \"...\"}}"}
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "reasoning_budget": 0,
            "thinking_budget_tokens": 0,
            "response_format": {"type": "json_object"}
        }
        try:
            response = requests.post(server_url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                return self._parse_question_response(response.json()["choices"][0]["message"])
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url} (этап 1): {e}")
        return None

    def _generate_answer_row_server(self, server_url: str, question: str, system_prompt: str, user_prompt: str) -> Optional[Dict]:
        """Запрос полного CoT-ответа к вашему инференс-серверу с пост-трансляцией (Этап 2)."""
        target_max_tokens = 1200 if self.context_size == 2048 else 5000
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.6,
            "max_tokens": target_max_tokens,
        }
        try:
            response = requests.post(server_url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                message_data = response.json()["choices"][0]["message"]
                reasoning = message_data.get("reasoning_content", "").strip()
                final_output = message_data.get("content", "").strip()
                
                if reasoning:
                    full_response = f"<Thought>\n{reasoning}\n</Thought>\n<output>\n{final_output}\n</output>"
                else:
                    full_response = final_output
                
                # Очистка вопроса
                clean_question = question[-1] if isinstance(question, (list, tuple)) else question
                clean_question = str(clean_question).strip().strip('"').strip("'")
                
                # Запуск пайплайна перевода (только для кастомных/англоязычных локальных моделей)
                final_response = self._process_translation_pipeline(server_url, full_response)
                
                return {
                    "system": system_prompt,
                    "prompt": clean_question,
                    "response": final_response,
                }
        except Exception as e:
            self.logger.warning(f"Ошибка запроса к {server_url} (этап 2): {e}")
        return None

    def _parse_question_response(self, choice: dict) -> Optional[str]:
        content_str = choice.get("content", "").strip()
        if not content_str: return None
        content_str = re.sub(r"^(?:json)?\s*", "", content_str)
        content_str = re.sub(r"\s*$", "", content_str)
        content_str = content_str.rstrip('",')
        try:
            data = json.loads(content_str)
            return data.get("prompt", "").strip()
        except json.JSONDecodeError:
            match = re.search(r'"prompt"\s*:\s*"([^"]+)"', content_str, re.DOTALL)
            if match: return match.group(1).strip()
        return None

    def _needs_translation(self, text: str, threshold: float = 0.3) -> bool:
        if not text: return False
        letters = [c for c in text if c.isalpha()]
        if not letters: return False
        eng_letters = sum(1 for c in letters if c.lower() in 'abcdefghijklmnopqrstuvwxyz')
        return (eng_letters / len(letters)) > threshold

    def _translate_block(self, server_url: str, text: str) -> str:
        if not text.strip() or not self._needs_translation(text): return text
        safe_text = text.replace("<Thought>", "=== START_THOUGHT ===").replace("</Thought>", "=== END_THOUGHT ===")
        safe_text = safe_text.replace("<output>", "=== START_OUTPUT ===").replace("</output>", "=== END_OUTPUT ===")
        
        system_prompt = (
            "Ты — профессиональный ИИ-переводчик научных публикаций. Переведи предоставленный текст на русский язык. "
            "КРИТИЧЕСКИ ВАЖНО: сохраняй разметку и маркеры структуры (=== START_THOUGHT ===, === END_THOUGHT ===) без изменений. "
            "Do NOT use reasoning. Provide the final translation immediately."
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Переведи этот научно-аналитический текст на русский язык:\n\n{safe_text}"}
            ],
            "temperature": 0.1, "max_tokens": 3500, "stop": ["<think>", "</think>"]
        }
        try:
            response = requests.post(server_url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                translated_text = response.json()["choices"][0]["message"].get("content", "").strip()
                return translated_text.replace("=== START_THOUGHT ===", "<Thought>").replace("=== END_THOUGHT ===", "</Thought>").replace("=== START_OUTPUT ===", "<output>").replace("=== END_OUTPUT ===", "</output>")
        except Exception as e:
            self.logger.warning(f"Ошибка перевода блока: {e}")
        return text

    def _process_translation_pipeline(self, server_url: str, full_response: str) -> str:
        if "Требование к формату:" in full_response:
            clean_response = full_response.split("Требование к формату:", 1)[1]
        else:
            clean_response = full_response
            
        thought_match = re.search(r"<Thought>(.*?)</Thought>", clean_response, re.DOTALL | re.IGNORECASE)
        output_match = re.search(r"<output>(.*?)</output>", full_response, re.DOTALL | re.IGNORECASE)
        
        if thought_match and output_match:
            translated_thought = self._translate_block(server_url, thought_match.group(1).strip())
            translated_output = self._translate_block(server_url, output_match.group(1).strip())
            return f"<Thought>\n{translated_thought}\n</Thought>\n<output>\n{translated_output}\n</output>"
        
        if self._needs_translation(full_response):
            return self._translate_block(server_url, full_response)
        return full_response

    def _save_questions_to_tmp(self, output_file: str, questions: Dict[int, str]) -> None:
        with open(output_file + ".tmp_questions", "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False)