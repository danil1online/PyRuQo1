import os
import re
import json
import requests
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# Импортируем компоненты нейросетевого парсера формул
from marker.convert import convert_single_pdf
from marker.models import load_models



# ==========================================
# 1. КОНФИГУРАЦИЯ СЕРВЕРОВ И ПУТЕЙ
# ==========================================
OUTPUT_FILE = "university_math_dataset.json"
PDF_FOLDER = "./university_pdfs_math" # Положите сюда ваши PDF

# ПОЛНЫЕ ИСПРАВЛЕННЫЕ АДРЕСА СЕРВЕРОВ llama.cpp
SERVERS_POOL = [
    "http://195.133.13.56:8079/v1/chat/completions",
    "http://195.63.145.3:8078/v1/chat/completions"
]

# Потокобезопасная очередь для свободных серверов
available_servers = queue.Queue()
for server in SERVERS_POOL:
    available_servers.put(server)

# ==========================================
# 2. МАТЕМАТИЧЕСКИЙ ПАРСИНГ И ЧАНКИНГ
# ==========================================
print("--> Загрузка нейросетей распознавания математического текста (Marker)...")
# Инициализация моделей Marker (выполняется 1 раз при старте)
marker_models = load_models()

def parse_pdf_to_math_markdown(file_path):
    """Превращает PDF в Markdown с сохранением сложных формул в формате LaTeX"""
    try:
        # Нейросетевое извлечение текста и формул (работает и для цифровых PDF, и для сканов)
        full_text, _, _ = convert_single_pdf(file_path, marker_models)
        
        # Отсекаем список литературы, чтобы не забивать контекст мусором
        lit_pattern = r'\b(Список литературы|References|Список источников)\b'
        if re.search(lit_pattern, full_text, re.IGNORECASE):
            full_text = re.split(lit_pattern, full_text, flags=re.IGNORECASE)[0]
            
        return full_text.strip()
    except Exception as e:
        print(f"\n[Ошибка] Не удалось распознать формулы в файле {os.path.basename(file_path)}: {e}")
        return ""

def split_math_chunks(text, max_chars=3500, overlap=500):
    """Нарезает текст на чанки, защищая блоки $$...$$ от разрыва формул посередине"""
    # Токенизируем текст, выделяя LaTeX блоки формул
    tokens = re.split(r'(\$\$.*?\$\$|\$.*?\$)', text, flags=re.DOTALL)
    
    chunks = []
    current_chunk = ""
    
    for token in tokens:
        # Если добавление следующего куска превышает лимит — закрываем чанк
        if len(current_chunk) + len(token) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Делаем нахлест (overlap) из конца предыдущего фрагмента для сохранения контекста
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else ""
            
        current_chunk += token
        
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

# ==========================================
# 3. ПОТОКОБЕЗОПАСНЫЙ ЗАПРОС К API СЕРВЕРОВ
# ==========================================
def worker_query_api(context_chunk):
    """
    Адаптированная функция для нативных o1/Reasoning моделей (Qwen 35B).
    Извлекает нативные мысли из reasoning_content и собирает идеальный датасет.
    """
    server_url = available_servers.get()
    
    # Системный промпт теперь простой и текстовый. Модели больше НЕ нужно генерировать JSON.
    system_prompt = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент научной статьи "
        "с формулами в формате LaTeX. Выбери из текста ключевое математическое уравнение или теоретический вывод. "
        "Сформулируй сложную задачу, требующую доказать, вывести или решить это уравнение. "
        "Сначала выдай сформулированную задачу, а затем напиши подробное итоговое решение. "
        "Для всех математических символов и формул используй СТРОГИЙ синтаксис LaTeX."
    )
    
    user_prompt = f"Фрагмент научной публикации с LaTeX-формулами:\n\"\"\"\n{context_chunk}\n\"\"\""
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2, # Низкая температура для строгости математики
        "max_tokens": 2500  # Даем запас токенов на вывод формул
        # "response_format" УБРАН НАВСЕГДА — он ломал reasoning-модель!
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        # Ставим таймаут в 5 минут, так как 35B модель на сложных чанках может думать долго
        response = requests.post(server_url, headers=headers, json=payload, timeout=300)
        
        if response.status_code == 200:
            result_json = response.json()
            choice = result_json["choices"][0]["message"]
            
            # 1. Извлекаем финальный текстовый ответ (там будут и задача, и решение)
            full_response_text = choice.get("content", "").strip()
            
            # 2. ИЗВЛЕКАЕМ НАТИВНЫЕ МЫСЛИ МОДЕЛИ (Reasoning)
            # Если llama.cpp отдает их в reasoning_content (как в curl)
            thought_text = choice.get("reasoning_content", "").strip()
            
            # Страховка: если сервер отдал мысли в другом поле
            if not thought_text:
                thought_text = choice.get("data", {}).get("reasoning_content", "").strip()
                
            # Если модель сгенерировала глубокие мысли
            if full_response_text and thought_text:
                # Пакуем данные в формат вашего датасета
                return {
                    "prompt": f"На основе фрагмента статьи решите аналитическую задачу: {context_chunk[:150]}...",
                    "thought": thought_text,
                    "response": full_response_text
                }
            elif full_response_text:
                # Если мыслей нет (вдруг отключились), пишем базовый лог рассуждения
                return {
                    "prompt": f"Решите математическую задачу на основе текста: {context_chunk[:150]}...",
                    "thought": "Анализ предоставленного математического контекста и извлечение формул.",
                    "response": full_response_text
                }
        else:
            print(f"\n[Ошибка API] Сервер {server_url} вернул код {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"\n[Сетевой сбой] Ошибка связи с сервером {server_url}: {e}")
    finally:
        # В любом случае возвращаем сервер в пул свободных
        available_servers.put(server_url)
        
    return None


# ==========================================
# 4. ДИСПЕТЧЕР КОНВЕЙЕРА (MAIN)
# ==========================================
def main():
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Создана папка {PDF_FOLDER}. Положите туда ваши PDF-файлы со статьями и запустите скрипт снова.")
        return

    # Сканируем папку на наличие PDF
    pdf_files = []
    for root, _, files in os.walk(PDF_FOLDER):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, f))
                
    if not pdf_files:
        print(f"В папке {PDF_FOLDER} не найдено ни одного PDF файла.")
        return
                
    print(f"--> Шаг 1: Распознавание формул и извлечение LaTeX из {len(pdf_files)} файлов...")
    all_chunks = []
    for file_path in pdf_files:
        print(f"Анализ разметки в: {os.path.basename(file_path)}")
        text = parse_pdf_to_math_markdown(file_path)
        if len(text) > 300:
            all_chunks.extend(split_math_chunks(text))
            
    if not all_chunks:
        print("--> Критическая ошибка: Не удалось извлечь математический текст. Завершение.")
        return

    dataset_rows = []
    num_threads = len(SERVERS_POOL)
    print(f"--> Шаг 2: Сформировано {len(all_chunks)} защищенных чанков. Запуск параллельной генерации на {num_threads} серверах...")
    
    # Запускаем пул потоков (строго по 1 потоку на каждый физический сервер)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker_query_api, chunk): chunk for chunk in all_chunks}
        
        for future in tqdm(as_completed(futures), total=len(all_chunks), desc="Математическая o1-генерация"):
            llm_data = future.result()
            
            # Проверяем структуру ответа сервера
            if llm_data and all(k in llm_data for k in ["prompt", "thought", "response"]):
                # Форматируем данные точь-в-точь как в оригинальном Dataset_of_Russian_thinking
                row = {
                    "system": "You are an AI assistant. Format your answers as follows: <Thought> Your thoughts (understanding, reasoning) </Thought> <output> Your answer </output>",
                    "prompt": llm_data["prompt"],
                    "response": f"<Thought>\n{llm_data['thought']}\n</Thought>\n<output>\n{llm_data['response']}\n</output>"
                }
                dataset_rows.append(row)
                
                # Автосохранение каждые 20 шагов для защиты прогресса от сбоев
                if len(dataset_rows) % 20 == 0:
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                        
    # Финальная фиксация данных на диск
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
        
    print(f"\n--> Конвейер успешно завершен! Собрано математических пар: {len(dataset_rows)}")
    print(f"Файл {OUTPUT_FILE} готов для передачи в train_qlora.py.")

if __name__ == "__main__":
    main()