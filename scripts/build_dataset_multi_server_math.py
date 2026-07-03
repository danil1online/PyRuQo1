import os
import re
import json
import requests
import queue
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



# ==========================================
# 1. КОНФИГУРАЦИЯ СЕРВЕРОВ И ПУТЕЙ
# ==========================================
OUTPUT_FILE = "university_thinking_dataset.json"
PDF_FOLDER = "./university_pdfs" # Положите сюда ваши PDF

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
# 2. ПАРСИНГ И ОЧИСТКА PDF
# ==========================================
def clean_academic_text(text):
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+(?:[\s,-]*\d+)*\]', '', text)
    return text.strip()

def process_and_parse_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        full_text = []
        for page in doc:
            text = page.get_text("text")
            if re.search(r'\b(Список литературы|References|Список источников)\b', text, re.IGNORECASE):
                split_text = re.split(r'\b(Список литературы|References|Список источников)\b', text, flags=re.IGNORECASE)
                full_text.append(split_text[0])
                break
            full_text.append(text)
        doc.close()
        
        extracted = clean_academic_text(" ".join(full_text))
        
        # Если это "слепой скан" - вызываем OCR
        if len(extracted) < 200:
            ocr_output = file_path.replace(".pdf", "_ocr.pdf")
            os.system(f"ocrmypdf '{file_path}' '{ocr_output}' -l rus --quiet")
            if os.path.exists(ocr_output):
                doc = fitz.open(ocr_output)
                full_text = [page.get_text("text") for page in doc]
                doc.close()
                os.remove(ocr_output)
                return clean_academic_text(" ".join(full_text))
        return extracted
    except Exception as e:
        return ""

def split_to_chunks(text, chunk_size=3500, overlap=500):
    chunks = []
    words = text.split(' ')
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = int(overlap / 6)
            current_chunk = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else []
            current_length = sum(len(w) + 1 for w in current_chunk)
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

# ==========================================
# 3. ПОТОКОБЕЗОПАСНЫЙ ЗАПРОС К API
# ==========================================
def worker_query_api(context_chunk):
    server_url = available_servers.get()
    
    system_prompt = (
        "Ты — профессор высшей математики и теоретической физики. Перед тобой фрагмент научной статьи "
        "с формулами в формате LaTeX. Выбери из текста ключевое математическое уравнение или теоретический вывод. "
        "Сформулируй сложную задачу, требующую доказать, вывести или решить это уравнение. "
        "В поле 'thought' пошагово распиши математическую логику решения, промежуточные преобразования и законы. "
        "В поле 'response' запиши финальный структурированный ответ и конечную формулу. "
        "И в вопросе, и в рассуждениях, и в ответе используй СТРОГИЙ синтаксис LaTeX для математических символов. "
        "Выведи результат СТРОГО в формате JSON с ключами: 'prompt', 'thought', 'response'."
    )
    user_prompt = f"Фрагмент научной публикации:\n\"\"\"\n{context_chunk}\n\"\"\""
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            content_str = response.json()["choices"]["message"]["content"]
            return json.loads(content_str)
    except Exception as e:
        print(f"\n[Ошибка] Сервер {server_url} не ответил.")
    finally:
        available_servers.put(server_url)
    return None

# ==========================================
# 4. ДИСПЕТЧЕР И ЗАПУСК
# ==========================================
def main():
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Создана папка {PDF_FOLDER}. Положите туда ваши PDF.")
        return

    pdf_files = []
    for root, _, files in os.walk(PDF_FOLDER):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, f))
                
    print(f"--> Шаг 1: Извлечение текстов из {len(pdf_files)} файлов...")
    all_chunks = []
    for file_path in tqdm(pdf_files, desc="Чтение документов"):
        text = process_and_parse_pdf(file_path)
        if len(text) > 300:
            all_chunks.extend(split_to_chunks(text))
            
    if not all_chunks:
        print("Нет данных для обработки. Завершение.")
        return

    dataset_rows = []
    num_threads = len(SERVERS_POOL)
    print(f"--> Шаг 2: Найдено {len(all_chunks)} чанков. Запуск генерации на {num_threads} серверах...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker_query_api, chunk): chunk for chunk in all_chunks}
        
        for future in tqdm(as_completed(futures), total=len(all_chunks), desc="Асинхронная o1 генерация"):
            llm_data = future.result()
            if llm_data and all(k in llm_data for k in ["prompt", "thought", "response"]):
                row = {
                    "system": "You are an AI assistant. Format your answers as follows: <Thought> Your thoughts (understanding, reasoning) </Thought> <output> Your answer </output>",
                    "prompt": llm_data["prompt"],
                    "response": f"<Thought>\n{llm_data['thought']}\n</Thought>\n<output>\n{llm_data['response']}\n</output>"
                }
                dataset_rows.append(row)
                if len(dataset_rows) % 20 == 0:
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
    print(f"\n--> Генерация завершена. Собрано строк: {len(dataset_rows)}")

if __name__ == "__main__":
    main()
