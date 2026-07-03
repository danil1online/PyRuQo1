import os
import re
import json
import requests
import fitz  # PyMuPDF
from tqdm import tqdm

# ==========================================
# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ ПУТЕЙ
# ==========================================
PDF_FOLDER = "./university_pdfs"          # Папка с вашими PDF-файлами
OUTPUT_FILE = "university_thinking_dataset.json" # Итоговый файл датасета

# Адрес вашего сервера llama.cpp с моделью o1
API_URL = "http://195.133.13.56:8079/v1/chat/completions"

# ==========================================
# 2. МОДУЛЬ ПАРСИНГА И ОЧИСТКИ ТЕКСТА
# ==========================================
def clean_academic_text(text):
    """Очищает текст от переносов, мусора верстки и литературы"""
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) # Склейка переносов
    text = re.sub(r'\s+', ' ', text)                     # Лишние пробелы
    text = re.sub(r'\[\d+(?:[\s,-]*\d+)*\]', '', text)   # Ссылки на литературу [1]
    return text.strip()

def process_and_parse_pdf(file_path):
    """Читает PDF, при необходимости запускает OCR для сканов"""
    try:
        doc = fitz.open(file_path)
        full_text = []
        for page in doc:
            text = page.get_text("text")
            # Если дошли до списка литературы — прекращаем чтение файла
            if re.search(r'\b(Список литературы|References|Список источников)\b', text, re.IGNORECASE):
                split_text = re.split(r'\b(Список литературы|References|Список источников)\b', text, flags=re.IGNORECASE)
                full_text.append(split_text[0])
                break
            full_text.append(text)
        doc.close()
        
        extracted = clean_academic_text(" ".join(full_text))
        
        # Если извлечено слишком мало букв, скорее всего это скан без текстового слоя
        if len(extracted) < 200:
            print(f"\n[OCR] Файл {os.path.basename(file_path)} похож на скан. Запуск распознавания...")
            ocr_output = file_path.replace(".pdf", "_ocr.pdf")
            # Вызываем системный ocrmypdf (распознает русский язык)
            os.system(f"ocrmypdf '{file_path}' '{ocr_output}' -l rus --quiet")
            
            # Читаем уже распознанный файл
            if os.path.exists(ocr_output):
                doc = fitz.open(ocr_output)
                full_text = [page.get_text("text") for page in doc]
                doc.close()
                os.remove(ocr_output) # Удаляем временный файл
                return clean_academic_text(" ".join(full_text))
        return extracted
    except Exception as e:
        print(f"\n Ошибка парсинга {file_path}: {e}")
        return ""

def split_to_chunks(text, chunk_size=3500, overlap=500):
    """Режет текст на куски с перекрытием (overlap)"""
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
# 3. МОДУЛЬ ЗАПРОСОВ К СЕРВЕРУ LLM (API)
# ==========================================
def query_o1_server(context_chunk):
    """Отправляет чанк в API llama.cpp и просит вернуть JSON с мыслями и ответом"""
    
    system_prompt = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи, "
        "придумать сложный аналитический вопрос к нему, детально расписать логику "
        "рассуждения и выдать ответ. Ты должен вернуть результат СТРОГО в формате JSON "
        "со следующими ключами: 'prompt', 'thought', 'response'."
    )
    
    user_prompt = f"Фрагмент научной публикации:\n\"\"\"\n{context_chunk}\n\"\"\""
    
    headers = {"Content-Type": "application/json"}
    
    # Конфигурация запроса. Так как нам нужен строгий JSON, 
    # активируем Grammar/JSON-mode (поддерживается в llama.cpp API)
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"} # Принуждает сервер отвечать валидным JSON
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            result_json = response.json()
            content_str = result_json["choices"][0]["message"]["content"]
            return json.loads(content_str)
        else:
            print(f"\nОшибка сервера API (Код {response.status_code}): {response.text}")
    except Exception as e:
        print(f"\nОшибка при отправке запроса в API: {e}")
    return None

# ==========================================
# 4. ОСНОВНОЙ ПРОЦЕСС СБОРКИ
# ==========================================
def main():
    if not os.path.exists(PDF_FOLDER):
        print(f"Папка {PDF_FOLDER} не найдена. Создайте её и загрузите туда PDF-документы.")
        return

    # Собираем файлы
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
            
    print(f"--> Шаг 2: Сформировано {len(all_chunks)} чанков. Начинаем генерацию через API...")
    
    dataset_rows = []
    
    # Отправляем чанки на сервер генерации
    for chunk in tqdm(all_chunks, desc="Генерация мыслей через o1"):
        llm_data = query_o1_server(chunk)
        
        if llm_data and all(k in llm_data for k in ["prompt", "thought", "response"]):
            # Упаковываем структуру точь-в-точь как в Dataset_of_Russian_thinking
            row = {
                "system": "You are an AI assistant. Format your answers as follows: <Thought> Your thoughts (understanding, reasoning) </Thought> <output> Your answer </output>",
                "prompt": llm_data["prompt"],
                "response": f"<Thought>\n{llm_data['thought']}\n</Thought>\n<output>\n{llm_data['response']}\n</output>"
            }
            dataset_rows.append(row)
            
            # Периодически сохраняем промежуточный результат на диск (на случай сбоя сети)
            if len(dataset_rows) % 10 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                    
    # Финальное сохранение
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
        
    print(f"\n--> Сборка датасета завершена успешно!")
    print(f"Файл {OUTPUT_FILE} содержит {len(dataset_rows)} строк и готов к загрузке в SFTTrainer.")

if __name__ == "__main__":
    main()
