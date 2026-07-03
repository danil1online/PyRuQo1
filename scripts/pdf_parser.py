import os
import re
import fitz  # Это и есть PyMuPDF
from tqdm import tqdm

def clean_academic_text(text):
    """Очистка научного текста от мусора форматирования"""
    if not text:
        return ""
    
    # 1. Исправляем разорванные дефисами переносы слов на стыке строк
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 2. Заменяем множественные пробелы и переносы на одиночные
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Удаляем явные ссылки на литературу вида, [12, 15], [1-3]
    text = re.sub(r'\[\d+(?:[\s,-]*\d+)*\]', '', text)
    
    # 4. Удаляем ссылки на рисунки и таблицы (например: "см. рис. 1")
    text = re.sub(r'\(см\.\s+рис\.\s+\d+\)|\(табл\.\s+\d+\)', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def parse_pdf_to_text(file_path):
    """Извлекает и очищает текст из одного PDF файла"""
    try:
        doc = fitz.open(file_path)
        full_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Базовая фильтрация: отсекаем список литературы, если он начался
            # Обычно в статьях это разделы "Литература", "References", "Список источников"
            if re.search(r'\b(Список литературы|References|Список источников)\b', text, re.IGNORECASE):
                # Берем текст страницы ДО этого заголовка и завершаем чтение файла
                split_text = re.split(r'\b(Список литературы|References|Список источников)\b', text, flags=re.IGNORECASE)
                full_text.append(split_text[0])
                break
                
            full_text.append(text)
            
        doc.close()
        return clean_academic_text(" ".join(full_text))
    except Exception as e:
        print(f"\nОшибка при чтении файла {file_path}: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=3500, overlap=500):
    """
    Нарезает текст на куски заданной длины с перекрытием (overlap).
    Перекрытие нужно, чтобы модель не теряла контекст на границах кусков.
    """
    chunks = []
    words = text.split(' ')
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1 # +1 для пробела
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Делаем шаг назад для реализации overlap (приблизительно по словам)
            overlap_words = int(overlap / 6) # Считаем, что среднее слово ~6 символов
            current_chunk = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else []
            current_length = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def get_all_university_chunks(root_folder, chunk_size=3500, overlap=500):
    """Сканирует папку, парсит все PDF и возвращает готовый массив чанков"""
    all_chunks = []
    pdf_files = []
    
    # Рекурсивно обходим все подпапки в поисках .pdf
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    print(f"--> Найдено {len(pdf_files)} PDF-файлов для обработки.")
    
    # Запускаем парсинг с визуальной полосой прогресса в консоли
    for file_path in tqdm(pdf_files, desc="Парсинг PDF"):
        file_text = parse_pdf_to_text(file_path)
        
        # Если файл пустой (например, это отсканированная картинка без распознанного слоя текста)
        if len(file_text) < 200:
            continue
            
        file_chunks = split_text_into_chunks(file_text, chunk_size, overlap)
        all_chunks.extend(file_chunks)
        
    print(f"--> Обработка завершена. Всего сформировано {len(all_chunks)} смысловых чанков.")
    return all_chunks

# Пример изолированного запуска модуля для проверки
if __name__ == "__main__":
    # Укажите путь к вашей папке с PDF
    FOLDER_PATH = "./my_university_pdfs" 
    
    # Создадим папку для теста, если её нет
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        print(f"Создана папка {FOLDER_PATH}. Положите туда несколько PDF и запустите снова.")
    else:
        chunks = get_all_university_chunks(FOLDER_PATH)
        if chunks:
            print(f"Пример первого чанка:\n{chunks[0][:500]}...")
