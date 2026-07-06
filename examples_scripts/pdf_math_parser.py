import os
import re
from marker.convert import convert_single_pdf
from marker.models import load_models

# Загружаем нейросети для распознавания текста и формул (выполняется 1 раз)
# Модели автоматически развернутся на вашей RTX 3090
print("--> Загрузка нейросетей распознавания математического текста...")
model_lst = load_models()

def parse_pdf_to_math_markdown(file_path):
    """
    Превращает PDF в Markdown, где все формулы переведены в LaTeX.
    Прекрасно обрабатывает как цифровые PDF, так и "слепые" сканы.
    """
    try:
        # Конвертируем PDF с помощью Marker
        full_text, _, _ = convert_single_pdf(file_path, model_lst)
        
        # Отсекаем список литературы, чтобы не забивать датасет мусором
        lit_pattern = r'\b(Список литературы|References|Список источников)\b'
        if re.search(lit_pattern, full_text, re.IGNORECASE):
            full_text = re.split(lit_pattern, full_text, flags=re.IGNORECASE)[0]
            
        return full_text.strip()
    except Exception as e:
        print(f"\n[Ошибка] Не удалось распознать формулы в {file_path}: {e}")
        return ""

def split_math_chunks(text, max_chars=3500, overlap=500):
    """
    Нарезает Markdown-текст на чанки, гарантируя, что LaTeX формулы
    внутри блоков $$...$$ не будут разорваны посередине.
    """
    # Ищем блоки формул, чтобы защитить их от разрыва
    tokens = re.split(r'(\$\$.*?\$\$|\$.*?\$)', text, flags=re.DOTALL)
    
    chunks = []
    current_chunk = ""
    
    for token in tokens:
        # Если добавление токена превышает лимит, сохраняем текущий чанк
        if len(current_chunk) + len(token) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Реализуем нахлест (overlap) из конца предыдущего чанка
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else ""
            
        current_chunk += token
        
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def get_all_math_chunks(root_folder):
    """Обходит папки и собирает математические чанки со всех PDF"""
    all_chunks = []
    pdf_files = []
    
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    print(f"--> Найдено {len(pdf_files)} PDF-файлов для математического анализа.")
    
    for file_path in pdf_files:
        print(f"Обработка формул в: {os.path.basename(file_path)}")
        markdown_text = parse_pdf_to_math_markdown(file_path)
        if len(markdown_text) > 300:
            file_chunks = split_math_chunks(markdown_text)
            all_chunks.extend(file_chunks)
            
    print(f"--> Математический парсинг завершен. Всего чанков: {len(all_chunks)}")
    return all_chunks
