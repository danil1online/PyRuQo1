import os
import re
import fitz  # Используем PyMuPDF только для быстрого разрезания страниц

# Папка, куда вы положили массивные сборники и журналы
JOURNALS_FOLDER = "./raw_journals"
# Папка, куда скрипт сложит нарезанные изолированные статьи
OUTPUT_FOLDER = "./university_pdfs"

def split_journal_to_articles(pdf_path):
    journal_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    
    article_start_pages = []
    
    print(f"Сканирование структуры журнала: {journal_name}")
    
    # Шаг 1: Ищем страницы, на которых начинаются новые статьи
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Маркеры начала новой статьи в академических журналах:
        # Наличие УДК, слова "Аннотация", "Abstract" или "Введение" в первых строках
        has_udc = re.search(r'\bУДК\b', text)
        has_annotation = re.search(r'\b(Аннотация|Abstract)\b', text, re.IGNORECASE)
        
        # Если нашли маркеры (особенно на первых 1000 символах страницы)
        if (has_udc or has_annotation) and page_num not in article_start_pages:
            article_start_pages.append(page_num)
            
    if not article_start_pages:
        # Если автоматические маркеры не сработали, добавим хотя бы первую страницу
        article_start_pages.append(0)
        
    # Добавляем в конец общее количество страниц, чтобы закрыть последнюю статью
    article_start_pages.append(len(doc))
    
    print(f"Найдено предположительно статей: {len(article_start_pages) - 1}")
    
    # Шаг 2: Нарезаем PDF на изолированные файлы
    for i in range(len(article_start_pages) - 1):
        start_page = article_start_pages[i]
        end_page = article_start_pages[i+1]
        
        # Пропускаем слишком короткие куски (меньше 3 страниц), скорее всего это оглавление или титульник
        if (end_page - start_page) < 3:
            continue
            
        # Создаем новый мини-PDF для конкретной статьи
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
        
        article_filename = f"{journal_name}_article_{i+1}.pdf"
        output_path = os.path.join(OUTPUT_FOLDER, article_filename)
        
        new_doc.save(output_path)
        new_doc.close()
        
    doc.close()

def main():
    if not os.path.exists(JOURNALS_FOLDER):
        os.makedirs(JOURNALS_FOLDER)
        print(f"Создана папка {JOURNALS_FOLDER}. Положите туда ваши сборники и журналы.")
        return
        
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    files = [f for f in os.listdir(JOURNALS_FOLDER) if f.lower().endswith('.pdf')]
    
    if not files:
        print(f"В папке {JOURNALS_FOLDER} нет файлов для разделения.")
        return
        
    for file in files:
        full_path = os.path.join(JOURNALS_FOLDER, file)
        split_journal_to_articles(full_path)
        
    print("\n--> Все сборники успешно разделены на изолированные статьи!")
    print(f"Результаты сохранены в {OUTPUT_FOLDER} и готовы к математическому парсингу.")

if __name__ == "__main__":
    main()
