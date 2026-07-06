import os
import re
import fitz  # PyMuPDF

JOURNALS_FOLDER = "./raw_journals"
OUTPUT_FOLDER = "./university_pdfs_journals"
# Жесткий лимит страниц: если маркеры не сработали, режем по столько страниц
FORCE_PAGES_PER_ARTICLE = 4 

def split_journal_to_articles(pdf_path):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[Ошибка] Не удалось открыть файл {pdf_path}: {e}")
        return

    article_start_pages = [0] # Первая страница — всегда старт
    total_pages = len(doc)
    
    print(f"Сканирование структуры журнала: {base_name} (Всего страниц: {total_pages})")
    
    # Шаг 1: Поиск маркеров начала новой статьи
    for page_num in range(1, total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()
        
        if not text:
            continue
            
        header_area = text[:800]
        
        has_udk = re.search(r'\bУДК\b', header_area)
        has_annotation = re.search(r'\b(Аннотация|Abstract|Ключевые слова|Keywords|Введение|Introduction)\b', header_area, re.IGNORECASE)
        has_copyright = re.search(r'©\s+\d{4}', header_area)
        
        if (has_udk or has_annotation or has_copyright):
            if page_num - article_start_pages[-1] >= 2:
                article_start_pages.append(page_num)
                
    if article_start_pages[-1] != total_pages:
        article_start_pages.append(total_pages)
        
    actual_articles_found = len(article_start_pages) - 1
    print(f"Найдено по текстовым маркерам: {actual_articles_found}")
    
    # АВТОМАТИЧЕСКАЯ ЗАЩИТА: Если маркеры нашли всего 1 кусок (сборник не разделился)
    if actual_articles_found <= 1 and total_pages > 8:
        print(f"[ВНИМАНИЕ] Сборник не разделился стандартным путем. Включается принудительное дробление по {FORCE_PAGES_PER_ARTICLE} страницы...")
        # Генерируем сетку страниц с шагом в 4 страницы
        article_start_pages = list(range(0, total_pages, FORCE_PAGES_PER_ARTICLE))
        if article_start_pages[-1] != total_pages:
            article_start_pages.append(total_pages)
            
    # Шаг 2: Нарезка и сохранение мини-PDF файлов
    saved_count = 0
    for i in range(len(article_start_pages) - 1):
        start_page = article_start_pages[i]
        end_page = article_start_pages[i+1]
        
        # Пропускаем «огрызки» менее 2 страниц, только если это не режим жесткой нарезки
        if (end_page - start_page) < 2 and (len(article_start_pages) - 1) == actual_articles_found:
            continue
            
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
        
        # Формируем имя: добавляем пометку "force", если сработал защитный алгоритм
        is_force = "force_" if actual_articles_found <= 1 else ""
        article_filename = f"{base_name}_{is_force}article_{saved_count + 1}.pdf"
        output_path = os.path.join(OUTPUT_FOLDER, article_filename)
        
        new_doc.save(output_path)
        new_doc.close()
        saved_count += 1
        
    doc.close()
    print(f"Успешно сохранено изолированных файлов: {saved_count}\n")

def main():
    if not os.path.exists(JOURNALS_FOLDER):
        os.makedirs(JOURNALS_FOLDER)
        print(f"Создана папка {JOURNALS_FOLDER}. Положите туда ваши журналы.")
        return
        
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    files = [f for f in os.listdir(JOURNALS_FOLDER) if f.lower().endswith('.pdf')]
    
    if not files:
        print(f"В папке {JOURNALS_FOLDER} нет файлов для обработки.")
        return
        
    for file in files:
        full_path = os.path.join(JOURNALS_FOLDER, file)
        split_journal_to_articles(full_path)
        
    print("--> Все сборники успешно обработаны конвейером!")

if __name__ == "__main__":
    main()