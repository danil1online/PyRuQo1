import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from pyruqo1.utils.logger import get_logger

# Жесткий лимит страниц: если маркеры не сработали, режем по столько страниц
FORCE_PAGES_PER_ARTICLE = 4


class JournalSplitter:
    """Разрезание сборников журналов на отдельные статьи по УДК/Аннотациям.

    Если маркеры не нашли границ (≤1 статья) и страниц >8, включается fallback
    — принудительная нарезка по FORCE_PAGES_PER_ARTICLE страниц.
    """

    ARTICLE_START_PATTERNS = [
        re.compile(r'УДК\s+\d+[.\d\s]*', re.IGNORECASE),
        re.compile(r'Аннотация\b', re.IGNORECASE),
        re.compile(r'Abstract\b', re.IGNORECASE),
        re.compile(r'Введение\b', re.IGNORECASE),
        re.compile(r'Introduction\b', re.IGNORECASE),
        re.compile(r'\bSection\b', re.IGNORECASE),
    ]

    def __init__(
        self,
        output_dir: str = "./university_pdfs",
        patterns: List[re.Pattern] = None,
        force_pages_per_article: int = FORCE_PAGES_PER_ARTICLE,
    ):
        self.output_dir = Path(output_dir)
        self.patterns = patterns or self.ARTICLE_START_PATTERNS
        self.force_pages = force_pages_per_article
        self.logger = get_logger()

    def _is_article_start(self, text: str) -> bool:
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False

    def split_pdf(self, input_path: str, output_dir: str = None) -> List[str]:
        input_path = Path(input_path)
        if not input_path.exists():
            self.logger.error(f"Файл не найден: {input_path}")
            return []

        output_dir = Path(output_dir) if output_dir else (self.output_dir / input_path.stem)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(input_path))
        total_pages = len(doc)
        base_name = input_path.stem

        self.logger.info(f"Сканирование структуры журнала: {base_name} (Всего страниц: {total_pages})")

        # Шаг 1: Поиск маркеров начала новой статьи
        article_start_pages = [0]  # Первая страница — всегда старт

        for page_num in range(1, total_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            header_area = text[:800]

            has_udk = re.search(r'\bУДК\b', header_area)
            has_annotation = re.search(
                r'\b(Аннотация|Abstract|Ключевые слова|Keywords|Введение|Introduction)\b',
                header_area, re.IGNORECASE
            )
            has_copyright = re.search(r'©\s+\d{4}', header_area)

            if (has_udk or has_annotation or has_copyright):
                if page_num - article_start_pages[-1] >= 2:
                    article_start_pages.append(page_num)

        if article_start_pages[-1] != total_pages:
            article_start_pages.append(total_pages)

        actual_articles_found = len(article_start_pages) - 1
        self.logger.info(f"Найдено по текстовым маркерам: {actual_articles_found}")

        # АВТОМАТИЧЕСКАЯ ЗАЩИТА: Если маркеры нашли всего 1 кусок (сборник не разделился)
        if actual_articles_found <= 1 and total_pages > 8:
            self.logger.warning(
                f"Сборник не разделился стандартным путем. "
                f"Включается принудительное дробление по {self.force_pages} страницы..."
            )
            article_start_pages = list(range(0, total_pages, self.force_pages))
            if article_start_pages[-1] != total_pages:
                article_start_pages.append(total_pages)

        is_force = actual_articles_found <= 1

        # Шаг 2: Нарезка и сохранение мини-PDF файлов
        saved_count = 0
        for i in range(len(article_start_pages) - 1):
            start_page = article_start_pages[i]
            end_page = article_start_pages[i + 1]

            # Пропускаем «огрызки» менее 2 страниц, только если это не режим жесткой нарезки
            if (end_page - start_page) < 2 and not is_force:
                continue

            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

            # Формируем имя: добавляем пометку "force", если сработал защитный алгоритм
            is_force_prefix = "force_" if is_force else ""
            article_filename = f"{base_name}_{is_force_prefix}article_{saved_count + 1:03d}.pdf"
            output_file = output_dir / article_filename

            new_doc.save(str(output_file))
            new_doc.close()
            saved_count += 1

        doc.close()
        self.logger.info(f"Успешно сохранено изолированных файлов: {saved_count}")
        self.logger.info(f"Сохранено в: {output_dir}")

        return [str(output_dir / f) for f in sorted(output_dir.glob("*.pdf"))]

    def split_folder(self, folder_path: str) -> List[str]:
        folder = Path(folder_path)
        if not folder.exists():
            self.logger.error(f"Папка не найдена: {folder_path}")
            return []

        pdf_files = sorted(folder.glob("*.pdf"))
        self.logger.info(f"Найдено {len(pdf_files)} журналов в {folder}")

        all_articles = []
        for pdf_file in tqdm(pdf_files, desc="Разрезание журналов"):
            articles = self.split_pdf(str(pdf_file))
            all_articles.extend(articles)

        self.logger.info(f"Всего статей: {len(all_articles)}")
        return all_articles
