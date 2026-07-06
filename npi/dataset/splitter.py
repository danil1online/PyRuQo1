import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from npi.utils.logger import get_logger


class JournalSplitter:
    """Разрезание сборников журналов на отдельные статьи по УДК/Аннотациям."""

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
    ):
        self.output_dir = Path(output_dir)
        self.patterns = patterns or self.ARTICLE_START_PATTERNS
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
        articles = []
        current_article_pages = []
        current_article_start_page = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if page_num == 0:
                current_article_start_page = 0
                current_article_pages = [page_num]
            elif self._is_article_start(text) and len(current_article_pages) > 1:
                articles.append((current_article_start_page, page_num - 1))
                current_article_start_page = page_num
                current_article_pages = [page_num]
            else:
                current_article_pages.append(page_num)

        if current_article_pages:
            articles.append((current_article_start_page, current_article_pages[-1]))

        doc.close()

        self.logger.info(f"Найдено {len(articles)} статей в {input_path.name}")

        output_files = []
        for i, (start, end) in enumerate(articles):
            if end - start < 0:
                continue
            article_doc = fitz.open(str(input_path))
            new_doc = fitz.open()

            for page_num in range(start, end + 1):
                new_doc.insert_pdf(article_doc, from_page=page_num, to_page=page_num)

            article_name = f"{input_path.stem}_article_{i+1:03d}.pdf"
            output_file = output_dir / article_name
            new_doc.save(str(output_file))
            new_doc.close()
            article_doc.close()
            output_files.append(str(output_file))

        self.logger.info(f"Сохранено {len(output_files)} статей в {output_dir}")
        return output_files

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
