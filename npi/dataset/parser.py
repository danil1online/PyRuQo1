import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from npi.utils.logger import get_logger


class PDFParser:
    """Парсинг PDF-файлов с очисткой текста и OCR fallback."""

    LITERATURE_PATTERNS = re.compile(
        r'\b(Список литературы|References|Список источников)\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        min_text_length: int = 200,
        chunk_size: int = 3500,
        overlap: int = 500,
        enable_ocr: bool = True,
    ):
        self.min_text_length = min_text_length
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_ocr = enable_ocr
        self.logger = get_logger()

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+(?:[\s,-]*\d+)*\]', '', text)
        return text.strip()

    def parse_pdf(self, file_path: str) -> Optional[str]:
        try:
            doc = fitz.open(file_path)
            full_text = []
            for page in doc:
                text = page.get_text("text")
                if self.LITERATURE_PATTERNS.search(text):
                    parts = self.LITERATURE_PATTERNS.split(text, maxsplit=1)
                    full_text.append(parts[0])
                    break
                full_text.append(text)
            doc.close()

            extracted = self.clean_text(" ".join(full_text))

            if len(extracted) < self.min_text_length and self.enable_ocr:
                self.logger.info(f"[OCR] {os.path.basename(file_path)} — скан, запускаю распознавание...")
                ocr_output = str(file_path).replace(".pdf", "_ocr.pdf")
                os.system(f"ocrmypdf '{file_path}' '{ocr_output}' -l rus --quiet 2>/dev/null")

                if os.path.exists(ocr_output):
                    doc = fitz.open(ocr_output)
                    full_text = [page.get_text("text") for page in doc]
                    doc.close()
                    os.remove(ocr_output)
                    extracted = self.clean_text(" ".join(full_text))

            return extracted if len(extracted) >= self.min_text_length else None

        except Exception as e:
            self.logger.error(f"Ошибка парсинга {file_path}: {e}")
            return None

    def parse_folder(self, folder_path: str, recursive: bool = True) -> List[str]:
        folder = Path(folder_path)
        if not folder.exists():
            self.logger.error(f"Папка не найдена: {folder_path}")
            return []

        pdf_files = []
        pattern = "**/*.pdf" if recursive else "*.pdf"
        for f in sorted(folder.glob(pattern)):
            if f.is_file():
                pdf_files.append(str(f))

        self.logger.info(f"Найдено {len(pdf_files)} PDF-файлов.")

        all_text = []
        for file_path in tqdm(pdf_files, desc="Парсинг PDF"):
            text = self.parse_pdf(file_path)
            if text:
                all_text.append(text)

        self.logger.info(f"Извлечено текста из {len(all_text)} файлов.")
        return all_text
