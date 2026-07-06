import os
import re
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from pyruqo1.utils.logger import get_logger


class MathParser:
    """Парсинг PDF с извлечением LaTeX-формул через Marker."""

    def __init__(
        self,
        chunk_size: int = 3500,
        overlap: int = 500,
        min_text_length: int = 300,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_text_length = min_text_length
        self.logger = get_logger()
        self._model_lst = None

    def _get_models(self):
        """Ленивая загрузка моделей Marker (выполняется 1 раз)."""
        if self._model_lst is not None:
            return self._model_lst

        try:
            from marker.models import load_models
        except ImportError:
            raise ImportError(
                "marker-pdf не установлен. Установите: pip install marker-pdf"
            )

        self.logger.info("Marker: загрузка нейросетей распознавания математического текста...")
        self._model_lst = load_models()
        self.logger.info("Marker: модели загружены.")
        return self._model_lst

    def _parse_pdf_with_marker(self, file_path: str) -> Optional[str]:
        try:
            from marker.convert import convert_single_pdf

            model_lst = self._get_models()

            full_text, _, _ = convert_single_pdf(file_path, model_lst)

            # Отсекаем список литературы
            lit_pattern = r'\b(Список литературы|References|Список источников)\b'
            if re.search(lit_pattern, full_text, re.IGNORECASE):
                full_text = re.split(lit_pattern, full_text, flags=re.IGNORECASE)[0]

            full_text = full_text.strip()

            if full_text and len(full_text) >= self.min_text_length:
                return full_text
            return None

        except Exception as e:
            self.logger.error(f"Ошибка Marker {file_path}: {e}")
            return None

    def parse_pdf(self, file_path: str) -> Optional[str]:
        return self._parse_pdf_with_marker(file_path)

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

        self.logger.info(f"Marker: найдено {len(pdf_files)} PDF-файлов.")

        all_text = []
        for file_path in tqdm(pdf_files, desc="Marker парсинг"):
            text = self.parse_pdf(file_path)
            if text:
                all_text.append(text)

        self.logger.info(f"Marker: извлечено текста из {len(all_text)} файлов.")
        return all_text


