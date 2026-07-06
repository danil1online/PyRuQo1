import os
import re
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from npi.utils.logger import get_logger


def _ensure_marker_weights(weights_dir: str = None):
    try:
        from marker.models import load_model
    except ImportError:
        raise ImportError(
            "marker-pdf не установлен. Установите: pip install marker-pdf"
        )


class MathParser:
    """Парсинг PDF с извлечением LaTeX-формул через Marker."""

    def __init__(
        self,
        chunk_size: int = 3500,
        overlap: int = 500,
        min_text_length: int = 200,
        marker_model_name: str = "pythia-0.1b",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_text_length = min_text_length
        self.marker_model_name = marker_model_name
        self.logger = get_logger()
        _ensure_marker_weights()

    def _parse_pdf_with_marker(self, file_path: str) -> Optional[str]:
        try:
            from marker.convert import convert_single_pdf
            from marker.models import (
                load_model,
                load_tokenizer,
                load_detector,
                load_segmenter,
                load_math_translator,
            )

            self.logger.info(f"Marker: загрузка модели ({self.marker_model_name})...")
            converter_cls = None
            try:
                from marker.convert import Converter
                converter_cls = Converter
            except ImportError:
                pass

            if converter_cls:
                renderer = converter_cls(
                    load_model,
                    load_tokenizer,
                    load_detector,
                    load_segmenter,
                    load_math_translator,
                )
                full_text, _, _ = renderer(file_path)
            else:
                full_text, _, _ = convert_single_pdf(file_path, load_model(self.marker_model_name))

            return full_text if full_text and len(full_text.strip()) >= self.min_text_length else None

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
