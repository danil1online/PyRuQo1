import os
import re
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

from pyruqo1.utils.logger import get_logger


# Паттерны заголовков разделов
SECTION_PATTERNS = {
    "introduction": [
        r"^введение\b", r"^introduction\b", r"^вводная\s*часть\b",
        r"^актуальность\b", r"^background\b",
        r"^цель\b", r"^objective\b", r"^aim\b", r"^задачи\b",
        r"^постановка\s*задачи\b", r"^goals\b",
    ],
    "methods": [
        r"^материалы\b", r"^материалы\s*и\s*методы\b",
        r"^методы\b", r"^materials\b", r"^materials\s*and\s*methods\b",
        r"^методы\s*исследования\b",
    ],
    "results": [
        r"^результаты\b", r"^результаты\s*исследования\b",
        r"^results\b", r"^discussion\b",
        r"^обсуждение\s*результатов\b",
    ],
    "conclusion": [
        r"^выводы\b", r"^заключение\b", r"^выводы\s*и\s*обсуждение\b",
        r"^основные\s*результаты\b",
        r"^conclusion\b", r"^conclusions\b", r"^summary\b",
        r"^conclusions\s*and\s*discussion\b",
    ],
}

SKIP_PATTERNS = [
    r"^удк\b", r"^удк\s*указатель\b",
    r"^авторы\b", r"^authors\b",
    r"^организация\b", r"^affiliation\b", r"^institution\b",
    r"^аннотация\b", r"^abstract\b",
    r"^ключевые\s*слова\b", r"^keywords\b",
    r"^список\b.*?литературы\b", r"^references\b", r"^bibliography\b",
]


class BaseParser:
    """Парсинг PDF для SFT-датасета: извлечение разделов статьи через Marker."""

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
            from marker.models import load_all_models
        except ImportError:
            raise ImportError(
                "marker-pdf не установлен. Установите: pip install marker-pdf"
            )

        self.logger.info("Marker: загрузка нейросетей распознавания академических текстов...")
        self._model_lst = load_all_models()
        self.logger.info("Marker: модели загружены.")
        return self._model_lst

    def _classify_header(self, text: str) -> Optional[str]:
        """Классификация заголовка по паттернам."""
        text_lower = text.strip().lower()
        for section, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, flags=re.IGNORECASE):
                    return section
        return None

    def _is_skip_section(self, text: str) -> bool:
        """Проверка, нужно ли пропустить раздел."""
        text_lower = text.strip().lower()
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
                return True
        return False

    def _extract_sections(self, full_text: str) -> Dict[str, str]:
        """Разбиение текста на секции по заголовкам."""
        lines = full_text.split('\n')
        sections = {
            "introduction": "",
            "methods": [],
            "results": [],
            "conclusion": "",
        }
        current_section = None
        current_text = ""
        in_content = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Проверяем, является ли строка заголовком
            classified = self._classify_header(stripped)
            skipped = self._is_skip_section(stripped)

            if classified:
                # Сохраняем предыдущую секцию
                if current_section and current_text:
                    if current_section in ("introduction", "conclusion"):
                        sections[current_section] = current_text.strip()
                    else:
                        sections[current_section].append(current_text.strip())
                    current_text = ""

                current_section = classified
                in_content = True
                continue

            if skipped:
                # Сохраняем предыдущую секцию если была
                if current_section and current_text:
                    if current_section in ("introduction", "conclusion"):
                        sections[current_section] = current_text.strip()
                    else:
                        sections[current_section].append(current_text.strip())
                    current_text = ""
                current_section = None
                in_content = False
                continue

            if in_content and current_section:
                if current_text:
                    current_text += "\n" + stripped
                else:
                    current_text = stripped

        # Сохраняем последнюю секцию
        if current_section and current_text:
            if current_section in ("introduction", "conclusion"):
                sections[current_section] = current_text.strip()
            else:
                sections[current_section].append(current_text.strip())

        return sections

    def _parse_pdf_with_marker(self, file_path: str) -> Optional[Dict]:
        """Парсинг одного PDF через Marker с извлечением разделов."""
        try:
            from marker.convert import convert_single_pdf

            model_lst = self._get_models()

            full_text, _, _ = convert_single_pdf(file_path, model_lst)

            # Отсекаем список литературы
            lit_pattern = r'\b(Список литературы|References|Список источников)\b'
            if re.search(lit_pattern, full_text, flags=re.IGNORECASE):
                full_text = re.split(lit_pattern, full_text, flags=re.IGNORECASE)[0]

            full_text = full_text.strip()

            if not full_text or len(full_text) < self.min_text_length:
                return None

            sections = self._extract_sections(full_text)

            # Проверяем наличие обязательных разделов
            if not sections["introduction"] or not sections["conclusion"]:
                self.logger.debug(f"Пропуск (нет введения или выводов): {file_path}")
                return None

            self.logger.debug(f"Извлечены разделы из {file_path}: intro={len(sections['introduction'])}, methods={len(sections['methods'])}, results={len(sections['results'])}, conclusion={len(sections['conclusion'])}")
            return sections

        except Exception as e:
            self.logger.error(f"Ошибка Marker {file_path}: {e}")
            return None

    def parse_pdf(self, file_path: str) -> Optional[Dict]:
        return self._parse_pdf_with_marker(file_path)

    def parse_folder(self, folder_path: str, recursive: bool = True) -> List[Dict]:
        folder = Path(folder_path)
        if not folder.exists():
            self.logger.error(f"Папка не найдена: {folder_path}")
            return []

        pdf_files = []
        pattern = "**/*.pdf" if recursive else "*.pdf"
        for f in sorted(folder.glob(pattern)):
            if f.is_file():
                pdf_files.append(str(f))

        self.logger.info(f"Base: найдено {len(pdf_files)} PDF-файлов.")

        all_sections = []
        for file_path in tqdm(pdf_files, desc="Base парсинг"):
            sections = self.parse_pdf(file_path)
            if sections:
                all_sections.append(sections)

        self.logger.info(f"Base: извлечено статей с разделами из {len(all_sections)} файлов.")
        return all_sections
