import re
from typing import List


class TextChunker:
    """Нарезка текста на чанки с перекрытием (overlap)."""

    def __init__(self, chunk_size: int = 3500, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        if not text or len(text) < self.chunk_size:
            return [text] if text else []

        chunks = []
        words = text.split(' ')
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                overlap_words = max(1, int(self.overlap / 6))
                current_chunk = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else []
                current_length = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class MathChunker:
    """Нарезка текста с защитой формул LaTeX (не разрывать $$...$$)."""

    FORMULA_PATTERN = re.compile(r'\$\$.*?\$\$', re.DOTALL)
    INLINE_MATH = re.compile(r'\$(?:.*?)\$')

    def __init__(self, chunk_size: int = 3500, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _extract_formulas(self, text: str) -> List[re.Match]:
        return list(self.FORMULA_PATTERN.finditer(text))

    def _formula_safe_chunks(self, text: str) -> List[str]:
        formulas = self._extract_formulas(text)
        if not formulas:
            chunker = TextChunker(self.chunk_size, self.overlap)
            return chunker.chunk(text)

        protected_text = text
        formula_map = {}
        for i, match in enumerate(formulas):
            placeholder = f"__FORMULA_{i}__"
            protected_text = protected_text.replace(match.group(), placeholder)
            formula_map[placeholder] = match.group()

        chunker = TextChunker(self.chunk_size, self.overlap)
        raw_chunks = chunker.chunk(protected_text)

        chunks = []
        for chunk in raw_chunks:
            for placeholder, formula in formula_map.items():
                chunk = chunk.replace(placeholder, formula)
            chunks.append(chunk)

        return chunks

    def chunk(self, text: str) -> List[str]:
        return self._formula_safe_chunks(text)
