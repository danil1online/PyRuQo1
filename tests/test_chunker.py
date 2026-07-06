import pytest
import sys
from unittest.mock import MagicMock

sys.modules['fitz'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['tokenizers'] = MagicMock()
sys.modules['bitsandbytes'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['marker'] = MagicMock()
sys.modules['marker.models'] = MagicMock()
sys.modules['marker.convert'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()
sys.modules['llama_cpp.convert'] = MagicMock()


def test_text_chunker_simple():
    from pyruqo1.dataset.chunker import TextChunker
    chunker = TextChunker(chunk_size=100, overlap=10)
    text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1


def test_text_chunker_small():
    from pyruqo1.dataset.chunker import TextChunker
    chunker = TextChunker(chunk_size=1000, overlap=50)
    text = "short text"
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_chunker_empty():
    from pyruqo1.dataset.chunker import TextChunker
    chunker = TextChunker(chunk_size=100, overlap=10)
    assert chunker.chunk("") == []
    assert chunker.chunk(None) == []


def test_math_chunker_no_formulas():
    from pyruqo1.dataset.chunker import MathChunker
    chunker = MathChunker(chunk_size=100, overlap=10)
    text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1


def test_math_chunker_with_formulas():
    from pyruqo1.dataset.chunker import MathChunker
    chunker = MathChunker(chunk_size=500, overlap=50)
    text = "Some text $$\\int_0^1 x^2 dx$$ more text $$\\frac{a}{b}$$ end"
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1


def test_text_chunker_overlap():
    from pyruqo1.dataset.chunker import TextChunker
    chunker = TextChunker(chunk_size=20, overlap=10)
    text = "one two three four five six seven eight"
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    for chunk in chunks:
        words = chunk.split()
        assert len(words) > 0
