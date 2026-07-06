import pytest
import sys
from unittest.mock import MagicMock, patch

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


def test_clean_text():
    from pyruqo1.dataset.parser import PDFParser
    parser = PDFParser()
    text = "hello   world"
    result = parser.clean_text(text)
    assert result == "hello world"


def test_clean_text_hyphens():
    from pyruqo1.dataset.parser import PDFParser
    parser = PDFParser()
    text = "nano-\n technology"
    result = parser.clean_text(text)
    assert "nanotechnology" in result


def test_clean_text_empty():
    from pyruqo1.dataset.parser import PDFParser
    parser = PDFParser()
    assert parser.clean_text("") == ""
    assert parser.clean_text(None) == ""


def test_parse_folder_not_found():
    from pyruqo1.dataset.parser import PDFParser
    parser = PDFParser()
    result = parser.parse_folder("/nonexistent/path/xyz")
    assert result == []
