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


def test_is_article_start_udk():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    assert splitter._is_article_start("УДК 517.5") is True
    assert splitter._is_article_start("УДК 621.3") is True


def test_is_article_start_abstract():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    assert splitter._is_article_start("Аннотация") is True
    assert splitter._is_article_start("Abstract") is True
    assert splitter._is_article_start("Введение") is True


def test_is_article_start_normal():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    assert splitter._is_article_start("Просто текст статьи") is False
    assert splitter._is_article_start("Lorem ipsum dolor sit amet") is False


def test_split_folder_not_found():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    result = splitter.split_folder("/nonexistent/path")
    assert result == []
