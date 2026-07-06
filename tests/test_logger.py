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


def test_get_logger():
    from pyruqo1.utils.logger import get_logger, progress_bar
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"
    assert len(logger.handlers) > 0


def test_get_logger_default():
    from pyruqo1.utils.logger import get_logger
    logger = get_logger()
    assert logger is not None


def test_progress_bar():
    from pyruqo1.utils.logger import progress_bar
    pb = progress_bar(total=10, description="test")
    assert pb is not None


def test_import_utils():
    from pyruqo1.utils import get_logger, progress_bar
    assert callable(get_logger)
    assert callable(progress_bar)
