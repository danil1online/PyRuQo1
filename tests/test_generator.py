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
sys.modules['requests'] = MagicMock()


def test_generator_init():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    assert gen.servers == ["http://localhost:8079/v1/chat/completions"]
    assert gen.temperature == 0.2
    assert gen.max_tokens == 2500


def test_generator_custom_servers():
    from pyruqo1.dataset.generator import DatasetGenerator
    servers = ["http://srv1:8079/v1/chat/completions", "http://srv2:8079/v1/chat/completions"]
    gen = DatasetGenerator(servers=servers)
    assert len(gen.servers) == 2


def test_next_server():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator(servers=["http://a:8079", "http://b:8079"])
    assert gen._get_next_server() == "http://a:8079"
    assert gen._get_next_server() == "http://b:8079"
    assert gen._get_next_server() == "http://a:8079"


def test_generate_row_failure():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    result = gen._generate_row("context", "system")
    assert result is None


def test_default_system_prompt():
    from pyruqo1.dataset.generator import DatasetGenerator
    assert "научный методолог" in DatasetGenerator.DEFAULT_SYSTEM_PROMPT.lower()


def test_math_system_prompt():
    from pyruqo1.dataset.generator import DatasetGenerator
    assert "LaTeX" in DatasetGenerator.MATH_SYSTEM_PROMPT
