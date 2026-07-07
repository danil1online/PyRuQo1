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


def test_generate_answer_row_failure():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    result = gen._generate_answer_row(
        server_url="http://localhost:8079",
        chunk="context",
        question="question?",
        system_prompt="system",
        user_prompt="user",
    )
    assert result is None


def test_default_question_system_prompt():
    from pyruqo1.dataset.generator import DatasetGenerator
    assert "научный методолог" in DatasetGenerator.DEFAULT_QUESTION_SYSTEM_PROMPT.lower()


def test_default_answer_system_prompt():
    from pyruqo1.dataset.generator import DatasetGenerator
    assert "научный методолог" in DatasetGenerator.DEFAULT_ANSWER_SYSTEM_PROMPT.lower()


def test_math_question_system_prompt():
    from pyruqo1.dataset.generator import DatasetGenerator
    assert "LaTeX" in DatasetGenerator.MATH_QUESTION_SYSTEM_PROMPT


def test_math_answer_system_prompt():
    from pyruqo1.dataset.generator import DatasetGenerator
    assert "LaTeX" in DatasetGenerator.MATH_ANSWER_SYSTEM_PROMPT


def test_parse_question_response():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    choice = {"content": '{"prompt": "Какова причина явления?"}'}
    result = gen._parse_question_response(choice)
    assert result == "Какова причина явления?"


def test_parse_question_response_empty():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    choice = {"content": ""}
    result = gen._parse_question_response(choice)
    assert result is None


def test_parse_question_response_invalid_json():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    choice = {"content": "not json"}
    result = gen._parse_question_response(choice)
    assert result is None


def test_parse_answer_response_with_content():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    choice = {
        "reasoning_content": "Размышления модели",
        "content": "<Thought>Логика</Thought> <output>Ответ</output>",
    }
    result = gen._parse_answer_response(choice)
    assert result == "<Thought>Логика</Thought> <output>Ответ</output>"


def test_parse_answer_response_only_reasoning_content():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    choice = {
        "reasoning_content": "Размышления и ответ",
        "content": "",
    }
    result = gen._parse_answer_response(choice)
    assert result == "Размышления и ответ"


def test_parse_answer_response_both_empty():
    from pyruqo1.dataset.generator import DatasetGenerator
    gen = DatasetGenerator()
    choice = {
        "reasoning_content": "",
        "content": "",
    }
    result = gen._parse_answer_response(choice)
    assert result is None
