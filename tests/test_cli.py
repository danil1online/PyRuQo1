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


def test_parse_servers_empty():
    from pyruqo1.cli import _parse_servers
    result = _parse_servers(None, None, ())
    assert result == ("http://localhost:8079/v1/chat/completions",)


def test_parse_servers_single():
    from pyruqo1.cli import _parse_servers
    result = _parse_servers(None, None, ("http://localhost:8079/v1/chat/completions",))
    assert result == ("http://localhost:8079/v1/chat/completions",)


def test_parse_servers_multiple_flags():
    from pyruqo1.cli import _parse_servers
    result = _parse_servers(None, None, ("http://srv1:8079", "http://srv2:8079"))
    assert len(result) == 2
    assert "http://srv1:8079" in result
    assert "http://srv2:8079" in result


def test_parse_servers_comma_separated():
    from pyruqo1.cli import _parse_servers
    result = _parse_servers(None, None, ("http://srv1:8079,http://srv2:8079",))
    assert len(result) == 2
    assert "http://srv1:8079" in result
    assert "http://srv2:8079" in result


def test_parse_servers_space_separated():
    from pyruqo1.cli import _parse_servers
    result = _parse_servers(None, None, ("http://srv1:8079 http://srv2:8079",))
    assert len(result) == 2
    assert "http://srv1:8079" in result
    assert "http://srv2:8079" in result


def test_train_dataset_type_micro():
    """dataset_type=micro задаёт пути из micro_datasets/."""
    from click.testing import CliRunner
    from pyruqo1.cli import train

    runner = CliRunner()
    result = runner.invoke(train, ["--model", "gigachat-20b", "--dataset-type", "micro", "--mode", "train_val"])
    # dataset-type принят без ошибки валидации
    assert result.exit_code == 0


def test_train_dataset_type_big():
    """dataset_type=big — по умолчанию."""
    from click.testing import CliRunner
    from pyruqo1.cli import train

    runner = CliRunner()
    result = runner.invoke(train, ["--model", "gigachat-20b", "--mode", "simple"])
    assert result.exit_code == 0
