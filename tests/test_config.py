import pytest
import sys
from unittest.mock import MagicMock, Mock

# Mock heavy dependencies before any imports
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


def test_config_import():
    from pyruqo1.config import load_config, save_config, deep_merge
    assert callable(load_config)
    assert callable(save_config)
    assert callable(deep_merge)


def test_deep_merge():
    from pyruqo1.config import deep_merge
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10, "e": 5}, "f": 6}
    result = deep_merge(base, override)
    assert result["a"]["b"] == 10
    assert result["a"]["c"] == 2
    assert result["a"]["e"] == 5
    assert result["d"] == 3
    assert result["f"] == 6


def test_load_config_with_model():
    from pyruqo1.config import load_config
    config = load_config(model_name="gigachat-20b")
    assert "model" in config
    assert "training" in config
    assert config["model"]["name"] == "ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16"
    assert config["lora"]["r"] == 16


def test_load_config_with_path(tmp_path):
    from pyruqo1.config import load_config
    import yaml
    test_config = tmp_path / "test.yaml"
    test_config.write_text(yaml.dump({"model": {"name": "test-model"}, "training": {"output_dir": "/tmp/test"}}))
    config = load_config(config_path=str(test_config))
    assert config["model"]["name"] == "test-model"


def test_load_config_defaults():
    from pyruqo1.config import load_config
    config = load_config()
    assert "model" in config
    assert "training" in config


def test_save_config(tmp_path):
    from pyruqo1.config import save_config
    config = {"model": {"name": "test"}, "training": {"output_dir": "/tmp"}}
    out = tmp_path / "out.yaml"
    save_config(config, str(out))
    import yaml
    loaded = yaml.safe_load(out.read_text())
    assert loaded["model"]["name"] == "test"
