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


def test_merger_init():
    from pyruqo1.merge.merger import LORAMerger
    config = {"model": {"name": "test-model", "trust_remote_code": True}}
    merger = LORAMerger(config)
    assert merger.config == config


def test_merger_manage_swap_always():
    """--manage-swap работает всегда, не только при --low-ram."""
    from pyruqo1.merge.merger import LORAMerger
    config = {
        "model": {"name": "base-model", "trust_remote_code": True},
        "merge": {"low_ram": False, "cpu_swap_gb": 40, "max_shard_size": "3GB"},
    }
    merger = LORAMerger(config)

    with patch("pyruqo1.merge.merger.managed_swap") as mock_swap:
        with patch.object(merger, "_do_merge") as mock_do_merge:
            merger.merge(manage_swap=True)
            mock_swap.assert_called_once()
            mock_do_merge.assert_called_once()


def test_merger_low_ram():
    """low_ram режим без --manage-swap."""
    from pyruqo1.merge.merger import LORAMerger
    config = {
        "model": {"name": "base-model", "trust_remote_code": True},
        "merge": {"low_ram": True, "cpu_swap_gb": 30, "max_shard_size": "3GB"},
    }
    merger = LORAMerger(config)

    with patch("pyruqo1.merge.merger.managed_swap") as mock_swap:
        with patch.object(merger, "_do_merge") as mock_do_merge:
            merger.merge(manage_swap=False)
            # low_ram тоже использует managed_swap, но manage_swap=False не должен вызывать _merge_with_swap
            mock_do_merge.assert_called_once()


def test_merger_standard():
    """Обычный merge без swap и low_ram."""
    from pyruqo1.merge.merger import LORAMerger
    config = {
        "model": {"name": "base-model", "trust_remote_code": True},
        "merge": {"low_ram": False, "cpu_swap_gb": 40},
    }
    merger = LORAMerger(config)

    with patch.object(merger, "_merge_standard") as mock_standard:
        merger.merge(manage_swap=False)
        mock_standard.assert_called_once()


def test_save_model():
    from pyruqo1.merge.merger import LORAMerger
    import tempfile
    config = {"model": {"name": "test", "trust_remote_code": True}}
    merger = LORAMerger(config)
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    with tempfile.TemporaryDirectory() as tmpdir:
        merger._save_model(mock_model, mock_tokenizer, tmpdir, "3GB")
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
