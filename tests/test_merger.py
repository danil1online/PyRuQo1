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


def test_merger_init():
    from npi.merge.merger import LORAMerger
    config = {"model": {"name": "test-model", "trust_remote_code": True}}
    merger = LORAMerger(config)
    assert merger.config == config


def test_merger_merge_params():
    from npi.merge.merger import LORAMerger
    config = {
        "model": {"name": "base-model", "trust_remote_code": True},
        "training": {"output_dir": "./lora_output"},
        "merge": {"output_dir": "./merged_output", "low_ram": False, "cpu_swap_gb": 40},
    }
    merger = LORAMerger(config)
    with patch.object(merger, "_merge_standard") as mock_standard:
        merger.merge()
        mock_standard.assert_called_once()


def test_merger_merge_low_ram():
    from npi.merge.merger import LORAMerger
    config = {
        "model": {"name": "base-model", "trust_remote_code": True},
        "merge": {"low_ram": True, "cpu_swap_gb": 30, "max_shard_size": "3GB"},
    }
    merger = LORAMerger(config)
    with patch("npi.merge.merger.managed_swap"):
        with patch.object(merger, "_save_model"):
            merger._merge_low_ram(
                "base-model", "./lora", "./merged",
                {"low_ram": True, "cpu_swap_gb": 30, "max_shard_size": "3GB"}, True,
            )


def test_save_model():
    from npi.merge.merger import LORAMerger
    import tempfile
    import os
    config = {"model": {"name": "test", "trust_remote_code": True}}
    merger = LORAMerger(config)
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    with tempfile.TemporaryDirectory() as tmpdir:
        merger._save_model(mock_model, mock_tokenizer, tmpdir, "3GB")
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
