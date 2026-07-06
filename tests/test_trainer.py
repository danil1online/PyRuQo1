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


def test_formatting_functions():
    from pyruqo1.training.formatting import format_single_example, formatting_prompts_func
    example = {
        "system": "You are a helper.",
        "prompt": "Hello?",
        "response": "Hi there!"
    }
    result = format_single_example(example)
    assert "text" in result
    assert "You are a helper" in result["text"]
    assert "Hello?" in result["text"]
    assert "Hi there!" in result["text"]


def test_format_dataset():
    from pyruqo1.training.formatting import format_dataset
    MockDataset = MagicMock()
    MockDataset.column_names = ["system", "prompt", "response"]
    result = format_dataset(MockDataset)
    assert result is not None


def test_trainer_init():
    from pyruqo1.training import NPITrainer
    config = {
        "model": {"name": "test-model", "trust_remote_code": True},
        "lora": {"r": 8, "lora_alpha": 16},
        "training": {"output_dir": "./test_out"},
        "dataset": {"train_file": "train.json", "val_file": "val.json"},
    }
    trainer = NPITrainer(config)
    assert trainer.config == config
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.trainer is None


def test_build_training_args():
    """Test that SFTConfig is called with correct kwargs."""
    from pyruqo1.training.config import build_training_args

    with patch("pyruqo1.training.config.SFTConfig") as MockSFTConfig:
        mock_instance = MagicMock()
        MockSFTConfig.return_value = mock_instance

        config = {
            "training": {
                "output_dir": "./test_output",
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1e-4,
                "num_train_epochs": 3,
                "bf16": True,
                "max_seq_length": 1024,
            }
        }
        args = build_training_args(config)

        MockSFTConfig.assert_called_once()
        kwargs = MockSFTConfig.call_args[1]
        assert kwargs["output_dir"] == "./test_output"
        assert kwargs["per_device_train_batch_size"] == 2
        assert kwargs["learning_rate"] == 1e-4
        assert kwargs["num_train_epochs"] == 3
        assert kwargs["bf16"] is True
        assert kwargs["max_seq_length"] == 1024


def test_build_training_args_defaults():
    """Test default values when config is empty."""
    from pyruqo1.training.config import build_training_args

    with patch("pyruqo1.training.config.SFTConfig") as MockSFTConfig:
        mock_instance = MagicMock()
        MockSFTConfig.return_value = mock_instance

        args = build_training_args({})

        kwargs = MockSFTConfig.call_args[1]
        assert kwargs["learning_rate"] == 2e-4
        assert kwargs["num_train_epochs"] == 1
        assert kwargs["max_seq_length"] == 2048
        assert kwargs["gradient_checkpointing"] is True


def test_build_training_args_full():
    """Test full configuration."""
    from pyruqo1.training.config import build_training_args

    with patch("pyruqo1.training.config.SFTConfig") as MockSFTConfig:
        mock_instance = MagicMock()
        MockSFTConfig.return_value = mock_instance

        config = {
            "training": {
                "output_dir": "./full_test",
                "gradient_checkpointing": False,
                "fp16": True,
                "bf16": False,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
                "save_strategy": "epoch",
                "save_steps": 50,
                "report_to": "wandb",
                "max_seq_length": 4096,
                "do_eval": True,
                "eval_strategy": "steps",
                "eval_steps": 25,
            }
        }
        args = build_training_args(config)

        kwargs = MockSFTConfig.call_args[1]
        assert kwargs["gradient_checkpointing"] is False
        assert kwargs["fp16"] is True
        assert kwargs["bf16"] is False
        assert kwargs["max_grad_norm"] == 1.0
        assert kwargs["warmup_ratio"] == 0.1
        assert kwargs["lr_scheduler_type"] == "cosine"
        assert kwargs["save_strategy"] == "epoch"
        assert kwargs["save_steps"] == 50
        assert kwargs["report_to"] == "wandb"
        assert kwargs["max_seq_length"] == 4096
        assert kwargs["do_eval"] is True
        assert kwargs["eval_steps"] == 25
