import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from pyruqo1.utils.logger import get_logger
from pyruqo1.training.formatting import format_dataset, formatting_prompts_func
from pyruqo1.training.config import build_training_args


class NPITrainer:
    """Единый Trainer для QLoRA-обучения разных моделей."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _load_model(self):
        self.logger.info(f"Загрузка модели: {self.config['model']['name']}")

        model_name = self.config['model']['name']
        trust_remote = self.config['model'].get('trust_remote_code', True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote,
        )
        self.model = prepare_model_for_kbit_training(self.model)

    def _setup_lora(self):
        lora_cfg = self.config.get("lora", {})
        self.logger.info("Настройка LoRA-адаптера...")

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _load_dataset(self):
        dataset_cfg = self.config.get("dataset", {})
        train_file = dataset_cfg.get("train_file", "university_train.json")
        val_file = dataset_cfg.get("val_file", "university_val.json")

        self.logger.info(f"Загрузка датасета: train={train_file}, val={val_file}")

        data_files = {
            "train": train_file,
            "validation": val_file,
        }

        dataset = load_dataset("json", data_files=data_files)
        self.logger.info(
            f"Загружен train: {len(dataset['train'])} строк, "
            f"validation: {len(dataset['validation'])} строк."
        )

        dataset = format_dataset(dataset["train"], list(dataset["train"].column_names))
        if "validation" in dataset:
            dataset["validation"] = format_dataset(
                dataset["validation"], list(dataset["validation"].column_names)
            )

        return dataset

    def _build_trainer(self, dataset, dataset_type: str = "big"):
        training_args = build_training_args(self.config, dataset_type=dataset_type)
        self.logger.info("Создание SFTTrainer...")

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            peft_config=LoraConfig(
                r=self.config.get("lora", {}).get("r", 16),
                lora_alpha=self.config.get("lora", {}).get("lora_alpha", 32),
                target_modules=self.config.get("lora", {}).get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                lora_dropout=self.config.get("lora", {}).get("lora_dropout", 0.05),
                bias=self.config.get("lora", {}).get("bias", "none"),
                task_type="CAUSAL_LM",
            ),
            args=training_args,
        )

    def train(self, dataset_type: str = "big"):
        self.logger.info(f"=== Начинается процесс обучения ===")
        self.logger.info(f"Тип датасета: {dataset_type}")

        self._load_model()
        self._setup_lora()
        dataset = self._load_dataset()
        self._build_trainer(dataset, dataset_type=dataset_type)

        self.logger.info("Запуск обучения...")
        self.trainer.train()

        output_dir = self.config.get("training", {}).get("output_dir", "./output")
        self.trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.logger.info(f"=== Обучение завершено! Адаптер сохранён в: {output_dir} ===")

    def save_lora(self, output_dir: str = None):
        output_dir = output_dir or self.config.get("training", {}).get("output_dir", "./output")
        self.trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"LoRA-адаптер сохранён в: {output_dir}")
