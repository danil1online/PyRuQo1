import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from pyruqo1.utils.logger import get_logger
from pyruqo1.utils.swap import managed_swap, get_free_ram_gb, get_managed_swap_path, remove_swap_file, DEFAULT_SWAP_PATH


class LORAMerger:
    """Слияние LoRA-адаптера с базовой моделью."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()

    def merge(
        self,
        base_model_name: str = None,
        lora_adapter_dir: str = None,
        output_dir: str = None,
        manage_swap: bool = False,
    ):
        base_model_name = base_model_name or self.config['model']['name']
        lora_adapter_dir = lora_adapter_dir or self.config.get("training", {}).get("output_dir", "./output")
        output_dir = output_dir or self.config.get("merge", {}).get("output_dir", "./merged_model")
        merge_cfg = self.config.get("merge", {})
        low_ram = merge_cfg.get("low_ram", False)

        self.logger.info(f"=== Начало слияния весов ===")
        self.logger.info(f"Базовая модель: {base_model_name}")
        self.logger.info(f"LoRA-адаптер: {lora_adapter_dir}")
        self.logger.info(f"Выход: {output_dir}")
        self.logger.info(f"Low RAM режим: {low_ram}")
        self.logger.info(f"Управление swap: {manage_swap}")

        trust_remote = self.config['model'].get('trust_remote_code', True)

        if manage_swap:
            self._merge_with_swap(base_model_name, lora_adapter_dir, output_dir, merge_cfg, trust_remote)
        elif low_ram:
            self._merge_low_ram(base_model_name, lora_adapter_dir, output_dir, merge_cfg, trust_remote)
        else:
            self._merge_standard(base_model_name, lora_adapter_dir, output_dir, trust_remote)

        self.logger.info(f"=== Слияние завершено! Модель в: {output_dir} ===")

    def _merge_standard(self, base_model_name, lora_adapter_dir, output_dir, trust_remote):
        self.logger.info("Загрузка модели на CPU (FP16)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=False,
            load_in_4bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            trust_remote_code=trust_remote,
        )

        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_dir, trust_remote_code=trust_remote)

        self.logger.info(f"Подключение LoRA-адаптера из: {lora_adapter_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_dir,
            device_map={"": "cpu"},
        )

        self.logger.info("Слияние матриц весов (Merge)...")
        merged_model = model.merge_and_unload()

        self._save_model(merged_model, tokenizer, output_dir)

    def _merge_low_ram(self, base_model_name, lora_adapter_dir, output_dir, merge_cfg, trust_remote):
        cpu_swap_gb = merge_cfg.get("cpu_swap_gb", 40)
        max_shard_size = merge_cfg.get("max_shard_size", "5GB")

        self.logger.info(f"Low RAM режим. Swap: {cpu_swap_gb} ГБ...")

        with managed_swap(size_gb=cpu_swap_gb):
            self._do_merge(base_model_name, lora_adapter_dir, output_dir, max_shard_size, trust_remote)

    def _merge_with_swap(self, base_model_name, lora_adapter_dir, output_dir, merge_cfg, trust_remote):
        cpu_swap_gb = merge_cfg.get("cpu_swap_gb", 40)
        max_shard_size = merge_cfg.get("max_shard_size", "5GB")

        self.logger.info(f"Swap включён. Swap: {cpu_swap_gb} ГБ...")

        with managed_swap(size_gb=cpu_swap_gb):
            self._do_merge(base_model_name, lora_adapter_dir, output_dir, max_shard_size, trust_remote)

    def _do_merge(self, base_model_name, lora_adapter_dir, output_dir, max_shard_size, trust_remote):
        self.logger.info("Загрузка модели слоями (low_cpu_mem_usage)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=False,
            load_in_4bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            trust_remote_code=trust_remote,
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_dir, trust_remote_code=trust_remote)

        self.logger.info(f"Подключение LoRA-адаптера из: {lora_adapter_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_dir,
            device_map={"": "cpu"},
        )

        self.logger.info("Послойное слияние весов...")
        merged_model = model.merge_and_unload()

        self._save_model(merged_model, tokenizer, output_dir, max_shard_size)

    def _save_model(self, model, tokenizer, output_dir, max_shard_size="5GB"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Сохранение модели в: {output_dir} (~40 ГБ)...")
        model.save_pretrained(
            str(output_path),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(str(output_path))
        self.logger.info(f"Модель и токенизатор сохранены.")
