import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
# Меняем базовую модель на YandexGPT-5-Lite-8B
MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-pretrain"
OUTPUT_DIR = "./o1_yandex_university_lora"

print("--> Начинается процесс подготовки к обучению...")

# ==========================================
# 2. ПОДГОТОВКА ДАТАСЕТА (ПЕРЕВОД В ЧАТ-ФОРМАТ CHATML)
# ==========================================
data_files = {
    "train": "university_train.json",
    "validation": "university_val.json"
}
dataset = load_dataset("json", data_files=data_files)
print(f"Загружен train: {len(dataset['train'])} строк, validation: {len(dataset['validation'])} строк.")

# ВНИМАНИЕ: Принудительно размечаем датасет под системные токены ChatML
def format_single_example(example):
    return {
        "text": (
            f"<|im_start|>system\n{example['system']}<|im_end|>\n"
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['response']}<|im_end|>"
        )
    }

print("Принудительное форматирование колонок датасета под ChatML...")
dataset = dataset.map(format_single_example, remove_columns=dataset["train"].column_names)

# ==========================================
# 3. НАСТРОЙКА КВАНТОВАНИЯ И ЗАГРУЗКА МОДЕЛИ
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Загружаем токенизатор YandexGPT
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    use_fast=False,   # <-- Добавьте эту строку
    legacy=False
)
tokenizer.pad_token = tokenizer.eos_token

# Загружаем саму модель YandexGPT с флагом trust_remote_code
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # Обязательно для кастомной архитектуры Яндекса
)

model = prepare_model_for_kbit_training(model)

# ==========================================
# 4. НАСТРОЙКА LORA АДАПТЕРА (РАСШИРЕННАЯ)
# ==========================================
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    # ВНИМАНИЕ: Для Llama-like архитектуры Яндекса добавляем проекции up_proj, down_proj, gate_proj.
    # Это позволит модели лучше усвоить математические токены.
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# 5. ГИПЕРПАРАМЕТРЫ ТРЕНИРОВКИ (ОПТИМИЗАЦИЯ ПОД VRAM)
# ==========================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    learning_rate=2e-4, 
    logging_steps=10, 
    num_train_epochs=1, 
    optim="paged_adamw_32bit", 
    gradient_checkpointing=True, 
    fp16=False,
    bf16=True, 
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    save_strategy="steps",
    save_steps=100, 
    report_to="none", 
    eval_strategy="steps", 
    eval_steps=50, 
    per_device_eval_batch_size=1, 
    do_eval=True, 
    max_seq_length=8192,       # Оставляем заученный вами лимит в 2048 токенов
    dataset_text_field="text",
)

# ==========================================
# 6. ЗАПУСК ОБУЧЕНИЯ
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    args=training_args,
)

print("--> Окружение настроено. Запуск процесса обучения YandexGPT...")
trainer.train()

# Сохраняем финальные обученные веса LoRA-адаптера
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"--> Обучение успешно завершено! Адаптер сохранен в: {OUTPUT_DIR}")
