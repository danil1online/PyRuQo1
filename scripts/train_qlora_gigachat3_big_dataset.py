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
from trl import SFTTrainer
from trl import SFTConfig  # Добавить эту строку в блок импортов

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
# Базовая модель от Сбера (20 миллиардов параметров)
MODEL_NAME = "ai-sage/GigaChat3-10B-A1.8B-base"
# Папка для сохранения итогового LoRA-адаптера
OUTPUT_DIR = "./o1_gigachat3_university_lora"
# Путь к вашему сгенерированному локальному файлу, если он один
DATASET_NAME = "university_text_dataset.json" 

print("--> Начинается процесс подготовки к обучению...")

# ==========================================
# 2. ПОДГОТОВКА ДАТАСЕТА И ФОРМАТИРОВАНИЕ
# ==========================================
# Загружаем ваш собственный сгенерированный JSON датасет
data_files = {
    "train": "university_train.json",
    "validation": "university_val.json"
}

dataset = load_dataset("json", data_files=data_files)
print(f"Загружен train: {len(dataset['train'])} строк, validation: {len(dataset['validation'])} строк.")

# Новая функция форматирования, которая обрабатывает ОДИН конкретный пример (строку)
def format_single_example(example):
    return {
        "text": (
            f"Система: {example['system']}\n\n"
            f"Пользователь: {example['prompt']}\n\n"
            f"Система: {example['response']}"
        )
    }

# Принудительно мапим датасет, создавая в нём ОДИН чистый столбец "text"
# Это полностью отключает внутренние "угадайки" библиотеки TRL
print("Принудительное форматирование колонок датасета...")
dataset = dataset.map(format_single_example, remove_columns=dataset["train"].column_names)

# Либо, если есть только один train-json:
#dataset = load_dataset("json", data_files=DATASET_NAME, split="train")

# Функция форматирования данных для модели-рассуждалки.
# Она упаковывает промпт и ответ в структуру с тегами <Thought> (Мысли).
# Измените ключи 'instruction', 'thought', 'output', если структура вашего датасета отличается.
# Новая функция форматирования для SFTTrainer под готовый JSON:
def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = (
            f"Система: {examples['system'][i]}\n\n"
            f"Пользователь: {examples['prompt'][i]}\n\n"
            f"Система: {examples['response'][i]}"
        )
        output_texts.append(text)
    return output_texts

# ==========================================
# 3. НАСТРОЙКА КВАНТОВАНИЯ И ЗАГРУЗКА МОДЕЛИ
# ==========================================
# Настройки BitsAndBytes для сжатия модели до 4 бит (QLoRA)
# Это критически важно, чтобы уместить 20B модель в 24 ГБ VRAM вашей RTX 3090.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # Высокоточный тип данных для квантования llm
    bnb_4bit_compute_dtype=torch.bfloat16, # Вычисления будут идти в BF16 (быстро на RTX 3090)
    bnb_4bit_use_double_quant=True        # Дополнительное квантование весов для экономии памяти
)

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Устанавливаем токен заполнения

# Загружаем саму модель в 4-битном режиме
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",                     # Автоматически займет GPU 0
    trust_remote_code=True
)

# Подготавливаем модель к обучению в пониженной точности
model = prepare_model_for_kbit_training(model)

# ==========================================
# 4. НАСТРОЙКА LORA АДАПТЕРА
# ==========================================
# Настраиваем конфигурацию PEFT/LoRA
peft_config = LoraConfig(
    r=16,          # Ранг матриц. 16 или 32 — золотой стандарт для обучения стилю/логике
    lora_alpha=32, # Коэффициент масштабирования (обычно берется как r * 2)
    # Нацеливаемся на слои внимания (Attention). Для архитектур типа GigaChat/Mistral
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Обучаем только базовые слои внимания
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Оборачиваем модель в LoRA-адаптер
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Выведет процент обучаемых параметров (~1-2%)

# ==========================================
# 5. ГИПЕРПАРАМЕТРЫ ТРЕНИРОВКИ (ОПТИМИЗАЦИЯ ПОД VRAM)
# ==========================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # Минимальный размер батча для экономии VRAM
    gradient_accumulation_steps=8,     # Накапливаем градиенты за 8 шагов (эффективный батч = 8)
    learning_rate=2e-4,                # Стандартный шаг обучения для LoRA
    logging_steps=10,                  # Как часто выводить логи в консоль
    num_train_epochs=1,                # 1 эпохи обычно достаточно для адаптации стиля
    # Страничный оптимизатор: сбрасывает излишки памяти в RAM вашего ПК (в ваши 256 ГБ)
    optim="paged_adamw_32bit",         
    # Экономит до 30% видеопамяти, не пересчитывая все веса на проходе вперед
    gradient_checkpointing=True,       
    fp16=False,
    bf16=True,                             # Обучение в полуточности
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    save_strategy="steps", 
    save_steps=100,                     # Сохранять чекпоинт каждые 100 шагов
    report_to="none",                   # Отключаем отправку логов в сторонние сервисы (wandb)
    eval_strategy="steps",              # Считать метрики валидации по шагам
    eval_steps=50,                      # Запускать валидацию каждые 50 шагов обучения
    per_device_eval_batch_size=1,      # Размер батча для валидации
    do_eval=True,                      # Включить режим оценки
    max_seq_length=2048,
    dataset_text_field="text",
)

# ==========================================
# 6. ЗАПУСК ОБУЧЕНИЯ
# ==========================================
# Используем SFTTrainer (Supervised Fine-Tuning) из библиотеки TRL
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    args=training_args, # Трейнер сам заберет длину контекста отсюда!
)

print("--> Окружение настроено. Запуск процесса обучения...")
trainer.train()

# Сохраняем финальные обученные веса LoRA-адаптера
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"--> Обучение успешно завершено! Адаптер сохранен в: {OUTPUT_DIR}")