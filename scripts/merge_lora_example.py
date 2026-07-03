import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. КОНФИГУРАЦИЯ ПУТЕЙ
# ==========================================
# Путь к оригинальной базовой модели от Сбера
BASE_MODEL_NAME = "ai-sage/GigaChat-20B-A3B-instruct-v1.5"

# Путь к папке, куда ваш скрипт обучения сохранил LoRA-адаптер
LORA_ADAPTER_DIR = "./o1_gigachat_university_lora"

# Путь, куда сохранить готовую объединенную модель (в формате Hugging Face FP16)
OUTPUT_DIR = "./merged_o1_gigachat_university"

print("--> Начало процесса слияния весов...")

# ==========================================
# 2. ЗАГРУЗКА БАЗОВОЙ МОДЕЛИ НА CPU
# ==========================================
print(f"Загрузка базовой модели: {BASE_MODEL_NAME}")
print("Используется системная RAM (256 ГБ), так как модель весит ~40 ГБ...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    load_in_8bit=False,                  # НЕ квантуем при слиянии
    load_in_4bit=False,                  # НЕ квантуем при слиянии
    torch_dtype=torch.float16,           # Базовый тип данных (FP16)
    device_map={"": "cpu"},              # Принудительно загружаем ВСЁ в оперативную память
    trust_remote_code=True
)

# Загружаем токенизатор из папки адаптера (или базовой)
tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, trust_remote_code=True)

# ==========================================
# 3. ПОДКЛЮЧЕНИЕ И СЛИЯНИЕ LORA
# ==========================================
print(f"Загрузка LoRA-адаптера из: {LORA_ADAPTER_DIR}")
# Оборачиваем базовую модель в PeftModel, подключая веса адаптера
model = PeftModel.from_pretrained(
    base_model, 
    LORA_ADAPTER_DIR,
    device_map={"": "cpu"}               # Также держим в RAM
)

print("Запуск процесса слияния матриц (Merge)...")
# Метод merge_and_unload физически складывает матрицы весов LoRA с базовыми
# и удаляет из памяти больше не нужные структуры адаптера.
merged_model = model.merge_and_unload()

# ==========================================
# 4. СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ==========================================
print(f"Сохранение объединенной модели в: {OUTPUT_DIR}")
print("Это займет несколько минут, так как на диск запишется около 40 ГБ данных...")

# Сохраняем веса новой полноценной модели
merged_model.save_pretrained(
    OUTPUT_DIR, 
    safe_serialization=True,             # Сохраняем в современном и безопасном формате .safetensors
    max_shard_size="5GB"                 # Разбиваем модель на файлы по 5 ГБ для удобства
)

# Не забываем сохранить токенизатор в ту же папку (он нужен для GGUF)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n--> Слияние успешно завершено!")
print(f"Полноразмерная модель готова к квантованию в GGUF. Путь: {OUTPUT_DIR}")
