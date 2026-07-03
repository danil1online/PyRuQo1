import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

BASE_MODEL_NAME = "ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16"
LORA_ADAPTER_DIR = "../o1_gigachat_university_lora"
OUTPUT_DIR = "../merged_o1_gigachat_university"

print("--> Запуск оптимизированного слияния для ПК с 64 ГБ RAM...")

# 1. Загружаем только токенизатор
tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, trust_remote_code=True)

# 2. Загружаем базовую модель с жестким ограничением памяти
print("Загрузка базовой модели частями (low_cpu_mem_usage)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    load_in_8bit=False,
    load_in_4bit=False,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,       # КРИТИЧЕСКИ ВАЖНО: загружает модель послойно, а не целиком в RAM сразу
    trust_remote_code=True
)

# 3. Подключаем адаптер
print("Подключение LoRA-адаптера...")
model = PeftModel.from_pretrained(
    base_model, 
    LORA_ADAPTER_DIR,
    device_map={"": "cpu"}
)

# 4. Проводим слияние
print("Слияние весов (Merge)...")
merged_model = model.merge_and_unload()

# Очищаем временный мусор в памяти перед сохранением
gc.collect()

# 5. Пошаговое сохранение на диск
print(f"Сохранение результата в {OUTPUT_DIR}...")
merged_model.save_pretrained(
    OUTPUT_DIR, 
    safe_serialization=True, 
    max_shard_size="2GB"          # Разбиваем на мелкие файлы по 2 ГБ, чтобы не перегружать RAM при записи
)

tokenizer.save_pretrained(OUTPUT_DIR)
print("--> Слияние успешно завершено!")
