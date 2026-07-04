import json
import random

# Исходные файлы из генераторов
TEXT_JSON = "university_text_dataset.json"
MATH_JSON = "university_math_dataset.json"

# Итоговые файлы для обучения
TRAIN_JSON = "university_train.json"
VAL_JSON = "university_val.json"

print("--> Начало процесса смешивания и разбиения датасетов...")

# 1. Читаем текстовый датасет
try:
    with open(TEXT_JSON, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    print(f"Загружено текстовых строк: {len(text_data)}")
except FileNotFoundError:
    text_data = []

# 2. Читаем математический датасет
try:
    with open(MATH_JSON, "r", encoding="utf-8") as f:
        math_data = json.load(f)
    print(f"Загружено математических строк: {len(math_data)}")
except FileNotFoundError:
    math_data = []

final_dataset = text_data + math_data

if not final_dataset:
    print("Ошибка: Нет данных для обработки.")
    exit(1)

# 3. Перемешиваем строки
random.seed(42)
random.shuffle(final_dataset)

# 4. РАСЧЕТ РАЗБИЕНИЯ (90% / 10%)
split_index = int(len(final_dataset) * 0.9)
train_dataset = final_dataset[:split_index]
val_dataset = final_dataset[split_index:]

# 5. Сохраняем файлы
with open(TRAIN_JSON, "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent=4)

with open(VAL_JSON, "w", encoding="utf-8") as f:
    json.dump(val_dataset, f, ensure_ascii=False, indent=4)

print(f"--> Успешно разделено!")
print(f"Для обучения (Train): {len(train_dataset)} строк сохранены в {TRAIN_JSON}")
print(f"Для валидации (Validation): {len(val_dataset)} строк сохранены in {VAL_JSON}")
