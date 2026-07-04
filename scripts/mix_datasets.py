import json
import random

# Пути к вашим сгенерированным файлам
TEXT_JSON = "university_thinking_dataset.json"
MATH_JSON = "university_math_dataset.json"
FINAL_JSON = "university_text_dataset.json" # Этот файл пойдет в train_qlora.py

print("--> Начало процесса смешивания датасетов...")

# Читаем текстовый датасет
try:
    with open(TEXT_JSON, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    print(f"Загружено текстовых строк: {len(text_data)}")
except FileNotFoundError:
    text_data = []
    print(f"[Предупреждение] Файл {TEXT_JSON} не найден, пропускаем.")

# Читаем математический датасет
try:
    with open(MATH_JSON, "r", encoding="utf-8") as f:
        math_data = json.load(f)
    print(f"Загружено математических строк: {len(math_data)}")
except FileNotFoundError:
    math_data = []
    print(f"[Предупреждение] Файл {MATH_JSON} не найден, пропускаем.")

# Склеиваем два массива в один общий список
final_dataset = text_data + math_data

if not final_dataset:
    print("Ошибка: Оба исходных файла пусты или не найдены. Смешивание отменено.")
    exit(1)

# КРИТИЧЕСКИ ВАЖНО: Перемешиваем строки случайным образом.
# Это нужно, чтобы в процессе обучения (в рамках одного батча) 
# модель видела и обычный текст, и формулы. Так она не будет "забывать" навыки.
random.seed(42) # Фиксируем зерно для воспроизводимости
random.shuffle(final_dataset)

# Сохраняем итоговый объединенный датасет
with open(FINAL_JSON, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)

print(f"--> Успешно создано. Итоговый датасет содержит {len(final_dataset)} строк.")
print(f"Файл готов к обучению: {FINAL_JSON}")
