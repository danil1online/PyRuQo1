import json
import random

# Исходные файлы из генераторов
TEXT_JSON = "university_thinking_dataset.json"
MATH_JSON = "university_math_dataset.json"

# Итоговые файлы для обучения
TRAIN_JSON = "university_train.json"
VAL_JSON = "university_val.json"


# ==========================================
# 1. ФУНКЦИЯ ОЧИСТКИ JSON ИЗ OUTPUT
# ==========================================
def clean_json_from_output(response_text):
    """
    Находит маркдаун-блок ```json внутри тега <output>, 
    десериализует его и превращает в красивый читаемый текст.
    """
    if "```json" in response_text:
        try:
            # Извлекаем контент, который находится строго между <output> и </output>
            parts_output = response_text.split("<output>")
            thought_part = parts_output[0]  # Всё, что до <output> (включая <Thought>...</Thought>)
            output_content = parts_output[1].split("</output>")[0].strip()
            
            # Очищаем от маркдаун-тегов кусок с JSON
            json_str = output_content.replace("```json", "").replace("```", "").strip()
            
            # Декодируем строку в Python-словарь
            data = json.loads(json_str)
            
            # Собираем текстовый ответ из полей JSON (учитываем возможную разницу в ключах)
            question = data.get('question', data.get('analytical_question', ''))
            answer = data.get('answer', '')
            
            clean_answer = f"**Аналитический вопрос:** {question}\n\n**Ответ и обоснование:** {answer}"
            
            # Собираем итоговую строку обратно с тегами
            return f"{thought_part}<output>\n{clean_answer}\n</output>"
        except Exception:
            # Если JSON был битым или split упал — возвращаем текст без изменений (страховка)
            return response_text
    return response_text


# ==========================================
# 2. ОСНОВНОЙ БЛОК ЧТЕНИЯ И СМЕШИВАНИЯ
# ==========================================
print("--> Начало процесса смешивания, очистки и разбиения датасетов...")

raw_final_dataset = []

# Читаем текстовый датасет
try:
    with open(TEXT_JSON, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    print(f"Загружено текстовых строк: {len(text_data)}")
    raw_final_dataset.extend(text_data)
except FileNotFoundError:
    print(f"[Предупреждение] Файл {TEXT_JSON} не найден, пропускаем.")

# Читаем математический датасет
try:
    with open(MATH_JSON, "r", encoding="utf-8") as f:
        math_data = json.load(f)
    print(f"Загружено математических строк: {len(math_data)}")
    raw_final_dataset.extend(math_data)
except FileNotFoundError:
    print(f"[Предупреждение] Файл {MATH_JSON} не найден, пропускаем.")

if not raw_final_dataset:
    print("Ошибка: Оба исходных файла пусты или не найдены. Смешивание отменено.")
    exit(1)


# ==========================================
# 3. ВЫЗОВ ОЧИСТКИ ДЛЯ КАЖДОЙ СТРОКИ
# ==========================================
print("Запуск распаковки JSON-оберток из полей ответов...")
processed_dataset = []

for row in raw_final_dataset:
    # Вызываем очистку для поля 'response' в каждой строке датасета
    cleaned_response = clean_json_from_output(row["response"])
    
    # Сохраняем обновленный результат
    row["response"] = cleaned_response
    processed_dataset.append(row)


# ==========================================
# 4. ПЕРЕМЕШИВАНИЕ И РАЗБИЕНИЕ (90% / 10%)
# ==========================================
# КРИТИЧЕСКИ ВАЖНО: Перемешиваем строки, чтобы модель видела формулы и текст вперемешку
random.seed(42) 
random.shuffle(processed_dataset)

split_index = int(len(processed_dataset) * 0.9)
train_dataset = processed_dataset[:split_index]
val_dataset = processed_dataset[split_index:]

# Сохраняем итоговые файлы
with open(TRAIN_JSON, "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent=4)

with open(VAL_JSON, "w", encoding="utf-8") as f:
    json.dump(val_dataset, f, ensure_ascii=False, indent=4)

print(f"--> Успешно выполнено!")
print(f"Очищено и подготовлено для Train: {len(train_dataset)} строк ({TRAIN_JSON})")
print(f"Очищено и подготовлено для Validation: {len(val_dataset)} строк ({VAL_JSON})")
