import os
import sys
import json
import random
from llama_cpp import Llama

# ==========================================
# 1. КОНФИГУРАЦИЯ ТЕСТА
# ==========================================
MODEL_PATH = "./university_model_Q4_K_M.gguf"
VAL_DATASET_PATH = "./university_val.json" # Берем задачи отсюда!
NUM_TEST_SAMPLES = 3                       # Сколько случайных задач взять на экзамен

if not os.path.exists(MODEL_PATH):
    print(f"[Ошибка] Файл модели не найден по пути: {MODEL_PATH}")
    sys.exit(1)

print("--> Инициализация модели GigaChat-20B в llama.cpp на CUDA...")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

# ==========================================
# 2. АВТОМАТИЧЕСКАЯ ЗАГРУЗКА ЗАДАНИЙ
# ==========================================
print(f"Загрузка проверочных заданий из {VAL_DATASET_PATH}...")
try:
    with open(VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Выбираем случайные задачи из валидационной выборки
    random.seed(42) # Фиксируем seed, чтобы при повторном запуске тесты были те же самые
    test_samples = random.sample(val_data, min(NUM_TEST_SAMPLES, len(val_data)))
except Exception as e:
    print(f"[Предупреждение] Не удалось загрузить {VAL_DATASET_PATH}: {e}")
    print("Используются резервные встроенные промпты.")
    # Резервный вариант, если файла валидации нет под рукой
    test_samples = [
        {"prompt": "Найди аналитическое решение дифференциального уравнения: dy/dx + 2xy = x * e^(-x^2) при y(0) = 1.", "response": "Эталон отсутствует"},
        {"prompt": "Выведи уравнение изменения предела прочности сплава от концентрации нанотрубок x.", "response": "Эталон отсутствует"}
    ]

SYSTEM_PROMPT = (
    "Вы — ИИ-помощник. Отформатируйте свои ответы следующим образом: "
    "<Thought> Ваши мысли (понимание, рассуждения) </Thought> <output> Ваш ответ </output>"
)

# ==========================================
# 3. ЗАПУСК ЭКЗАМЕНА
# ==========================================
print("\n--> Запуск валидационного тестирования...\n")

for i, sample in enumerate(test_samples, 1):
    user_query = sample["prompt"] if isinstance(sample, dict) else sample
    
    print("=" * 80)
    print(f"ЗАДАНИЕ №{i} ИЗ ВАЛИДАЦИОННОЙ ВЫБОРКИ:")
    print(user_query)
    print("=" * 80)
    
    # Если мы загрузили данные из JSON, покажем эталон для сравнения человеком
    if isinstance(sample, dict) and "response" in sample:
        print("\n[ЭТАЛОН ИЗ ДАТАСЕТА (Как ответила o1-генератор)]:")
        print(sample["response"])
        print("-" * 40)
    
    print("\n[ОТВЕТ ВАШЕЙ ДООБУЧЕННОЙ МОДЕЛИ]:\n")
    
    full_prompt = (
        f"Система: {SYSTEM_PROMPT}\n\n"
        f"Пользователь: {user_query}\n\n"
        f"Система: "
    )
    full_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    response_stream = llm(
        full_prompt,
        max_tokens=1000,
        temperature=0.2,
        repeat_penalty=1.2,
        stop=["Пользователь:", "Система:", "<|im_end|>", "</output>"],
        stream=True
    )
    
    #print("<Thought>", end="", flush=True)
    for chunk in response_stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            if "text" in choice:
                print(choice["text"], end="", flush=True)
            elif "delta" in choice and "content" in choice["delta"]:
                print(choice["delta"]["content"], end="", flush=True)
                
    print("\n\n")

print("--> Экзамен завершен. Оцените точность формул и логики вашей модели!")