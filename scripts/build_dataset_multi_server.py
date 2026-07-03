import os
import json
import requests
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 1. ОБНОВЛЕННАЯ КОНФИГУРАЦИЯ СЕРВЕРОВ
# ==========================================
OUTPUT_FILE = "university_thinking_dataset.json"

# Список ваших серверов llama.cpp
SERVERS_POOL = [
    "http://195.133.13",
    "http://195.63.145"
]

# Потокобезопасная очередь для свободных серверов
available_servers = queue.Queue()
for server in SERVERS_POOL:
    available_servers.put(server)

# ==========================================
# 2. ПОТОКОБЕЗОПАСНАЯ ФУНКЦИЯ ЗАПРОСА К API
# ==========================================
def worker_query_api(context_chunk):
    """Берет свободный сервер из очереди, делает запрос и возвращает результат"""
    # Ждем и забираем адрес свободного сервера
    server_url = available_servers.get()
    
    system_prompt = (
        "Ты — ведущий научный методолог. Твоя задача — изучить фрагмент статьи, "
        "придумать сложный аналитический вопрос к нему, детально расписать логику "
        "рассуждения и выдать ответ. Ты должен вернуть результат СТРОГО в формате JSON "
        "со следующими ключами: 'prompt', 'thought', 'response'."
    )
    user_prompt = f"Фрагмент научной публикации:\n\"\"\"\n{context_chunk}\n\"\"\""
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        # Ставим таймаут повыше, так как сервера не самые мощные
        response = requests.post(server_url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            content_str = response.json()["choices"]["message"]["content"]
            return json.loads(content_str)
    except Exception as e:
        # Логируем ошибку конкретного сервера, но не роняем скрипт
        print(f"\n[Ошибка] Сервер {server_url} не ответил или вернул некорректный JSON.")
    finally:
        # В ЛЮБОМ СЛУЧАЕ возвращаем сервер обратно в пул свободных
        available_servers.put(server_url)
        
    return None

# ==========================================
# 3. МНОГОПОТОЧНЫЙ ДИСПЕТЧЕР
# ==========================================
def run_multi_server_generation(all_chunks):
    dataset_rows = []
    print(f"--> Запуск параллельной генерации на {len(SERVERS_POOL)} серверах...")
    
    # Количество потоков строго равно количеству серверов (в данном случае 2)
    # Это исключает отправку «параллельных» запросов на один слабый сервер
    num_threads = len(SERVERS_POOL)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Отправляем все чанки на выполнение
        futures = {executor.submit(worker_query_api, chunk): chunk for chunk in all_chunks}
        
        # tqdm красиво отображает общий прогресс по мере завершения потоков
        for future in tqdm(as_completed(futures), total=len(all_chunks), desc="Асинхронная o1 генерация"):
            llm_data = future.result()
            
            if llm_data and all(k in llm_data for k in ["prompt", "thought", "response"]):
                row = {
                    "system": "You are an AI assistant. Format your answers as follows: <Thought> Your thoughts (understanding, reasoning) </Thought> <output> Your answer </output>",
                    "prompt": llm_data["prompt"],
                    "response": f"<Thought>\n{llm_data['thought']}\n</Thought>\n<output>\n{llm_data['response']}\n</output>"
                }
                dataset_rows.append(row)
                
                # Потокобезопасное сохранение каждые 20 строк
                if len(dataset_rows) % 20 == 0:
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
                        
    # Финальная запись
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
        
    print(f"\n--> Генерация завершена. Собрано строк: {len(dataset_rows)}")
