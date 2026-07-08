# PyRuQo1 — конвейер обучения reasoning-моделей

Обучение российских LLM (GigaChat-20B / GigaChat3-10B / YandexGPT-8B) методом QLoRA на синтетическом датасете с цепочками рассуждений (Chain-of-Thought).

## Конфигурация

Встроенные конфиги в `pyruqo1/config/`:
- `gigachat-20b.yaml` — GigaChat-20B-A3B (основной)
- `gigachat3-10b.yaml` — GigaChat3-10B-A1.8B
- `ygpt-5-lite-8b.yaml` — YandexGPT-5-Lite-8B

Пользовательские оверрайды: создайте `configs/<model_name>.yaml` для переопределения параметров.

## Быстрый старт

### 1. Установка зависимостей ОС

```bash
sudo apt update && sudo apt install ocrmypdf tesseract-ocr-rus build-essential cmake git python3-pip -y
```

### 2. Установка библиотеки

**Комплектование датасета** (torch 2.4.1 + marker-pdf 0.3.10):
```bash
python3 -m venv dsenv
source dsenv/bin/activate
pip install -e ".[ds,test]"
```

**Обучение** (torch 2.6.0, без marker-pdf):
```bash
python3 -m venv trainenv
source trainenv/bin/activate
pip install -e ".[train,test]"
```

**Тестирование GGUF** (опционально):
```bash
# При активированном trainenv выполнить
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python
```

### 3. Проверка системы

```bash
pyruqo1 check
```

### 4. Разрезание журналов

```bash
pyruqo1 split --input ./raw_journals --output-dir ./university_pdfs
```

### 5. Генерация датасета

**Гуманитарные тексты (один сервер, простой текст):**
```bash
pyruqo1 generate --input ./university_pdfs --mode simple --servers http://localhost:8079/v1/chat/completions --context-size 2048
```

**Математические тексты (LaTeX-формулы, несколько серверов):**
```bash
pyruqo1 generate --input ./math_pdfs --mode math --servers 'http://localhost:8079/v1/chat/completions,http://192.168.2.52:8181/v1/chat/completions'
```

Серверы задаются двумя способами:
- Повторяемый флаг: `--servers http://a --servers http://b`
- Один флаг с разделителем: `--servers 'http://a,http://b'`

Если серверы не заданы, будет выполнена попытка обратиться к GigaChat. Пользователь должен будет ввести `GigaChat Authorization Key (Client Secret)`. Данный режим реализован ввиду того, что reasoning-self-hosted модели, у которых можно учиться рассуждениям, сами рассуждения генерируют на английском языке. GigaChat не показывает рассуждений. В скриптах реализована принудительная имитация `Thought` и `output`. 

Режимы:
- `simple` — 1 сервер + гуманитарный текст (PDFParser)
- `math` — 1 сервер + математические тексты с LaTeX (Marker-парсер)

Размер контекста задается под будущую модель:
- для большой модели gigachat-20b на VRAM 24 Gb большой контекст невозможен, поэтому `--context-size 2048` (задан по умолчанию)
- для моделей gigachat3-10b, ygpt-5-lite-8b используется второй вариант `--context-size 8192`

### 6. Continual Pre-Training (CPT) — опционально

CPT дообучает модель на сыром доменном тексте (статьи из журналов) перед SFT.

**6a. Разрезание журналов:**
```bash
pyruqo1 split --input ./raw_journals --output-dir ./university_pdfs
```

**6b. Генерация CPT-датасета (сырой текст с LaTeX через marker-pdf):**
```bash
pyruqo1 generate --input ./university_pdfs --mode cpt --output cpt_dataset.json
```

**6c. CPT-обучение (LoRA rank=64, lr=1e-4, cosine scheduler):**
```bash
pyruqo1 train --model gigachat-20b --train-type cpt --train-file cpt_dataset.json
```

**6d. Merge CPT-адаптера с базовой моделью:**
```bash
pyruqo1 merge --model gigachat-20b --lora-dir ./o1_gigachat_university_lora --output-dir ./cpt_merged_model
```

Далее используйте `cpt_merged_model` как базовую модель для SFT (см. шаг 7 ниже).

### 7. SFT из структуры статей (Base)

Base-режим извлекает разделы статьи и формирует SFT-датасет без LLM:
- **Введение + Цель** → `prompt`
- **Материалы и методы + Результаты** → `<Thought>` (каждый абзац отдельный)
- **Выводы / Заключение** → `<output>`

Стандартная статья: УДК → Авторы → Организация → Аннотация → Ключевые слова → **Введение** → **Цель** → **Материалы и методы** → **Результаты** → **Выводы** → Список литературы.

**7a. Разрезание журналов:**
```bash
pyruqo1 split --input ./raw_journals --output-dir ./university_pdfs
```

**7b. Генерация Base-датасета (гуманитарный):**
```bash
pyruqo1 generate --input ./university_pdfs --mode base-hum
```

**7c. Генерация Base-датасета (математический):**
```bash
pyruqo1 generate --input ./university_pdfs --mode base-math
```

**7d. Обучение:**
```bash
pyruqo1 train --model gigachat-20b --train-file university_base_dataset.json
```

> **Примечание:** Статьи без разделов "Введение" или "Выводы/Заключение" пропускаются.

### 8. Объединение датасетов

**В один файл:**
```bash
pyruqo1 mixds university_thinking_dataset.json university_math_dataset.json --mode simple
```

**В два файла (train + val):**
```bash
pyruqo1 mixds university_thinking_dataset.json university_math_dataset.json --mode train_val
```
или можно
```bash
pyruqo1 mixds university_thinking_dataset.json --mode train_val
```

### 9. Обучение

**По одному файлу (university_train.json из корня):**
```bash
pyruqo1 train --model gigachat-20b --mode simple
```

**По двум файлам (train + val):**
```bash
pyruqo1 train --model gigachat-20b --mode train_val --train-file university_train.json --val-file university_val.json
```

**С параметрами micro (для маленьких датасетов):**
```bash
pyruqo1 train --model gigachat-20b --mode simple --dataset-type micro
```

**С параметрами micro + два файла:**
```bash
pyruqo1 train --model gigachat-20b --mode train_val --train-file university_train.json --val-file university_val.json --dataset-type micro
```

Флаг `-D, --dataset-type` управляет **параметрами обучения**:

| Параметр | `micro` (маленький датасет) | `big` (большой датасет) |
|---|---|---|
| `save_strategy` | `epoch` | `steps` |
| `logging_steps` | `2` | `10` |
| `eval_steps` | `2` | `50` |
| `save_steps` | `None` | `100` |

Для микро-датасета чекпоинты сохраняются раз в эпоху, логирование и валидация — каждые 2 шага (так как эпоха короткая).
Для большого датасета — чекпоинты каждые 100 шагов, логирование каждые 10, валидация каждые 50 шагов.

### 10. Слияние LoRA

```bash
pyruqo1 merge --model gigachat-20b --lora-dir ./o1_gigachat_university_lora --output-dir ./merged_o1_gigachat_university_lora
```
Параметры:
- `--config`, `-c`, `config_path`, Путь к YAML-конфигу, default=None
- `--model`, `-m`, `model_name`, Имя модели для загрузки дефолтного конфига, default=None
- `--base-model`, Путь/имя базовой модели, default=None
- `--lora-dir`, Путь к LoRA-адаптеру, default="./output"
- `--output-dir`, Директория для объединённой модели, default="./merged_model"
- `--manage-swap`, Автоматически управлять swap (создать/удалить), is_flag=True
- `--low-ram`, Режим низкого RAM (< 64 ГБ), is_flag=True

### 9. Конвертация в GGUF

```bash
# Деактивируем использованное для обучения моделей окружение trainenv, если нужно
deactivate

# Клонирование проекта llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Сборка проекта с поддержкой CUDA под RTX 3090
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Установка зависимостей конвертера. Сначала отключаем текущее окружение, потом создаем новое и ставим все в него
python3 -m venv llamacppenv
source llamacppenv/bin/activate
pip3 install -r requirements.txt

# Конвертация Hugging Face + обученный адаптер в GGUF(BF16)
python3 convert_hf_to_gguf.py ../merged_o1_gigachat_university_lora --outfile ../o1_gigachat_university_bf16.gguf

# Финальное квантование (Сжатие модели)
./build/bin/llama-quantize ../o1_gigachat_university_bf16.gguf ../o1_gigachat_university_Q4_K_M.gguf Q4_K_M

# В корневой директории появится готовый к локальному инференсу файл o1_gigachat_university_Q4_K_M.gguf размером около 13–14 ГБ
```

### 11. Тестирование GGUF

```bash
# Деактивируем llamacppenv (если нужно)
deactivate
# Активируем окружение для тестирования
cd ..
source trainenv/bin/activate

pyruqo1 test-gguf --model gigachat-20b --modelfile ./o1_gigachat_university_Q4_K_M.gguf --val-file ./university_val.json
```

**Параметры:**
- `--model`, `-m` — тип модели: `gigachat-20b`, `gigachat3-10b`, `ygpt-5-lite-8b`
- `--modelfile` — путь к GGUF-файлу модели
- `--val-file` — путь к валидационному датасету (JSON)
- `--res`, `-r` — путь к файлу для сохранения результатов (по умолчанию: `gguf_test_results_{model_type}.json`)
- `--num-samples`, `-n` — количество случайных заданий для теста (по умолчанию: 3)

**Пример с сохранением результатов:**
```bash
pyruqo1 test-gguf --model gigachat3-10b --modelfile ./o1_gigachat3_Q4_K_M.gguf --val-file ./university_val.json --res ./results_gigachat3.json
```

## CLI

```
pyruqo1 check      # Проверка системы
pyruqo1 split      # Разрезание журналов на статьи
pyruqo1 generate   # Генерация датасета из PDF
pyruqo1 mixds      # Объединение датасетов
pyruqo1 train      # QLoRA-обучение
pyruqo1 merge      # Слияние LoRA-адаптера
pyruqo1 test-gguf  # Тестирование GGUF-модели
```

## Python API

```python
from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer

config = load_config(model_name="gigachat-20b")
trainer = NPITrainer(config)
trainer.train()
```

```python
from pyruqo1.dataset import PDFParser, TextChunker, DatasetGenerator

parser = PDFParser()
texts = parser.parse_folder("./pdfs")

chunker = TextChunker(chunk_size=3500, overlap=500)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

generator = DatasetGenerator(servers=["http://localhost:8079/v1/chat/completions"])
generator.generate_from_chunks(chunks, "dataset.json", mode="simple")
```

**CPT (Continual Pre-Training) через Python API:**

```python
from pyruqo1.dataset import CPTParser, CPTChunker

parser = CPTParser(chunk_size=3500, overlap=500)
chunks = parser.parse_folder("./pdfs")

chunker = CPTChunker(chunk_size=3500, overlap=500)
raw_texts = []
for chunk in chunks:
    raw_texts.extend(chunker.chunk(chunk))

import json
cpt_data = [{"text": t} for t in raw_texts]
with open("cpt_dataset.json", "w", encoding="utf-8") as f:
    json.dump(cpt_data, f, ensure_ascii=False, indent=4)
```

**Base (SFT из структуры статьи) через Python API:**

```python
from pyruqo1.dataset import BaseParser

parser = BaseParser(chunk_size=3500, overlap=500)
sections_list = parser.parse_folder("./pdfs")

import json

dataset_rows = []
for sections in sections_list:
    prompt_text = sections.get("introduction", "").strip()
    conclusion_text = sections.get("conclusion", "").strip()

    if not prompt_text or not conclusion_text:
        continue

    methods_paragraphs = sections.get("methods", [])
    results_paragraphs = sections.get("results", [])

    thought_parts = []
    for paragraph in methods_paragraphs:
        p = paragraph.strip()
        if p:
            thought_parts.append(f"<Thought>\n{p}\n</Thought>")
    for paragraph in results_paragraphs:
        p = paragraph.strip()
        if p:
            thought_parts.append(f"<Thought>\n{p}\n</Thought>")

    response_text = "\n\n".join(thought_parts) + f"\n\n<output>\n{conclusion_text}\n</output>"

    dataset_rows.append({
        "system": "Ты ученый. Проанализируй задачу и реши ее.",
        "prompt": prompt_text,
        "response": response_text,
    })

with open("base_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_rows, f, ensure_ascii=False, indent=4)
```

```python
from pyruqo1.merge import LORAMerger
from pyruqo1.utils.swap import get_managed_swap_path, remove_swap_file

config = load_config(model_name="gigachat-20b")
merger = LORAMerger(config)
merger.merge(manage_swap=True)

# После merge можно отключить swap
swap_path = get_managed_swap_path()
if swap_path:
    remove_swap_file(swap_path)
```

## Структура

```
pyruqo1/                   # основная библиотека
├── config/            # YAML-конфиги моделей
├── utils/             # логгер, системные утилиты, swap
├── dataset/           # парсинг, чанкинг, генерация датасета
│   ├── parser.py      # PDFParser (простой текст)
│   ├── math_parser.py # MathParser (LaTeX через marker)
│   ├── cpt_parser.py  # CPTParser (сырой текст для CPT)
│   ├── base_parser.py # BaseParser (структура статьи для SFT)
│   ├── chunker.py     # TextChunker, MathChunker, CPTChunker
│   ├── generator.py   # DatasetGenerator (SFT-генерация)
│   └── splitter.py    # JournalSplitter
├── training/          # QLoRA-обучение (SFT + CPT)
├── merge/             # слияние LoRA
├── gguf/              # тестирование GGUF-моделей
└── cli.py             # CLI (click)

examples_scripts/      # старые скрипты (сохранены для совместимости)
configs/               # пользовательские YAML-оверрайды (опционально)
logs_example/          # примеры результатов обучения
micro_datasets/        # микро-датасеты, которые были использованы для получения примеров результатов обучения
tests/                 # базовые тесты
```

## Зависимости

### 1. Комплектование датасета (DS)

torch 2.4.1 + marker-pdf 0.3.10:
```bash
pip install -e ".[ds]"
```

### 2. Обучение (Train)

torch 2.6.0, без marker-pdf:
```bash
pip install -e ".[train]"
```

### 3. Тестирование GGUF

llama-cpp-python с поддержкой CUDA (устанавливается из исходников llama.cpp):
```bash
# См. инструкцию в разделе "Конвертация в GGUF"
```

Для тестирования GGUF-моделей используется `llama-cpp-python`, который устанавливается через `pip3 install llama-cpp-python` в окружении `llamacppenv`.

## Управление swap

Для merge и GGUF конвертации требуется ~80 ГБ дискового пространства и 60+ ГБ RAM.

```bash
# Автоматическое управление swap (создать перед, удалить после)
pyruqo1 merge --model gigachat-20b --manage-swap
```

## Минимальные технические требования

- Ubuntu 22.04
- 64 RAM
- 24 VRAM

## Подробнее

- Подробная документация, пошаговое руководство и примеры результатов — в [EXAMPLESINFO.md](EXAMPLESINFO.md).
- Запуск local-host-llm в [LOCALHOSTLLM.md](LOCALHOSTLLM.md)


## License

MIT
