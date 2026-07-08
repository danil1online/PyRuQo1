# PyRuQo1 — конвейер обучения reasoning-моделей

Обучение российских LLM (GigaChat-20B / GigaChat3-10B / YandexGPT-8B) методом QLoRA на синтетических и реальных датасетах с цепочками рассуждений (Chain-of-Thought).

## Конфигурации моделей

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

### 5. Генерация датасета через "образцовые" LLM

Данный режим предполагает, что 
1. Чанк статьи передается в образцовую модель (запущенную локально, или в GigaChat) для генерации вопросов
2. Чанк статьи вместе с вопросом подается в образцовую модель для генерации ответа "с рассуждениями"
3. Системный промпт, вопросы и ответы объединяются в датасет.

Далее приведены примеры запуска и пояснения  

**Гуманитарные тексты (один сервер, простой текст, одна запись датасета -- до 8192 токенов):**
```bash
pyruqo1 generate --input ./university_pdfs --mode simple --servers http://localhost:8079/v1/chat/completions --context-size 8192
```

**Математические тексты (LaTeX-формулы, несколько серверов):**
```bash
pyruqo1 generate --input ./math_pdfs --mode math --servers 'http://localhost:8079/v1/chat/completions,http://192.168.2.52:8181/v1/chat/completions'
```

Серверы задаются двумя способами:
- Повторяемый флаг: `--servers http://a --servers http://b`
- Один флаг с разделителем: `--servers 'http://a,http://b'`

Если серверы (`--servers`) не заданы, будет выполнена попытка обратиться к `GigaChat`. Пользователь должен будет ввести `GigaChat Authorization Key (Client Secret)`. Данный режим реализован ввиду того, что reasoning-self-hosted модели, у которых можно учиться рассуждениям, сами рассуждения генерируют на английском языке. GigaChat не показывает рассуждений. В скриптах реализована принудительная имитация `Thought` и `output`. 

Режимы:
- `simple` — 1 сервер + гуманитарный текст (PDFParser)
- `math` — 1 сервер + математические тексты с LaTeX (Marker-парсер)

Размер контекста задается под будущую модель:
- для большой модели gigachat-20b на VRAM 24 Gb большой контекст невозможен, поэтому `--context-size 2048` (задан по умолчанию)
- для моделей gigachat3-10b, ygpt-5-lite-8b используется второй вариант `--context-size 8192`

### 6. Генерация датасета парсингом текста статьи

Стандартная статья содержит УДК → Авторы → Организация → Аннотация → Ключевые слова → **Введение** → **Цель** → **Материалы и методы** → **Результаты** → **Выводы** → Список литературы. Поэтому данный режим предполагает, что текст статьи уже структурирован под то, чтобы оформить его в датасет `system prompt` | `prompt` | `response`: 

- В `system prompt` или "Ты ученый социолог-экономист. Проанализируй задачу и реши ее.", или "Ты ученый в области математики, физики, техники и информатики. Проанализируй задачу и реши ее."
- Содержание разделов "Введение" + "Цель" --> в `prompt`
- Каждый отдельный абзац разделов "Материалы и методы" и "Результаты" --> в отдельный <Thought>...</Thought> в `response`
- Содержание разделов "Выводы" / "Заключение" --> в `<output>...</output>` в `response`

**6a. Генерация Base-датасета (гуманитарный):**
```bash
pyruqo1 generate --input ./university_pdfs --mode base-hum
```

**6b. Генерация Base-датасета (математический):**
```bash
pyruqo1 generate --input ./university_pdfs --mode base-math
```

> **Примечание:** Статьи без разделов "Введение" или "Выводы/Заключение" пропускаются.

### 7. Объединение датасетов / разделение на train & validation

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

### 8. Обучение (SFT)

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

### 9. Слияние LoRA

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

### 10. Конвертация в GGUF

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

### (Опционально) 12. Sequential Fine-Tuning через пользовательские оверрайды конфигов на примере Continual Pre-Training (CPT) --> SFT

В общем случае для Sequential Fine-Tuning необходимо:
- Шаг 1 — обучить первый адаптер:
```bash
pyruqo1 train -m gigachat-20b --train-file dataset1.json --output-dir ./lora_step1
```
- Шаг 2 — слить адаптер с базовой моделью:
```bash
pyruqo1 merge -m gigachat-20b --lora-dir ./lora_step1 --output-dir ./merged_step1
```
- Шаг 3 — обучить второй адаптер на мерджнутой модели. CLI train не имеет флага --base-model, поэтому нужно создать пользовательский конфиг:

Создать configs/merged_step1.yaml на основе конфига реальной базовой модели, в котором заменить:
```sh
...
model:
...
  name: "./merged_step1"
...
```
Запустить обучение:
```bash
pyruqo1 train -m merged_step1 --train-file dataset2.json --output-dir ./lora_step2
```
Как работает

[LORAMerger](pyruqo1/merge/merger.py:29) принимает base_model_name — по умолчанию из config['model']['name'], но можно переопределить. [NPITrainer](pyruqo1/training/trainer.py:39) загружает модель из config['model']['name'] — поэтому достаточно в конфиге указать путь к мерджнутой модели. Конфиги из configs/<model_name>.yaml имеют приоритет над встроенными [см](pyruqo1/config/__init__.py:32-40).

Важные нюансы Sequential Fine-Tuning
- После нескольких итераций может накапливаться catastrophic forgetting — модель может "забыть" навыки из ранних данных
- На каждой итерации модель становится больше (требует больше RAM/VRAM при merge)
- Рекомендуется валидировать модель на отдельных данных после каждой итерации

*На основе Sequential Fine-Tuning может быть реализовано как SFT --> SFT, так и CPT --> SFT*

***CPT --> SFT***

Этап 1. Загрузка «базы знаний» (Continual Pre-training / CPT)* Цель: научить модель новому языку, терминам, формулам и специфическим сокращениям из статей. Необходимо подобрать научные статьи, разрезать их на куски (чанки) и обучить модель в режиме сырого предсказания следующего токена (как при обучении базовых моделей). На этом этапе не используются вопросы и ответы. Этот этап закладывает фундамент знаний. Для этого этапа у LoRA
- выставляют большой ранг (например, [rank](pyruqo1/training/trainer.py#L127) = 64 или 128), чтобы модель могла физически "вместить" новые концепты;
- LR Scheduler Type `cosine` (сейчас `constant`). Постоянный шаг обучения хорош для быстрой подгонки под формат (SFT) на небольшом датасете. При CPT на сырых научных статьях модели критически важно плавно снижать скорость обучения к концу процесса. Это помогает ей зафиксировать глубокие веса знаний и не позволяет "забыть" то, что она выучила в начале.
- Learning Rate (Скорость обучения). Сейчас (SFT) `2.0e-4` (`0.0002`). Нужно для CPT уменьшить в 2–5 раз, например, до `5.0e-5` или `1.0e-4`. Поскольку для CPT ранг LoRA будет увеличен (с `r: 16` до `64` или `128`), матрица адаптера станет больше. Слишком высокий learning rate при обучении на неразмеченном тексте может привести к "взрыву градиента" или к тому, что модель начнет ломать свои базовые языковые навыки (перестанет связно говорить по-русски).
- Warmup Ratio (Разогрев). Сейчас (SFT) `0.03` Нужно для CPT увеличить до `0.05` – `0.10`. Модели нужно больше времени (больше шагов), чтобы плавно адаптироваться к совершенно новому для нее стилю научных статей и формул, прежде чем выйти на пиковый `learning rate`.
- Epochs (Эпохи). Сейчас (SFT) 1. Нужно для CPT оставить 1 эпоху (максимум 2, если датасет маленький). Прогон по одним и тем же научным текстам более одного-двух раз при предсказании следующего токена быстро приведет к зазубриванию (overfitting). Модель начнет выдавать точные цитаты из статей вместо понимания физики. Лучше расширить сам датасет статьями, чем крутить эпохи.

Этап 2. Обучение решать задачи (Instruction Fine-Tuning / SFT)*  Цель: научить модель применять законы электротехники и отвечать на ваши вопросы. Модель после CPT обучения будет просто «продолжать» введенный текст, генерируя псевдонаучный бред. Поэтому поверх первого обучения запускают второе — на датасете формата «Вопрос — Решение — Ответ». Этот этап уже реализован.

После этого модель будет готова к *включению в RAG (поисковую систему)*, чтобы во время ответа она не придумывала, а подтягивала конкретную статью или справочник и брала точные коэффициенты оттуда.


Таким образом, CPT дообучает модель на сыром доменном тексте (статьи из журналов) перед SFT. Основные команды перечислены ниже

**12a. Генерация CPT-датасета (сырой текст с LaTeX через marker-pdf):**
```bash
pyruqo1 generate --input ./university_pdfs --mode cpt --output cpt_dataset.json
```

**12b. CPT-обучение (LoRA rank=64, lr=1e-4, cosine scheduler):**
```bash
pyruqo1 train --model gigachat3-10b --train-type cpt --train-file cpt_dataset.json
```

**12c. Merge CPT-адаптера с базовой моделью:**
```bash
pyruqo1 merge --model gigachat3-10b --lora-dir ./o1_gigachat3_university_lora --output-dir ./cpt_merged_model
```

Далее используем `cpt_merged_model` как базовую модель для SFT.

**12d. Создаем configs/merged_step1.yaml на основе конфига реальной базовой модели.**
```bash
nano configs/merged_step1.yaml
```

В данном примере это была gigachat3-10b. Для нее:
```sh
model:
  name: "./cpt_merged_model"
  trust_remote_code: true

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  lora_dropout: 0.05
  bias: "none"

training:
  output_dir: "./o1_gigachat3_university_lora"
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  logging_steps: 10
  num_train_epochs: 1
  optim: "paged_adamw_32bit"
  gradient_checkpointing: true
  fp16: false
  bf16: true
  max_grad_norm: 0.3
  warmup_ratio: 0.03
  lr_scheduler_type: "constant"
  save_strategy: "steps"
  save_steps: 100
  report_to: "none"
  do_eval: true
  eval_strategy: "steps"
  eval_steps: 50
  per_device_eval_batch_size: 1
  max_seq_length: 8192
  dataset_text_field: "text"

dataset:
  train_file: "university_train.json"
  val_file: "university_val.json"

merge:
  low_ram: false
  cpu_swap_gb: 30
  output_dir: "./merged_o1_gigachat3_university"
  safe_serialization: true
  max_shard_size: "5GB"

gguf:
  quantization: "Q4_K_M"
  output_dir: "./gguf"
  use_mlock: false
  vocab_only: false
```


**12e. Обучение:**
```bash
pyruqo1 train --model merged_step1 --train-file university_train.json
```

Получаем уже SFT-адаптер в `./merged_o1_gigachat3_university` и если нужно повторяем шаги `merge` - `gguf` - `test`.

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
