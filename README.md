# PyRuQo1 — конвейер обучения reasoning-моделей

Обучение российских LLM (GigaChat-20B / GigaChat3-10B / YandexGPT-8B) методом QLoRA на синтетическом датасете с цепочками рассуждений (Chain-of-Thought).

## Быстрый старт

### 1. Установка зависимостей ОС

```bash
sudo apt update && sudo apt install ocrmypdf tesseract-ocr-rus build-essential cmake git python3-pip -y
```

### 2. Установка библиотеки

**Комплектование датасета** (torch 2.4.1 + marker-pdf 0.3.10):
```bash
pip install -e ".[ds,test]"
```

**Обучение** (torch 2.6.0, без marker-pdf):
```bash
pip install -e ".[train,test]"
```

**GGUF конвертация** (опционально):
```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e ".[gguf]"
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

### 6. Объединение датасетов

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

### 7. Обучение

**По одному файлу (university_train.json):**
```bash
pyruqo1 train --model gigachat-20b --mode simple
```

**По двум файлам (train + val):**
```bash
pyruqo1 train --model gigachat-20b --mode train_val --train-file university_train.json --val-file university_val.json
```

**На микро-датасете (из micro_datasets/):**
```bash
pyruqo1 train --model gigachat-20b --dataset-type micro
```

Флаг `-D, --dataset-type` принимает `micro` (файлы в `micro_datasets/`) или `big` (файлы в корне проекта).

**Различия режимов обучения:**

| Параметр | `micro` (50 строк) | `big` (1000+ строк) |
|---|---|---|
| `save_strategy` | `epoch` | `steps` |
| `logging_steps` | `2` | `10` |
| `eval_steps` | `2` | `50` |
| `save_steps` | `None` | `100` |

Для микро-датасета чекпоинты сохраняются раз в эпоху, логирование и валидация — каждые 2 шага (так как эпоха короткая).
Для большого датасета — чекпоинты каждые 100 шагов, логирование каждые 10, валидация каждые 50 шагов.

### 8. Слияние LoRA

```bash
pyruqo1 merge --model gigachat-20b --manage-swap
```

### 9. Конвертация в GGUF

```bash
pyruqo1 gguf --model ./merged_model --quant Q4_K_M --managed-swap
```

## CLI

```
pyruqo1 check      # Проверка системы
pyruqo1 split      # Разрезание журналов на статьи
pyruqo1 generate   # Генерация датасета из PDF
pyruqo1 mixds      # Объединение датасетов
pyruqo1 train      # QLoRA-обучение
pyruqo1 merge      # Слияние LoRA-адаптера
pyruqo1 gguf       # Конвертация в GGUF
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
├── training/          # QLoRA-обучение
├── merge/             # слияние LoRA
├── gguf/              # конвертация в GGUF
└── cli.py             # CLI (click)

examples_scripts/      # старые скрипты (сохранены для совместимости)
configs/               # пользовательские YAML-оверрайды (опционально)
logs_example/          # примеры результатов обучения
micro_datasets/        # микро-датасеты
tests/                 # базовые тесты
```

## Конфигурация

Встроенные конфиги в `pyruqo1/config/`:
- `gigachat-20b.yaml` — GigaChat-20B-A3B (основной)
- `gigachat3-10b.yaml` — GigaChat3-10B-A1.8B
- `ygpt-5-lite-8b.yaml` — YandexGPT-5-Lite-8B

Пользовательские оверрайды: создайте `configs/<model_name>.yaml` для переопределения параметров.

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

### 3. GGUF конвертация

llama-cpp-python с поддержкой CUDA:
```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e ".[gguf]"
```

`llama-cpp-python` через pip покрывает оба шага: `convert_hf_to_gguf` и `llama-quantize`. Исходники llama.cpp из репозитория не требуются.

## Управление swap

Для merge и GGUF конвертации требуется ~80 ГБ дискового пространства и 60+ ГБ RAM.

```bash
# Автоматическое управление swap (создать перед, удалить после)
pyruqo1 merge --model gigachat-20b --manage-swap
pyruqo1 gguf --model ./merged_model --managed-swap
```

## Подробнее

Подробная документация, пошаговое руководство и примеры результатов — в [EXAMPLESINFO.md](EXAMPLESINFO.md).

## License

MIT
