# NPI Reasoning — конвейер обучения reasoning-моделей

Обучение российских LLM (GigaChat-20B / GigaChat3-10B / YandexGPT-8B) методом QLoRA на синтетическом датасете с цепочками рассуждений (Chain-of-Thought).

## Быстрый старт

```bash
# Установка
pip install -e ".[test]"        # основной стек
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e ".[gguf,test]"  # + GGUF конвертация

# Проверка системы
npi check

# Обучение
npi train --model gigachat-20b

# Генерация датасета
npi generate --input ./pdfs --mode simple --servers http://localhost:8079/v1/chat/completions

# Разрезание журналов
npi split --input ./journals/

# Слияние LoRA
npi merge --model gigachat-20b --manage-swap

# Конвертация в GGUF
npi gguf --model ./merged_model --quant Q4_K_M
```

## CLI

```
npi train      # QLoRA-обучение
npi generate   # генерация датасета из PDF
npi parse      # парсинг PDF в чанки
npi split      # разрезание журналов на статьи
npi mix        # объединение датасетов + split train/val
npi merge      # слияние LoRA-адаптера
npi gguf       # конвертация в GGUF
npi check      # проверка системы
```

## Python API

```python
from npi.config import load_config
from npi.training import NPITrainer

config = load_config(model_name="gigachat-20b")
trainer = NPITrainer(config)
trainer.train()
```

```python
from npi.dataset import PDFParser, TextChunker, DatasetGenerator

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
from npi.merge import LORAMerger

config = load_config(model_name="gigachat-20b")
merger = LORAMerger(config)
merger.merge(manage_swap=True)
```

## Структура

```
npi/                   # основная библиотека
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

Встроенные конфиги в `npi/config/`:
- `gigachat-20b.yaml` — GigaChat-20B-A3B (основной)
- `gigachat3-10b.yaml` — GigaChat3-10B-A1.8B
- `ygpt-5-lite-8b.yaml` — YandexGPT-5-Lite-8B

Пользовательские оверрайды: создайте `configs/<model_name>.yaml` для переопреждения параметров.

## Зависимости

**Основные** (обучение + генерация датасета):
```bash
pip install -e "."
```

**GGUF** (опционально, для конвертации):
```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e ".[gguf]"
```

## Управление swap

Для merge и GGUF конвертации требуется ~80 ГБ дискового пространства и 60+ ГБ RAM.

```bash
# Автоматическое управление swap
npi merge --model gigachat-20b --manage-swap

# Ручное управление (Python API)
from npi.utils.swap import managed_swap

with managed_swap(size_gb=40):
    merger.merge()
```

## Подробнее

Подробная документация, пошаговое руководство и примеры результатов — в [EXAMPLESINFO.md](EXAMPLESINFO.md).

## License

MIT
