import click
from pathlib import Path

from pyruqo1.config import load_config
from pyruqo1.utils.logger import get_logger
from pyruqo1.utils.system import check_system_requirements, print_system_report
from pyruqo1.utils.swap import get_managed_swap_path, remove_swap_file


def _parse_servers(ctx, param, value):
    """Обработка --servers: поддерживаем как повтор флага, так и запятую/пробел в одном значении."""
    if not value:
        return ("http://localhost:8079/v1/chat/completions",)
    servers = []
    for v in value:
        # Разделяем по запятой или пробелу
        for part in v.replace(",", " ").split():
            part = part.strip()
            if part:
                servers.append(part)
    return tuple(servers) if servers else ("http://localhost:8079/v1/chat/completions",)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PyRuQo1 — конвейер обучения reasoning-моделей (GigaChat / GigaChat3 / YandexGPT)."""
    pass


@cli.command()
@click.option("--config", "-c", "config_path", help="Путь к YAML-конфигу модели", default=None)
@click.option("--model", "-m", "model_name", help="Имя модели (gigachat-20b, gigachat3-10b, ygpt-5-lite-8b)", default=None)
@click.option("--train-file", help="Файл обучающего датасета", default=None)
@click.option("--val-file", help="Файл валидационного датасета", default=None)
@click.option("--output-dir", help="Директория для сохранения LoRA-адаптера", default=None)
@click.option("--mode", "-M", "mode", type=click.Choice(["simple", "train_val"]), default="simple", help="Режим: simple — один train-файл, train_val — два файла (train + val)")
@click.option("--system-report", is_flag=True, help="Показать отчёт о системе")
def train(config_path, model_name, train_file, val_file, output_dir, mode, system_report):
    """Запуск QLoRA-обучения."""
    if system_report:
        print_system_report()
        return

    config = load_config(config_path, model_name)

    if mode == "train_val":
        if not train_file:
            train_file = config.get("dataset", {}).get("train_file", "university_train.json")
        if not val_file:
            val_file = config.get("dataset", {}).get("val_file", "university_val.json")
        if not train_file or not val_file:
            get_logger().error("Для train_val режима нужно указать --train-file и --val-file")
            return
    else:
        # simple mode — только один train-файл
        if not train_file:
            train_file = config.get("dataset", {}).get("train_file", "university_train.json")
        if not train_file:
            get_logger().error("Для simple режима нужно указать --train-file")
            return
        val_file = None

    if output_dir:
        config.setdefault("training", {})["output_dir"] = output_dir

    config.setdefault("dataset", {})["train_file"] = train_file
    if val_file:
        config["dataset"]["val_file"] = val_file
    elif "val_file" in config.get("dataset", {}):
        del config["dataset"]["val_file"]

    requirements = check_system_requirements(min_ram_gb=16, min_vram_gb=8)
    if requirements.get("warnings"):
        for w in requirements["warnings"]:
            get_logger().warning(w)

    from pyruqo1.training import NPITrainer
    trainer = NPITrainer(config)
    trainer.train()


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Папка с PDF-файлами или сборник журналов")
@click.option("--output", "-o", "output_file", default=None, help="Выходной JSON-файл датасета (по умолчанию: university_thinking_dataset.json / university_math_dataset.json)")
@click.option("--mode", "-M", "mode", type=click.Choice(["simple", "multi_server", "math"]), default="simple", help="Режим генерации: simple — 1 сервер + гумманитарный текст, multi_server — несколько серверов, math — 1 сервер + математические тексты с LaTeX")
@click.option("--servers", "-s", "servers", multiple=True, callback=_parse_servers, help="URL серверов llama.cpp (повторяемый флаг: --servers http://a --servers http://b; или через запятую: --servers 'http://a,http://b')")
@click.option("--chunk-size", default=3500, help="Размер чанка в символах")
@click.option("--overlap", default=500, help="Перекрывание чанков в символах")
@click.option("--enable-ocr", is_flag=True, default=True, help="Включить OCR для сканов")
@click.option("--recursive", is_flag=True, default=True, help="Рекурсивный обход папок")
def generate(input_path, output_file, mode, servers, chunk_size, overlap, enable_ocr, recursive):
    """Генерация датасета из PDF-файлов через API.

    Примеры:
      pyruqo1 generate --input ./pdfs --mode simple --servers http://localhost:8079/v1/chat/completions
      pyruqo1 generate --input ./pdfs --mode math --servers http://srv1:8079 http://srv2:8079
      pyruqo1 generate --input ./pdfs --mode simple --servers 'http://srv1:8079,http://srv2:8079'
    """
    if not output_file:
        output_file = "university_math_dataset.json" if mode == "math" else "university_thinking_dataset.json"

    input_dir = Path(input_path)

    if mode == "math":
        from pyruqo1.dataset import MathParser, MathChunker
        parser = MathParser(chunk_size=chunk_size, overlap=overlap)
        chunker = MathChunker(chunk_size=chunk_size, overlap=overlap)
    else:
        from pyruqo1.dataset import PDFParser, TextChunker
        parser = PDFParser(chunk_size=chunk_size, overlap=overlap, enable_ocr=enable_ocr)
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    if not input_dir.is_dir():
        get_logger().error(f"Директория не найдена: {input_path}")
        return

    texts = parser.parse_folder(str(input_dir), recursive=recursive)

    all_chunks = []
    for text in texts:
        all_chunks.extend(chunker.chunk(text))

    get_logger().info(f"Сформировано {len(all_chunks)} чанков.")

    from pyruqo1.dataset import DatasetGenerator
    generator = DatasetGenerator(servers=list(servers))
    generator.generate_from_chunks(all_chunks, output_file, mode=mode)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Путь к PDF-файлу или папке журналов")
@click.option("--output-dir", "-o", "output_dir", default="./university_pdfs", help="Директория для статей (по умолчанию: ./university_pdfs)")
def split(input_path, output_dir):
    """Разрезание сборников журналов на отдельные статьи."""
    from pyruqo1.dataset import JournalSplitter
    splitter = JournalSplitter(output_dir=output_dir)
    input_path = Path(input_path)

    if input_path.is_file():
        splitter.split_pdf(str(input_path), output_dir)
    elif input_path.is_dir():
        splitter.split_folder(str(input_path))
    else:
        get_logger().error(f"Не найден: {input_path}")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Путь к PDF-файлу или папке")
@click.option("--mode", "-M", "mode", type=click.Choice(["text", "math"]), default="text", help="Режим парсинга: text — обычный текст, math — LaTeX формулы")
@click.option("--output", "-o", "output_file", default=None, help="Выходной JSON-файл")
@click.option("--chunk-size", default=3500, help="Размер чанка")
@click.option("--overlap", default=500, help="Перекрывание")
def parse(input_path, mode, output_file, chunk_size, overlap):
    """Парсинг PDF в чанки (текст или LaTeX)."""
    input_path = Path(input_path)
    if not input_path.exists():
        get_logger().error(f"Не найден: {input_path}")
        return

    if mode == "math":
        from pyruqo1.dataset import MathParser, MathChunker
        parser = MathParser()
        chunker = MathChunker(chunk_size=chunk_size, overlap=overlap)
    else:
        from pyruqo1.dataset import PDFParser, TextChunker
        parser = PDFParser()
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    texts = parser.parse_folder(str(input_path))

    all_chunks = []
    for text in texts:
        all_chunks.extend(chunker.chunk(text))

    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)
        get_logger().info(f"Сохранено {len(all_chunks)} чанков в {output_file}")
    else:
        for i, chunk in enumerate(all_chunks):
            print(f"--- Чанк {i+1} ({len(chunk)} символов) ---")
            print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
            print()


@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--mode", "-M", "mode", type=click.Choice(["simple", "train_val"]), default="train_val", help="Режим: simple — один объединённый файл, train_val — два файла (train + val)")
@click.option("--split", "-s", "split_ratio", type=float, default=0.9, help="Доля train (для train_val режима, по умолчанию: 0.9)")
def mixds(files, mode, split_ratio):
    """Объединение датасетов. simple — один файл, train_val — два файла (train + val)."""
    import json
    import random

    all_rows = []
    for f in files:
        path = Path(f)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    all_rows.extend(data)
                else:
                    all_rows.append(data)

    get_logger().info(f"Объединено {len(all_rows)} строк.")

    if mode == "simple":
        out_file = "university_dataset.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=4)
        get_logger().info(f"Объединённый датасет: {out_file} ({len(all_rows)} строк)")
    else:
        split_idx = int(len(all_rows) * split_ratio)
        random.shuffle(all_rows)

        train_data = all_rows[:split_idx]
        val_data = all_rows[split_idx:]

        train_file = "university_train.json"
        val_file = "university_val.json"

        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=4)

        get_logger().info(f"Train: {len(train_data)} строк -> {train_file}")
        get_logger().info(f"Val: {len(val_data)} строк -> {val_file}")


@cli.command()
@click.option("--config", "-c", "config_path", help="Путь к YAML-конфигу", default=None)
@click.option("--model", "-m", "model_name", help="Имя модели для загрузки дефолтного конфига", default=None)
@click.option("--base-model", help="Путь/имя базовой модели", default=None)
@click.option("--lora-dir", help="Путь к LoRA-адаптеру", default=None)
@click.option("--output-dir", help="Директория для объединённой модели", default=None)
@click.option("--manage-swap", is_flag=True, help="Автоматически управлять swap (создать/удалить)")
@click.option("--low-ram", is_flag=True, help="Режим низкого RAM (< 64 ГБ)")
def merge(config_path, model_name, base_model, lora_dir, output_dir, manage_swap, low_ram):
    """Слияние LoRA-адаптера с базовой моделью."""
    config = load_config(config_path, model_name)

    if low_ram:
        config.setdefault("merge", {})["low_ram"] = True

    from pyruqo1.merge import LORAMerger
    merger = LORAMerger(config)
    merger.merge(
        base_model_name=base_model,
        lora_adapter_dir=lora_dir,
        output_dir=output_dir,
        manage_swap=manage_swap,
    )


@cli.command()
@click.option("--model", "-m", "model_path", required=True, help="Путь к объединённой модели")
@click.option("--quant", "-q", "quantization", default=None, help="Квантование (Q4_K_M, Q5_K_M, Q8_0 и т.д.)")
@click.option("--output-dir", "-o", "output_dir", default=None, help="Директория для GGUF")
@click.option("--config", "-c", "config_path", help="Путь к YAML-конфигу для дефолтных значений", default=None)
@click.option("--managed-swap", is_flag=True, help="Автоматически управлять swap (отключит swap после конвертации)")
def gguf(model_path, quant, output_dir, config_path, managed_swap):
    """Конвертация модели в GGUF."""
    if not config_path:
        config = {}
    else:
        config = load_config(config_path)

    from pyruqo1.gguf import GGUFConverter
    converter = GGUFConverter(config)
    converter.convert(
        model_path=model_path,
        quantization=quant,
        output_dir=output_dir,
        managed_swap=managed_swap,
    )


@cli.command()
@click.option("--min-ram", type=float, default=16, help="Минимальная RAM для проверки")
@click.option("--min-vram", type=float, default=8, help="Минимальная VRAM для проверки")
def check(min_ram, min_vram):
    """Проверка системы на соответствие требованиям."""
    report = check_system_requirements(min_ram_gb=min_ram, min_vram_gb=min_vram)
    click.echo(f"\nRAM: {report.get('free_ram_gb', 'N/A')} ГБ свободно (из {report.get('total_ram_gb', 'N/A')} ГБ)")
    click.echo(f"GPU: {'OK' if report.get('gpu_ok') else 'NOT FOUND'}")
    if report.get("warnings"):
        for w in report["warnings"]:
            click.echo(f"  WARNING: {w}")


if __name__ == "__main__":
    cli()
