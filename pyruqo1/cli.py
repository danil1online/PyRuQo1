import click
from pathlib import Path

from pyruqo1.config import load_config
from pyruqo1.utils.logger import get_logger
from pyruqo1.utils.system import check_system_requirements, print_system_report
from pyruqo1.utils.swap import get_managed_swap_path, remove_swap_file


def _parse_servers(ctx, param, value):
    """Обработка --servers: по умолчанию возвращает ['gigachat'].
    
    Поддерживает как повтор флага, так и запятую/пробел в одном значении.
    """
    if not value:
        # Если флаг не передан, возвращаем дефолтный список с gigachat
        return ['gigachat']
        
    servers = []
    for v in value:
        # Разделяем по запятой или пробелу
        for part in v.replace(",", " ").split():
            part = part.strip()
            if part:
                servers.append(part)
                
    # Если флаг передали, но он оказался пустым, тоже подменяем на gigachat
    return list(servers) if servers else ['gigachat']

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PyRuQo1 — конвейер обучения reasoning-моделей (GigaChat / GigaChat3 / YandexGPT)."""
    pass


@cli.command()
@click.option("--config", "-c", "config_path", help="Путь к YAML-конфигу модели", default=None)
@click.option("--model", "-m", "model_name", help="Имя модели (gigachat-20b, gigachat3-10b, ygpt-5-lite-8b)", default=None)
@click.option("--dataset-type", "-D", "dataset_type", type=click.Choice(["micro", "big"]), default="big", help="Параметры обучения: micro — epoch-чеки для маленьких датасетов, big — step-чеки для больших")
@click.option("--train-file", help="Файл обучающего датасета", default=None)
@click.option("--val-file", help="Файл валидационного датасета", default=None)
@click.option("--output-dir", help="Директория для сохранения LoRA-адаптера", default=None)
@click.option("--mode", "-M", "mode", type=click.Choice(["simple", "train_val"]), default="simple", help="Режим: simple — один train-файл (из конфига), train_val — два файла (указать --train-file и --val-file)")
@click.option("--train-type", type=click.Choice(["sft", "cpt"]), default="sft", help="Тип обучения: sft — instruction fine-tuning, cpt — continual pre-training")
@click.option("--system-report", is_flag=True, help="Показать отчёт о системе")
def train(config_path, model_name, dataset_type, train_file, val_file, output_dir, mode, system_report, train_type):
    """Запуск QLoRA-обучения."""
    if system_report:
        print_system_report()
        return

    config = load_config(config_path, model_name)

    # Определяем пути к датасету
    if not train_file:
        train_file = config.get("dataset", {}).get("train_file", "university_train.json")

    if mode == "train_val":
        if not train_file or not val_file:
            get_logger().error("Для train_val режима нужно указать --train-file и --val-file")
            return
    else:
        if not train_file:
            get_logger().error("Для simple режима нужно указать --train-file")
            return
        val_file = None

    if output_dir:
        config.setdefault("training", {})["output_dir"] = output_dir

    config.setdefault("training", {})["train_type"] = train_type
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
    trainer.train(dataset_type=dataset_type)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Папка с PDF-файлами или сборник журналов")
@click.option("--output", "-o", "output_file", default=None, help="Выходной JSON-файл датасета")
@click.option("--mode", "-M", "mode", type=click.Choice(["simple", "math", "cpt", "base"]), default="simple", help="Режим текста: simple — гуманитарный, math — математический с LaTeX, cpt — сырой текст для continual pre-training, base — структура статьи (введение→prompt, методы+результаты→thoughts, выводы→output)")
@click.option("--servers", "-s", "servers", multiple=True, callback=_parse_servers, help="URL серверов llama.cpp")
@click.option("--chunk-size", default=3500, help="Размер чанка в символах")
@click.option("--overlap", default=500, help="Перекрывание чанков в символах")
@click.option("--enable-ocr", is_flag=True, default=True, help="Включить OCR для сканов")
@click.option("--recursive", is_flag=True, default=True, help="Рекурсивный обход папок")
@click.option(
    '--context-size', 
    type=click.Choice(['2048', '8192']), 
    default='2048', 
    help='Целевой размер контекста обучаемой модели (влияет на лаконичность ответов)'
)
@click.option("--base-hum", is_flag=True, default=False, help="Base-режим: гуманитарный системный промпт (социолог-экономист)")
@click.option("--base-math", is_flag=True, default=False, help="Base-режим: математический системный промпт (математика, физика, информатика)")
# ДОБАВЛЕН context_size В АРГУМЕНТЫ ФУНКЦИИ НИЖЕ:
def generate(input_path, output_file, mode, servers, chunk_size, overlap, enable_ocr, recursive, context_size, base_hum, base_math):
    """Генерация датасета из PDF-файлов через API."""
    if not output_file:
        if mode == "math":
            output_file = "university_math_dataset.json"
        elif mode == "cpt":
            output_file = "university_cpt_dataset.json"
        elif mode == "base":
            output_file = "university_base_dataset.json"
        else:
            output_file = "university_thinking_dataset.json"

    input_dir = Path(input_path)

    if mode == "cpt":
        from pyruqo1.dataset import CPTParser, CPTChunker
        parser = CPTParser(chunk_size=chunk_size, overlap=overlap)
        chunker = CPTChunker(chunk_size=chunk_size, overlap=overlap)
    elif mode == "math":
        from pyruqo1.dataset import MathParser, MathChunker
        parser = MathParser(chunk_size=chunk_size, overlap=overlap)
        chunker = MathChunker(chunk_size=chunk_size, overlap=overlap)
    elif mode == "base":
        from pyruqo1.dataset import BaseParser
        parser = BaseParser(chunk_size=chunk_size, overlap=overlap)
    else:
        from pyruqo1.dataset import PDFParser, TextChunker
        parser = PDFParser(chunk_size=chunk_size, overlap=overlap, enable_ocr=enable_ocr)
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    if not input_dir.is_dir():
        get_logger().error(f"Директория не найдена: {input_path}")
        return

    if mode == "base":
        _save_base_dataset(parser, input_dir, recursive, output_file, base_hum, base_math)
    elif mode == "cpt":
        texts = parser.parse_folder(str(input_dir), recursive=recursive)

        all_chunks = []
        for text in texts:
            all_chunks.extend(chunker.chunk(text))

        get_logger().info(f"Сформировано {len(all_chunks)} чанков.")

        _save_cpt_dataset(all_chunks, output_file)
    else:
        texts = parser.parse_folder(str(input_dir), recursive=recursive)

        all_chunks = []
        for text in texts:
            all_chunks.extend(chunker.chunk(text))

        get_logger().info(f"Сформировано {len(all_chunks)} чанков.")

        from pyruqo1.dataset import DatasetGenerator
        generator = DatasetGenerator(servers=list(servers), context_size=int(context_size))
        generator.generate_from_chunks(all_chunks, output_file, mode=mode)


def _save_base_dataset(parser, input_dir, recursive, output_file, base_hum, base_math):
    """Генерация SFT-датасета из структуры статей (введение→prompt, методы+результаты→thoughts, выводы→output)."""
    import json

    if base_hum:
        system_prompt = "Ты ученый социолог-экономист. Проанализируй задачу и реши ее."
    elif base_math:
        system_prompt = "Ты ученый в области математики, физики, техники и информатики. Проанализируй задачу и реши ее."
    else:
        system_prompt = "Ты ученый. Проанализируй задачу и реши ее."

    sections_list = parser.parse_folder(str(input_dir), recursive=recursive)

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
            "system": system_prompt,
            "prompt": prompt_text,
            "response": response_text,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=4)

    get_logger().info(f"Base-датасет сохранён: {output_file} ({len(dataset_rows)} строк)")


def _save_cpt_dataset(chunks: list, output_file: str) -> None:
    """Сохранение CPT-датасета в JSONL с полем text."""
    import json

    cpt_data = [{"text": chunk} for chunk in chunks]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cpt_data, f, ensure_ascii=False, indent=4)

    get_logger().info(f"CPT-датасет сохранён: {output_file} ({len(cpt_data)} строк)")

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
def gguf(model_path, quantization, output_dir, config_path, managed_swap):
    """Конвертация модели в GGUF."""
    if not config_path:
        config = {}
    else:
        config = load_config(config_path)

    from pyruqo1.gguf import GGUFConverter
    converter = GGUFConverter(config)
    converter.convert(
        model_path=model_path,
        quantization=quantization,
        output_dir=output_dir,
        managed_swap=managed_swap,
    )


@cli.command()
@click.option("--model", "-m", "model_type", required=True, type=click.Choice(["gigachat-20b", "gigachat3-10b", "ygpt-5-lite-8b"]), help="Тип модели (gigachat-20b, gigachat3-10b, ygpt-5-lite-8b)")
@click.option("--modelfile", required=True, help="Путь к GGUF-файлу модели")
@click.option("--val-file", required=True, help="Путь к валидационному датасету (JSON)")
@click.option("--res", "-r", "res_file", default=None, help="Путь к файлу для сохранения результатов")
@click.option("--num-samples", "-n", "num_samples", default=3, help="Количество случайных заданий для теста")
def test_gguf(model_type, modelfile, val_file, res_file, num_samples):
    """Тестирование GGUF-модели на валидационном датасете."""
    from pyruqo1.gguf import GGUFTester

    try:
        tester = GGUFTester(
            model_path=modelfile,
            model_type=model_type,
            val_file=val_file,
            num_samples=num_samples,
            res_file=res_file,
        )
        output_file = tester.run()
        get_logger().info(f"Тестирование завершено. Результаты: {output_file}")
    except Exception as e:
        get_logger().error(f"Ошибка тестирования: {e}")


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
