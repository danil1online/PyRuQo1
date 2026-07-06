import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from typing import Optional

console = Console()

def get_logger(name: Optional[str] = None, level: str = "INFO"):
    import logging
    logger = logging.getLogger(name or "npi")
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def progress_bar(total: int, description: str = ""):
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        console=console,
    )
