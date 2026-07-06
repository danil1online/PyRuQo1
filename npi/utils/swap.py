import os
import subprocess
import tempfile
from contextlib import contextmanager

logger = None

def get_logger():
    global logger
    if logger is None:
        from npi.utils.logger import get_logger
        logger = get_logger()
    return logger


DEFAULT_SWAP_PATH = "/tmp/npi_swapfile"
DEFAULT_SWAP_SIZE_GB = 40


def create_swap_file(size_gb: int = DEFAULT_SWAP_SIZE_GB, path: str = DEFAULT_SWAP_PATH) -> str:
    logger = get_logger()
    path = os.path.abspath(path)

    if os.path.exists(path):
        logger.warning(f"Swap-файл уже существует: {path}")
        return path

    free_disk = get_free_disk_gb(os.path.dirname(path) or "/tmp")
    if free_disk < size_gb + 5:
        raise RuntimeError(
            f"Недостаточно места на диске. Свободно: {free_disk:.1f} ГБ, требуется: {size_gb + 5:.1f} ГБ"
        )

    logger.info(f"Создание swap-файла: {path} ({size_gb} ГБ)...")

    try:
        subprocess.run(
            ["dd", "if=/dev/zero", f"of={path}", f"bs=1G", f"count={size_gb}"],
            check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ошибка создания swap-файла: {e.stderr.decode()}")

    subprocess.run(["chmod", "600", path], check=True)
    subprocess.run(["mkswap", path], check=True)

    try:
        subprocess.run(["swapon", path], check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"Не удалось активировать swap. Запустите с sudo:\n"
            f"  sudo dd if=/dev/zero of={path} bs=1G count={size_gb}\n"
            f"  sudo chmod 600 {path}\n"
            f"  sudo mkswap {path}\n"
            f"  sudo swapon {path}"
        )

    logger.info(f"Swap активирован: {path} ({size_gb} ГБ)")
    return path


def remove_swap_file(path: str = DEFAULT_SWAP_PATH) -> None:
    logger = get_logger()

    if not os.path.exists(path):
        return

    logger.info(f"Отключение и удаление swap: {path}...")

    try:
        subprocess.run(["swapoff", path], check=True)
    except subprocess.CalledProcessError:
        logger.warning(f"Не удалось отключить swap: {path}. Возможно, он уже отключён.")

    try:
        os.remove(path)
        logger.info(f"Swap удалён: {path}")
    except OSError as e:
        logger.warning(f"Не удалось удалить swap-файл: {e}")


def get_free_disk_gb(path: str = "/tmp") -> float:
    disk = os.statvfs(path)
    return disk.f_bavail * disk.f_frsize / (1024 ** 3)


@contextmanager
def managed_swap(size_gb: int = DEFAULT_SWAP_SIZE_GB, path: str = DEFAULT_SWAP_PATH):
    free_ram = get_free_ram_gb()
    logger = get_logger()

    if free_ram < 60:
        logger.warning(f"Доступно RAM: {free_ram:.1f} ГБ (требуется >= 60 ГБ). Активация swap...")
        swap_path = create_swap_file(size_gb, path)
    else:
        logger.info(f"Доступно RAM: {free_ram:.1f} ГБ. Swap не требуется.")
        swap_path = None

    try:
        yield
    finally:
        if swap_path:
            remove_swap_file(swap_path)


def get_free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / (1024 ** 3)
