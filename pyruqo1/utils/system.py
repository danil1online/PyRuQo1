import os
import psutil
import subprocess
from typing import Tuple, Optional

logger = None

def get_logger():
    global logger
    if logger is None:
        from pyruqo1.utils.logger import get_logger
        logger = get_logger()
    return logger


def get_free_ram_gb() -> float:
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def get_total_ram_gb() -> float:
    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3)


def get_swap_info() -> dict:
    swap = psutil.swap_memory()
    return {
        "total_gb": swap.total / (1024 ** 3),
        "used_gb": swap.used / (1024 ** 3),
        "free_gb": swap.free / (1024 ** 3),
        "percent": swap.percent,
    }


def get_free_disk_gb(path: str = "/tmp") -> float:
    disk = os.statvfs(path)
    free_bytes = disk.f_bavail * disk.f_frsize
    return free_bytes / (1024 ** 3)


def check_gpu_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_gpu_info() -> list:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append({
                    "name": parts[0],
                    "total_gb": int(parts[1]) / 1024,
                    "free_gb": int(parts[2]) / 1024,
                })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


def check_system_requirements(min_ram_gb: float = 16, min_vram_gb: float = 8, require_gpu: bool = True) -> dict:
    logger = get_logger()
    status = {
        "ram_ok": False,
        "gpu_ok": False,
        "swap_available": False,
        "warnings": [],
    }

    free_ram = get_free_ram_gb()
    total_ram = get_total_ram_gb()
    status["ram_ok"] = free_ram >= min_ram_gb

    if require_gpu:
        gpus = get_gpu_info()
        if gpus:
            total_vram = sum(g["total_gb"] for g in gpus)
            status["gpu_ok"] = total_vram >= min_vram_gb
            status["total_vram_gb"] = total_vram
            status["gpu_names"] = [g["name"] for g in gpus]
        else:
            status["gpu_ok"] = False
            status["warnings"].append("GPU NVIDIA не обнаружен. Обучение/слияние будет идти на CPU (очень медленно).")
    else:
        status["gpu_ok"] = True

    if not status["ram_ok"]:
        status["warnings"].append(
            f"Свободно RAM: {free_ram:.1f} ГБ (требуется минимум {min_ram_gb} ГБ). "
            f"Используйте --manage-swap для создания swap-файла."
        )

    swap = get_swap_info()
    status["swap"] = swap
    if swap["total_gb"] > 0:
        status["swap_available"] = True

    free_disk = get_free_disk_gb()
    status["free_disk_gb"] = free_disk
    if free_disk < 50:
        status["warnings"].append(
            f"Свободно на диске: {free_disk:.1f} ГБ. Для merge требуется ~80 ГБ."
        )

    return status


def print_system_report():
    import json
    report = {
        "free_ram_gb": round(get_free_ram_gb(), 2),
        "total_ram_gb": round(get_total_ram_gb(), 2),
        "swap": get_swap_info(),
        "gpu": get_gpu_info(),
        "free_disk_gb": round(get_free_disk_gb(), 2),
    }
    for key, value in report.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for gpu in value:
                print(f"    - {json.dumps(gpu)}")
        elif isinstance(value, dict) and key != "swap":
            print(f"  {key}: {json.dumps(value)}")
        else:
            print(f"  {key}: {json.dumps(value)}")
