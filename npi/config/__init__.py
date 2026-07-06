import os
import yaml
from pathlib import Path
from copy import deepcopy

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _get_package_config_dir() -> Path:
    return Path(__file__).parent


def _find_package_config(name: str) -> Path:
    config_dir = _get_package_config_dir()
    for ext in [".yaml", ".yml"]:
        config_path = config_dir / f"{name}{ext}"
        if config_path.exists():
            return config_path
    return None


def load_config(
    config_path: str = None,
    model_name: str = None,
    user_config_dir: str = "configs",
) -> dict:
    """
    Загрузка конфигурации с приоритетами:
    1. Если передан config_path — загружаем из файла
    2. Ищем встроенный конфиг в npi/config/<model_name>.yaml
    3. Ищем пользовательский оверрайды в configs/<model_name>.yaml
    4. Глубоко объединяем (deep merge)

    Приоритет загрузки (перекрывают друг друга):
    1. config_path (явный файл)
    2. configs/<model_name>.yaml (пользовательские оверрайды)
    3. npi/config/<model_name>.yaml (встроенный конфиг)
    4. npi/config/default.yaml (дефолты)
    """
    base_config = _load_yaml(str(_get_package_config_dir() / "default.yaml"))

    if model_name:
        package_config_path = _find_package_config(model_name)
        if package_config_path:
            base_config = deep_merge(base_config, _load_yaml(str(package_config_path)))

    user_config_path = None
    if user_config_dir and model_name:
        user_config_path = Path(user_config_dir) / f"{model_name}.yaml"
        if not user_config_path.exists():
            user_config_path = Path(user_config_dir) / f"{model_name}.yml"

    if user_config_path and user_config_path.exists():
        base_config = deep_merge(base_config, _load_yaml(str(user_config_path)))

    if config_path:
        base_config = deep_merge(base_config, _load_yaml(config_path))

    return base_config


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=Loader)
    return data if data else {}


def save_config(config: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
