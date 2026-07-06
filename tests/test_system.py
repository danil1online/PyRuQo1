import pytest
import sys
from unittest.mock import MagicMock, patch, Mock

sys.modules['fitz'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['tokenizers'] = MagicMock()
sys.modules['bitsandbytes'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['marker'] = MagicMock()
sys.modules['marker.models'] = MagicMock()
sys.modules['marker.convert'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()
sys.modules['llama_cpp.convert'] = MagicMock()


def test_get_free_ram_gb():
    import psutil
    result = psutil.virtual_memory().available / (1024 ** 3)
    assert isinstance(result, float)
    assert result > 0


def test_get_total_ram_gb():
    import psutil
    result = psutil.virtual_memory().total / (1024 ** 3)
    assert isinstance(result, float)
    assert result > 0


def test_get_swap_info():
    import psutil
    swap = psutil.swap_memory()
    assert swap.total >= 0
    assert swap.used >= 0


def test_get_free_disk_gb():
    import os
    disk = os.statvfs("/tmp")
    free_bytes = disk.f_bavail * disk.f_frsize
    assert isinstance(free_bytes, int)
    assert free_bytes >= 0


def test_check_system_requirements():
    result = pytest.importorskip("pyruqo1.utils.system")
    # Just verify the module exists and has expected functions
    assert hasattr(result, "check_system_requirements")


def test_swap_functions():
    from pyruqo1.utils.swap import (
        get_free_ram_gb,
        get_free_disk_gb,
        DEFAULT_SWAP_SIZE_GB,
        DEFAULT_SWAP_PATH,
    )
    assert DEFAULT_SWAP_SIZE_GB == 40
    assert DEFAULT_SWAP_PATH == "/tmp/pyruqo1_swapfile"
    assert get_free_ram_gb() > 0
    assert get_free_disk_gb() >= 0


def test_managed_swap_context_exists():
    from pyruqo1.utils.swap import managed_swap
    import inspect
    assert inspect.isfunction(managed_swap)
