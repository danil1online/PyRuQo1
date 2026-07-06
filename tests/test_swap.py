import pytest
import sys
from unittest.mock import MagicMock, patch

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


def test_swap_defaults():
    from pyruqo1.utils.swap import DEFAULT_SWAP_SIZE_GB, DEFAULT_SWAP_PATH
    assert DEFAULT_SWAP_SIZE_GB == 40
    assert DEFAULT_SWAP_PATH == "/tmp/npi_swapfile"


def test_swap_functions_exist():
    from pyruqo1.utils.swap import (
        create_swap_file,
        remove_swap_file,
        managed_swap,
        get_free_ram_gb,
        get_free_disk_gb,
    )
    assert callable(create_swap_file)
    assert callable(remove_swap_file)
    assert callable(managed_swap)
    assert callable(get_free_ram_gb)
    assert callable(get_free_disk_gb)


def test_get_free_ram_gb():
    from pyruqo1.utils.swap import get_free_ram_gb
    import psutil
    result = get_free_ram_gb()
    expected = psutil.virtual_memory().available / (1024 ** 3)
    assert abs(result - expected) < 0.1


def test_get_free_disk_gb():
    from pyruqo1.utils.swap import get_free_disk_gb
    result = get_free_disk_gb("/tmp")
    assert isinstance(result, float)
    assert result >= 0
