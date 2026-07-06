import os
import subprocess
from pathlib import Path

from pyruqo1.utils.logger import get_logger


class GGUFConverter:
    """Конвертация модели в формат GGUF через llama-cpp-python."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()

    def _check_llama_cpp(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python не установлен. Установите:\n"
                'CMAKE_ARGS="-DGGML_CUDA=ON" pip install -r requirements-gguf.txt'
            )

    def convert(
        self,
        model_path: str = None,
        quantization: str = None,
        output_dir: str = None,
    ):
        model_path = model_path or self.config.get("merge", {}).get("output_dir", "./merged_model")
        quantization = quantization or self.config.get("gguf", {}).get("quantization", "Q4_K_M")
        output_dir = output_dir or self.config.get("gguf", {}).get("output_dir", "./gguf")

        self.logger.info(f"=== Конвертация в GGUF ===")
        self.logger.info(f"Модель: {model_path}")
        self.logger.info(f"Квантование: {quantization}")
        self.logger.info(f"Выход: {output_dir}")

        self._check_llama_cpp()

        from llama_cpp.convert import convert_llama_to_gguf

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_file = output_path / f"model.gguf"

        self.logger.info("Конвертация...")
        convert_llama_to_gguf(model_path, str(model_file), outtype=self._quant_to_dtype(quantization))

        self.logger.info(f"GGUF сохранён: {model_file}")

        # Квантование
        quantized_file = output_path / f"model-{quantization}.gguf"
        self.logger.info(f"Квантование {quantization}...")

        try:
            from llama_cpp import GGMLQuantizationType, quantize_file
            quantize_file(str(model_file), str(quantized_file), quantization=self._get_quant_type(quantization))
            self.logger.info(f"Квантованная модель: {quantized_file}")
        except ImportError:
            self.logger.warning("llama_cpp.quantize_file не доступен. Попробуйте через llama.cpp CLI.")
            subprocess.run(
                ["llama-quantize", str(model_file), str(quantized_file), quantization],
                check=True,
            )

        self.logger.info(f"=== GGUF готов: {quantized_file} ===")

    def _quant_to_dtype(self, quant: str):
        mapping = {
            "F16": "f16",
            "F32": "f32",
            "Q4_0": "q4_0",
            "Q4_1": "q4_1",
            "Q5_0": "q5_0",
            "Q5_1": "q5_1",
            "Q8_0": "q8_0",
        }
        return mapping.get(quant, "q4_0")

    def _get_quant_type(self, quant: str):
        from llama_cpp import GGMLQuantizationType

        mapping = {
            "Q4_0": GGMLQuantizationType.GGML_TYPE_Q4_0,
            "Q4_1": GGMLQuantizationType.GGML_TYPE_Q4_1,
            "Q5_0": GGMLQuantizationType.GGML_TYPE_Q5_0,
            "Q5_1": GGMLQuantizationType.GGML_TYPE_Q5_1,
            "Q8_0": GGMLQuantizationType.GGML_TYPE_Q8_0,
            "Q4_K_M": GGMLQuantizationType.GGML_TYPE_Q4_K,
            "Q5_K_M": GGMLQuantizationType.GGML_TYPE_Q5_K,
            "Q6_K": GGMLQuantizationType.GGML_TYPE_Q6_K,
            "Q2_K": GGMLQuantizationType.GGML_TYPE_Q2_K,
        }
        return mapping.get(quant, GGMLQuantizationType.GGML_TYPE_Q4_0)
