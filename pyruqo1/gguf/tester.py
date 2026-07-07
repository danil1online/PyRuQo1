import os
import sys
import json
import random
from pathlib import Path
from typing import Optional, List, Dict

from llama_cpp import Llama


MODEL_CONFIGS = {
    "gigachat-20b": {
        "n_ctx": 2048,
        "max_tokens": 1000,
        "description": "GigaChat-20B-A3B",
    },
    "gigachat3-10b": {
        "n_ctx": 8192,
        "max_tokens": 7000,
        "description": "GigaChat3-10B-A1.8B",
    },
    "ygpt-5-lite-8b": {
        "n_ctx": 8192,
        "max_tokens": 7000,
        "description": "YandexGPT-5-Lite-8B",
    },
}

SYSTEM_PROMPT = (
    "Вы — ИИ-помощник. Отформатируйте свои ответы следующим образом: "
    "<Thought> Ваши мысли (понимание, рассуждения) </Thought> <output> Ваш ответ </output>"
)


class GGUFTester:
    """Тестирование GGUF-моделей на валидационном датасете."""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        val_file: str,
        num_samples: int = 3,
        res_file: Optional[str] = None,
    ):
        if model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Неизвестный тип модели: {model_type}. "
                f"Доступные: {list(MODEL_CONFIGS.keys())}"
            )

        self.model_path = model_path
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.val_file = val_file
        self.num_samples = num_samples
        self.res_file = res_file

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Валидационный файл не найден: {val_file}")

    def load_model(self) -> Llama:
        print(f"--> Загрузка модели {self.config['description']} из {self.model_path}...")
        llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_ctx=self.config["n_ctx"],
            verbose=False,
        )
        print(f"--> Модель загружена. n_ctx={self.config['n_ctx']}, max_tokens={self.config['max_tokens']}")
        return llm

    def load_val_data(self) -> List[Dict]:
        print(f"--> Загрузка валидационных заданий из {self.val_file}...")
        with open(self.val_file, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        if not isinstance(val_data, list):
            raise ValueError(f"Валидационный файл должен содержать список. Найдено: {type(val_data)}")

        return val_data

    def build_prompt(self, user_query: str) -> str:
        return (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_query}<|end|>\n"
            f"<|assistant|>\n"
        )

    def run_test(self, llm: Llama, val_data: List[Dict]) -> List[Dict]:
        print(f"\n--> Запуск тестирования (выборка: {self.num_samples} из {len(val_data)} заданий)...\n")

        random.seed(42)
        test_samples = random.sample(val_data, min(self.num_samples, len(val_data)))

        results = []

        for i, sample in enumerate(test_samples, 1):
            user_query = sample["prompt"] if isinstance(sample, dict) else sample
            expected_response = sample.get("response", "Эталон отсутствует")

            print("=" * 80)
            print(f"ЗАДАНИЕ №{i} ИЗ ВАЛИДАЦИОННОЙ ВЫБОРКИ:")
            print(user_query)
            print("=" * 80)
            print(f"\n[ЭТАЛОН ИЗ ДАТАСЕТА]:")
            print(expected_response)
            print("-" * 40)

            full_prompt = self.build_prompt(user_query)

            response_stream = llm(
                full_prompt,
                max_tokens=self.config["max_tokens"],
                temperature=0.2,
                repeat_penalty=1.2,
                stop=["<|user|>", "<|system|>", "<|end|>", "</output>"],
                stream=True,
            )

            model_response = ""
            for chunk in response_stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    if "text" in choice:
                        print(choice["text"], end="", flush=True)
                        model_response += choice["text"]
                    elif "delta" in choice and "content" in choice["delta"]:
                        print(choice["delta"]["content"], end="", flush=True)
                        model_response += choice["delta"]["content"]

            print("\n")
            print(f"\n[ОТВЕТ МОДЕЛИ {self.config['description']}]:")
            print(model_response)
            print("=" * 80)
            print()

            results.append({
                "task_number": i,
                "prompt": user_query,
                "expected_response": expected_response,
                "model_response": model_response,
            })

        return results

    def save_results(self, results: List[Dict]) -> str:
        if not self.res_file:
            self.res_file = f"gguf_test_results_{self.model_type}.json"

        with open(self.res_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_type": self.model_type,
                "model_description": self.config["description"],
                "model_path": self.model_path,
                "val_file": self.val_file,
                "num_samples": self.num_samples,
                "results": results,
            }, f, ensure_ascii=False, indent=4)

        print(f"--> Результаты сохранены в: {self.res_file}")
        return self.res_file

    def run(self) -> str:
        llm = self.load_model()
        val_data = self.load_val_data()
        results = self.run_test(llm, val_data)
        return self.save_results(results)
