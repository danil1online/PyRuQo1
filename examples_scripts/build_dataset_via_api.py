#!/usr/bin/env python3
"""Генерация датасета через API (1 сервер). npi generate --input ./pdfs"""
from npi.config import load_config
from npi.dataset import PDFParser, TextChunker, DatasetGenerator

config = load_config(model_name="gigachat-20b")
parser = PDFParser()
texts = parser.parse_folder("./university_pdfs")

chunker = TextChunker(
    chunk_size=config["training"]["max_seq_length"] * 2,
    overlap=500,
)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

generator = DatasetGenerator(servers=["http://192.168.2.52:8079/v1/chat/completions"])
generator.generate_from_chunks(chunks, "university_thinking_dataset.json", mode="simple")
