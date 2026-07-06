#!/usr/bin/env python3
"""Генерация датасета (простой вариант). npi generate --input ./pdfs"""
from npi.dataset import PDFParser, TextChunker, DatasetGenerator

parser = PDFParser()
texts = parser.parse_folder("./university_pdfs")

chunker = TextChunker(chunk_size=3500, overlap=500)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

generator = DatasetGenerator(servers=["http://localhost:8079/v1/chat/completions"])
generator.generate_from_chunks(chunks, "university_thinking_dataset.json", mode="simple")
