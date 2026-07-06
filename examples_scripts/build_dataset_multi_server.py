#!/usr/bin/env python3
"""Генерация датасета (мульти-сервер). npi generate --input ./pdfs --servers http://srv1:8079/v1/chat/completions http://srv2:8079/v1/chat/completions"""
from npi.dataset import PDFParser, TextChunker, DatasetGenerator

parser = PDFParser()
texts = parser.parse_folder("./university_pdfs")

chunker = TextChunker(chunk_size=3500, overlap=500)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

servers = ["http://192.168.2.52:8079/v1/chat/completions", "http://192.168.2.53:8079/v1/chat/completions"]
generator = DatasetGenerator(servers=servers)
generator.generate_from_chunks(chunks, "university_thinking_dataset.json", mode="simple")
