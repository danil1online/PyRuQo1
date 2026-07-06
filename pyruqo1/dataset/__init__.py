def __getattr__(name):
    if name == "PDFParser":
        from pyruqo1.dataset.parser import PDFParser
        return PDFParser
    elif name == "MathParser":
        from pyruqo1.dataset.math_parser import MathParser
        return MathParser
    elif name == "JournalSplitter":
        from pyruqo1.dataset.splitter import JournalSplitter
        return JournalSplitter
    elif name == "TextChunker":
        from pyruqo1.dataset.chunker import TextChunker
        return TextChunker
    elif name == "MathChunker":
        from pyruqo1.dataset.chunker import MathChunker
        return MathChunker
    elif name == "DatasetGenerator":
        from pyruqo1.dataset.generator import DatasetGenerator
        return DatasetGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PDFParser",
    "MathParser",
    "JournalSplitter",
    "TextChunker",
    "MathChunker",
    "DatasetGenerator",
]
