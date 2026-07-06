def __getattr__(name):
    if name == "PDFParser":
        from npi.dataset.parser import PDFParser
        return PDFParser
    elif name == "MathParser":
        from npi.dataset.math_parser import MathParser
        return MathParser
    elif name == "JournalSplitter":
        from npi.dataset.splitter import JournalSplitter
        return JournalSplitter
    elif name == "TextChunker":
        from npi.dataset.chunker import TextChunker
        return TextChunker
    elif name == "MathChunker":
        from npi.dataset.chunker import MathChunker
        return MathChunker
    elif name == "DatasetGenerator":
        from npi.dataset.generator import DatasetGenerator
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
