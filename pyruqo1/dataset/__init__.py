def __getattr__(name):
    if name == "PDFParser":
        from pyruqo1.dataset.parser import PDFParser
        return PDFParser
    elif name == "MathParser":
        from pyruqo1.dataset.math_parser import MathParser
        return MathParser
    elif name == "CPTParser":
        from pyruqo1.dataset.cpt_parser import CPTParser
        return CPTParser
    elif name == "BaseParser":
        from pyruqo1.dataset.base_parser import BaseParser
        return BaseParser
    elif name == "JournalSplitter":
        from pyruqo1.dataset.splitter import JournalSplitter
        return JournalSplitter
    elif name == "TextChunker":
        from pyruqo1.dataset.chunker import TextChunker
        return TextChunker
    elif name == "MathChunker":
        from pyruqo1.dataset.chunker import MathChunker
        return MathChunker
    elif name == "CPTChunker":
        from pyruqo1.dataset.chunker import CPTChunker
        return CPTChunker
    elif name == "DatasetGenerator":
        from pyruqo1.dataset.generator import DatasetGenerator
        return DatasetGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PDFParser",
    "MathParser",
    "CPTParser",
    "BaseParser",
    "JournalSplitter",
    "TextChunker",
    "MathChunker",
    "CPTChunker",
    "DatasetGenerator",
]
