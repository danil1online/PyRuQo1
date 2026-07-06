def __getattr__(name):
    if name == "LORAMerger":
        from pyruqo1.merge.merger import LORAMerger
        return LORAMerger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LORAMerger"]
