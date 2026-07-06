def __getattr__(name):
    if name == "GGUFConverter":
        from npi.gguf.converter import GGUFConverter
        return GGUFConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GGUFConverter"]
