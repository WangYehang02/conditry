"""FMGAD local-prior package."""

__all__ = ["ResFlowGAD"]


def __getattr__(name: str):
    if name == "ResFlowGAD":
        from fmgad.detector import ResFlowGAD

        return ResFlowGAD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
