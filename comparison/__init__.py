"""模型对比和微调模块。"""

__all__ = ["ModelComparisonGUI"]


def __getattr__(name):
    if name == "ModelComparisonGUI":
        from .comparison_gui import ModelComparisonGUI

        return ModelComparisonGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
