"""预训练运行器的惰性导出。

数据与候选词模块也会被主实验E复用；惰性导入可避免加载轻量特征代码时
提前载入实验编排器并形成循环依赖。
"""

from typing import Any

__all__ = ["D1ContextMLMRunner", "D2WordPretrainingRunner", "D5ExperimentRunner"]


def __getattr__(name: str) -> Any:
    if name == "D1ContextMLMRunner":
        from .d1_runner import D1ContextMLMRunner

        return D1ContextMLMRunner
    if name == "D2WordPretrainingRunner":
        from .d2_runner import D2WordPretrainingRunner

        return D2WordPretrainingRunner
    if name == "D5ExperimentRunner":
        from .d5_runner import D5ExperimentRunner

        return D5ExperimentRunner
    raise AttributeError(name)
