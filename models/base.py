from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from data.corpus import SequenceChunk


class SequenceTagger(ABC):
    """模型后端接口；CRF、神经模型和消融模型共享同一实验层。"""

    @abstractmethod
    def fit(self, train: list[SequenceChunk], dev: list[SequenceChunk] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError
