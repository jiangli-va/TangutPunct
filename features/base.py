from __future__ import annotations

from abc import ABC, abstractmethod

from data.corpus import SequenceChunk


class FeatureExtractor(ABC):
    """所有 CRF 特征组的稳定接口；新增消融特征时实现此类即可。"""

    @abstractmethod
    def transform(self, sequence: SequenceChunk) -> list[dict[str, object]]:
        raise NotImplementedError
