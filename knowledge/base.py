from __future__ import annotations

from abc import ABC, abstractmethod

from data.corpus import SequenceChunk


FeatureMatrix = tuple[tuple[float, ...], ...]


class KnowledgeFeatureProvider(ABC):
    """显式知识的折内接口。

    ``fit_transform``必须为外层训练样本生成防泄漏特征；``transform``只可
    使用已经拟合的训练折统计量处理开发集、测试集或新文本。
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        raise NotImplementedError

    @abstractmethod
    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, object]:
        """返回可随模型checkpoint保存、供未来推理重建的统计状态。"""

        raise NotImplementedError


class CompositeKnowledgeProvider(KnowledgeFeatureProvider):
    """按字符位置拼接多个独立知识通道。"""

    def __init__(self, providers: list[KnowledgeFeatureProvider]) -> None:
        if not providers:
            raise ValueError("知识组合器至少需要一个特征提供器")
        self.providers = tuple(providers)

    @property
    def dimension(self) -> int:
        return sum(provider.dimension for provider in self.providers)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            name for provider in self.providers for name in provider.feature_names
        )

    @staticmethod
    def _concatenate(parts: list[list[FeatureMatrix]]) -> list[FeatureMatrix]:
        if not parts:
            return []
        sequence_count = len(parts[0])
        if any(len(part) != sequence_count for part in parts):
            raise ValueError("知识通道返回的序列数量不一致")
        output: list[FeatureMatrix] = []
        for sequence_index in range(sequence_count):
            matrices = [part[sequence_index] for part in parts]
            length = len(matrices[0])
            if any(len(matrix) != length for matrix in matrices):
                raise ValueError("知识通道返回的字符长度不一致")
            output.append(
                tuple(
                    tuple(
                        value
                        for matrix in matrices
                        for value in matrix[position]
                    )
                    for position in range(length)
                )
            )
        return output

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        return self._concatenate(
            [provider.fit_transform(sequences) for provider in self.providers]
        )

    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        return self._concatenate(
            [provider.transform(sequences) for provider in self.providers]
        )

    def metadata(self) -> dict[str, object]:
        return {
            "总维度": self.dimension,
            "特征名称": list(self.feature_names),
            "通道": [provider.metadata() for provider in self.providers],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "composite_knowledge",
            "providers": [provider.state_dict() for provider in self.providers],
        }
