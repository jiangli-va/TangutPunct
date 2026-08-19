"""主实验E的显式知识特征。

知识提取与神经模型解耦：每个提供器只负责折内拟合和逐字符向量生成，
组合器负责拼接。E2—E8通过独立提供器和模型头继续扩展。
"""

from .base import CompositeKnowledgeProvider, KnowledgeFeatureProvider
from .context import ContextStatisticsProvider
from .domain import LocalDomainDistributionProvider
from .lexicon import LexiconGapLatticeProvider
from .pos import POSKnowledgeProvider, POSRelationKnowledgeProvider
from .segmentation import SegmentationKnowledgeProvider

__all__ = [
    "CompositeKnowledgeProvider",
    "ContextStatisticsProvider",
    "KnowledgeFeatureProvider",
    "LocalDomainDistributionProvider",
    "LexiconGapLatticeProvider",
    "POSKnowledgeProvider",
    "POSRelationKnowledgeProvider",
    "SegmentationKnowledgeProvider",
]
