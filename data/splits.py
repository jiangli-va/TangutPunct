from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .corpus import PreparedDocument


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_ids: tuple[str, ...]
    dev_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def validate(self) -> None:
        train, dev, test = map(set, (self.train_ids, self.dev_ids, self.test_ids))
        if train & dev or train & test or dev & test:
            raise ValueError(f"第 {self.fold} 折存在跨集合的卷")


class VolumeCrossValidator:
    """固定随机种子的文献级分层 K 折。

    分层字段是文献的 ``domain``，使经书和世俗文献尽量均匀出现在
    每折测试集和开发集中；一卷绝不跨 train/dev/test。
    """

    def __init__(self, n_splits: int = 5, dev_ratio: float = 0.125, seed: int = 42) -> None:
        if n_splits < 2:
            raise ValueError("n_splits 至少为 2")
        if not 0 < dev_ratio < 1:
            raise ValueError("dev_ratio 必须在 (0, 1) 内")
        self.n_splits = n_splits
        self.dev_ratio = dev_ratio
        self.seed = seed

    def split(self, documents: list[PreparedDocument]) -> list[FoldSplit]:
        ids = [document.document_id for document in documents]
        if len(ids) < self.n_splits:
            raise ValueError("卷数少于折数")
        by_domain: dict[str, list[str]] = defaultdict(list)
        for document in documents:
            by_domain[document.domain].append(document.document_id)

        test_folds: list[list[str]] = [[] for _ in range(self.n_splits)]
        for domain_index, domain in enumerate(sorted(by_domain)):
            domain_ids = by_domain[domain][:]
            random.Random(self.seed + domain_index).shuffle(domain_ids)
            for index, document_id in enumerate(domain_ids):
                test_folds[index % self.n_splits].append(document_id)

        results: list[FoldSplit] = []
        for fold, test_ids in enumerate(test_folds, 1):
            test_set = set(test_ids)
            remaining_by_domain: dict[str, list[str]] = defaultdict(list)
            for document in documents:
                if document.document_id not in test_set:
                    remaining_by_domain[document.domain].append(document.document_id)
            train_ids: list[str] = []
            dev_ids: list[str] = []
            for domain_index, domain in enumerate(sorted(remaining_by_domain)):
                remaining = remaining_by_domain[domain]
                random.Random(self.seed + fold * 100 + domain_index).shuffle(remaining)
                dev_size = round(len(remaining) * self.dev_ratio)
                if dev_size == 0 and remaining:
                    dev_size = 1
                # 至少给训练集留一卷。
                dev_size = min(dev_size, max(0, len(remaining) - 1))
                dev_ids.extend(remaining[:dev_size])
                train_ids.extend(remaining[dev_size:])
            split = FoldSplit(
                fold=fold,
                train_ids=tuple(train_ids),
                dev_ids=tuple(dev_ids),
                test_ids=tuple(sorted(test_ids)),
            )
            split.validate()
            results.append(split)
        return results
