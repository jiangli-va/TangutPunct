from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from data.corpus import SequenceChunk
from tasks import OUTSIDE

from .base import SequenceTagger


LOGGER = logging.getLogger(__name__)
POSITION_LABEL = "P"
ContextKey = tuple[str, ...]


class BackoffNGramTagger(SequenceTagger):
    """双向字符n-gram分类器，供B2的位置和类别阶段共用。"""

    def __init__(
        self,
        punctuation_labels: Iterable[str],
        n: int = 3,
        min_support: int = 5,
        alpha: float = 0.5,
        backoff_k: float = 10.0,
        threshold_start: float = 0.05,
        threshold_end: float = 0.95,
        threshold_step: float = 0.05,
        use_left: bool = True,
        use_right: bool = True,
        use_cross_gap: bool = True,
    ) -> None:
        self.punctuation_labels = tuple(punctuation_labels)
        self.n = n
        self.min_support = min_support
        self.alpha = alpha
        self.backoff_k = backoff_k
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.threshold_step = threshold_step
        self.use_left = use_left
        self.use_right = use_right
        self.use_cross_gap = use_cross_gap

        self.task = "unfitted"
        self.classes: tuple[str, ...] = ()
        self.class_counts: Counter[str] = Counter()
        self.context_counts: dict[ContextKey, Counter[str]] = {}
        self.position_threshold = 0.5
        self.dev_position_f1 = 0.0
        self._fitted = False

    @staticmethod
    def _boundary_token(chunk: SequenceChunk, position: int) -> str:
        if position < chunk.block_start:
            return "<BOS>" if chunk.block_start == 0 else "<CUT>"
        if position >= chunk.block_end:
            return (
                "<EOS>"
                if chunk.block_end == chunk.document_length
                else "<CUT>"
            )
        return chunk.document_tokens[position]

    def _context_families(
        self, chunk: SequenceChunk, local_index: int
    ) -> dict[str, list[ContextKey]]:
        index = chunk.offset + local_index
        families: dict[str, list[ContextKey]] = defaultdict(list)
        if self.use_left:
            for order in range(1, self.n + 1):
                values = tuple(
                    self._boundary_token(chunk, position)
                    for position in range(index - order + 1, index + 1)
                )
                families["L"].append(("L", str(order), *values))
        if self.use_right:
            for order in range(1, self.n + 1):
                values = tuple(
                    self._boundary_token(chunk, position)
                    for position in range(index + 1, index + order + 1)
                )
                families["R"].append(("R", str(order), *values))
        if self.use_cross_gap:
            for order in range(2, self.n + 1):
                for left_size in range(1, order):
                    right_size = order - left_size
                    left = tuple(
                        self._boundary_token(chunk, position)
                        for position in range(index - left_size + 1, index + 1)
                    )
                    right = tuple(
                        self._boundary_token(chunk, position)
                        for position in range(index + 1, index + right_size + 1)
                    )
                    families["X"].append(
                        ("X", str(order), str(left_size), *left, "|", *right)
                    )
        return dict(families)

    def _all_contexts(
        self, chunk: SequenceChunk, local_index: int
    ) -> Iterable[ContextKey]:
        for keys in self._context_families(chunk, local_index).values():
            yield from keys

    @staticmethod
    def _upstream_positions(chunk: SequenceChunk) -> tuple[str, ...] | None:
        channels = dict(chunk.document_feature_channels)
        return channels.get("position")

    def _infer_task(self, train: list[SequenceChunk]) -> None:
        observed = {
            label
            for chunk in train
            for label in chunk.labels
            if label != OUTSIDE
        }
        if observed == {POSITION_LABEL}:
            self.task = "position"
            self.classes = (OUTSIDE, POSITION_LABEL)
            return
        punctuation = observed & set(self.punctuation_labels)
        if punctuation:
            self.task = "punctuation_type"
            # 固定保留七种类别，使每折的平滑、并列决策和Macro评价一致。
            self.classes = self.punctuation_labels
            return
        raise ValueError(f"无法从训练标签识别n-gram阶段：{sorted(observed)}")

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        if not train:
            raise ValueError("n-gram训练序列不能为空")
        self._infer_task(train)
        context_counts: dict[ContextKey, Counter[str]] = defaultdict(Counter)
        class_counts: Counter[str] = Counter()

        for chunk in train:
            for local_index, gold in enumerate(chunk.labels):
                if self.task == "punctuation_type":
                    # 严格第二阶段：只在训练集真实标点位置学习七分类。
                    if gold not in self.punctuation_labels:
                        continue
                    label = gold
                else:
                    label = POSITION_LABEL if gold == POSITION_LABEL else OUTSIDE
                class_counts[label] += 1
                for key in self._all_contexts(chunk, local_index):
                    context_counts[key][label] += 1

        if not class_counts:
            raise ValueError("当前训练折没有可供n-gram学习的标签")
        self.class_counts = class_counts
        self.context_counts = dict(context_counts)
        self._fitted = True

        if self.task == "position":
            self.position_threshold, self.dev_position_f1 = self._select_threshold(
                dev or []
            )
            LOGGER.info(
                "位置n-gram完成：n=%d，上下文=%d，开发集阈值=%.2f，开发集位置F1=%.4f",
                self.n,
                len(self.context_counts),
                self.position_threshold,
                self.dev_position_f1,
            )
        else:
            LOGGER.info(
                "类别n-gram完成：n=%d，上下文=%d，训练标点=%d",
                self.n,
                len(self.context_counts),
                sum(self.class_counts.values()),
            )

    def _distribution(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.get(label, 0) for label in self.classes)
        denominator = total + self.alpha * len(self.classes)
        return {
            label: (counts.get(label, 0) + self.alpha) / denominator
            for label in self.classes
        }

    def _selected_contexts(
        self, chunk: SequenceChunk, local_index: int
    ) -> list[tuple[Counter[str], int]]:
        selected: list[tuple[Counter[str], int]] = []
        for keys in self._context_families(chunk, local_index).values():
            candidates = []
            for key in keys:
                counts = self.context_counts.get(key)
                if counts is None:
                    continue
                support = sum(counts.values())
                if support >= self.min_support:
                    candidates.append((int(key[1]), support, key, counts))
            if candidates:
                _, support, _, counts = max(
                    candidates, key=lambda item: (item[0], item[1], item[2])
                )
                selected.append((counts, support))
        return selected

    def probabilities(
        self, chunk: SequenceChunk, local_index: int
    ) -> dict[str, float]:
        """返回当前位置经平滑和最长可靠上下文回退后的类别概率。"""
        if not self._fitted:
            raise RuntimeError("n-gram模型尚未训练")
        prior = self._distribution(self.class_counts)
        totals = dict(prior)
        total_weight = 1.0
        for counts, support in self._selected_contexts(chunk, local_index):
            distribution = self._distribution(counts)
            weight = support / (support + self.backoff_k)
            total_weight += weight
            for label in self.classes:
                totals[label] += weight * distribution[label]
        return {label: value / total_weight for label, value in totals.items()}

    def _threshold_values(self) -> list[float]:
        values = []
        value = self.threshold_start
        while value <= self.threshold_end + 1e-12:
            values.append(round(value, 10))
            value += self.threshold_step
        return values

    @staticmethod
    def _position_metrics(gold: list[bool], predicted: list[bool]) -> tuple[float, float]:
        tp = sum(expected and actual for expected, actual in zip(gold, predicted))
        fp = sum(not expected and actual for expected, actual in zip(gold, predicted))
        fn = sum(expected and not actual for expected, actual in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, f1

    def _select_threshold(self, dev: list[SequenceChunk]) -> tuple[float, float]:
        if not dev:
            return 0.5, 0.0
        gold: list[bool] = []
        scores: list[float] = []
        for chunk in dev:
            for local_index, label in enumerate(chunk.labels):
                gold.append(label == POSITION_LABEL)
                scores.append(self.probabilities(chunk, local_index)[POSITION_LABEL])

        best_threshold = 0.5
        best_precision = -1.0
        best_f1 = -1.0
        for threshold in self._threshold_values():
            precision, f1 = self._position_metrics(
                gold, [score >= threshold for score in scores]
            )
            # F1并列时先取精确率更高、再取阈值更高的方案。
            if (f1, precision, threshold) > (
                best_f1,
                best_precision,
                best_threshold,
            ):
                best_threshold = threshold
                best_precision = precision
                best_f1 = f1
        return best_threshold, best_f1

    def _best_class(self, probabilities: dict[str, float]) -> str:
        return max(
            self.classes,
            key=lambda label: (probabilities[label], -self.classes.index(label)),
        )

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        if not self._fitted:
            raise RuntimeError("n-gram模型尚未训练")
        predictions: list[list[str]] = []
        for chunk in sequences:
            upstream = self._upstream_positions(chunk)
            if self.task == "punctuation_type" and upstream is None:
                raise ValueError("类别n-gram预测缺少上游position通道")
            labels = []
            for local_index in range(len(chunk.tokens)):
                absolute_index = chunk.offset + local_index
                probabilities = self.probabilities(chunk, local_index)
                if self.task == "position":
                    labels.append(
                        POSITION_LABEL
                        if probabilities[POSITION_LABEL] >= self.position_threshold
                        else OUTSIDE
                    )
                elif upstream is not None and upstream[absolute_index] == POSITION_LABEL:
                    labels.append(self._best_class(probabilities))
                else:
                    labels.append(OUTSIDE)
            predictions.append(labels)
        return predictions

    @staticmethod
    def _display_context(key: ContextKey) -> str:
        if key[0] == "X":
            return "".join(key[3:])
        return "".join(key[2:])

    def metadata(self) -> dict[str, object]:
        most_common_contexts = sorted(
            self.context_counts.items(),
            key=lambda item: (-sum(item[1].values()), item[0]),
        )[:20]
        return {
            "模型阶段": "位置" if self.task == "position" else "具体类别",
            "n": self.n,
            "最小支持数": self.min_support,
            "平滑系数": self.alpha,
            "回退权重k": self.backoff_k,
            "上下文规则数": len(self.context_counts),
            "训练标签频次": dict(self.class_counts),
            "开发集位置阈值": (
                self.position_threshold if self.task == "position" else None
            ),
            "开发集位置F1": self.dev_position_f1 if self.task == "position" else None,
            "高频上下文示例": [
                {
                    "方向": key[0],
                    "阶数": int(key[1]),
                    "上下文": self._display_context(key),
                    "频次": sum(counts.values()),
                }
                for key, counts in most_common_contexts
            ],
        }

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
