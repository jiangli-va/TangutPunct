from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data.corpus import SequenceChunk
from tasks import OUTSIDE

from .base import SequenceTagger


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnedRule:
    label: str
    confidence: float
    support: int
    total: int


@dataclass(frozen=True)
class _RuleDocument:
    document_id: str
    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    blocks: tuple[tuple[int, int], ...]


def _documents_from_chunks(chunks: Iterable[SequenceChunk]) -> list[_RuleDocument]:
    """把模型接口中的分块重新拼成文献，同时保留真实 TAB 边界。"""
    grouped: dict[str, list[SequenceChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)

    documents: list[_RuleDocument] = []
    for document_id, parts in grouped.items():
        ordered = sorted(parts, key=lambda item: item.offset)
        tokens = ordered[0].document_tokens
        labels = [OUTSIDE] * len(tokens)
        covered = [False] * len(tokens)
        for chunk in ordered:
            start = chunk.offset
            end = start + len(chunk.labels)
            if end > len(tokens):
                raise ValueError(f"{document_id} 的序列块超出文献范围")
            labels[start:end] = chunk.labels
            covered[start:end] = [True] * len(chunk.labels)
        if not all(covered):
            raise ValueError(f"{document_id} 的序列块没有覆盖全部正文")
        blocks = tuple(
            sorted({(chunk.block_start, chunk.block_end) for chunk in ordered})
        )
        documents.append(
            _RuleDocument(document_id, tokens, tuple(labels), blocks)
        )
    return documents


class LengthMajorityRuleTagger(SequenceTagger):
    """B0：按训练集片段中位长度放置标点，类别使用多数规则。"""

    def __init__(
        self,
        labels: Iterable[str],
        force_document_final_period: bool = True,
    ) -> None:
        self.label_order = tuple(labels)
        if not self.label_order:
            raise ValueError("规则模型至少需要一个目标标点")
        self.force_document_final_period = force_document_final_period
        self.median_segment_length = 1
        self.global_majority = self.label_order[0]
        self.length_majorities: dict[int, str] = {}
        self.label_counts: Counter[str] = Counter()
        self._fitted = False

    def _label_rank(self, label: str) -> int:
        try:
            return self.label_order.index(label)
        except ValueError:
            return len(self.label_order)

    def _majority(self, counts: Counter[str]) -> str:
        candidates = [
            (count, -self._label_rank(label), label)
            for label, count in counts.items()
            if label in self.label_order
        ]
        if not candidates:
            return self.global_majority
        return max(candidates)[2]

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        del dev  # 基础规则没有可在开发集上拟合的参数。
        documents = _documents_from_chunks(train)
        segment_lengths: list[int] = []
        by_length: dict[int, Counter[str]] = defaultdict(Counter)
        label_counts: Counter[str] = Counter()

        for document in documents:
            for block_start, block_end in document.blocks:
                distance = 0
                for index in range(block_start, block_end):
                    distance += 1
                    label = document.labels[index]
                    if label not in self.label_order:
                        continue
                    segment_lengths.append(distance)
                    by_length[distance][label] += 1
                    label_counts[label] += 1
                    distance = 0

        if not segment_lengths or not label_counts:
            raise ValueError("训练折中没有可学习的停顿标点")
        self.label_counts = label_counts
        self.global_majority = self._majority(label_counts)
        self.median_segment_length = max(
            1, int(statistics.median(segment_lengths))
        )
        self.length_majorities = {
            length: self._majority(counts) for length, counts in by_length.items()
        }
        self._fitted = True
        LOGGER.info(
            "规则统计完成：片段中位长度=%d，全局多数标点=%s，训练标点=%d",
            self.median_segment_length,
            self.global_majority,
            sum(label_counts.values()),
        )

    def _final_label(self) -> str:
        return "。" if "。" in self.label_order else self.global_majority

    def _length_label(self, distance: int) -> str:
        return self.length_majorities.get(distance, self.global_majority)

    def _predict_document(self, document: _RuleDocument) -> list[str]:
        labels = [OUTSIDE] * len(document.tokens)
        for block_start, block_end in document.blocks:
            distance = 0
            for index in range(block_start, block_end):
                distance += 1
                if distance >= self.median_segment_length:
                    labels[index] = self.global_majority
                    distance = 0
        if labels and self.force_document_final_period:
            labels[-1] = self._final_label()
        return labels

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        if not self._fitted:
            raise RuntimeError("规则模型尚未训练")
        documents = _documents_from_chunks(sequences)
        predicted = {
            document.document_id: self._predict_document(document)
            for document in documents
        }
        return [
            predicted[chunk.document_id][chunk.offset : chunk.offset + len(chunk.tokens)]
            for chunk in sequences
        ]

    def metadata(self) -> dict[str, object]:
        return {
            "规则类型": "长度—多数类",
            "片段中位长度": self.median_segment_length,
            "全局多数标点": self.global_majority,
            "训练折标点频次": dict(self.label_counts),
            "文献末尾强制句号": self.force_document_final_period,
        }

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class JointRuleTagger(LengthMajorityRuleTagger):
    """B1：直接联合长度、特征字词和重复结构规则预测具体标点。"""

    CUE_PRIORITIES = {"L2": 400, "LR": 350, "L1": 300, "R1": 250}
    STRUCTURE_PRIORITY = 325

    def __init__(
        self,
        labels: Iterable[str],
        cue_min_support: int = 8,
        cue_min_confidence: float = 0.70,
        structure_min_support: int = 8,
        structure_min_confidence: float = 0.55,
        cue_max_length: int = 2,
        force_document_final_period: bool = True,
    ) -> None:
        super().__init__(labels, force_document_final_period)
        self.cue_min_support = cue_min_support
        self.cue_min_confidence = cue_min_confidence
        self.structure_min_support = structure_min_support
        self.structure_min_confidence = structure_min_confidence
        self.cue_max_length = cue_max_length
        self.cue_rules: dict[str, LearnedRule] = {}
        self.structure_rules: dict[str, LearnedRule] = {}

    @staticmethod
    def _cue_keys(
        tokens: tuple[str, ...], index: int, block_start: int, block_end: int
    ) -> tuple[str, ...]:
        keys = [f"L1:{tokens[index]}"]
        if index > block_start:
            keys.append(f"L2:{tokens[index - 1]}{tokens[index]}")
        if index + 1 < block_end:
            keys.append(f"R1:{tokens[index + 1]}")
            keys.append(f"LR:{tokens[index]}{tokens[index + 1]}")
        return tuple(keys)

    def _selected_cue_keys(
        self, tokens: tuple[str, ...], index: int, block_start: int, block_end: int
    ) -> tuple[str, ...]:
        keys = self._cue_keys(tokens, index, block_start, block_end)
        if self.cue_max_length == 1:
            return tuple(key for key in keys if key.startswith(("L1:", "R1:")))
        return keys

    @staticmethod
    def _structure_keys(
        tokens: tuple[str, ...], index: int, block_start: int, block_end: int
    ) -> tuple[str, ...]:
        keys: list[str] = []
        if index + 1 < block_end and tokens[index] == tokens[index + 1]:
            keys.append("相邻同字_A|A")
        if (
            index > block_start
            and index + 1 < block_end
            and tokens[index - 1] == tokens[index + 1]
        ):
            keys.append("跨界重复_A B|A")
        if (
            index > block_start
            and index + 2 < block_end
            and tokens[index - 1 : index + 1] == tokens[index + 1 : index + 3]
        ):
            keys.append("二字重复_AB|AB")
        return tuple(keys)

    def _learn_rules(
        self,
        counts: dict[str, Counter[str]],
        min_support: int,
        min_confidence: float,
    ) -> dict[str, LearnedRule]:
        learned: dict[str, LearnedRule] = {}
        for key, values in counts.items():
            total = sum(values.values())
            punctuation = Counter(
                {label: count for label, count in values.items() if label in self.label_order}
            )
            if not punctuation or total == 0:
                continue
            label = self._majority(punctuation)
            support = punctuation[label]
            confidence = support / total
            if support >= min_support and confidence >= min_confidence:
                learned[key] = LearnedRule(label, confidence, support, total)
        return learned

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        super().fit(train, dev)
        documents = _documents_from_chunks(train)
        cue_counts: dict[str, Counter[str]] = defaultdict(Counter)
        structure_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for document in documents:
            for block_start, block_end in document.blocks:
                for index in range(block_start, block_end):
                    # 文献末尾已经由硬规则处理，不把它泄漏成某个末字的词汇规则。
                    if index == len(document.tokens) - 1:
                        continue
                    label = (
                        document.labels[index]
                        if document.labels[index] in self.label_order
                        else OUTSIDE
                    )
                    for key in self._selected_cue_keys(
                        document.tokens, index, block_start, block_end
                    ):
                        cue_counts[key][label] += 1
                    for key in self._structure_keys(
                        document.tokens, index, block_start, block_end
                    ):
                        structure_counts[key][label] += 1

        self.cue_rules = self._learn_rules(
            cue_counts, self.cue_min_support, self.cue_min_confidence
        )
        self.structure_rules = self._learn_rules(
            structure_counts,
            self.structure_min_support,
            self.structure_min_confidence,
        )
        LOGGER.info(
            "联合规则生成完成：特征字词规则=%d，重复结构规则=%d",
            len(self.cue_rules),
            len(self.structure_rules),
        )

    def _best_learned_rule(
        self,
        tokens: tuple[str, ...],
        index: int,
        block_start: int,
        block_end: int,
    ) -> LearnedRule | None:
        matches: list[tuple[int, float, int, int, LearnedRule]] = []
        for key in self._selected_cue_keys(tokens, index, block_start, block_end):
            rule = self.cue_rules.get(key)
            if rule is None:
                continue
            kind = key.split(":", 1)[0]
            matches.append(
                (
                    self.CUE_PRIORITIES[kind],
                    rule.confidence,
                    rule.support,
                    -self._label_rank(rule.label),
                    rule,
                )
            )
        for key in self._structure_keys(tokens, index, block_start, block_end):
            rule = self.structure_rules.get(key)
            if rule is not None:
                matches.append(
                    (
                        self.STRUCTURE_PRIORITY,
                        rule.confidence,
                        rule.support,
                        -self._label_rank(rule.label),
                        rule,
                    )
                )
        return max(matches, default=None, key=lambda item: item[:4])[-1] if matches else None

    def _predict_document(self, document: _RuleDocument) -> list[str]:
        labels = [OUTSIDE] * len(document.tokens)
        for block_start, block_end in document.blocks:
            distance = 0
            for index in range(block_start, block_end):
                distance += 1
                learned = self._best_learned_rule(
                    document.tokens, index, block_start, block_end
                )
                if learned is not None:
                    labels[index] = learned.label
                elif distance >= self.median_segment_length:
                    labels[index] = self._length_label(distance)
                if labels[index] != OUTSIDE:
                    distance = 0
        if labels and self.force_document_final_period:
            labels[-1] = self._final_label()
        return labels

    @staticmethod
    def _rule_preview(rules: dict[str, LearnedRule], limit: int = 20) -> list[dict[str, object]]:
        ordered = sorted(
            rules.items(),
            key=lambda item: (-item[1].support, -item[1].confidence, item[0]),
        )
        return [
            {
                "规则": key,
                "标点": rule.label,
                "置信度": round(rule.confidence, 6),
                "支持数": rule.support,
                "出现数": rule.total,
            }
            for key, rule in ordered[:limit]
        ]

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata.update(
            {
                "规则类型": "长度＋特征字词＋重复结构联合规则",
                "特征字词规则数": len(self.cue_rules),
                "重复结构规则数": len(self.structure_rules),
                "特征规则阈值": {
                    "最小支持数": self.cue_min_support,
                    "最小置信度": self.cue_min_confidence,
                },
                "结构规则阈值": {
                    "最小支持数": self.structure_min_support,
                    "最小置信度": self.structure_min_confidence,
                },
                "高频特征规则示例": self._rule_preview(self.cue_rules),
                "重复结构规则": self._rule_preview(self.structure_rules),
            }
        )
        return metadata
