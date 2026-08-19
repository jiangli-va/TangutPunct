from __future__ import annotations

import logging
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from config import ContextKnowledgeConfig
from data.corpus import SequenceChunk, is_tangut

from .base import FeatureMatrix, KnowledgeFeatureProvider


LOGGER = logging.getLogger(__name__)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return max(ordered[lower], 1.0e-12)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return max(value, 1.0e-12)


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )


@dataclass(frozen=True)
class _ContextStatistics:
    unigram: Counter[str]
    bigram: Counter[tuple[str, str]]
    right_neighbors: dict[str, Counter[str]]
    left_neighbors: dict[str, Counter[str]]
    total_bigrams: int
    log_frequency: dict[tuple[str, str], float]
    association: dict[tuple[str, str], float]
    right_entropy: dict[str, float]
    left_entropy: dict[str, float]
    frequency_scale: float
    association_scale: float
    entropy_scale: float


class ContextStatisticsProvider(KnowledgeFeatureProvider):
    """E3：由未截长训练文献估计的左右间隔8维上下文统计。

    训练特征使用文献级内部OOF；开发、测试和推理只读取当前外层训练
    文献拟合出的完整统计。统计源去掉标点后连接正文，但缺字、TAB及
    其他非正文字符会阻断bigram，且绝不跨文献。
    """

    def __init__(
        self,
        config: ContextKnowledgeConfig,
        missing_characters: tuple[str, ...],
        seed: int,
    ) -> None:
        self.config = config
        self.missing_characters = frozenset(missing_characters)
        self.seed = seed
        self.source_mapping = dict(config.source_mapping)
        self._source_lines: dict[Path, list[str]] = {}
        self._statistics: _ContextStatistics | None = None
        self._training_metadata: dict[str, object] = {}

    @property
    def dimension(self) -> int:
        return 8

    @property
    def feature_names(self) -> tuple[str, ...]:
        association = self.config.association
        return (
            "context_left_log_frequency",
            f"context_left_{association}",
            "context_left_previous_right_entropy",
            "context_left_current_left_entropy",
            "context_right_log_frequency",
            f"context_right_{association}",
            "context_right_current_right_entropy",
            "context_right_next_left_entropy",
        )

    @staticmethod
    def _document_stem(document_id: str) -> str:
        marker = "_volume_"
        if marker not in document_id:
            raise ValueError(f"E3无法从文献ID解析语料stem：{document_id}")
        return document_id.rsplit(marker, 1)[0]

    def _line(self, chunk: SequenceChunk) -> str:
        stem = self._document_stem(chunk.document_id)
        path = self.source_mapping.get(stem)
        if path is None:
            available = "、".join(sorted(self.source_mapping))
            raise ValueError(
                f"E3没有为{stem}配置未截长语料；已配置：{available}"
            )
        lines = self._source_lines.get(path)
        if lines is None:
            lines = path.read_text(encoding="utf-8").splitlines()
            self._source_lines[path] = lines
        index = chunk.volume_number - 1
        if not 0 <= index < len(lines):
            raise ValueError(
                f"{chunk.document_id}卷号{chunk.volume_number}超出{path}的{len(lines)}行"
            )
        return lines[index]

    def _segments(self, text: str) -> list[tuple[str, ...]]:
        segments: list[tuple[str, ...]] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                segments.append(tuple(current))
                current.clear()

        for character in text:
            if character in self.missing_characters or character == "\t":
                flush()
            elif is_tangut(character):
                current.append(character)
            elif character.isspace():
                flush()
            elif unicodedata.category(character)[0] in {"P", "S"}:
                # 去掉金标准标点，不让“此处原本有标点”成为统计边界。
                continue
            else:
                flush()
        flush()
        return segments

    def _document_segments(
        self, sequences: list[SequenceChunk]
    ) -> dict[str, list[tuple[str, ...]]]:
        representative: dict[str, SequenceChunk] = {}
        for chunk in sequences:
            representative.setdefault(chunk.document_id, chunk)
        return {
            document_id: self._segments(self._line(chunk))
            for document_id, chunk in representative.items()
        }

    def _association(
        self,
        pair: tuple[str, str],
        count: int,
        unigram: Counter[str],
        total_bigrams: int,
    ) -> float:
        left, right = pair
        left_count = max(unigram[left], 1)
        right_count = max(unigram[right], 1)
        metric = self.config.association
        if metric == "dpmi":
            discounted = count - self.config.dpmi_discount
            if discounted <= 0 or total_bigrams <= 0:
                return 0.0
            return max(
                0.0,
                math.log(discounted)
                - math.log(left_count)
                - math.log(right_count)
                + math.log(total_bigrams),
            )
        if metric == "dice":
            return 2.0 * count / (left_count + right_count)
        expected = left_count * right_count / max(total_bigrams, 1)
        return (count - expected) / math.sqrt(count)

    def _compute_statistics(
        self, documents: dict[str, list[tuple[str, ...]]], document_ids: set[str]
    ) -> _ContextStatistics:
        unigram: Counter[str] = Counter()
        bigram: Counter[tuple[str, str]] = Counter()
        right_neighbors: dict[str, Counter[str]] = defaultdict(Counter)
        left_neighbors: dict[str, Counter[str]] = defaultdict(Counter)
        for document_id in document_ids:
            for segment in documents[document_id]:
                unigram.update(segment)
                for left, right in zip(segment, segment[1:]):
                    bigram[(left, right)] += 1
                    right_neighbors[left][right] += 1
                    left_neighbors[right][left] += 1
        total_bigrams = sum(bigram.values())
        log_frequency = {
            pair: math.log1p(count) for pair, count in bigram.items()
        }
        association = {
            pair: self._association(pair, count, unigram, total_bigrams)
            for pair, count in bigram.items()
        }
        right_entropy = {
            character: _entropy(counts)
            for character, counts in right_neighbors.items()
        }
        left_entropy = {
            character: _entropy(counts)
            for character, counts in left_neighbors.items()
        }
        frequency_scale = max(log_frequency.values(), default=1.0)
        if self.config.association == "dice":
            association_scale = 1.0
        elif self.config.association == "t_score":
            association_scale = _percentile(
                [abs(value) for value in association.values()],
                self.config.clipping_percentile,
            )
        else:
            association_scale = _percentile(
                [value for value in association.values() if value > 0],
                self.config.clipping_percentile,
            )
        entropy_scale = math.log(max(len(unigram), 2))
        return _ContextStatistics(
            unigram,
            bigram,
            dict(right_neighbors),
            dict(left_neighbors),
            total_bigrams,
            log_frequency,
            association,
            right_entropy,
            left_entropy,
            max(frequency_scale, 1.0e-12),
            max(association_scale, 1.0e-12),
            max(entropy_scale, 1.0e-12),
        )

    def _normalized_association(
        self, pair: tuple[str, str], statistics: _ContextStatistics
    ) -> float:
        value = statistics.association.get(pair, 0.0)
        if self.config.association == "dice":
            return min(max(value, 0.0), 1.0)
        scaled = value / statistics.association_scale
        if self.config.association == "t_score":
            return min(max(scaled, -1.0), 1.0)
        return min(max(scaled, 0.0), 1.0)

    def _gap(
        self, left: str, right: str, statistics: _ContextStatistics
    ) -> tuple[float, float, float, float]:
        if not is_tangut(left) or not is_tangut(right):
            return (0.0, 0.0, 0.0, 0.0)
        pair = (left, right)
        return (
            statistics.log_frequency.get(pair, 0.0) / statistics.frequency_scale,
            self._normalized_association(pair, statistics),
            statistics.right_entropy.get(left, 0.0) / statistics.entropy_scale,
            statistics.left_entropy.get(right, 0.0) / statistics.entropy_scale,
        )

    def _extract(
        self, chunk: SequenceChunk, statistics: _ContextStatistics
    ) -> FeatureMatrix:
        context_start = max(chunk.block_start, chunk.offset - 1)
        context_end = min(
            chunk.block_end, chunk.offset + len(chunk.tokens) + 1
        )
        context = chunk.document_tokens[context_start:context_end]
        local_start = chunk.offset - context_start
        rows: list[tuple[float, ...]] = []
        for local_position in range(len(chunk.tokens)):
            position = local_start + local_position
            left = (
                self._gap(context[position - 1], context[position], statistics)
                if position > 0
                else (0.0, 0.0, 0.0, 0.0)
            )
            right = (
                self._gap(context[position], context[position + 1], statistics)
                if position + 1 < len(context)
                else (0.0, 0.0, 0.0, 0.0)
            )
            rows.append(left + right)
        return tuple(rows)

    def _document_folds(self, sequences: list[SequenceChunk]) -> dict[str, int]:
        by_domain: dict[str, list[str]] = defaultdict(list)
        for chunk in sequences:
            if chunk.document_id not in by_domain[chunk.domain]:
                by_domain[chunk.domain].append(chunk.document_id)
        assignment: dict[str, int] = {}
        for domain_index, domain in enumerate(sorted(by_domain)):
            document_ids = sorted(by_domain[domain])
            random.Random(self.seed + domain_index).shuffle(document_ids)
            for index, document_id in enumerate(document_ids):
                assignment[document_id] = index % self.config.inner_folds
        return assignment

    @staticmethod
    def _chunk_key(chunk: SequenceChunk) -> tuple[str, int, int]:
        return chunk.document_id, chunk.offset, len(chunk.tokens)

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if not sequences:
            raise ValueError("E3训练序列不能为空")
        documents = self._document_segments(sequences)
        fold_of = self._document_folds(sequences)
        matrices: dict[tuple[str, int, int], FeatureMatrix] = {}
        active_folds = sorted(set(fold_of.values()))
        all_ids = set(documents)
        for fold in active_folds:
            reference_ids = {
                document_id
                for document_id, assigned_fold in fold_of.items()
                if assigned_fold != fold
            }
            statistics = self._compute_statistics(documents, reference_ids)
            for chunk in sequences:
                if fold_of[chunk.document_id] == fold:
                    matrices[self._chunk_key(chunk)] = self._extract(chunk, statistics)
        self._statistics = self._compute_statistics(documents, all_ids)
        visible_pairs = 0
        seen_pairs = 0
        for chunk in sequences:
            matrix = matrices[self._chunk_key(chunk)]
            for row in matrix:
                if row[4] > 0:
                    seen_pairs += 1
                visible_pairs += 1
        self._training_metadata = {
            "文献数": len(documents),
            "unigram种类": len(self._statistics.unigram),
            "unigram总数": sum(self._statistics.unigram.values()),
            "bigram种类": len(self._statistics.bigram),
            "bigram总数": self._statistics.total_bigrams,
            "OOF当前间隔已见率": seen_pairs / max(visible_pairs, 1),
            "内部OOF折数": len(active_folds),
        }
        LOGGER.info(
            "E3上下文统计：关联度=%s，维度=8，未截长训练文献=%d，"
            "bigram=%d类/%d次，OOF当前间隔已见率=%.2f%%",
            self.config.association,
            len(documents),
            len(self._statistics.bigram),
            self._statistics.total_bigrams,
            100 * float(self._training_metadata["OOF当前间隔已见率"]),
        )
        return [matrices[self._chunk_key(chunk)] for chunk in sequences]

    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if self._statistics is None:
            raise RuntimeError("E3上下文提供器尚未在训练文献上拟合")
        return [self._extract(chunk, self._statistics) for chunk in sequences]

    def metadata(self) -> dict[str, object]:
        return {
            "名称": "未截长语料间隔上下文统计",
            "维度": self.dimension,
            "关联度": self.config.association,
            "关联度归一化": (
                "原值已位于[0,1]"
                if self.config.association == "dice"
                else "绝对值分位数对称截断到[-1,1]"
                if self.config.association == "t_score"
                else "正值分位数截断到[0,1]"
            ),
            "截断分位数": self.config.clipping_percentile,
            "dPMI折扣": self.config.dpmi_discount,
            "未截长语料映射": {
                stem: str(path) for stem, path in self.config.source_mapping
            },
            "训练统计": dict(self._training_metadata),
        }

    def state_dict(self) -> dict[str, object]:
        if self._statistics is None:
            raise RuntimeError("E3上下文提供器尚未拟合，不能保存统计状态")
        statistics = self._statistics
        return {
            "type": "context_statistics",
            "feature_names": list(self.feature_names),
            "association": self.config.association,
            "unigram": dict(statistics.unigram),
            "bigram": [(left, right, count) for (left, right), count in statistics.bigram.items()],
            "normalization": {
                "frequency_scale": statistics.frequency_scale,
                "association_scale": statistics.association_scale,
                "entropy_scale": statistics.entropy_scale,
                "clipping_percentile": self.config.clipping_percentile,
            },
            "metadata": dict(self._training_metadata),
        }
