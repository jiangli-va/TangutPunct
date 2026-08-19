from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from config import DomainKnowledgeConfig, WordPretrainingConfig
from data.corpus import SequenceChunk
from pretraining.data import MLMSequence
from pretraining.word_candidates import WordCandidate, build_word_candidates

from .base import FeatureMatrix, KnowledgeFeatureProvider


LOGGER = logging.getLogger(__name__)
JINGSHU = "jingshu"
SHISU = "shisu"
NEUTRAL = (0.5, 0.5)


@dataclass(frozen=True)
class _DomainStatistics:
    candidate_counts: dict[str, Counter[str]]
    character_counts: dict[str, Counter[str]]
    candidate_opportunities: dict[str, Counter[int]]
    character_totals: Counter[str]


@dataclass(frozen=True)
class _TextUnit:
    document_id: str
    domain: str
    tokens: tuple[str, ...]


def _chunk_key(chunk: SequenceChunk) -> tuple[str, int, int]:
    return chunk.document_id, chunk.offset, len(chunk.tokens)


class LocalDomainDistributionProvider(KnowledgeFeatureProvider):
    """E1：候选词覆盖位置的经书/世俗局部分布。

    候选词集合沿用D2的筛选方法，但领域概率只由当前外层训练折统计。
    训练特征再按文献做内部OOF，避免一部文献用自身领域标签构造输入。
    """

    def __init__(
        self,
        config: DomainKnowledgeConfig,
        word_config: WordPretrainingConfig,
        seed: int,
    ) -> None:
        self.config = config
        self.word_config = word_config
        self.seed = seed
        self.candidates: dict[str, WordCandidate] = {}
        self._terms_by_length: dict[int, set[str]] = {}
        self._statistics: _DomainStatistics | None = None
        self._candidate_summary: dict[str, object] = {}
        self._training_character_coverage = 0.0

    @property
    def dimension(self) -> int:
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("local_domain_jingshu", "local_domain_shisu")

    @staticmethod
    def _validate_domains(sequences: list[SequenceChunk]) -> None:
        unknown = Counter(
            chunk.domain
            for chunk in sequences
            if chunk.domain not in {JINGSHU, SHISU}
        )
        if unknown:
            details = "、".join(f"{name}={count}" for name, count in unknown.items())
            raise ValueError(f"E1只能统计jingshu/shisu领域，发现：{details}")

    @staticmethod
    def _text_units(sequences: list[SequenceChunk]) -> list[_TextUnit]:
        """还原TAB训练块，避免128字模型切块截断跨边界候选词。"""

        units: dict[tuple[str, int, int], _TextUnit] = {}
        for chunk in sequences:
            key = (chunk.document_id, chunk.block_start, chunk.block_end)
            units[key] = _TextUnit(
                chunk.document_id,
                chunk.domain,
                chunk.document_tokens[chunk.block_start : chunk.block_end],
            )
        return list(units.values())

    @classmethod
    def _mlm_sequences(cls, sequences: list[SequenceChunk]) -> list[MLMSequence]:
        return [
            MLMSequence(unit.document_id, unit.domain, unit.tokens)
            for unit in cls._text_units(sequences)
        ]

    def _build_candidates(self, sequences: list[SequenceChunk]) -> None:
        self.candidates, self._candidate_summary = build_word_candidates(
            self._mlm_sequences(sequences),
            self.word_config,
            mode=self.config.candidate_mode,
        )
        if not self.candidates:
            raise ValueError("E1没有得到候选词，请检查D2辞书路径和候选阈值")
        grouped: dict[int, set[str]] = defaultdict(set)
        for term in self.candidates:
            grouped[len(term)].add(term)
        self._terms_by_length = dict(grouped)

    def _matches(self, tokens: tuple[str, ...]) -> list[tuple[int, int, str]]:
        text = "".join(tokens)
        matches: list[tuple[int, int, str]] = []
        for length, terms in self._terms_by_length.items():
            for start in range(max(0, len(text) - length + 1)):
                term = text[start : start + length]
                if term in terms:
                    matches.append((start, start + length, term))
        return matches

    def _compute_statistics(
        self, sequences: list[SequenceChunk]
    ) -> _DomainStatistics:
        candidate_counts = {JINGSHU: Counter(), SHISU: Counter()}
        character_counts = {JINGSHU: Counter(), SHISU: Counter()}
        opportunities = {JINGSHU: Counter(), SHISU: Counter()}
        character_totals: Counter[str] = Counter()
        lengths = tuple(self._terms_by_length)
        for unit in self._text_units(sequences):
            domain = unit.domain
            character_counts[domain].update(unit.tokens)
            character_totals[domain] += len(unit.tokens)
            for length in lengths:
                opportunities[domain][length] += max(0, len(unit.tokens) - length + 1)
            candidate_counts[domain].update(
                term for _, _, term in self._matches(unit.tokens)
            )
        return _DomainStatistics(
            candidate_counts,
            character_counts,
            opportunities,
            character_totals,
        )

    def _normalized_distribution(
        self,
        jingshu_count: int,
        shisu_count: int,
        jingshu_total: int,
        shisu_total: int,
    ) -> tuple[float, float]:
        alpha = self.config.smoothing
        jingshu_rate = (jingshu_count + alpha) / (jingshu_total + 2 * alpha)
        shisu_rate = (shisu_count + alpha) / (shisu_total + 2 * alpha)
        rate_total = jingshu_rate + shisu_rate
        raw_jingshu = jingshu_rate / rate_total if rate_total else 0.5
        support = jingshu_count + shisu_count
        strength = support / (support + self.config.shrinkage) if support else 0.0
        jingshu = 0.5 + strength * (raw_jingshu - 0.5)
        return jingshu, 1.0 - jingshu

    def _candidate_distribution(
        self, term: str, statistics: _DomainStatistics
    ) -> tuple[float, float]:
        length = len(term)
        return self._normalized_distribution(
            statistics.candidate_counts[JINGSHU][term],
            statistics.candidate_counts[SHISU][term],
            statistics.candidate_opportunities[JINGSHU][length],
            statistics.candidate_opportunities[SHISU][length],
        )

    def _character_distribution(
        self, character: str, statistics: _DomainStatistics
    ) -> tuple[float, float]:
        return self._normalized_distribution(
            statistics.character_counts[JINGSHU][character],
            statistics.character_counts[SHISU][character],
            statistics.character_totals[JINGSHU],
            statistics.character_totals[SHISU],
        )

    def _extract(
        self, chunk: SequenceChunk, statistics: _DomainStatistics
    ) -> FeatureMatrix:
        # 最多向两侧扩展“最长候选词-1”个字，使人工模型切块边缘仍能
        # 接收到跨边界词的领域信号；绝不越过TAB硬边界。
        margin = max(self._terms_by_length, default=1) - 1
        context_start = max(chunk.block_start, chunk.offset - margin)
        context_end = min(
            chunk.block_end,
            chunk.offset + len(chunk.tokens) + margin,
        )
        context = chunk.document_tokens[context_start:context_end]
        local_start = chunk.offset - context_start
        local_end = local_start + len(chunk.tokens)
        accumulated = [[0.0, 0.0] for _ in chunk.tokens]
        counts = [0] * len(chunk.tokens)
        for start, end, term in self._matches(context):
            overlap_start = max(start, local_start)
            overlap_end = min(end, local_end)
            if overlap_start >= overlap_end:
                continue
            distribution = self._candidate_distribution(term, statistics)
            for context_position in range(overlap_start, overlap_end):
                position = context_position - local_start
                accumulated[position][0] += distribution[0]
                accumulated[position][1] += distribution[1]
                counts[position] += 1
        output = []
        for position, character in enumerate(chunk.tokens):
            if counts[position]:
                output.append(
                    (
                        accumulated[position][0] / counts[position],
                        accumulated[position][1] / counts[position],
                    )
                )
            else:
                output.append(self._character_distribution(character, statistics))
        return tuple(output)

    def _document_folds(self, sequences: list[SequenceChunk]) -> dict[str, int]:
        by_domain: dict[str, list[str]] = defaultdict(list)
        domain_by_document: dict[str, str] = {}
        for chunk in sequences:
            previous = domain_by_document.setdefault(chunk.document_id, chunk.domain)
            if previous != chunk.domain:
                raise ValueError(f"{chunk.document_id}在同一折中出现多个领域")
        for document_id, domain in domain_by_document.items():
            by_domain[domain].append(document_id)
        assignment: dict[str, int] = {}
        for domain_index, domain in enumerate(sorted(by_domain)):
            document_ids = sorted(by_domain[domain])
            random.Random(self.seed + domain_index).shuffle(document_ids)
            for index, document_id in enumerate(document_ids):
                assignment[document_id] = index % self.config.inner_folds
        return assignment

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if not sequences:
            raise ValueError("E1训练序列不能为空")
        self._validate_domains(sequences)
        self._build_candidates(sequences)
        fold_of = self._document_folds(sequences)
        matrices: dict[tuple[str, int, int], FeatureMatrix] = {}
        active_folds = sorted(set(fold_of.values()))
        for fold in active_folds:
            reference = [
                chunk
                for chunk in sequences
                if fold_of[chunk.document_id] != fold
            ]
            held_out = [
                chunk
                for chunk in sequences
                if fold_of[chunk.document_id] == fold
            ]
            statistics = self._compute_statistics(reference)
            for chunk in held_out:
                matrices[_chunk_key(chunk)] = self._extract(chunk, statistics)
        self._statistics = self._compute_statistics(sequences)
        covered = 0
        for unit in self._text_units(sequences):
            hit = [False] * len(unit.tokens)
            for start, end, _ in self._matches(unit.tokens):
                for position in range(start, end):
                    hit[position] = True
            covered += sum(hit)
        character_total = sum(len(unit.tokens) for unit in self._text_units(sequences))
        self._training_character_coverage = covered / max(character_total, 1)
        LOGGER.info(
            "E1局部领域知识：候选词=%d，训练字符覆盖率=%.2f%%，内部OOF=%d折（文献级）",
            len(self.candidates),
            100 * self._training_character_coverage,
            len(active_folds),
        )
        return [matrices[_chunk_key(chunk)] for chunk in sequences]

    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if self._statistics is None:
            raise RuntimeError("E1领域提供器尚未在外层训练折上拟合")
        return [self._extract(chunk, self._statistics) for chunk in sequences]

    def metadata(self) -> dict[str, object]:
        return {
            "名称": "局部词汇领域分布",
            "维度": self.dimension,
            "候选模式": self.config.candidate_mode,
            "候选词数": len(self.candidates),
            "训练字符覆盖率": self._training_character_coverage,
            "内部OOF折数": self.config.inner_folds,
            "平滑": self.config.smoothing,
            "低频收缩强度": self.config.shrinkage,
            "无候选词回退": "训练折字符级领域分布；未见字符为[0.5, 0.5]",
            "候选摘要": self._candidate_summary,
        }

    def state_dict(self) -> dict[str, object]:
        if self._statistics is None:
            raise RuntimeError("E1领域提供器尚未拟合，不能保存统计状态")
        characters = (
            set(self._statistics.character_counts[JINGSHU])
            | set(self._statistics.character_counts[SHISU])
        )
        return {
            "type": "local_domain_distribution",
            "feature_names": list(self.feature_names),
            "candidate_terms": sorted(self.candidates),
            "candidate_distributions": {
                term: self._candidate_distribution(term, self._statistics)
                for term in sorted(self.candidates)
            },
            "character_distributions": {
                character: self._character_distribution(
                    character, self._statistics
                )
                for character in sorted(characters)
            },
            "unknown_character_distribution": NEUTRAL,
            "config": {
                "inner_folds": self.config.inner_folds,
                "smoothing": self.config.smoothing,
                "shrinkage": self.config.shrinkage,
                "candidate_mode": self.config.candidate_mode,
            },
        }
