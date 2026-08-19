from __future__ import annotations

import logging
from collections import Counter, defaultdict

from config import LexiconKnowledgeConfig
from data.corpus import SequenceChunk
from pretraining.word_candidates import read_dictionary_terms, read_tongyin_terms

from .base import FeatureMatrix, KnowledgeFeatureProvider


LOGGER = logging.getLogger(__name__)

RELATIONS = ("end", "start", "cross")
SOURCES = ("dictionary", "tongyin")


class LexiconGapLatticeProvider(KnowledgeFeatureProvider):
    """E2：把所有重叠辞书词映射为字符后间隔的软格网。

    标签位置 ``i`` 表示第 ``i`` 个字符后的间隔。对于辞书匹配 ``[s, e)``：
    ``e-1`` 是词尾间隔，``s-1`` 是词首前间隔，而 ``[s, e-1)`` 中的
    间隔均被该词跨越。这里只提供候选结构，不执行最大匹配或硬分词。
    """

    def __init__(self, config: LexiconKnowledgeConfig) -> None:
        self.config = config
        self.lengths = tuple(sorted(config.candidate_lengths))
        self._terms_by_length: dict[int, dict[str, frozenset[str]]] = {}
        self._source_counts: Counter[str] = Counter()
        self._training_statistics: dict[str, float | int] = {}
        self._loaded = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        length_features = tuple(
            f"lexicon_{relation}_{length}"
            for relation in RELATIONS
            for length in self.lengths
        )
        if not self.config.use_source_features:
            return length_features
        source_features = tuple(
            f"lexicon_{source}_{relation}"
            for source in SOURCES
            for relation in RELATIONS
        )
        return length_features + source_features

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def _load(self) -> None:
        if self._loaded:
            return
        if self.config.dictionary_path is None or self.config.tongyin_path is None:
            raise ValueError("E2需要《西夏文词典》和《同音》的路径")
        for path in (self.config.dictionary_path, self.config.tongyin_path):
            if not path.exists():
                raise FileNotFoundError(f"找不到E2辞书资源：{path}")
        dictionary = read_dictionary_terms(
            self.config.dictionary_path, self.lengths
        )
        tongyin = read_tongyin_terms(self.config.tongyin_path, self.lengths)
        grouped: dict[int, dict[str, set[str]]] = defaultdict(dict)
        for source, terms in (("dictionary", dictionary), ("tongyin", tongyin)):
            for term in terms:
                grouped[len(term)].setdefault(term, set()).add(source)
        self._terms_by_length = {
            length: {
                term: frozenset(sources) for term, sources in terms.items()
            }
            for length, terms in grouped.items()
        }
        overlap = dictionary & tongyin
        self._source_counts = Counter(
            dictionary=len(dictionary),
            tongyin=len(tongyin),
            overlap=len(overlap),
            union=len(dictionary | tongyin),
        )
        if not self._source_counts["union"]:
            raise ValueError("E2没有从辞书中读取到符合长度要求的西夏文词条")
        self._loaded = True

    def _filter_to_observed(self, sequences: list[SequenceChunk]) -> None:
        if self.config.include_unseen_terms:
            return
        texts = {
            (chunk.document_id, chunk.block_start, chunk.block_end): "".join(
                chunk.document_tokens[chunk.block_start : chunk.block_end]
            )
            for chunk in sequences
        }
        observed = {
            term
            for terms in self._terms_by_length.values()
            for term in terms
            if any(term in text for text in texts.values())
        }
        self._terms_by_length = {
            length: {term: sources for term, sources in terms.items() if term in observed}
            for length, terms in self._terms_by_length.items()
        }

    def _matches(
        self, tokens: tuple[str, ...]
    ) -> list[tuple[int, int, frozenset[str]]]:
        text = "".join(tokens)
        matches: list[tuple[int, int, frozenset[str]]] = []
        for length, terms in self._terms_by_length.items():
            for start in range(max(0, len(text) - length + 1)):
                sources = terms.get(text[start : start + length])
                if sources is not None:
                    matches.append((start, start + length, sources))
        return matches

    def _set_relation(
        self,
        rows: list[list[float]],
        context_gap: int,
        local_start: int,
        relation: str,
        length: int,
        sources: frozenset[str],
    ) -> None:
        position = context_gap - local_start
        if not 0 <= position < len(rows):
            return
        relation_index = RELATIONS.index(relation)
        length_index = self.lengths.index(length)
        rows[position][relation_index * len(self.lengths) + length_index] = 1.0
        if self.config.use_source_features:
            source_offset = len(RELATIONS) * len(self.lengths)
            for source in sources:
                source_index = SOURCES.index(source)
                rows[position][source_offset + source_index * len(RELATIONS) + relation_index] = 1.0

    def _extract(self, chunk: SequenceChunk) -> FeatureMatrix:
        margin = max(self.lengths) - 1
        context_start = max(chunk.block_start, chunk.offset - margin)
        context_end = min(
            chunk.block_end, chunk.offset + len(chunk.tokens) + margin
        )
        context = chunk.document_tokens[context_start:context_end]
        local_start = chunk.offset - context_start
        rows = [[0.0] * self.dimension for _ in chunk.tokens]
        for start, end, sources in self._matches(context):
            length = end - start
            self._set_relation(
                rows, end - 1, local_start, "end", length, sources
            )
            if start > 0:
                self._set_relation(
                    rows, start - 1, local_start, "start", length, sources
                )
            for gap in range(start, end - 1):
                self._set_relation(
                    rows, gap, local_start, "cross", length, sources
                )
        return tuple(tuple(row) for row in rows)

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if not sequences:
            raise ValueError("E2训练序列不能为空")
        self._load()
        self._filter_to_observed(sequences)
        matrices = [self._extract(chunk) for chunk in sequences]
        total = sum(len(matrix) for matrix in matrices)
        active = sum(any(row) for matrix in matrices for row in matrix)
        overlapping = sum(
            sum(row[: len(RELATIONS) * len(self.lengths)]) > 1
            for matrix in matrices
            for row in matrix
        )
        self._training_statistics = {
            "间隔总数": total,
            "有词典信号间隔数": active,
            "间隔覆盖率": active / max(total, 1),
            "多重格网间隔数": overlapping,
            "多重格网间隔率": overlapping / max(total, 1),
        }
        LOGGER.info(
            "E2软词典格网：词条=%d（西夏文词典=%d、同音=%d、交集=%d），"
            "维度=%d，训练间隔覆盖率=%.2f%%，多重格网间隔率=%.2f%%",
            self._source_counts["union"],
            self._source_counts["dictionary"],
            self._source_counts["tongyin"],
            self._source_counts["overlap"],
            self.dimension,
            100 * float(self._training_statistics["间隔覆盖率"]),
            100 * float(self._training_statistics["多重格网间隔率"]),
        )
        return matrices

    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        self._load()
        return [self._extract(chunk) for chunk in sequences]

    def metadata(self) -> dict[str, object]:
        self._load()
        return {
            "名称": "间隔中心软词典格网",
            "维度": self.dimension,
            "表示": self.config.representation,
            "候选词长": list(self.lengths),
            "使用辞书来源特征": self.config.use_source_features,
            "保留训练集未见词": self.config.include_unseen_terms,
            "词条统计": dict(self._source_counts),
            "训练格网统计": dict(self._training_statistics),
        }

    def state_dict(self) -> dict[str, object]:
        self._load()
        return {
            "type": "lexicon_gap_lattice",
            "feature_names": list(self.feature_names),
            "terms": {
                term: sorted(sources)
                for length in sorted(self._terms_by_length)
                for term, sources in sorted(self._terms_by_length[length].items())
            },
            "config": {
                "candidate_lengths": list(self.lengths),
                "representation": self.config.representation,
                "use_source_features": self.config.use_source_features,
                "include_unseen_terms": self.config.include_unseen_terms,
            },
        }
