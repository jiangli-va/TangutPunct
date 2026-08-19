from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass

from config import DataAugmentationConfig
from data.corpus import PreparedDocument


LOGGER = logging.getLogger(__name__)
LabelMap = dict[str, list[str]]


@dataclass(frozen=True)
class _SentenceUnit:
    index: int
    tokens: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class AugmentationResult:
    documents: tuple[PreparedDocument, ...]
    metadata: dict[str, object]


class SentenceConcatenationAugmenter:
    """在外层训练折内拼接同一文献的完整句子。

    只在金标准 ``。？！`` 后切句。句内字符、标签及各句的原文顺序均不
    改变；TAB块末尾没有句间标点的残片不会进入增强池。
    """

    def __init__(
        self,
        config: DataAugmentationConfig,
        sentence_end_marks: tuple[str, ...],
        seed: int,
    ) -> None:
        self.config = config
        self.sentence_end_marks = frozenset(sentence_end_marks)
        self.seed = seed

    def _sentences(
        self,
        document: PreparedDocument,
        labels: list[str],
    ) -> tuple[list[_SentenceUnit], int]:
        if len(labels) != len(document.tokens):
            raise ValueError(f"{document.document_id}的增强标签长度与正文不一致")
        units: list[_SentenceUnit] = []
        incomplete_tails = 0
        boundaries = (0, *document.cut_offsets, len(document.tokens))
        sentence_index = 0
        for block_start, block_end in zip(boundaries, boundaries[1:]):
            start = block_start
            for end in range(block_start, block_end):
                if labels[end] not in self.sentence_end_marks:
                    continue
                units.append(
                    _SentenceUnit(
                        sentence_index,
                        document.tokens[start : end + 1],
                        tuple(labels[start : end + 1]),
                    )
                )
                sentence_index += 1
                start = end + 1
            if start < block_end:
                # 该残片可能由TAB处删除长片段产生，不能伪造句末标签。
                incomplete_tails += 1
        return units, incomplete_tails

    def augment(
        self,
        documents: list[PreparedDocument],
        labels: LabelMap,
        target_count: int,
        fold: int,
    ) -> AugmentationResult:
        if target_count <= 0:
            return AugmentationResult(
                (),
                {
                    "目标合成块": max(target_count, 0),
                    "实际合成块": 0,
                    "完整句子": 0,
                    "跳过的不完整TAB块尾": 0,
                    "低于最小长度的回退块": 0,
                },
            )

        by_document: dict[str, list[_SentenceUnit]] = {}
        document_lookup = {document.document_id: document for document in documents}
        incomplete_tails = 0
        complete_sentences = 0
        for document in documents:
            units, tails = self._sentences(
                document, labels[document.document_id]
            )
            incomplete_tails += tails
            complete_sentences += len(units)
            usable = [
                unit
                for unit in units
                if len(unit.tokens) <= self.config.max_characters
            ]
            if len(usable) >= self.config.min_sentences:
                by_document[document.document_id] = usable
        if not by_document:
            raise ValueError("当前训练折没有可供完整句子拼接的文献")

        rng = random.Random(self.seed + fold * 100_003)
        document_ids = sorted(by_document)
        weights = [len(by_document[document_id]) for document_id in document_ids]
        selected: list[tuple[str, tuple[int, ...], tuple[str, ...], tuple[str, ...]]] = []
        signatures: set[tuple[str, tuple[int, ...]]] = set()
        fallback_count = 0

        def sample(require_minimum_length: bool) -> bool:
            document_id = rng.choices(document_ids, weights=weights, k=1)[0]
            units = by_document[document_id]
            maximum = min(self.config.max_sentences, len(units))
            if maximum < self.config.min_sentences:
                return False
            count = rng.randint(self.config.min_sentences, maximum)
            chosen = sorted(
                rng.sample(units, count), key=lambda unit: unit.index
            )
            signature = (document_id, tuple(unit.index for unit in chosen))
            if signature in signatures:
                return False
            length = sum(len(unit.tokens) for unit in chosen)
            if length > self.config.max_characters:
                return False
            if require_minimum_length and length < self.config.min_characters:
                return False
            tokens = tuple(token for unit in chosen for token in unit.tokens)
            combined_labels = tuple(
                label for unit in chosen for label in unit.labels
            )
            signatures.add(signature)
            selected.append((document_id, signature[1], tokens, combined_labels))
            return True

        # 第一轮遵守期望长度；组合不足时允许较短的2—4句块，以避免小文献
        # 被完全排除。两轮都严格遵守最大长度、完整句界和同文献约束。
        attempt_limit = max(target_count * 200, 2_000)
        attempts = 0
        while len(selected) < target_count and attempts < attempt_limit:
            sample(require_minimum_length=True)
            attempts += 1
        if len(selected) < target_count:
            before_fallback = len(selected)
            attempts = 0
            while len(selected) < target_count and attempts < attempt_limit:
                sample(require_minimum_length=False)
                attempts += 1
            fallback_count = len(selected) - before_fallback

        augmented: list[PreparedDocument] = []
        sentence_counts: Counter[int] = Counter()
        domain_counts: Counter[str] = Counter()
        for index, (source_id, sentence_ids, tokens, combined_labels) in enumerate(
            selected, 1
        ):
            source = document_lookup[source_id]
            sentence_counts[len(sentence_ids)] += 1
            domain_counts[source.domain] += 1
            augmented.append(
                PreparedDocument(
                    document_id=(
                        f"e9_aug_fold_{fold:02d}_{index:06d}_{source.document_id}"
                    ),
                    volume_number=source.volume_number,
                    tokens=tokens,
                    labels=combined_labels,
                    domain=source.domain,
                    source_path=f"augmented:{source.source_path}",
                    source_line=source.source_line,
                )
            )
        if len(augmented) < target_count:
            LOGGER.warning(
                "[E9-Aug][外折%d] 可用的不重复整句组合不足：目标%d，实际%d",
                fold,
                target_count,
                len(augmented),
            )
        metadata: dict[str, object] = {
            "方法": self.config.method,
            "范围": "同一文献",
            "保持原文句序": self.config.preserve_order,
            "目标合成块": target_count,
            "实际合成块": len(augmented),
            "完整句子": complete_sentences,
            "可增强文献": len(by_document),
            "跳过的不完整TAB块尾": incomplete_tails,
            "低于最小长度的回退块": fallback_count,
            "每块句数": dict(sorted(sentence_counts.items())),
            "领域合成块": dict(sorted(domain_counts.items())),
            "最小字符数": self.config.min_characters,
            "最大字符数": self.config.max_characters,
            "随机种子": self.seed + fold * 100_003,
        }
        return AugmentationResult(tuple(augmented), metadata)
