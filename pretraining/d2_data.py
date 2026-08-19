from __future__ import annotations

import math
import random

import torch

from models.tangut_encoder import MASK_TOKEN

from .data import IGNORE_INDEX, MLMDataset
from .word_candidates import CandidateOccurrence, WordCandidate


class WordAwareMLMDataset(MLMDataset):
    """D2动态数据集：普通MLM与候选词整词遮盖共享同一字符目标比例。"""

    def __init__(
        self,
        *args: object,
        occurrences: list[tuple[CandidateOccurrence, ...]],
        candidates: dict[str, WordCandidate],
        whole_word_probability: float,
        ranking_negatives: int,
        enable_ranking: bool,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if len(occurrences) != len(self.sequences):
            raise ValueError("D2候选出现位置与预训练序列数量不一致")
        self.occurrences = occurrences
        self.candidates = candidates
        self.whole_word_probability = whole_word_probability
        self.ranking_negatives = ranking_negatives
        self.enable_ranking = enable_ranking

    @staticmethod
    def _overlaps(start: int, end: int, positions: set[int]) -> bool:
        return any(position in positions for position in range(start, end))

    def _word_masked_positions(
        self, index: int, length: int, target_count: int, rng: random.Random
    ) -> set[int]:
        selected: set[int] = set()
        available = list(self.occurrences[index])
        while available and len(selected) < target_count:
            weights = [
                item.candidate.confidence
                * math.sqrt(min(item.candidate.frequency, 1000))
                for item in available
            ]
            occurrence = rng.choices(available, weights=weights, k=1)[0]
            positions = set(range(occurrence.start, occurrence.end))
            available = [
                item
                for item in available
                if item.end <= occurrence.start or item.start >= occurrence.end
            ]
            if selected and len(selected | positions) > target_count:
                continue
            selected.update(positions)
        if len(selected) < target_count:
            remaining = [position for position in range(length) if position not in selected]
            rng.shuffle(remaining)
            selected.update(remaining[: target_count - len(selected)])
        return selected

    def _ranking_example(
        self,
        index: int,
        original: list[int],
        masked: set[int],
        rng: random.Random,
    ) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], float, bool, int, str]:
        if not self.enable_ranking:
            return (0, 1), tuple((0, 1) for _ in range(self.ranking_negatives)), 0.0, False, 0, "none"
        positives = [
            item
            for item in self.occurrences[index]
            if not self._overlaps(item.start, item.end, masked)
        ]
        if not positives:
            return (0, 1), tuple((0, 1) for _ in range(self.ranking_negatives)), 0.0, False, 0, "none"
        weights = [item.candidate.confidence for item in positives]
        positive = rng.choices(positives, weights=weights, k=1)[0]
        span_length = positive.end - positive.start
        token_text = "".join(self.sequences[index].tokens)
        possible = [
            (start, start + span_length)
            for start in range(max(0, len(original) - span_length + 1))
            if start != positive.start
            and not self._overlaps(start, start + span_length, masked)
            and token_text[start : start + span_length] not in self.candidates
        ]
        if not possible:
            possible = [
                (start, start + span_length)
                for start in range(max(0, len(original) - span_length + 1))
                if start != positive.start
                and not self._overlaps(start, start + span_length, masked)
            ]
        if not possible:
            return (0, 1), tuple((0, 1) for _ in range(self.ranking_negatives)), 0.0, False, 0, "none"
        negatives = tuple(rng.choice(possible) for _ in range(self.ranking_negatives))
        lexical = set(positive.candidate.sources) & {"dictionary", "tongyin"}
        source_group = (
            "lexicon_both"
            if len(lexical) == 2
            else "lexicon_single"
            if lexical
            else "statistics_only"
        )
        return (
            (positive.start, positive.end),
            negatives,
            positive.candidate.confidence,
            True,
            positive.end - positive.start,
            source_group,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        original = list(self.encoded[index])
        rng = self._rng(index)
        target_count = max(1, round(len(original) * self.mask_ratio))
        use_word_masking = (
            bool(self.occurrences[index])
            and rng.random() < self.whole_word_probability
        )
        positions = (
            self._word_masked_positions(index, len(original), target_count, rng)
            if use_word_masking
            else self._masked_positions(len(original), target_count, rng)
        )
        input_ids = original[:]
        labels = [IGNORE_INDEX] * len(original)
        for position in positions:
            labels[position] = original[position]
            draw = rng.random()
            if draw < self.mask_replace_probability:
                input_ids[position] = self.vocabulary[MASK_TOKEN]
            elif draw < self.mask_replace_probability + self.random_replace_probability:
                input_ids[position] = rng.choice(self.replacement_ids)

        positive, negatives, confidence, rank_valid, candidate_length, candidate_source = self._ranking_example(
            index, original, positions, rng
        )
        sequence = self.sequences[index]
        return {
            "input_ids": tuple(input_ids),
            "labels": tuple(labels),
            "document_id": sequence.document_id,
            "domain": sequence.domain,
            "word_masked": use_word_masking,
            "positive_span": positive,
            "negative_spans": negatives,
            "candidate_confidence": confidence,
            "rank_valid": rank_valid,
            "candidate_length": candidate_length,
            "candidate_source": candidate_source,
        }


def collate_word_mlm(
    batch: list[dict[str, object]], pad_id: int
) -> dict[str, object]:
    lengths = [len(item["input_ids"]) for item in batch]  # type: ignore[arg-type]
    max_length = max(lengths)
    input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, item in enumerate(batch):
        length = lengths[row]
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "padding_mask": input_ids.eq(pad_id),
        "document_ids": [str(item["document_id"]) for item in batch],
        "domains": [str(item["domain"]) for item in batch],
        "word_masked": torch.tensor(
            [bool(item["word_masked"]) for item in batch], dtype=torch.bool
        ),
        "positive_spans": torch.tensor(
            [item["positive_span"] for item in batch], dtype=torch.long
        ),
        "negative_spans": torch.tensor(
            [item["negative_spans"] for item in batch], dtype=torch.long
        ),
        "candidate_confidence": torch.tensor(
            [float(item["candidate_confidence"]) for item in batch], dtype=torch.float
        ),
        "rank_valid": torch.tensor(
            [bool(item["rank_valid"]) for item in batch], dtype=torch.bool
        ),
        "candidate_lengths": torch.tensor(
            [int(item["candidate_length"]) for item in batch], dtype=torch.long
        ),
        "candidate_sources": [str(item["candidate_source"]) for item in batch],
    }
