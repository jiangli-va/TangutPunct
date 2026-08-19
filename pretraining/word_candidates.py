from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from config import WordPretrainingConfig
from data.corpus import is_tangut

from .data import MLMSequence


@dataclass(frozen=True)
class WordCandidate:
    term: str
    frequency: int
    sources: tuple[str, ...]
    confidence: float
    min_dpmi: float
    left_entropy: float
    right_entropy: float

    @property
    def length(self) -> int:
        return len(self.term)

    def serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateOccurrence:
    start: int
    end: int
    candidate: WordCandidate


def _valid_term(value: object, lengths: set[int]) -> str | None:
    if not isinstance(value, str):
        return None
    term = "".join(character for character in value if is_tangut(character))
    if len(term) not in lengths or len(term) != len(value):
        return None
    return term


def read_dictionary_terms(path: Path, lengths: Iterable[int]) -> set[str]:
    """读取《西夏文词典》term_list中的多字词。"""

    allowed = set(lengths)
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        for item in entry.get("term_list", []):
            if not isinstance(item, dict):
                continue
            term = _valid_term(item.get("term_character"), allowed)
            if term is not None:
                terms.add(term)
    return terms


def read_tongyin_terms(path: Path, lengths: Iterable[int]) -> set[str]:
    """读取《同音》的显式词字段，并由“本字＋词余＋位置”复原词形。"""

    allowed = set(lengths)
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        head = entry.get("xixia_character")
        for value in entry.values():
            if not isinstance(value, dict):
                continue
            explicit = _valid_term(value.get("词"), allowed)
            if explicit is not None:
                terms.add(explicit)
            remainder = value.get("词余")
            position = str(value.get("位置", ""))
            if not isinstance(head, str) or not isinstance(remainder, str):
                continue
            reconstructed = (
                head + remainder if position == "1" else remainder + head
                if position == "2"
                else ""
            )
            term = _valid_term(reconstructed, allowed)
            if term is not None:
                terms.add(term)
    return terms


def _entropy(values: Counter[str]) -> float:
    total = sum(values.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total) for count in values.values()
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _ngram_statistics(
    sequences: list[MLMSequence], lengths: tuple[int, ...]
) -> tuple[
    dict[int, Counter[str]],
    dict[int, int],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
]:
    required_lengths = set(lengths)
    for length in lengths:
        required_lengths.update(range(1, length))
    counts = {length: Counter() for length in sorted(required_lengths)}
    totals = {length: 0 for length in sorted(required_lengths)}
    for sequence in sequences:
        text = "".join(sequence.tokens)
        for length in counts:
            number = max(0, len(text) - length + 1)
            totals[length] += number
            counts[length].update(
                text[start : start + length] for start in range(number)
            )

    minimums = {length: 1 for length in lengths}
    eligible = {
        term
        for length in lengths
        for term, frequency in counts[length].items()
        if frequency >= minimums[length]
    }
    left: dict[str, Counter[str]] = defaultdict(Counter)
    right: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        text = "".join(sequence.tokens)
        for length in lengths:
            for start in range(max(0, len(text) - length + 1)):
                term = text[start : start + length]
                if term not in eligible:
                    continue
                if start:
                    left[term][text[start - 1]] += 1
                if start + length < len(text):
                    right[term][text[start + length]] += 1
    return counts, totals, left, right


def _minimum_dpmi(
    term: str, counts: dict[int, Counter[str]], totals: dict[int, int]
) -> float:
    term_count = counts[len(term)][term]
    term_total = max(totals[len(term)], 1)
    term_probability = term_count / term_total
    values: list[float] = []
    for split in range(1, len(term)):
        left = term[:split]
        right = term[split:]
        left_probability = counts[len(left)][left] / max(totals[len(left)], 1)
        right_probability = counts[len(right)][right] / max(totals[len(right)], 1)
        denominator = max(left_probability * right_probability, 1e-15)
        values.append(math.log(term_probability / denominator))
    return min(values)


def build_word_candidates(
    sequences: list[MLMSequence],
    config: WordPretrainingConfig,
    mode: str = "fusion",
) -> tuple[dict[str, WordCandidate], dict[str, object]]:
    """只用训练文献统计候选；mode为lexicon时不加入纯统计候选。"""

    if mode not in {"lexicon", "fusion"}:
        raise ValueError(f"不支持的D2候选模式：{mode}")
    if config.dictionary_path is None or config.tongyin_path is None:
        raise ValueError("D2需要配置dictionary_path和tongyin_path")
    for path in (config.dictionary_path, config.tongyin_path):
        if not path.exists():
            raise FileNotFoundError(f"找不到D2辞书资源：{path}")

    lengths = tuple(sorted(config.candidate_lengths))
    dictionary = read_dictionary_terms(config.dictionary_path, lengths)
    tongyin = read_tongyin_terms(config.tongyin_path, lengths)
    counts, totals, left, right = _ngram_statistics(sequences, lengths)
    frequency_thresholds = dict(
        zip(lengths, config.statistical_min_frequencies)
    )
    observed_lexicon = {
        term for term in dictionary | tongyin if counts[len(term)][term] > 0
    }
    statistical_pool = {
        term
        for length in lengths
        for term, frequency in counts[length].items()
        if frequency >= frequency_thresholds[length]
    }
    pool = observed_lexicon | (statistical_pool if mode == "fusion" else set())

    candidates: dict[str, WordCandidate] = {}
    rejected = Counter()
    for term in pool:
        frequency = counts[len(term)][term]
        sources: list[str] = []
        if term in dictionary:
            sources.append("dictionary")
        if term in tongyin:
            sources.append("tongyin")
        dpmi = _minimum_dpmi(term, counts, totals)
        left_entropy = _entropy(left[term])
        right_entropy = _entropy(right[term])
        statistical = (
            frequency >= frequency_thresholds[len(term)]
            and dpmi >= config.statistical_min_dpmi
            and min(left_entropy, right_entropy) >= config.statistical_min_entropy
        )
        if statistical:
            sources.append("statistics")

        lexicon_sources = set(sources) & {"dictionary", "tongyin"}
        minimum_frequency = (
            config.intersection_min_frequency
            if len(lexicon_sources) == 2
            else config.dictionary_min_frequency
        )
        if lexicon_sources and frequency < minimum_frequency:
            rejected["lexicon_low_frequency"] += 1
            continue
        if not lexicon_sources and not statistical:
            rejected["statistics_threshold"] += 1
            continue

        statistical_score = 0.7 * _sigmoid(
            dpmi - config.statistical_min_dpmi
        ) + 0.3 * min(1.0, min(left_entropy, right_entropy) / 2.0)
        if len(lexicon_sources) == 2:
            confidence = 0.9 + 0.1 * statistical_score
        elif lexicon_sources:
            confidence = 0.7 + 0.2 * statistical_score
        else:
            confidence = 0.5 + 0.3 * statistical_score
        candidates[term] = WordCandidate(
            term=term,
            frequency=frequency,
            sources=tuple(sources),
            confidence=min(confidence, 1.0),
            min_dpmi=dpmi,
            left_entropy=left_entropy,
            right_entropy=right_entropy,
        )

    source_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    for candidate in candidates.values():
        length_counts[candidate.length] += 1
        source_counts["+".join(candidate.sources)] += 1
    summary: dict[str, object] = {
        "dictionary_types": len(dictionary),
        "tongyin_types": len(tongyin),
        "dictionary_tongyin_overlap": len(dictionary & tongyin),
        "observed_lexicon_types": len(observed_lexicon),
        "candidate_types": len(candidates),
        "candidate_occurrences": sum(item.frequency for item in candidates.values()),
        "by_length": {str(key): value for key, value in sorted(length_counts.items())},
        "by_source": dict(sorted(source_counts.items())),
        "rejected": dict(rejected),
    }
    return candidates, summary


def find_candidate_occurrences(
    sequences: list[MLMSequence], candidates: dict[str, WordCandidate]
) -> list[tuple[CandidateOccurrence, ...]]:
    by_length: dict[int, set[str]] = defaultdict(set)
    for term in candidates:
        by_length[len(term)].add(term)
    output: list[tuple[CandidateOccurrence, ...]] = []
    for sequence in sequences:
        text = "".join(sequence.tokens)
        occurrences: list[CandidateOccurrence] = []
        for length, terms in by_length.items():
            for start in range(max(0, len(text) - length + 1)):
                term = text[start : start + length]
                if term in terms:
                    occurrences.append(
                        CandidateOccurrence(start, start + length, candidates[term])
                    )
        output.append(tuple(sorted(occurrences, key=lambda item: (item.start, item.end))))
    return output
