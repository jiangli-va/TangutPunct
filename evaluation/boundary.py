from __future__ import annotations

import statistics

from tasks import BOUNDARY
from .common import prf


DEFAULT_LENGTH_BINS = ((1, 10), (11, 20), (21, 40), (41, None))


def _bin_name(length: int, bins: tuple[tuple[int, int | None], ...]) -> str:
    for lower, upper in bins:
        if length >= lower and (upper is None or length <= upper):
            return f"{lower}+" if upper is None else f"{lower}-{upper}"
    raise ValueError(f"句长 {length} 不在任何区间")


def _sentence_length_at(labels: list[str]) -> list[int]:
    result = [0] * len(labels)
    start = 0
    for index, label in enumerate(labels):
        if label == BOUNDARY:
            length = index - start + 1
            for position in range(start, index + 1):
                result[position] = length
            start = index + 1
    if start < len(labels):
        length = len(labels) - start
        for position in range(start, len(labels)):
            result[position] = length
    return result


def evaluate_boundary(
    gold_by_document: dict[str, list[str]],
    predicted_by_document: dict[str, list[str]],
    length_bins: tuple[tuple[int, int | None], ...] = DEFAULT_LENGTH_BINS,
) -> dict[str, object]:
    totals = [0, 0, 0]  # tp, predicted, gold
    per_document: dict[str, dict[str, float | int]] = {}
    bucket_counts = {(_bin_name(lower, length_bins)): [0, 0, 0] for lower, _ in length_bins}

    for document_id, gold in gold_by_document.items():
        predicted = predicted_by_document[document_id]
        if len(gold) != len(predicted):
            raise ValueError(f"{document_id} 的预测长度与金标准不一致")
        tp = sum(g == p == BOUNDARY for g, p in zip(gold, predicted))
        pred_n = predicted.count(BOUNDARY)
        gold_n = gold.count(BOUNDARY)
        per_document[document_id] = prf(tp, pred_n, gold_n)
        totals[0] += tp
        totals[1] += pred_n
        totals[2] += gold_n

        lengths = _sentence_length_at(gold)
        for index, (g, p) in enumerate(zip(gold, predicted)):
            bucket = _bin_name(lengths[index], length_bins)
            bucket_counts[bucket][0] += int(g == p == BOUNDARY)
            bucket_counts[bucket][1] += int(p == BOUNDARY)
            bucket_counts[bucket][2] += int(g == BOUNDARY)

    document_f1 = [float(metrics["f1"]) for metrics in per_document.values()]
    return {
        "boundary": prf(*totals),
        "by_sentence_length": {
            bucket: prf(*counts) for bucket, counts in bucket_counts.items()
        },
        "by_document": per_document,
        "document_f1_mean": statistics.mean(document_f1) if document_f1 else 0.0,
        "document_f1_std": statistics.pstdev(document_f1) if len(document_f1) > 1 else 0.0,
    }
