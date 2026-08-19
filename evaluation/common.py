from __future__ import annotations


def prf(true_positive: int, predicted_positive: int, gold_positive: int) -> dict[str, float | int]:
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / gold_positive if gold_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "predicted": predicted_positive,
        "gold": gold_positive,
    }


def flatten(sequences: list[list[str]]) -> list[str]:
    return [label for sequence in sequences for label in sequence]

