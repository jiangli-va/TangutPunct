from __future__ import annotations

from collections.abc import Iterable

from tasks import OUTSIDE
from .common import flatten, prf


def evaluate_punctuation(
    gold_by_document: dict[str, list[str]],
    predicted_by_document: dict[str, list[str]],
    sentence_marks: frozenset[str] = frozenset("。？！"),
    class_labels: Iterable[str] | None = None,
) -> dict[str, object]:
    gold = flatten(list(gold_by_document.values()))
    predicted = flatten([predicted_by_document[key] for key in gold_by_document])
    if len(gold) != len(predicted):
        raise ValueError("预测长度与金标准不一致")

    correct = sum(g == p for g, p in zip(gold, predicted))
    total = len(gold)

    gold_positions = sum(label != OUTSIDE for label in gold)
    pred_positions = sum(label != OUTSIDE for label in predicted)
    position_tp = sum(g != OUTSIDE and p != OUTSIDE for g, p in zip(gold, predicted))

    observed_classes = (set(gold) | set(predicted)) - {OUTSIDE}
    if class_labels is None:
        classes = sorted(observed_classes)
    else:
        configured_classes = set(class_labels)
        unexpected = observed_classes - configured_classes
        if unexpected:
            raise ValueError(f"评价中出现未配置类别：{sorted(unexpected)}")
        classes = sorted(configured_classes)
    per_class: dict[str, dict[str, float | int]] = {}
    micro_tp = micro_pred = micro_gold = 0
    for label in classes:
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        pred_n = predicted.count(label)
        gold_n = gold.count(label)
        tn = sum(g != label and p != label for g, p in zip(gold, predicted))
        metrics = prf(tp, pred_n, gold_n)
        metrics.update(
            {
                # 该类别对其余全部标签的 one-vs-rest Accuracy；包含 O，
                # 只作辅助指标，不参与该类别 P/R/F1。
                "accuracy": (tp + tn) / total if total else 0.0,
                "tn": tn,
                "total": total,
            }
        )
        per_class[label] = metrics
        micro_tp += tp
        micro_pred += pred_n
        micro_gold += gold_n

    macro_f1 = (
        sum(float(metrics["f1"]) for metrics in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )
    labels = [OUTSIDE] + classes
    confusion = {
        gold_label: {
            pred_label: sum(g == gold_label and p == pred_label for g, p in zip(gold, predicted))
            for pred_label in labels
        }
        for gold_label in labels
    }

    gold_boundary = [any(mark in label for mark in sentence_marks) for label in gold]
    pred_boundary = [any(mark in label for mark in sentence_marks) for label in predicted]
    boundary_tp = sum(g and p for g, p in zip(gold_boundary, pred_boundary))
    return {
        # Accuracy 按所有字符间隔的完整标签计算，因此包含大量 O。
        # 它只作为辅助指标，绝不并入下面任何 P/R/F1 的统计。
        "accuracy": {
            "value": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        },
        "punctuation_position": prf(position_tp, pred_positions, gold_positions),
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro": prf(micro_tp, micro_pred, micro_gold),
        "confusion_matrix": {"labels": labels, "matrix": confusion},
        "sentence_boundary_from_period_question_exclamation": prf(
            boundary_tp, sum(pred_boundary), sum(gold_boundary)
        ),
    }
