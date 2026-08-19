from __future__ import annotations

import statistics
from collections import Counter
from typing import Iterable


METRIC_NAMES = {"precision": "精确率", "recall": "召回率", "f1": "F1"}


def mean_std(values: Iterable[float]) -> str:
    numbers = list(values)
    mean = statistics.mean(numbers) if numbers else 0.0
    # 报告五折结果的总体标准差（ddof=0）。
    std = statistics.pstdev(numbers) if len(numbers) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_inspection(
    task: str,
    document_count: int,
    character_count: int,
    labels: Counter[str],
    folds: list[tuple[int, int, int, int]],
) -> str:
    task_name = "自动断句" if task == "boundary" else "断句与标点"
    corpus = markdown_table(
        ["项目", "数值"],
        [["任务", task_name], ["文献卷数", document_count], ["正文字符数", character_count]],
    )
    label_rows = [[label, count] for label, count in labels.most_common()]
    # 联合标签可能很多；默认只展示最高频的12类，其余合并。
    if len(label_rows) > 12:
        shown = label_rows[:12]
        shown.append(["其他联合标签", sum(row[1] for row in label_rows[12:])])
        label_rows = shown
    fold_rows = [[fold, train, dev, test] for fold, train, dev, test in folds]
    return (
        f"## 数据检查：{task_name}\n\n{corpus}\n\n"
        f"### 标签分布\n\n{markdown_table(['标签', '数量'], label_rows)}\n\n"
        f"### 五折文献级划分\n\n"
        f"{markdown_table(['折', '训练卷', '开发卷', '测试卷'], fold_rows)}"
    )


def _fold_metric(folds: list[dict[str, object]], section: str, metric: str) -> list[float]:
    return [float(fold["metrics"][section][metric]) for fold in folds]  # type: ignore[index]


def render_boundary_results(result: dict[str, object]) -> str:
    folds: list[dict[str, object]] = result["folds"]  # type: ignore[assignment]
    primary_rows = [
        [METRIC_NAMES[metric], mean_std(_fold_metric(folds, "boundary", metric))]
        for metric in ("precision", "recall", "f1")
    ]
    buckets = list(folds[0]["metrics"]["by_sentence_length"])  # type: ignore[index]
    length_rows = [
        [
            bucket,
            mean_std(
                float(fold["metrics"]["by_sentence_length"][bucket]["f1"])  # type: ignore[index]
                for fold in folds
            ),
        ]
        for bucket in buckets
    ]
    aggregate = result["aggregate"]  # type: ignore[assignment]
    document_value = (
        f"{float(aggregate['document_f1_mean']):.4f} ± "
        f"{float(aggregate['document_f1_std']):.4f}"
    )
    return (
        "## 自动断句五折测试结果\n\n"
        + markdown_table(["指标", "五折均值 ± 标准差"], primary_rows)
        + "\n\n### 不同句长\n\n"
        + markdown_table(["句长（字）", "F1（五折均值 ± 标准差）"], length_rows)
        + "\n\n### 各卷表现\n\n"
        + markdown_table(["指标", "均值 ± 标准差"], [["各卷 F1", document_value]])
    )


def render_punctuation_results(result: dict[str, object]) -> str:
    folds: list[dict[str, object]] = result["folds"]  # type: ignore[assignment]
    rows: list[list[str]] = []
    for title, section in (
        ("任意标点位置", "punctuation_position"),
        ("标点类别 Micro", "micro"),
        ("由。？！推导的句界", "sentence_boundary_from_period_question_exclamation"),
    ):
        for metric in ("precision", "recall", "f1"):
            rows.append([title, METRIC_NAMES[metric], mean_std(_fold_metric(folds, section, metric))])
    rows.append(
        [
            "标点类别 Macro",
            "F1",
            mean_std(float(fold["metrics"]["macro_f1"]) for fold in folds),  # type: ignore[index]
        ]
    )

    aggregate = result["aggregate"]  # type: ignore[assignment]
    main_classes = [
        label
        for label, metrics in aggregate["per_class"].items()
        if int(metrics["gold"]) >= 10
    ]
    main_classes.sort(key=lambda label: int(aggregate["per_class"][label]["gold"]), reverse=True)
    class_rows = []
    for label in main_classes:
        class_rows.append(
            [
                label,
                *[
                    mean_std(
                        float(fold["metrics"]["per_class"].get(label, {}).get(metric, 0.0))  # type: ignore[index]
                        for fold in folds
                    )
                    for metric in ("precision", "recall", "f1")
                ],
            ]
        )
    return (
        "## 断句与标点五折测试结果\n\n"
        + markdown_table(["评价对象", "指标", "五折均值 ± 标准差"], rows)
        + "\n\n### 主要标点类别（语料频次不少于10）\n\n"
        + markdown_table(
            ["标点标签", "精确率", "召回率", "F1"],
            class_rows,
        )
        + "\n\n> 完整逐类结果与混淆矩阵见 `results.json`。"
    )


def render_results(result: dict[str, object]) -> str:
    return (
        render_boundary_results(result)
        if result["task"] == "boundary"
        else render_punctuation_results(result)
    )
