from __future__ import annotations

from collections import Counter
from typing import Iterable

from config import ExperimentConfig
from data.corpus import PreparedDocument
from data.labels import select_punctuation
from data.splits import FoldSplit
from experiments.specs import (
    INTRA_GROUP_LABEL,
    SENTENCE_GROUP_LABEL,
    ExperimentSpec,
)
from reporting import markdown_table, mean_std
from tasks import OUTSIDE


DOMAIN_NAMES = {"overall": "总体", "jingshu": "经书", "shisu": "世俗"}
CONDITION_NAMES = {
    "direct": "直接预测",
    "oracle": "金标准上游（Oracle）",
    "predicted": "预测上游（实际流水线）",
}
GRADE_LABEL_NAMES = {
    INTRA_GROUP_LABEL: "句内停顿（，、：；）",
    SENTENCE_GROUP_LABEL: "句间停顿（。？！）",
}


def _count_domains(documents: Iterable[PreparedDocument]) -> Counter[str]:
    return Counter(document.domain for document in documents)


def render_main_inspection(
    config: ExperimentConfig,
    specs: list[ExperimentSpec],
    documents: list[PreparedDocument],
    folds: list[FoldSplit],
    grade: int = 1,
) -> str:
    if grade not in {1, 2}:
        raise ValueError("grade 只支持 1 或 2")
    source_rows = []
    for source in config.data.sources:
        selected = [doc for doc in documents if doc.source_path == str(source.path)]
        source_rows.append(
            [source.path.name, DOMAIN_NAMES.get(source.domain, source.domain), len(selected), sum(len(doc.tokens) for doc in selected)]
        )
    stage_rows = []
    for spec in specs:
        default_backend = spec.model_name or config.model.name
        stage_backends = [stage.model_name or default_backend for stage in spec.stages]
        backend_text = (
            stage_backends[0]
            if len(set(stage_backends)) == 1
            else " → ".join(stage_backends)
        )
        stage_rows.append(
            [
                spec.name.upper(),
                spec.display_name,
                backend_text,
                " → ".join(stage.display_name for stage in spec.stages),
                "直接" if len(spec.stages) == 1 else "金标准 / 预测",
            ]
        )
    by_id = {document.document_id: document for document in documents}
    split_rows = []
    for split in folds:
        cells: list[object] = [split.fold]
        for ids in (split.train_ids, split.dev_ids, split.test_ids):
            counts = _count_domains(by_id[item] for item in ids)
            cells.append(
                f"{len(ids)}（经{counts['jingshu']}/世{counts['shisu']}）"
            )
        total = len(documents)
        cells.append(
            f"{len(split.train_ids)/total:.1%} / {len(split.dev_ids)/total:.1%} / {len(split.test_ids)/total:.1%}"
        )
        split_rows.append(cells)

    pause = set(config.punctuation.sentence_pause) | set(
        config.punctuation.intra_sentence_pause
    )
    label_counts = Counter(
        select_punctuation(label, pause)
        for document in documents
        for label in document.labels
    )
    if grade == 2:
        sentence = set(config.punctuation.sentence_pause)
        intra = set(config.punctuation.intra_sentence_pause)
        grouped_counts: Counter[str] = Counter()
        for label, count in label_counts.items():
            if label == OUTSIDE:
                grouped_counts[OUTSIDE] += count
            elif label in sentence:
                grouped_counts[SENTENCE_GROUP_LABEL] += count
            elif label in intra:
                grouped_counts[INTRA_GROUP_LABEL] += count
        label_counts = grouped_counts
    label_rows = [
        [GRADE_LABEL_NAMES.get(label, label), count]
        for label, count in label_counts.most_common()
    ]
    adjacent_pause = Counter(
        label
        for doc in documents
        for label in doc.labels
        if label != OUTSIDE and sum(mark in pause for mark in label) > 1
    )
    warning = ""
    if adjacent_pause:
        examples = "、".join(f"{label}×{count}" for label, count in adjacent_pause.most_common())
        warning = (
            "\n\n> 警告：发现同一字后含多个停顿标点的联合标签 "
            f"{sum(adjacent_pause.values())} 处（{examples}）。七类停顿任务要求每个位置至多一个停顿标点，请先清洗语料。"
        )
    if specs and all(spec.name.startswith("b") for spec in specs):
        group_name = "主实验 B"
    elif specs and all(spec.name.startswith("c") for spec in specs):
        group_name = "主实验 C"
    elif specs and all(spec.name.startswith("d") for spec in specs):
        group_name = "主实验 D（自动标点下游评测）"
    elif specs and all(spec.name.startswith("e") for spec in specs):
        group_name = "主实验 E（外部知识消融）"
    elif specs and all(spec.name.startswith("f") for spec in specs):
        group_name = "主实验 F（粗粒度标点直接训练）"
    else:
        group_name = "主实验 A"
    augmentation_section = ""
    if any(spec.augmentation for spec in specs):
        augmentation = config.data_augmentation
        augmentation_section = (
            "\n\n### E9-Aug训练折内增强\n\n"
            + markdown_table(
                ["设置", "值"],
                [
                    ["方法", augmentation.method],
                    ["原始块之外的增强比例", f"{augmentation.ratio:.2f}"],
                    ["每块完整句数", f"{augmentation.min_sentences}—{augmentation.max_sentences}"],
                    ["每块正文长度", f"{augmentation.min_characters}—{augmentation.max_characters}"],
                    ["抽样范围", "同一文献"],
                    ["保持原文句序", "是" if augmentation.preserve_order else "否"],
                    ["开发/测试增强", "否"],
                    ["E3统计拟合", "仅原始外层训练折"],
                ],
            )
            + "\n\n> 只按 `。？！` 提取完整句子；TAB块末尾残片不进入增强池。"
        )
    return (
        f"## {group_name}：运行前检查\n\n"
        + markdown_table(["语料", "领域", "文献数", "正文字符"], source_rows)
        + "\n\n### 实验阶段\n\n"
        + markdown_table(
            ["实验", "方案", "模型后端", "阶段", "上游特征与门控"],
            stage_rows,
        )
        + augmentation_section
        + "\n\n### 五折文献级分层划分\n\n"
        + markdown_table(
            ["折", "训练文献", "开发文献", "测试文献", "实际比例"], split_rows
        )
        + (
            "\n\n### 七种停顿标点标签分布（grade=1）\n\n"
            if grade == 1
            else "\n\n### 句内/句间停顿标签分布（grade=2）\n\n"
        )
        + markdown_table(["标签", "数量"], label_rows)
        + "\n\n> Accuracy 包含 `O`，只作为辅助指标；所有 P/R/F1 均排除 `O`。"
        + warning
    )


def _fold_values(
    folds: list[dict[str, object]],
    condition: str,
    domain: str,
    section: str,
    metric: str | None = None,
) -> list[float]:
    values = []
    for fold in folds:
        group = fold["final_metrics"][condition].get(domain)  # type: ignore[index,union-attr]
        if group is None:
            continue
        value = group[section] if metric is None else group[section][metric]
        values.append(float(value))
    return values


def render_main_summary(result: dict[str, object]) -> str:
    folds: list[dict[str, object]] = result["folds"]  # type: ignore[assignment]
    grade = int(result.get("evaluation_grade", 1))
    grade_name = (
        "七种具体停顿标点" if grade == 1 else "句内/句间两类停顿标点"
    )
    rows = []
    for condition in result["conditions"]:  # type: ignore[union-attr]
        for domain in ("overall", "jingshu", "shisu"):
            if not _fold_values(folds, condition, domain, "micro", "f1"):
                continue
            rows.append(
                [
                    CONDITION_NAMES[condition],
                    DOMAIN_NAMES[domain],
                    mean_std(_fold_values(folds, condition, domain, "punctuation_position", "precision")),
                    mean_std(_fold_values(folds, condition, domain, "punctuation_position", "recall")),
                    mean_std(_fold_values(folds, condition, domain, "punctuation_position", "f1")),
                    mean_std(_fold_values(folds, condition, domain, "micro", "f1")),
                    mean_std(_fold_values(folds, condition, domain, "macro_f1")),
                    mean_std(
                        _fold_values(folds, condition, domain, "accuracy", "value")
                    ),
                    mean_std(
                        _fold_values(
                            folds,
                            condition,
                            domain,
                            "sentence_boundary_from_period_question_exclamation",
                            "f1",
                        )
                    ),
                ]
            )
    return (
        f"## {result['display_name']}：五折测试结果（grade={grade}，{grade_name}）\n\n"
        + markdown_table(
            [
                "输入条件",
                "测试集",
                "位置P",
                "位置R",
                "位置F1",
                "类别Micro-F1",
                "Macro-F1",
                "Accuracy（含O）",
                "句界F1",
            ],
            rows,
        )
        + (
            "\n\n> grade=1：按七种具体停顿标点评价；Macro-F1 固定对七类取宏平均。"
            if grade == 1
            else (
                "\n\n> grade=2：模型直接训练并预测O/句内/句间；不存在七类到两类的评价后映射。"
                if result.get("training_target") == "pause_group"
                else "\n\n> grade=2：将七种预测映射为句内/句间两类后评价；同组内具体标点混淆视为正确。"
            )
        )
        + " 数值为五折均值 ± 1 个总体标准差，保留四位小数。Accuracy 按所有位置的完整标签计算，包含 `O`，不进入任何 P/R/F1。"
        + (
            " grade=2 的句界F1与‘句间停顿’类别F1定义相同。"
            if grade == 2
            else ""
        )
    )


def render_stage_details(result: dict[str, object]) -> str:
    folds: list[dict[str, object]] = result["folds"]  # type: ignore[assignment]
    stage_names = list(folds[0]["stage_metrics"][result["conditions"][0]])  # type: ignore[index]
    rows = []
    display_names = result.get("stage_display_names", {})
    for condition in result["conditions"]:  # type: ignore[union-attr]
        for stage in stage_names:
            values = [
                float(fold["stage_metrics"][condition][stage]["overall"]["micro"]["f1"])  # type: ignore[index]
                for fold in folds
            ]
            position = [
                float(fold["stage_metrics"][condition][stage]["overall"]["punctuation_position"]["f1"])  # type: ignore[index]
                for fold in folds
            ]
            rows.append(
                [
                    CONDITION_NAMES[condition],
                    display_names.get(stage, stage),  # type: ignore[union-attr]
                    mean_std(position),
                    mean_std(values),
                ]
            )
    return "### 各阶段总体表现\n\n" + markdown_table(
        ["输入条件", "阶段", "位置F1", "类别Micro-F1"], rows
    )


def render_class_details(result: dict[str, object], minimum_gold: int = 1) -> str:
    """按总体/经书/世俗写出当前 grade 的类别指标，不在命令行刷屏。"""
    grade = int(result.get("evaluation_grade", 1))
    folds: list[dict[str, object]] = result["folds"]  # type: ignore[assignment]
    sections = []
    for condition in result["conditions"]:  # type: ignore[union-attr]
        domain_sections = []
        for domain in ("overall", "jingshu", "shisu"):
            domain_folds = [
                fold
                for fold in folds
                if domain in fold["final_metrics"][condition]  # type: ignore[operator,index]
            ]
            if not domain_folds:
                continue
            gold_counts: Counter[str] = Counter()
            for fold in domain_folds:
                per_class = fold["final_metrics"][condition][domain]["per_class"]  # type: ignore[index]
                for label, metrics in per_class.items():
                    gold_counts[label] += int(metrics["gold"])
            labels = [
                label
                for label, count in gold_counts.most_common()
                if count >= minimum_gold
            ]
            rows = []
            for label in labels:
                metric_values = []
                for metric in ("precision", "recall", "f1"):
                    metric_values.append(
                        mean_std(
                            float(
                                fold["final_metrics"][condition][domain]["per_class"]  # type: ignore[index]
                                .get(label, {})
                                .get(metric, 0.0)
                            )
                            for fold in domain_folds
                        )
                    )
                accuracy_values = []
                for fold in domain_folds:
                    group_metrics = fold["final_metrics"][condition][domain]  # type: ignore[index]
                    class_metrics = group_metrics["per_class"].get(label, {})
                    if "accuracy" in class_metrics:
                        accuracy_values.append(float(class_metrics["accuracy"]))
                        continue
                    # 兼容增加类别 Accuracy 之前生成的 results.json：可由
                    # 已保存的混淆矩阵直接恢复，无需重新训练模型。
                    confusion = group_metrics.get("confusion_matrix", {}).get(
                        "matrix", {}
                    )
                    if not confusion:
                        accuracy_values.append(0.0)
                        continue
                    total = sum(
                        int(count)
                        for predicted_counts in confusion.values()
                        for count in predicted_counts.values()
                    )
                    tp = int(confusion.get(label, {}).get(label, 0))
                    fp = sum(
                        int(predicted_counts.get(label, 0))
                        for gold_label, predicted_counts in confusion.items()
                        if gold_label != label
                    )
                    fn = sum(
                        int(count)
                        for predicted_label, count in confusion.get(label, {}).items()
                        if predicted_label != label
                    )
                    tn = total - tp - fp - fn
                    accuracy_values.append((tp + tn) / total if total else 0.0)
                rows.append(
                    [
                        GRADE_LABEL_NAMES.get(label, label),
                        gold_counts[label],
                        *metric_values,
                        mean_std(accuracy_values),
                    ]
                )
            domain_sections.append(
                f"##### {DOMAIN_NAMES[domain]}测试集\n\n"
                + markdown_table(
                    [
                        "停顿类别",
                        "金标准数",
                        "精确率",
                        "召回率",
                        "F1",
                        "Accuracy（含O）",
                    ],
                    rows,
                )
            )
        sections.append(
            f"#### {CONDITION_NAMES[condition]}\n\n"
            + "\n\n".join(domain_sections)
        )
    frequency_note = (
        "全部在相应测试集中出现过的停顿类别"
        if minimum_gold == 1
        else f"相应测试集金标准频次不少于 {minimum_gold} 的停顿类别"
    )
    grade_note = (
        "当前按七种具体停顿标点分别统计"
        if grade == 1
        else "当前按句内停顿（，、：；）和句间停顿（。？！）两类统计"
    )
    return (
        f"### 各领域逐类别指标（{frequency_note}）\n\n"
        f"> {grade_note}；引号和书名号不参与本表。`O` 不参与类别 P/R/F1。每类 Accuracy 按该类与其余全部标签的 one-vs-rest 方式计算，包含 `O`，仅作辅助指标。金标准数是五折测试集计数之和。\n\n"
        + "\n\n".join(sections)
    )


def render_model_details(result: dict[str, object]) -> str:
    """规则与n-gram给出每折可解释参数；CRF实验不额外显示。"""
    rule_rows = []
    ngram_rows = []
    neural_rows = []
    for fold in result["folds"]:  # type: ignore[union-attr]
        metadata_by_stage = fold.get("model_metadata", {})
        for stage, metadata in metadata_by_stage.items():
            stage_name = result.get("stage_display_names", {}).get(stage, stage)
            if "神经编码器" in metadata:
                threshold = metadata.get("开发集位置阈值")
                special = "—"
                if metadata.get("候选阈值") is not None:
                    special = f"候选阈值={float(metadata['候选阈值']):.2f}"
                if metadata.get("软融合alpha") is not None:
                    special = f"alpha={float(metadata['软融合alpha']):g}"
                if metadata.get("位置损失权重") is not None:
                    special = f"位置损失权重={float(metadata['位置损失权重']):g}"
                if metadata.get("最终融合权重") is not None:
                    special = (
                        f"词性残差 λ={float(metadata['最终融合权重']):g}；"
                        f"残差epoch={metadata.get('词性残差最佳epoch', '—')}"
                    )
                neural_rows.append(
                    [
                        fold["fold"],
                        stage_name,
                        metadata["神经编码器"],
                        metadata.get("解码层", "逐位置Softmax"),
                        metadata.get("参数量", "—"),
                        metadata.get("最佳epoch", "—"),
                        f"{float(metadata.get('最佳开发集loss', 0.0)):.6f}",
                        "—" if threshold is None else f"{float(threshold):.2f}",
                        special,
                    ]
                )
            elif "n" in metadata:
                threshold = metadata.get("开发集位置阈值")
                dev_f1 = metadata.get("开发集位置F1")
                ngram_rows.append(
                    [
                        fold["fold"],
                        stage_name,
                        metadata["n"],
                        metadata.get("上下文规则数", "—"),
                        "—" if threshold is None else f"{float(threshold):.2f}",
                        "—" if dev_f1 is None else f"{float(dev_f1):.4f}",
                    ]
                )
            elif "片段中位长度" in metadata:
                rule_rows.append(
                    [
                        fold["fold"],
                        stage_name,
                        metadata.get("片段中位长度", "—"),
                        metadata.get("全局多数标点", "—"),
                        metadata.get("特征字词规则数", "—"),
                        metadata.get("重复结构规则数", "—"),
                    ]
                )
    if not rule_rows and not ngram_rows and not neural_rows:
        return ""
    sections = []
    if rule_rows:
        sections.append(
            "### 各折规则摘要\n\n"
            + markdown_table(
                ["折", "阶段", "中位长度", "多数标点", "字词规则数", "结构规则数"],
                rule_rows,
            )
        )
    if ngram_rows:
        sections.append(
            "### 各折n-gram摘要\n\n"
            + markdown_table(
                ["折", "阶段", "n", "上下文数", "位置阈值", "开发集位置F1"],
                ngram_rows,
            )
        )
    if neural_rows:
        sections.append(
            "### 各折神经模型摘要\n\n"
            + markdown_table(
                [
                    "折",
                    "阶段",
                    "编码器",
                    "解码层",
                    "参数量",
                    "最佳epoch",
                    "开发集loss",
                    "位置阈值",
                    "C组关键参数",
                ],
                neural_rows,
            )
        )
    return "\n\n".join(sections)


def render_propagation_details(result: dict[str, object]) -> str:
    folds = [
        fold
        for fold in result["folds"]  # type: ignore[index]
        if fold.get("propagation_diagnostics")
    ]
    if not folds:
        return ""
    keys = []
    for fold in folds:
        for key in fold["propagation_diagnostics"]:
            if key not in keys:
                keys.append(key)
    rows = []
    for key in keys:
        values = [
            float(fold["propagation_diagnostics"][key])
            for fold in folds
            if key in fold["propagation_diagnostics"]
        ]
        if not values:
            continue
        is_rate = any(mark in key for mark in ("率", "F1", "精确率", "召回率"))
        summary = mean_std(values) if is_rate else str(int(sum(values)))
        rows.append([key, summary])
    return (
        "### 误差传播诊断\n\n"
        + markdown_table(["诊断量", "五折汇总"], rows)
        + "\n\n> 比率和F1报告五折均值 ± 标准差；实例数量为五折测试集总数。"
    )


def render_main_report(result: dict[str, object]) -> str:
    sections = [
        render_main_summary(result),
        render_stage_details(result),
        render_model_details(result),
        render_propagation_details(result),
        render_class_details(result),
    ]
    return (
        "\n\n".join(section for section in sections if section)
        + "\n\n> 完整混淆矩阵保存在 `results.json`，命令行不显示 JSON。"
    )
