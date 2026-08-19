from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from config import ExperimentConfig
from data.corpus import CorpusReader, PreparedDocument, SequenceChunk
from data.labels import select_punctuation
from data.splits import FoldSplit, VolumeCrossValidator
from evaluation import evaluate_punctuation
from models.factory import build_model
from tasks import OUTSIDE, Task

from .specs import (
    INTRA_GROUP_LABEL,
    POSITION_LABEL,
    SENTENCE_GROUP_LABEL,
    ExperimentSpec,
    StageSpec,
)


LOGGER = logging.getLogger(__name__)
LabelMap = dict[str, list[str]]


def validate_evaluation_grade(grade: int) -> None:
    if grade not in {1, 2}:
        raise ValueError(f"未知评价等级 {grade!r}；只支持 1（七类）或 2（句内/句间两类）")


def evaluation_grade_name(grade: int) -> str:
    validate_evaluation_grade(grade)
    return "七种具体停顿标点" if grade == 1 else "句内/句间两类停顿标点"


def project_evaluation_labels(
    labels: LabelMap,
    config: ExperimentConfig,
    grade: int,
) -> LabelMap:
    """将七种标点标签投影到指定评价体系；不改变训练标签。"""
    validate_evaluation_grade(grade)
    if grade == 1:
        return {document_id: list(values) for document_id, values in labels.items()}

    sentence = set(config.punctuation.sentence_pause)
    intra = set(config.punctuation.intra_sentence_pause)
    projected: LabelMap = {}
    for document_id, values in labels.items():
        grouped = []
        for label in values:
            if label == OUTSIDE:
                grouped.append(OUTSIDE)
            elif label in {SENTENCE_GROUP_LABEL, INTRA_GROUP_LABEL}:
                # F1的训练与预测本身已经是粗粒度标签，不再做七类投影。
                grouped.append(label)
            elif label in sentence:
                grouped.append(SENTENCE_GROUP_LABEL)
            elif label in intra:
                grouped.append(INTRA_GROUP_LABEL)
            else:
                raise ValueError(f"{document_id} 的评价标签无法归入句内/句间：{label!r}")
        projected[document_id] = grouped
    return projected


def evaluate_final_by_domain(
    gold: LabelMap,
    predicted: LabelMap,
    test_documents: list[PreparedDocument],
    config: ExperimentConfig,
    grade: int,
) -> dict[str, dict[str, object]]:
    """按grade评价输出；F1的grade=2输入已经是句内/句间标签。"""
    gold_for_evaluation = project_evaluation_labels(gold, config, grade)
    predicted_for_evaluation = project_evaluation_labels(predicted, config, grade)
    if grade == 1:
        class_labels = frozenset(
            config.punctuation.sentence_pause
            + config.punctuation.intra_sentence_pause
        )
        sentence_marks = frozenset(config.punctuation.sentence_pause)
    else:
        class_labels = frozenset({INTRA_GROUP_LABEL, SENTENCE_GROUP_LABEL})
        sentence_marks = frozenset({SENTENCE_GROUP_LABEL})
    return evaluate_by_domain(
        gold_for_evaluation,
        predicted_for_evaluation,
        test_documents,
        class_labels=class_labels,
        sentence_marks=sentence_marks,
    )


def read_main_documents(config: ExperimentConfig) -> list[PreparedDocument]:
    reader = CorpusReader(
        config.data.paths,
        config.data.boundary_punctuation,
        config.data.missing_characters,
        config.data.missing_volume_numbers,
        config.data.ignored_editorial_symbols,
        domains=config.data.domains,
    )
    # 所有方案都从同一份“完整联合标签”派生阶段金标准。
    return reader.read(Task.PUNCTUATION)


def select_documents(
    documents: Iterable[PreparedDocument], ids: Iterable[str]
) -> list[PreparedDocument]:
    wanted = set(ids)
    return [document for document in documents if document.document_id in wanted]


def make_chunks(
    documents: Iterable[PreparedDocument], max_length: int
) -> list[SequenceChunk]:
    return [chunk for document in documents for chunk in document.chunks(max_length)]


def join_chunk_labels(
    chunks: list[SequenceChunk], labels: list[list[str]] | None = None
) -> LabelMap:
    grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        values = labels[index] if labels is not None else list(chunk.labels)
        grouped[chunk.document_id].append((chunk.offset, values))
    return {
        document_id: [label for _, part in sorted(parts) for label in part]
        for document_id, parts in grouped.items()
    }


def stage_gold_labels(
    document: PreparedDocument,
    stage: StageSpec,
    config: ExperimentConfig,
) -> list[str]:
    pause_labels = [
        select_punctuation(label, stage.punctuation) for label in document.labels
    ]
    if stage.target == "pause_type":
        return pause_labels
    if stage.target == "pause_position":
        return [POSITION_LABEL if label != OUTSIDE else OUTSIDE for label in pause_labels]
    if stage.target == "pause_group":
        sentence = set(config.punctuation.sentence_pause)
        intra = set(config.punctuation.intra_sentence_pause)
        grouped = []
        for label in pause_labels:
            has_sentence = any(mark in label for mark in sentence)
            has_intra = any(mark in label for mark in intra)
            if has_sentence and has_intra:
                raise ValueError(f"同一位置同时含句间和句内停顿标点：{label!r}")
            if has_sentence:
                grouped.append(SENTENCE_GROUP_LABEL)
            elif has_intra:
                grouped.append(INTRA_GROUP_LABEL)
            else:
                grouped.append(OUTSIDE)
        return grouped
    raise ValueError(f"未知阶段目标 {stage.target!r}")


def stage_class_labels(
    stage: StageSpec, config: ExperimentConfig
) -> frozenset[str]:
    if stage.target == "pause_position":
        return frozenset({POSITION_LABEL})
    if stage.target == "pause_group":
        return frozenset({SENTENCE_GROUP_LABEL, INTRA_GROUP_LABEL})
    if stage.target == "pause_type":
        return stage.punctuation
    raise ValueError(f"未知阶段目标 {stage.target!r}")


def constrain_stage_predictions(
    stage: StageSpec,
    predicted: LabelMap,
    upstream_channels: dict[str, LabelMap],
    config: ExperimentConfig,
) -> LabelMap:
    """按上游位置/大类对级联阶段做确定性门控。"""
    sentence = set(config.punctuation.sentence_pause)
    intra = set(config.punctuation.intra_sentence_pause)
    constrained: LabelMap = {}
    for document_id, labels in predicted.items():
        mask = (
            upstream_channels[stage.mask_upstream][document_id]
            if stage.mask_upstream
            else None
        )
        groups = (
            upstream_channels[stage.group_upstream][document_id]
            if stage.group_upstream
            else None
        )
        if mask is not None and len(mask) != len(labels):
            raise ValueError(f"{document_id} 的位置门控长度与预测长度不一致")
        if groups is not None and len(groups) != len(labels):
            raise ValueError(f"{document_id} 的大类门控长度与预测长度不一致")
        values = []
        for index, label in enumerate(labels):
            if mask is not None and mask[index] == OUTSIDE:
                values.append(OUTSIDE)
                continue
            if groups is not None and label != OUTSIDE:
                group = groups[index]
                compatible = (
                    group == SENTENCE_GROUP_LABEL and label in sentence
                ) or (group == INTRA_GROUP_LABEL and label in intra)
                if not compatible:
                    values.append(OUTSIDE)
                    continue
            values.append(label)
        constrained[document_id] = values
    return constrained


def _annotate_documents(
    documents: Iterable[PreparedDocument],
    gold: LabelMap,
    channels: dict[str, LabelMap],
) -> list[PreparedDocument]:
    result = []
    for document in documents:
        result.append(
            document.annotated(
                gold[document.document_id],
                (
                    (channel_name, values[document.document_id])
                    for channel_name, values in channels.items()
                ),
            )
        )
    return result


def _domain_counts(documents: Iterable[PreparedDocument]) -> str:
    counts = Counter(document.domain for document in documents)
    return "、".join(f"{domain}={count}" for domain, count in sorted(counts.items()))


def _label_summary(documents: Iterable[PreparedDocument]) -> str:
    counts = Counter(label for document in documents for label in document.labels)
    punctuation = [(label, count) for label, count in counts.most_common() if label != OUTSIDE]
    shown = punctuation[:8]
    return "、".join(f"{label}={count}" for label, count in shown) or "无标点标签"


def evaluate_by_domain(
    gold: LabelMap,
    predicted: LabelMap,
    test_documents: list[PreparedDocument],
    class_labels: frozenset[str] | None = None,
    sentence_marks: frozenset[str] = frozenset("。？！"),
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for group in ("overall", "jingshu", "shisu"):
        ids = [
            document.document_id
            for document in test_documents
            if group == "overall" or document.domain == group
        ]
        if not ids:
            continue
        groups[group] = evaluate_punctuation(
            {document_id: gold[document_id] for document_id in ids},
            {document_id: predicted[document_id] for document_id in ids},
            sentence_marks=sentence_marks,
            class_labels=class_labels,
        )
    return groups


def compose_final_predictions(
    spec: ExperimentSpec,
    stage_predictions: dict[str, LabelMap],
) -> LabelMap:
    final_stage = spec.stages[-1].name
    return {
        document_id: list(labels)
        for document_id, labels in stage_predictions[final_stage].items()
    }


class MainExperimentRunner:
    """主实验与规则基线共用的文献级交叉验证编排器。"""

    def __init__(
        self, config: ExperimentConfig, spec: ExperimentSpec, grade: int = 1
    ) -> None:
        validate_evaluation_grade(grade)
        self.config = config
        self.spec = spec
        self.grade = grade

    def run(self, output_dir: Path) -> dict[str, object]:
        started_at = time.perf_counter()
        documents = read_main_documents(self.config)
        classified = set(self.config.punctuation.sentence_pause) | set(
            self.config.punctuation.intra_sentence_pause
        ) | set(self.config.punctuation.structural)
        unclassified = Counter(
            mark
            for document in documents
            for label in document.labels
            if label != OUTSIDE
            for mark in label.lstrip("^")
            if mark not in classified and mark != "|"
        )
        if unclassified:
            details = "、".join(
                f"{mark}×{count}" for mark, count in unclassified.most_common()
            )
            raise ValueError(
                f"语料含未归类标点（{details}）；请先在 punctuation 配置中归类"
            )
        pause = set(self.config.punctuation.sentence_pause) | set(
            self.config.punctuation.intra_sentence_pause
        )
        invalid_pause_labels = Counter(
            selected
            for document in documents
            for label in document.labels
            if (selected := select_punctuation(label, pause))
            not in ({OUTSIDE} | pause)
        )
        if invalid_pause_labels:
            details = "、".join(
                f"{label}×{count}"
                for label, count in invalid_pause_labels.most_common()
            )
            raise ValueError(
                "停顿标点实验要求每个位置至多一个目标标点；"
                f"发现非法停顿联合标签：{details}"
            )
        splitter = VolumeCrossValidator(
            self.config.folds, self.config.dev_ratio, self.config.seed
        )
        folds = splitter.split(documents)
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("实验：%s", self.spec.display_name)
        LOGGER.info(
            "评价体系：grade=%d（%s）；Accuracy 含 O，仅作辅助指标",
            self.grade,
            evaluation_grade_name(self.grade),
        )
        LOGGER.info(
            "语料：%d 部文献，%d 个正文字符，领域[%s]",
            len(documents),
            sum(len(document.tokens) for document in documents),
            _domain_counts(documents),
        )
        LOGGER.info(
            "划分：%d 折文献级分层交叉验证；目标比例 train/dev/test ≈ 7:1:2；随机种子=%d",
            self.config.folds,
            self.config.seed,
        )
        LOGGER.info(
            "阶段：%s",
            " → ".join(stage.display_name for stage in self.spec.stages),
        )
        backend = self.spec.model_name or self.config.model.name
        stage_backends = {
            stage.name: stage.model_name or backend for stage in self.spec.stages
        }
        if len(set(stage_backends.values())) == 1:
            LOGGER.info("模型后端：%s", backend)
        else:
            LOGGER.info(
                "模型后端（按阶段）：%s",
                "；".join(
                    f"{stage.display_name}={stage_backends[stage.name]}"
                    for stage in self.spec.stages
                ),
            )
        if backend == "crf":
            LOGGER.info(
                "CRF 参数：c1=%g，c2=%g，max_iterations=%d；字窗口=±%d，上游特征窗口=±%d",
                self.config.model.c1,
                self.config.model.c2,
                self.config.model.max_iterations,
                self.config.features.window,
                self.config.features.stage_feature_window,
            )
        elif backend == "ngram":
            LOGGER.info(
                "n-gram参数：n=%d，最小支持数=%d，alpha=%g，回退k=%g；上下文[左=%s、右=%s、跨间隙=%s]",
                self.config.ngram.n,
                self.config.ngram.min_support,
                self.config.ngram.alpha,
                self.config.ngram.backoff_k,
                "开" if self.config.ngram.use_left else "关",
                "开" if self.config.ngram.use_right else "关",
                "开" if self.config.ngram.use_cross_gap else "关",
            )
        elif backend in {"bilstm", "bilstm_crf", "random_transformer"}:
            LOGGER.info(
                "神经参数：随机可训练字符嵌入=%d维，batch=%d，epochs=%d，lr=%g，patience=%d，min_delta=%g，device=%s",
                self.config.neural.embedding_dim,
                self.config.neural.batch_size,
                self.config.neural.epochs,
                self.config.neural.learning_rate,
                self.config.neural.patience,
                self.config.neural.min_delta,
                self.config.neural.device,
            )
            if self.config.neural.device == "auto":
                LOGGER.info(
                    "自动选卡条件：GPU利用率≤%d%%，空闲显存≥%.2f GiB；候选中优先空闲显存最多者",
                    self.config.neural.cuda_max_utilization,
                    self.config.neural.cuda_min_free_memory_gb,
                )
            if backend in {"bilstm", "bilstm_crf"}:
                LOGGER.info(
                    "BiLSTM：hidden=%d/方向，layers=%d，dropout=%g，解码层=%s",
                    self.config.neural.bilstm_hidden_dim,
                    self.config.neural.bilstm_layers,
                    self.config.neural.dropout,
                    "线性链CRF/Viterbi"
                    if backend == "bilstm_crf"
                    else "逐位置Softmax",
                )
            else:
                LOGGER.info(
                    "Random-Transformer：layers=%d，heads=%d，ff_dim=%d，dropout=%g；无预训练权重",
                    self.config.neural.transformer_layers,
                    self.config.neural.transformer_heads,
                    self.config.neural.transformer_ff_dim,
                    self.config.neural.dropout,
                )
        elif backend in {
            "tangut_encoder",
            "tangut_encoder_crf",
            "tangut_encoder_bilstm",
            "tangut_encoder_bilstm_knowledge",
            "tangut_encoder_bilstm_lexicon",
            "tangut_encoder_bilstm_lexicon_context",
            "tangut_encoder_bilstm_lexicon_context_segmentation",
            "tangut_encoder_bilstm_lexicon_context_pos",
            "tangut_encoder_bilstm_lexicon_context_pos_relation",
            "tangut_encoder_bilstm_lexicon_context_pos_relation_direct",
        }:
            LOGGER.info(
                "TangutEncoder checkpoint：%s",
                self.config.pretraining.checkpoint,
            )
            LOGGER.info(
                "下游微调：batch=%d，epochs=%d，编码器lr=%g，下游头lr=%g，冻结编码器=%d epoch，patience=%d",
                self.config.neural.batch_size,
                self.config.neural.epochs,
                self.config.pretraining.downstream_encoder_learning_rate,
                self.config.pretraining.downstream_head_learning_rate,
                self.config.pretraining.downstream_freeze_epochs,
                self.config.neural.patience,
            )
            if backend == "tangut_encoder_crf":
                LOGGER.info(
                    "D3结构：TangutEncoder → 线性发射层 → 线性链CRF/Viterbi；位置阶段不搜索Softmax阈值"
                )
            elif backend in {
                "tangut_encoder_bilstm",
                "tangut_encoder_bilstm_knowledge",
                "tangut_encoder_bilstm_lexicon",
                "tangut_encoder_bilstm_lexicon_context",
                "tangut_encoder_bilstm_lexicon_context_segmentation",
                "tangut_encoder_bilstm_lexicon_context_pos",
                "tangut_encoder_bilstm_lexicon_context_pos_relation",
                "tangut_encoder_bilstm_lexicon_context_pos_relation_direct",
            }:
                LOGGER.info(
                    "D4结构：TangutEncoder → BiLSTM → 逐位置Softmax；BiLSTM hidden=%d/方向，layers=%d，dropout=%g",
                    self.config.neural.bilstm_hidden_dim,
                    self.config.neural.bilstm_layers,
                    self.config.neural.dropout,
                )
                if backend == "tangut_encoder_bilstm_knowledge":
                    domain = self.config.knowledge.domain
                    LOGGER.info(
                        "E1显式领域知识：逐字符2维[经书倾向, 世俗倾向]；候选=%s；内部OOF=%d折（文献级）；平滑=%g；低频收缩=%g",
                        domain.candidate_mode,
                        domain.inner_folds,
                        domain.smoothing,
                        domain.shrinkage,
                    )
                if backend in {
                    "tangut_encoder_bilstm_lexicon",
                    "tangut_encoder_bilstm_lexicon_context",
                    "tangut_encoder_bilstm_lexicon_context_segmentation",
                    "tangut_encoder_bilstm_lexicon_context_pos",
                    "tangut_encoder_bilstm_lexicon_context_pos_relation",
                    "tangut_encoder_bilstm_lexicon_context_pos_relation_direct",
                }:
                    lexicon = self.config.knowledge.lexicon
                    LOGGER.info(
                        "E2软词典格网：候选长度=%s，来源特征=%s，保留重叠候选",
                        "/".join(map(str, lexicon.candidate_lengths)),
                        "启用" if lexicon.use_source_features else "关闭",
                    )
                if backend in {
                    "tangut_encoder_bilstm_lexicon_context",
                    "tangut_encoder_bilstm_lexicon_context_segmentation",
                    "tangut_encoder_bilstm_lexicon_context_pos",
                    "tangut_encoder_bilstm_lexicon_context_pos_relation",
                    "tangut_encoder_bilstm_lexicon_context_pos_relation_direct",
                }:
                    context = self.config.knowledge.context
                    LOGGER.info(
                        "E3上下文统计：8维左右间隔特征，关联强度=%s，内部OOF=%d折，截断分位数=%.1f",
                        context.association,
                        context.inner_folds,
                        context.clipping_percentile,
                    )
                if backend == "tangut_encoder_bilstm_lexicon_context_segmentation":
                    segmentation = self.config.knowledge.segmentation
                    LOGGER.info(
                        "E4软分词知识：表示=%s，维度=%d，融合=%s；重叠策略=%s，最短精确匹配=%d字",
                        segmentation.representation,
                        segmentation.dimension,
                        segmentation.fusion,
                        segmentation.overlap_policy,
                        segmentation.min_overlap_length,
                    )
                    LOGGER.info(
                        "E4分词资源：model=%s；lexicon=%s；gap=%s；人工标注审计源=%d份",
                        segmentation.model_path,
                        segmentation.lexicon_path,
                        segmentation.gap_path,
                        len(segmentation.annotation_paths),
                    )
                if backend == "tangut_encoder_bilstm_lexicon_context_pos":
                    pos = self.config.knowledge.pos
                    LOGGER.info(
                        "E5软词性知识：CRF-Joint-full的36类边缘概率→7组；"
                        "字符表示=%s，融合=%s，原始维度=%d，投影维度=%s，"
                        "整通道dropout=%.2f",
                        pos.representation,
                        pos.fusion,
                        pos.raw_dimension,
                        str(pos.projection_dimension)
                        if pos.fusion == "projected"
                        else "不适用（原值直接拼接）",
                        pos.channel_dropout,
                    )
                    LOGGER.info(
                        "E5词性资源：model=%s；lexicon_state=%s；manifest=%s；"
                        "重叠策略=%s，最短精确匹配=%d字",
                        pos.model_path,
                        pos.lexicon_state_path,
                        pos.manifest_path,
                        pos.overlap_policy,
                        pos.min_overlap_length,
                    )
                elif "tangut_encoder_bilstm_lexicon_context_pos" in stage_backends.values():
                    pos = self.config.knowledge.pos
                    LOGGER.info(
                        "E6阶段限定：位置阶段保持E3，不输入词性；仅具体标点类别阶段"
                        "拼接当前字符%d维软词性特征（%s，整通道dropout=%.2f）",
                        pos.raw_dimension,
                        pos.fusion,
                        pos.channel_dropout,
                    )
                if (
                    "tangut_encoder_bilstm_lexicon_context_pos_relation_direct"
                    in stage_backends.values()
                ):
                    LOGGER.info(
                        "E7两阶段直接注入：位置与类别阶段都将BIES×36类构成的"
                        "76维左词结束/右词开始关系与E3的23维知识直接拼接，均以"
                        "99维输入各自的BiLSTM；无投影、无门控、无残差、无lambda选择；"
                        "词性关系无整通道dropout（训练与推理始终输入）",
                    )
                if (
                    "tangut_encoder_bilstm_lexicon_context_pos_relation"
                    in stage_backends.values()
                ):
                    relation = self.config.knowledge.pos_relation
                    LOGGER.info(
                        "E8两阶段注入：位置与类别阶段均保留BIES×36类联合词性边缘"
                        "概率，构造左词结束/右词开始关系；词性嵌入=%d维，关系隐藏=%d维，"
                        "每个阶段分别先拟合E3基础模型、再冻结主体拟合后期门控残差",
                        relation.tag_embedding_dim,
                        relation.relation_hidden_dim,
                    )
                    LOGGER.info(
                        "细粒度词性残差训练：epochs=%d，lr=%g，patience=%d，通道dropout=%.2f；"
                        "开发集融合权重候选=%s（含0可退回E3；位置阶段按位置F1联合选阈值）",
                        relation.epochs,
                        relation.learning_rate,
                        relation.patience,
                        relation.channel_dropout,
                        "/".join(
                            f"{value:g}"
                            for value in relation.fusion_weight_candidates
                        ),
                    )
            else:
                LOGGER.info("D1/D2下游结构：TangutEncoder → 逐位置Softmax")
            LOGGER.info(
                "序列上限=%d（不得超过预训练位置嵌入上限）",
                self.config.max_sequence_length,
            )
        else:
            LOGGER.info(
                "规则参数：特征支持数≥%d、置信度≥%.2f；结构支持数≥%d、置信度≥%.2f",
                self.config.rules.cue_min_support,
                self.config.rules.cue_min_confidence,
                self.config.rules.structure_min_support,
                self.config.rules.structure_min_confidence,
            )
        if len(self.spec.stages) > 1:
            LOGGER.info(
                "误差传播对照：下游模型用金标准上游训练；测试时 oracle 使用金标准特征和门控，predicted 使用前序模型预测"
            )

        gold_by_stage = {
            stage.name: {
                document.document_id: stage_gold_labels(document, stage, self.config)
                for document in documents
            }
            for stage in self.spec.stages
        }
        fold_results = [
            self._run_fold(documents, gold_by_stage, split, output_dir)
            for split in folds
        ]
        result: dict[str, object] = {
            "experiment": self.spec.name,
            "display_name": self.spec.display_name,
            "model_name": backend,
            "stage_model_names": stage_backends,
            "conditions": list(self.spec.conditions),
            "evaluation_grade": self.grade,
            "evaluation_name": evaluation_grade_name(self.grade),
            "target_labels": (
                sorted(pause)
                if self.grade == 1
                else [INTRA_GROUP_LABEL, SENTENCE_GROUP_LABEL]
            ),
            "raw_target_labels": sorted(pause),
            "stage_display_names": {
                stage.name: stage.display_name for stage in self.spec.stages
            },
            "folds": fold_results,
        }
        (output_dir / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "%s 五折实验全部完成，总耗时 %.1f 秒：%s",
            self.spec.name.upper(),
            time.perf_counter() - started_at,
            output_dir,
        )
        return result

    def _run_fold(
        self,
        documents: list[PreparedDocument],
        gold_by_stage: dict[str, LabelMap],
        split: FoldSplit,
        output_dir: Path,
    ) -> dict[str, object]:
        fold_started = time.perf_counter()
        partitions = {
            "train": select_documents(documents, split.train_ids),
            "dev": select_documents(documents, split.dev_ids),
            "test": select_documents(documents, split.test_ids),
        }
        total = len(documents)
        LOGGER.info(
            "[%d/%d] 文献 train/dev/test=%d/%d/%d (%.1f%%/%.1f%%/%.1f%%)",
            split.fold,
            self.config.folds,
            len(partitions["train"]),
            len(partitions["dev"]),
            len(partitions["test"]),
            100 * len(partitions["train"]) / total,
            100 * len(partitions["dev"]) / total,
            100 * len(partitions["test"]) / total,
        )
        LOGGER.info(
            "[%d/%d] 领域 train[%s] dev[%s] test[%s]",
            split.fold,
            self.config.folds,
            _domain_counts(partitions["train"]),
            _domain_counts(partitions["dev"]),
            _domain_counts(partitions["test"]),
        )
        LOGGER.debug("[%d/%d] 测试文献：%s", split.fold, self.config.folds, ", ".join(split.test_ids))

        predictions: dict[str, dict[str, dict[str, LabelMap]]] = {
            condition: {} for condition in self.spec.conditions
        }
        stage_metrics: dict[str, dict[str, object]] = {
            condition: {} for condition in self.spec.conditions
        }
        model_metadata: dict[str, object] = {}
        fold_path = output_dir / f"fold_{split.fold}"

        for stage_index, stage in enumerate(self.spec.stages):
            # 下游模型始终用上游金标准特征训练，然后在同一个模型上
            # 分别以金标准/预测特征测试。因而两个条件的差值不会被
            # “换了一个下游模型”所混淆，可直接解释为测试时误差传播。
            gold_channels = {
                upstream: gold_by_stage[upstream] for upstream in stage.upstream
            }
            train_documents = _annotate_documents(
                partitions["train"], gold_by_stage[stage.name], gold_channels
            )
            dev_documents = _annotate_documents(
                partitions["dev"], gold_by_stage[stage.name], gold_channels
            )
            train_chunks = make_chunks(
                train_documents, self.config.max_sequence_length
            )
            dev_chunks = make_chunks(dev_documents, self.config.max_sequence_length)
            LOGGER.info(
                "[%d/%d][%s] 训练上游特征=%s；标签摘要：%s",
                split.fold,
                self.config.folds,
                stage.display_name,
                "金标准：" + "、".join(stage.upstream) if stage.upstream else "无",
                _label_summary(train_documents),
            )
            stage_backend = stage.model_name or self.spec.model_name or self.config.model.name
            LOGGER.info(
                "[%d/%d][%s] 开始拟合 %s：%d 个训练块，%d 个开发块",
                split.fold,
                self.config.folds,
                stage.display_name,
                stage_backend,
                len(train_chunks),
                len(dev_chunks),
            )
            model = build_model(self.config, stage_backend)
            training_started = time.perf_counter()
            model.fit(train_chunks, dev_chunks)
            LOGGER.info(
                "[%d/%d][%s] 训练完成，耗时 %.1f 秒",
                split.fold,
                self.config.folds,
                stage.display_name,
                time.perf_counter() - training_started,
            )
            extension = getattr(model, "file_extension", ".joblib")
            model.save(fold_path / f"{stage.name}{extension}")
            metadata = getattr(model, "metadata", None)
            if callable(metadata):
                model_metadata[stage.name] = metadata()

            test_gold = {
                document.document_id: gold_by_stage[stage.name][document.document_id]
                for document in partitions["test"]
            }
            # A1 只有 direct；多阶段的第一阶段没有上游特征，预测一次即可共用。
            prediction_conditions = (
                (self.spec.conditions[0],)
                if stage_index == 0
                else self.spec.conditions
            )
            for condition in prediction_conditions:
                test_channels = {
                    upstream: (
                        gold_by_stage[upstream]
                        if condition == "oracle"
                        else predictions[condition][upstream]["test"]
                    )
                    for upstream in stage.upstream
                }
                if stage.upstream:
                    LOGGER.info(
                        "[%d/%d][%s][%s] 测试上游特征=%s",
                        split.fold,
                        self.config.folds,
                        stage.display_name,
                        condition,
                        "金标准" if condition == "oracle" else "前一阶段预测",
                    )
                test_documents = _annotate_documents(
                    partitions["test"], gold_by_stage[stage.name], test_channels
                )
                test_chunks = make_chunks(
                    test_documents, self.config.max_sequence_length
                )
                predicted_map = join_chunk_labels(
                    test_chunks, model.predict(test_chunks)
                )
                predicted_map = constrain_stage_predictions(
                    stage, predicted_map, test_channels, self.config
                )
                predictions[condition][stage.name] = {"test": predicted_map}
                stage_metrics[condition][stage.name] = evaluate_by_domain(
                    test_gold,
                    predicted_map,
                    partitions["test"],
                    class_labels=stage_class_labels(stage, self.config),
                    sentence_marks=frozenset(self.config.punctuation.sentence_pause),
                )
                overall = stage_metrics[condition][stage.name]["overall"]
                LOGGER.info(
                    "[%d/%d][%s][%s] 测试 Micro-F1=%.4f，任意标点位置 F1=%.4f",
                    split.fold,
                    self.config.folds,
                    stage.display_name,
                    condition,
                    overall["micro"]["f1"],
                    overall["punctuation_position"]["f1"],
                )
            if stage_index == 0 and len(self.spec.conditions) > 1:
                shared = predictions[self.spec.conditions[0]][stage.name]
                shared_metrics = stage_metrics[self.spec.conditions[0]][stage.name]
                for condition in self.spec.conditions[1:]:
                    predictions[condition][stage.name] = shared
                    stage_metrics[condition][stage.name] = shared_metrics

        final_metrics: dict[str, object] = {}
        final_stage = self.spec.stages[-1]
        gold_pause = {
            document.document_id: gold_by_stage[final_stage.name][document.document_id]
            for document in partitions["test"]
        }
        for condition in self.spec.conditions:
            test_stage_predictions = {
                stage.name: predictions[condition][stage.name]["test"]
                for stage in self.spec.stages
            }
            final_predictions = compose_final_predictions(
                self.spec, test_stage_predictions
            )
            final_metrics[condition] = evaluate_final_by_domain(
                gold_pause,
                final_predictions,
                partitions["test"],
                self.config,
                self.grade,
            )
            overall = final_metrics[condition]["overall"]
            LOGGER.info(
                "[%d/%d][最终][%s] 总体：位置 F1=%.4f，Micro-F1=%.4f，Macro-F1=%.4f，句界 F1=%.4f，Accuracy=%.4f（含O）",
                split.fold,
                self.config.folds,
                condition,
                overall["punctuation_position"]["f1"],
                overall["micro"]["f1"],
                overall["macro_f1"],
                overall["sentence_boundary_from_period_question_exclamation"]["f1"],
                overall["accuracy"]["value"],
            )

        result: dict[str, object] = {
            "fold": split.fold,
            "train_documents": list(split.train_ids),
            "dev_documents": list(split.dev_ids),
            "test_documents": list(split.test_ids),
            "stage_metrics": stage_metrics,
            "final_metrics": final_metrics,
            "model_metadata": model_metadata,
        }
        fold_path.mkdir(parents=True, exist_ok=True)
        (fold_path / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "[%d/%d] 本折完成，总耗时 %.1f 秒",
            split.fold,
            self.config.folds,
            time.perf_counter() - fold_started,
        )
        return result
