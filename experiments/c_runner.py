from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from config import ExperimentConfig
from data.augmentation import SentenceConcatenationAugmenter
from data.corpus import PreparedDocument, SequenceChunk
from data.splits import FoldSplit, VolumeCrossValidator
from models.cascade import CascadeBiLSTMTagger
from models.factory import build_model
from models.neural import NeuralSequenceTagger
from tasks import OUTSIDE

from .runner import (
    LabelMap,
    _annotate_documents,
    _domain_counts,
    evaluate_by_domain,
    evaluate_final_by_domain,
    evaluation_grade_name,
    join_chunk_labels,
    make_chunks,
    read_main_documents,
    select_documents,
    stage_gold_labels,
    validate_evaluation_grade,
)
from .specs import POSITION_LABEL, ExperimentSpec


LOGGER = logging.getLogger(__name__)
ScoreMap = dict[str, list[float]]


@runtime_checkable
class _MultiTaskTagger(Protocol):
    """C4/E9共享编排器所需的最小多任务模型接口。"""

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None: ...

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]: ...

    def predict_position(
        self, sequences: list[SequenceChunk]
    ) -> list[list[str]]: ...

    def metadata(self) -> dict[str, object]: ...

    def save(self, path: Path) -> None: ...


@runtime_checkable
class _AugmentationAwareMultiTaskTagger(_MultiTaskTagger, Protocol):
    """E9-Aug额外要求：知识统计与实际优化样本使用不同输入集。"""

    def fit_augmented(
        self,
        original_train: list[SequenceChunk],
        augmented_train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None: ...


def join_chunk_scores(
    chunks: list[SequenceChunk], scores: list[list[float]]
) -> ScoreMap:
    if len(chunks) != len(scores):
        raise ValueError("序列块数量与概率输出数量不一致")
    grouped: dict[str, list[tuple[int, list[float]]]] = {}
    for chunk, values in zip(chunks, scores):
        if len(chunk.tokens) != len(values):
            raise ValueError(f"{chunk.document_id} 的概率长度与序列长度不一致")
        grouped.setdefault(chunk.document_id, []).append((chunk.offset, values))
    return {
        document_id: [value for _, part in sorted(parts) for value in part]
        for document_id, parts in grouped.items()
    }


def chunk_scores(chunks: list[SequenceChunk], scores: ScoreMap) -> list[list[float]]:
    values = []
    for chunk in chunks:
        document_scores = scores[chunk.document_id]
        values.append(
            document_scores[chunk.offset : chunk.offset + len(chunk.tokens)]
        )
    return values


def probability_labels(scores: ScoreMap, threshold: float) -> LabelMap:
    return {
        document_id: [
            POSITION_LABEL if probability >= threshold else OUTSIDE
            for probability in values
        ]
        for document_id, values in scores.items()
    }


def gold_probabilities(position_gold: LabelMap) -> ScoreMap:
    return {
        document_id: [1.0 if label == POSITION_LABEL else 0.0 for label in labels]
        for document_id, labels in position_gold.items()
    }


def _candidate_counts(
    gold: LabelMap, candidate: LabelMap
) -> dict[str, int | float]:
    tp = fp = fn = 0
    for document_id, gold_labels in gold.items():
        predicted = candidate[document_id]
        tp += sum(g == POSITION_LABEL and p == POSITION_LABEL for g, p in zip(gold_labels, predicted))
        fp += sum(g == OUTSIDE and p == POSITION_LABEL for g, p in zip(gold_labels, predicted))
        fn += sum(g == POSITION_LABEL and p == OUTSIDE for g, p in zip(gold_labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "候选真阳性": tp,
        "候选假阳性": fp,
        "候选漏检": fn,
        "候选精确率": precision,
        "候选召回率": recall,
    }


class CExperimentRunner:
    """C2—C4专用编排器；C1继续复用原B4硬级联运行器。"""

    def __init__(
        self, config: ExperimentConfig, spec: ExperimentSpec, grade: int = 1
    ) -> None:
        if spec.strategy not in {"candidate_reject", "soft_cascade", "multitask"}:
            raise ValueError(f"CExperimentRunner不支持策略 {spec.strategy!r}")
        self.config = config
        self.spec = spec
        validate_evaluation_grade(grade)
        if spec.stages[-1].target == "pause_group" and grade != 2:
            raise ValueError(
                "F1直接训练句内/句间两类，只支持--grade 2；"
                "它不能还原或评价七种具体标点"
            )
        self.grade = grade
        self.pause = frozenset(
            config.punctuation.sentence_pause
            + config.punctuation.intra_sentence_pause
        )

    def run(self, output_dir: Path) -> dict[str, object]:
        started = time.perf_counter()
        documents = read_main_documents(self.config)
        folds = VolumeCrossValidator(
            self.config.folds, self.config.dev_ratio, self.config.seed
        ).split(documents)
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
            "划分：%d折文献级分层交叉验证，train/dev/test≈7:1:2，seed=%d",
            self.config.folds,
            self.config.seed,
        )
        LOGGER.info(
            "BiLSTM参数：hidden=%d/方向，layers=%d，dropout=%g，max_length=%d",
            self.config.neural.bilstm_hidden_dim,
            self.config.neural.bilstm_layers,
            self.config.neural.dropout,
            self.config.max_sequence_length,
        )
        if self.spec.strategy in {"candidate_reject", "soft_cascade"}:
            LOGGER.info(
                "上游折外训练：位置模型在内折训练集上训练，预测内折测试集，"
                "下游只看到折外（未见文献）的上游预测"
            )
        if self.spec.strategy == "candidate_reject":
            LOGGER.info(
                "C2：开发集候选召回目标=%.2f；下游标签=O/拒绝＋七种停顿标点",
                self.config.cascade.candidate_recall_target,
            )
        elif self.spec.strategy == "soft_cascade":
            LOGGER.info(
                "C3：训练alpha=%g；开发集候选alpha=%s；无硬候选门控",
                self.config.cascade.soft_train_alpha,
                "/".join(map(str, self.config.cascade.soft_alpha_candidates)),
            )
        elif self.spec.name == "e9":
            LOGGER.info(
                "E9：TangutEncoder＋E3软词典格网/t-score上下文＋共享BiLSTM；"
                "完整标点头为主任务，位置辅助损失权重=%g",
                self.config.cascade.multitask_position_loss_weight,
            )
            LOGGER.info(
                "E9 TangutEncoder checkpoint：%s；编码器lr=%g，下游lr=%g，冻结编码器=%d epoch",
                self.config.pretraining.checkpoint,
                self.config.pretraining.downstream_encoder_learning_rate,
                self.config.pretraining.downstream_head_learning_rate,
                self.config.pretraining.downstream_freeze_epochs,
            )
            LOGGER.info(
                "E9知识防泄漏：词典为固定外部资源；上下文统计仅在每个外折训练文献拟合，"
                "训练块使用内部%d折OOF特征，开发/测试只做transform；关联强度=%s",
                self.config.knowledge.context.inner_folds,
                self.config.knowledge.context.association,
            )
        elif self.spec.name == "e9_aug":
            augmentation = self.config.data_augmentation
            if not augmentation.enabled:
                raise ValueError(
                    "E9-Aug需要在配置中设置data_augmentation.enabled=true"
                )
            LOGGER.info(
                "E9-Aug：严格复用E9模型；原始训练块全部保留，另生成约%.2f倍合成块",
                augmentation.ratio,
            )
            LOGGER.info(
                "增强约束：仅外层训练折、同一文献、完整句子%d—%d句、"
                "保持原文次序、长度%d—%d；开发/测试集不增强",
                augmentation.min_sentences,
                augmentation.max_sentences,
                augmentation.min_characters,
                augmentation.max_characters,
            )
            LOGGER.info(
                "E9-Aug知识防泄漏：E3 t-score只用原始训练文献拟合；"
                "合成块、开发集和测试集都只做transform"
            )
        elif self.spec.name == "f1":
            LOGGER.info(
                "F1：严格复用E9的D2 TangutEncoder＋E3词典/t-score上下文＋"
                "共享BiLSTM；主头直接训练O/句内/句间，位置辅助损失权重=%g",
                self.config.cascade.multitask_position_loss_weight,
            )
            LOGGER.info(
                "F1与E9唯一核心差异：E9主头训练O＋七种标点后再映射评价；"
                "F1主头从训练开始就不区分组内具体标点"
            )
            LOGGER.info(
                "F1知识防泄漏：固定外部词典；上下文统计只在外折训练文献拟合，"
                "训练块使用内部%d折OOF特征；关联强度=%s",
                self.config.knowledge.context.inner_folds,
                self.config.knowledge.context.association,
            )
        else:
            LOGGER.info(
                "C4：一个共享BiLSTM；完整标点头为主任务，位置辅助损失权重=%g",
                self.config.cascade.multitask_position_loss_weight,
            )

        type_stage = self.spec.stages[-1]
        position_stage = self.spec.stages[0] if len(self.spec.stages) > 1 else None
        type_gold = {
            document.document_id: stage_gold_labels(document, type_stage, self.config)
            for document in documents
        }
        position_gold = (
            {
                document.document_id: stage_gold_labels(
                    document, position_stage, self.config
                )
                for document in documents
            }
            if position_stage is not None
            else {
                document_id: [
                    POSITION_LABEL if label != OUTSIDE else OUTSIDE for label in labels
                ]
                for document_id, labels in type_gold.items()
            }
        )
        fold_results = [
            self._run_fold(
                documents, type_gold, position_gold, split, output_dir
            )
            for split in folds
        ]
        result: dict[str, object] = {
            "experiment": self.spec.name,
            "display_name": self.spec.display_name,
            "model_name": self.spec.model_name,
            "conditions": list(self.spec.conditions),
            "evaluation_grade": self.grade,
            "evaluation_name": evaluation_grade_name(self.grade),
            "target_labels": (
                sorted(self.pause)
                if self.grade == 1
                else ["INTRA", "SENTENCE"]
            ),
            "raw_target_labels": (
                ["INTRA", "SENTENCE"]
                if type_stage.target == "pause_group"
                else sorted(self.pause)
            ),
            "training_target": type_stage.target,
            "stage_display_names": {
                stage.name: stage.display_name for stage in self.spec.stages
            },
            "augmentation": self.spec.augmentation,
            "folds": fold_results,
        }
        (output_dir / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "%s五折实验完成，总耗时%.1f秒：%s",
            self.spec.name.upper(),
            time.perf_counter() - started,
            output_dir,
        )
        return result

    def _annotated_chunks(
        self,
        documents: list[PreparedDocument],
        labels: LabelMap,
    ) -> list[SequenceChunk]:
        annotated = _annotate_documents(documents, labels, {})
        return make_chunks(annotated, self.config.max_sequence_length)

    def _fit_position_model(
        self,
        train_documents: list[PreparedDocument],
        dev_documents: list[PreparedDocument],
        position_gold: LabelMap,
    ) -> NeuralSequenceTagger:
        train_chunks = self._annotated_chunks(train_documents, position_gold)
        dev_chunks = self._annotated_chunks(dev_documents, position_gold)
        model = build_model(self.config, "bilstm")
        if not isinstance(model, NeuralSequenceTagger):
            raise TypeError("C组上游必须是BiLSTM Softmax位置模型")
        model.fit(train_chunks, dev_chunks)
        return model

    def _score_documents(
        self,
        model: NeuralSequenceTagger,
        documents: list[PreparedDocument],
        position_gold: LabelMap,
    ) -> ScoreMap:
        chunks = self._annotated_chunks(documents, position_gold)
        return join_chunk_scores(chunks, model.predict_position_probabilities(chunks))

    def _out_of_fold_scores(
        self,
        outer_train: list[PreparedDocument],
        position_gold: LabelMap,
        outer_fold: int,
    ) -> ScoreMap:
        inner_folds = min(self.config.cascade.inner_folds, len(outer_train))
        if inner_folds < 2:
            raise ValueError("外层训练文献不足，无法生成折外上游预测")
        splits = VolumeCrossValidator(
            inner_folds,
            self.config.dev_ratio,
            self.config.seed + outer_fold * 1000,
        ).split(outer_train)
        scores: ScoreMap = {}
        for inner in splits:
            inner_train = select_documents(outer_train, inner.train_ids)
            inner_dev = select_documents(outer_train, inner.dev_ids)
            inner_test = select_documents(outer_train, inner.test_ids)
            LOGGER.info(
                "[外折%d][折外%d/%d] 位置模型文献 train/dev/heldout=%d/%d/%d",
                outer_fold,
                inner.fold,
                inner_folds,
                len(inner_train),
                len(inner_dev),
                len(inner_test),
            )
            model = self._fit_position_model(inner_train, inner_dev, position_gold)
            scores.update(self._score_documents(model, inner_test, position_gold))
        missing = {document.document_id for document in outer_train} - set(scores)
        if missing:
            raise RuntimeError(f"折外预测缺少文献：{sorted(missing)}")
        return scores

    def _threshold_values(self) -> list[float]:
        values = []
        value = self.config.neural.position_threshold_start
        while value <= self.config.neural.position_threshold_end + 1e-12:
            values.append(round(value, 10))
            value += self.config.neural.position_threshold_step
        return values

    def _candidate_threshold(
        self, dev_scores: ScoreMap, position_gold: LabelMap
    ) -> tuple[float, dict[str, int | float]]:
        target = self.config.cascade.candidate_recall_target
        best: tuple[float, float, float] | None = None
        fallback: tuple[float, float, float] | None = None
        for threshold in self._threshold_values():
            candidate = probability_labels(dev_scores, threshold)
            stats = _candidate_counts(
                {document_id: position_gold[document_id] for document_id in dev_scores},
                candidate,
            )
            precision = float(stats["候选精确率"])
            recall = float(stats["候选召回率"])
            fallback = max(fallback or (-1.0, -1.0, threshold), (recall, precision, threshold))
            if recall >= target:
                best = max(best or (-1.0, -1.0, threshold), (precision, threshold, recall))
        if best is not None:
            threshold = best[1]
        elif fallback is not None:
            threshold = fallback[2]
            LOGGER.warning(
                "开发集所有阈值都未达到候选召回目标%.2f，退回召回率最高的阈值%.2f",
                target,
                threshold,
            )
        else:
            raise RuntimeError("没有可用的候选阈值")
        candidate = probability_labels(dev_scores, threshold)
        stats = _candidate_counts(
            {document_id: position_gold[document_id] for document_id in dev_scores},
            candidate,
        )
        return threshold, stats

    def _run_fold(
        self,
        documents: list[PreparedDocument],
        type_gold: LabelMap,
        position_gold: LabelMap,
        split: FoldSplit,
        output_dir: Path,
    ) -> dict[str, object]:
        started = time.perf_counter()
        partitions = {
            "train": select_documents(documents, split.train_ids),
            "dev": select_documents(documents, split.dev_ids),
            "test": select_documents(documents, split.test_ids),
        }
        LOGGER.info(
            "[%d/%d] 文献train/dev/test=%d/%d/%d；领域test[%s]",
            split.fold,
            self.config.folds,
            len(partitions["train"]),
            len(partitions["dev"]),
            len(partitions["test"]),
            _domain_counts(partitions["test"]),
        )
        fold_path = output_dir / f"fold_{split.fold}"
        fold_path.mkdir(parents=True, exist_ok=True)
        if self.spec.strategy == "multitask":
            result = self._run_multitask_fold(
                partitions, type_gold, position_gold, split, fold_path
            )
        else:
            result = self._run_cascade_fold(
                partitions, type_gold, position_gold, split, fold_path
            )
        (fold_path / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "[%d/%d] 本折完成，耗时%.1f秒",
            split.fold,
            self.config.folds,
            time.perf_counter() - started,
        )
        return result

    def _run_cascade_fold(
        self,
        partitions: dict[str, list[PreparedDocument]],
        type_gold: LabelMap,
        position_gold: LabelMap,
        split: FoldSplit,
        fold_path: Path,
    ) -> dict[str, object]:
        upstream = self._fit_position_model(
            partitions["train"], partitions["dev"], position_gold
        )
        train_scores = self._out_of_fold_scores(
            partitions["train"], position_gold, split.fold
        )
        dev_scores = self._score_documents(upstream, partitions["dev"], position_gold)
        test_scores = self._score_documents(upstream, partitions["test"], position_gold)
        upstream.save(fold_path / "position.pt")
        train_chunks = self._annotated_chunks(partitions["train"], type_gold)
        dev_chunks = self._annotated_chunks(partitions["dev"], type_gold)
        test_chunks = self._annotated_chunks(partitions["test"], type_gold)
        train_probabilities = chunk_scores(train_chunks, train_scores)
        dev_probabilities = chunk_scores(dev_chunks, dev_scores)
        test_probabilities = chunk_scores(test_chunks, test_scores)

        if self.spec.strategy == "candidate_reject":
            threshold, dev_candidate_stats = self._candidate_threshold(
                dev_scores, position_gold
            )
            LOGGER.info(
                "[%d/%d][C2] 候选阈值=%.2f；开发集候选P=%.4f、R=%.4f",
                split.fold,
                self.config.folds,
                threshold,
                dev_candidate_stats["候选精确率"],
                dev_candidate_stats["候选召回率"],
            )
            downstream = build_model(self.config, "bilstm_candidate_reject")
        else:
            threshold = None
            dev_candidate_stats = {}
            downstream = build_model(self.config, "bilstm_soft_cascade")
        if not isinstance(downstream, CascadeBiLSTMTagger):
            raise TypeError("C2/C3下游模型类型不正确")
        downstream.fit_with_upstream(
            train_chunks,
            train_probabilities,
            dev_chunks,
            dev_probabilities,
            candidate_threshold=threshold,
        )

        if self.spec.strategy == "soft_cascade":
            scored_alphas = []
            dev_gold = {
                document.document_id: type_gold[document.document_id]
                for document in partitions["dev"]
            }
            for alpha in self.config.cascade.soft_alpha_candidates:
                prediction = join_chunk_labels(
                    dev_chunks,
                    downstream.predict_with_upstream(
                        dev_chunks, dev_probabilities, alpha=alpha
                    ),
                )
                metrics = evaluate_by_domain(
                    dev_gold,
                    prediction,
                    partitions["dev"],
                    class_labels=self.pause,
                    sentence_marks=frozenset(self.config.punctuation.sentence_pause),
                )["overall"]
                scored_alphas.append(
                    (float(metrics["micro"]["f1"]), float(metrics["punctuation_position"]["f1"]), alpha)
                )
            downstream.selected_alpha = max(scored_alphas)[2]
            LOGGER.info(
                "[%d/%d][C3] 开发集选择软融合alpha=%g（候选=%s）",
                split.fold,
                self.config.folds,
                downstream.selected_alpha,
                "/".join(map(str, self.config.cascade.soft_alpha_candidates)),
            )
        downstream.save(fold_path / f"{self.spec.stages[-1].name}.pt")

        test_ids = [document.document_id for document in partitions["test"]]
        test_type_gold = {document_id: type_gold[document_id] for document_id in test_ids}
        test_position_gold = {
            document_id: position_gold[document_id] for document_id in test_ids
        }
        gold_scores = gold_probabilities(test_position_gold)
        gold_probability_chunks = chunk_scores(test_chunks, gold_scores)
        predicted_final = join_chunk_labels(
            test_chunks,
            downstream.predict_with_upstream(
                test_chunks,
                test_probabilities,
                candidate_threshold=threshold,
            ),
        )
        oracle_final = join_chunk_labels(
            test_chunks,
            downstream.predict_with_upstream(
                test_chunks,
                gold_probability_chunks,
                candidate_threshold=0.5 if threshold is not None else None,
            ),
        )
        if threshold is not None:
            predicted_position = probability_labels(test_scores, threshold)
        else:
            predicted_position = join_chunk_labels(test_chunks, upstream.predict(test_chunks))
        final_metrics = {
            "oracle": evaluate_final_by_domain(
                test_type_gold,
                oracle_final,
                partitions["test"],
                self.config,
                self.grade,
            ),
            "predicted": evaluate_final_by_domain(
                test_type_gold,
                predicted_final,
                partitions["test"],
                self.config,
                self.grade,
            ),
        }
        position_metrics = {
            "oracle": evaluate_by_domain(
                test_position_gold,
                test_position_gold,
                partitions["test"],
                class_labels=frozenset({POSITION_LABEL}),
            ),
            "predicted": evaluate_by_domain(
                test_position_gold,
                predicted_position,
                partitions["test"],
                class_labels=frozenset({POSITION_LABEL}),
            ),
        }
        stage_metrics = {
            condition: {
                "position": position_metrics[condition],
                self.spec.stages[-1].name: final_metrics[condition],
            }
            for condition in self.spec.conditions
        }

        candidate_stats = _candidate_counts(test_position_gold, predicted_position)
        diagnostics: dict[str, int | float] = dict(candidate_stats)
        diagnostics.update(
            {f"开发集{name}": value for name, value in dev_candidate_stats.items()}
        )
        if self.spec.strategy == "candidate_reject":
            diagnostics["下游拒绝的候选假阳性"] = sum(
                gold == OUTSIDE and candidate == POSITION_LABEL and final == OUTSIDE
                for document_id in test_ids
                for gold, candidate, final in zip(
                    test_position_gold[document_id],
                    predicted_position[document_id],
                    predicted_final[document_id],
                )
            )
            diagnostics["上游漏检中被恢复"] = 0
        else:
            diagnostics["上游漏检中被恢复"] = sum(
                gold_type != OUTSIDE
                and upstream_label == OUTSIDE
                and final == gold_type
                for document_id in test_ids
                for gold_type, upstream_label, final in zip(
                    test_type_gold[document_id],
                    predicted_position[document_id],
                    predicted_final[document_id],
                )
            )
            diagnostics["上游假阳性中被拒绝"] = sum(
                gold_position == OUTSIDE
                and upstream_label == POSITION_LABEL
                and final == OUTSIDE
                for document_id in test_ids
                for gold_position, upstream_label, final in zip(
                    test_position_gold[document_id],
                    predicted_position[document_id],
                    predicted_final[document_id],
                )
            )
        overall = final_metrics["predicted"]["overall"]
        LOGGER.info(
            "[%d/%d][最终][predicted] 位置F1=%.4f，Micro-F1=%.4f，Macro-F1=%.4f，句界F1=%.4f，Accuracy=%.4f（含O）",
            split.fold,
            self.config.folds,
            overall["punctuation_position"]["f1"],
            overall["micro"]["f1"],
            overall["macro_f1"],
            overall["sentence_boundary_from_period_question_exclamation"]["f1"],
            overall["accuracy"]["value"],
        )
        return {
            "fold": split.fold,
            "train_documents": list(split.train_ids),
            "dev_documents": list(split.dev_ids),
            "test_documents": list(split.test_ids),
            "stage_metrics": stage_metrics,
            "final_metrics": final_metrics,
            "model_metadata": {
                "position": upstream.metadata(),
                self.spec.stages[-1].name: downstream.metadata(),
            },
            "propagation_diagnostics": diagnostics,
        }

    def _run_multitask_fold(
        self,
        partitions: dict[str, list[PreparedDocument]],
        type_gold: LabelMap,
        position_gold: LabelMap,
        split: FoldSplit,
        fold_path: Path,
    ) -> dict[str, object]:
        original_train_chunks = self._annotated_chunks(
            partitions["train"], type_gold
        )
        dev_chunks = self._annotated_chunks(partitions["dev"], type_gold)
        test_chunks = self._annotated_chunks(partitions["test"], type_gold)
        model = build_model(self.config, self.spec.model_name or "bilstm_multitask")
        if not isinstance(model, _MultiTaskTagger):
            raise TypeError(f"{self.spec.name.upper()}模型没有实现多任务接口")
        augmentation_metadata: dict[str, object] | None = None
        if self.spec.augmentation == "sentence_concatenation":
            if not isinstance(model, _AugmentationAwareMultiTaskTagger):
                raise TypeError("E9-Aug模型没有实现防泄漏增强训练接口")
            target_count = int(
                round(
                    len(original_train_chunks)
                    * self.config.data_augmentation.ratio
                )
            )
            augmenter = SentenceConcatenationAugmenter(
                self.config.data_augmentation,
                self.config.punctuation.sentence_pause,
                self.config.seed,
            )
            augmented = augmenter.augment(
                partitions["train"], type_gold, target_count, split.fold
            )
            augmented_chunks = [
                chunk
                for document in augmented.documents
                for chunk in document.chunks(self.config.max_sequence_length)
            ]
            augmentation_metadata = dict(augmented.metadata)
            augmentation_metadata.update(
                {
                    "原始训练块": len(original_train_chunks),
                    "增强训练块": len(augmented_chunks),
                    "开发集增强": False,
                    "测试集增强": False,
                    "E3统计拟合样本": "仅原始外层训练折",
                }
            )
            LOGGER.info(
                "[%d/%d][E9-Aug] 原始训练块=%d，合成块=%d；"
                "完整句=%d，可增强文献=%d，回退短块=%d",
                split.fold,
                self.config.folds,
                len(original_train_chunks),
                len(augmented_chunks),
                augmented.metadata["完整句子"],
                augmented.metadata.get("可增强文献", 0),
                augmented.metadata["低于最小长度的回退块"],
            )
            model.fit_augmented(
                original_train_chunks, augmented_chunks, dev_chunks
            )
        else:
            model.fit(original_train_chunks, dev_chunks)
        final_prediction = join_chunk_labels(test_chunks, model.predict(test_chunks))
        position_prediction = join_chunk_labels(
            test_chunks, model.predict_position(test_chunks)
        )
        test_ids = [document.document_id for document in partitions["test"]]
        test_type_gold = {document_id: type_gold[document_id] for document_id in test_ids}
        test_position_gold = {
            document_id: position_gold[document_id] for document_id in test_ids
        }
        final_metrics = {
            "direct": evaluate_final_by_domain(
                test_type_gold,
                final_prediction,
                partitions["test"],
                self.config,
                self.grade,
            )
        }
        position_metrics = evaluate_by_domain(
            test_position_gold,
            position_prediction,
            partitions["test"],
            class_labels=frozenset({POSITION_LABEL}),
        )
        derived_position = {
            document_id: [
                POSITION_LABEL if label != OUTSIDE else OUTSIDE
                for label in final_prediction[document_id]
            ]
            for document_id in test_ids
        }
        diagnostics = {
            "辅助位置头F1": position_metrics["overall"]["punctuation_position"]["f1"],
            "由主分类头推导的位置F1": evaluate_by_domain(
                test_position_gold,
                derived_position,
                partitions["test"],
                class_labels=frozenset({POSITION_LABEL}),
            )["overall"]["punctuation_position"]["f1"],
        }
        model.save(fold_path / "multitask.pt")
        overall = final_metrics["direct"]["overall"]
        LOGGER.info(
            "[%d/%d][%s最终] 位置F1=%.4f，Micro-F1=%.4f，Macro-F1=%.4f，句界F1=%.4f，Accuracy=%.4f（含O）",
            split.fold,
            self.config.folds,
            self.spec.name.upper(),
            overall["punctuation_position"]["f1"],
            overall["micro"]["f1"],
            overall["macro_f1"],
            overall["sentence_boundary_from_period_question_exclamation"]["f1"],
            overall["accuracy"]["value"],
        )
        result = {
            "fold": split.fold,
            "train_documents": list(split.train_ids),
            "dev_documents": list(split.dev_ids),
            "test_documents": list(split.test_ids),
            "stage_metrics": {
                "direct": {
                    "position": position_metrics,
                    "multitask": final_metrics["direct"],
                }
            },
            "final_metrics": final_metrics,
            "model_metadata": {"multitask": model.metadata()},
            "propagation_diagnostics": diagnostics,
        }
        if augmentation_metadata is not None:
            result["augmentation"] = augmentation_metadata
        return result
