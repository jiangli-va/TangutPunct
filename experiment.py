from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from config import ExperimentConfig
from data.corpus import CorpusReader, PreparedDocument, SequenceChunk
from data.splits import VolumeCrossValidator
from evaluation import evaluate_boundary, evaluate_punctuation
from models.factory import build_model
from tasks import Task


LOGGER = logging.getLogger(__name__)


def _select(documents: list[PreparedDocument], ids: tuple[str, ...]) -> list[PreparedDocument]:
    wanted = set(ids)
    return [document for document in documents if document.document_id in wanted]


def _chunks(documents: list[PreparedDocument], max_length: int) -> list[SequenceChunk]:
    return [chunk for document in documents for chunk in document.chunks(max_length)]


def _join_by_document(
    chunks: list[SequenceChunk], labels: list[list[str]] | None = None
) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        values = labels[index] if labels is not None else list(chunk.labels)
        grouped[chunk.document_id].append((chunk.offset, values))
    return {
        document_id: [label for _, part in sorted(parts) for label in part]
        for document_id, parts in grouped.items()
    }


class CrossValidationExperiment:
    def __init__(self, config: ExperimentConfig, task: Task) -> None:
        self.config = config
        self.task = task

    def run(self, output_dir: Path) -> dict[str, object]:
        started_at = time.perf_counter()
        reader = CorpusReader(
            self.config.data.paths,
            self.config.data.boundary_punctuation,
            self.config.data.missing_characters,
            self.config.data.missing_volume_numbers,
            self.config.data.ignored_editorial_symbols,
        )
        documents = reader.read(self.task)
        LOGGER.info(
            "任务=%s；读取 %d 卷、%d 个正文字符",
            self.task.value,
            len(documents),
            sum(len(document.tokens) for document in documents),
        )
        splitter = VolumeCrossValidator(
            self.config.folds, self.config.dev_ratio, self.config.seed
        )
        folds = splitter.split(documents)
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_results: list[dict[str, object]] = []
        all_gold: dict[str, list[str]] = {}
        all_predictions: dict[str, list[str]] = {}

        for split in folds:
            fold_started_at = time.perf_counter()
            train = _chunks(_select(documents, split.train_ids), self.config.max_sequence_length)
            dev = _chunks(_select(documents, split.dev_ids), self.config.max_sequence_length)
            test = _chunks(_select(documents, split.test_ids), self.config.max_sequence_length)
            LOGGER.info(
                "[%d/%d] train/dev/test=%d/%d/%d 卷，序列块=%d/%d/%d",
                split.fold,
                len(folds),
                len(split.train_ids),
                len(split.dev_ids),
                len(split.test_ids),
                len(train),
                len(dev),
                len(test),
            )
            LOGGER.debug("[%d/%d] 测试卷：%s", split.fold, len(folds), ", ".join(split.test_ids))
            model = build_model(self.config)
            training_started_at = time.perf_counter()
            LOGGER.info("[%d/%d] 开始构建特征并训练 CRF……", split.fold, len(folds))
            model.fit(train, dev)
            LOGGER.info(
                "[%d/%d] 训练完成，耗时 %.1f 秒；开始预测……",
                split.fold,
                len(folds),
                time.perf_counter() - training_started_at,
            )
            predictions = model.predict(test)
            gold_by_document = _join_by_document(test)
            predicted_by_document = _join_by_document(test, predictions)
            all_gold.update(gold_by_document)
            all_predictions.update(predicted_by_document)
            metrics = (
                evaluate_boundary(gold_by_document, predicted_by_document)
                if self.task is Task.BOUNDARY
                else evaluate_punctuation(gold_by_document, predicted_by_document)
            )
            result = {
                "fold": split.fold,
                "train_documents": list(split.train_ids),
                "dev_documents": list(split.dev_ids),
                "test_documents": list(split.test_ids),
                "metrics": metrics,
            }
            fold_results.append(result)
            (output_dir / f"fold_{split.fold}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            model.save(output_dir / f"fold_{split.fold}.joblib")
            primary = (
                metrics["boundary"]["f1"]
                if self.task is Task.BOUNDARY
                else metrics["micro"]["f1"]
            )
            LOGGER.info(
                "[%d/%d] 完成：F1=%.4f，总耗时 %.1f 秒；结果已保存",
                split.fold,
                len(folds),
                primary,
                time.perf_counter() - fold_started_at,
            )

        aggregate = (
            evaluate_boundary(all_gold, all_predictions)
            if self.task is Task.BOUNDARY
            else evaluate_punctuation(all_gold, all_predictions)
        )
        summary = {"task": self.task.value, "aggregate": aggregate, "folds": fold_results}
        (output_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("五折实验全部完成，总耗时 %.1f 秒：%s", time.perf_counter() - started_at, output_dir)
        return summary
