from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from data.corpus import PreparedDocument
from data.labels import select_punctuation
from data.splits import FoldSplit, VolumeCrossValidator
from experiments.runner import (
    evaluate_by_domain,
    evaluate_final_by_domain,
    join_chunk_labels,
    make_chunks,
    read_main_documents,
    select_documents,
)
from experiments.specs import INTRA_GROUP_LABEL, POSITION_LABEL, SENTENCE_GROUP_LABEL
from models.d5_punctuation import D5PunctuationModel
from models.tangut_encoder import (
    PAD_TOKEN,
    load_tangut_encoder_checkpoint,
)
from models.tangut_tagger import TangutEncoderSequenceTagger
from tasks import OUTSIDE

from .d1_runner import _linear_schedule, _resolve_device, _set_seed
from .d5_data import D5Dataset, collate_d5
from .data import IGNORE_INDEX


LOGGER = logging.getLogger(__name__)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _gold_maps(
    documents: list[PreparedDocument], config: ExperimentConfig
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    pause = frozenset(
        config.punctuation.intra_sentence_pause
        + config.punctuation.sentence_pause
    )
    punctuation = {
        document.document_id: [
            select_punctuation(label, pause) for label in document.labels
        ]
        for document in documents
    }
    position = {
        document_id: [
            POSITION_LABEL if label != OUTSIDE else OUTSIDE for label in labels
        ]
        for document_id, labels in punctuation.items()
    }
    return punctuation, position


def _annotate(
    documents: list[PreparedDocument],
    labels: dict[str, list[str]],
    position: dict[str, list[str]] | None = None,
) -> list[PreparedDocument]:
    return [
        document.annotated(
            labels[document.document_id],
            ()
            if position is None
            else (("position", position[document.document_id]),),
        )
        for document in documents
    ]


class D5FoldTrainer:
    """在一个外层折内训练D5；测试文献绝不进入优化或早停。"""

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_path: Path,
        fold: int,
    ) -> None:
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.fold = fold
        self.cfg = config.punctuation_pretraining
        self.pause_labels = tuple(
            dict.fromkeys(
                config.punctuation.sentence_pause
                + config.punctuation.intra_sentence_pause
            )
        )
        self.intra_indices = tuple(
            index
            for index, label in enumerate(self.pause_labels)
            if label in config.punctuation.intra_sentence_pause
        )
        self.sentence_indices = tuple(
            index
            for index, label in enumerate(self.pause_labels)
            if label in config.punctuation.sentence_pause
        )
        self.device, self.device_selection = _resolve_device(config.pretraining)
        self.model: D5PunctuationModel | None = None
        self.vocabulary: dict[str, int] = {}
        self.parent_checkpoint: dict[str, object] = {}
        self.best_step = 0
        self.best_dev_loss = float("inf")
        self.position_threshold = 0.5
        self.history: list[dict[str, float | int]] = []

    def _load_model(self) -> D5PunctuationModel:
        encoder, vocabulary, checkpoint = load_tangut_encoder_checkpoint(
            self.checkpoint_path
        )
        self.vocabulary = vocabulary
        self.parent_checkpoint = checkpoint
        return D5PunctuationModel(
            encoder,
            len(self.pause_labels),
            self.config.pretraining.dropout,
        ).to(self.device)

    def _datasets(
        self,
        train_documents: list[PreparedDocument],
        dev_documents: list[PreparedDocument],
    ) -> tuple[D5Dataset, D5Dataset]:
        train_chunks = make_chunks(train_documents, self.config.max_sequence_length)
        dev_chunks = make_chunks(dev_documents, self.config.max_sequence_length)
        return (
            D5Dataset(
                train_chunks,
                self.vocabulary,
                self.config,
                dynamic=True,
                seed=self.config.seed + self.fold * 1000,
            ),
            D5Dataset(
                dev_chunks,
                self.vocabulary,
                self.config,
                dynamic=False,
                seed=self.config.seed + self.fold * 1000 + 500,
            ),
        )

    @staticmethod
    def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not bool((labels != IGNORE_INDEX).any()):
            # 某个batch可能没有停顿标点；粗类/七类头该批不更新，但其他
            # 三个任务仍正常训练，不能让全IGNORE交叉熵产生NaN。
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def _losses(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("D5模型尚未建立")
        original = batch["original_ids"].to(self.device)  # type: ignore[union-attr]
        padding = original.eq(self.vocabulary[PAD_TOKEN])
        position_logits, group_logits, type_logits = self.model.punctuation_logits(
            original, padding
        )
        masked = batch["input_ids"].to(self.device)  # type: ignore[union-attr]
        mlm_labels = batch["labels"].to(self.device)  # type: ignore[union-attr]
        mlm_logits = self.model.mlm_logits(
            masked, batch["padding_mask"].to(self.device)  # type: ignore[union-attr]
        )
        losses = {
            "mlm": self._ce(mlm_logits, mlm_labels),
            "position": self._ce(
                position_logits,
                batch["position_labels"].to(self.device),  # type: ignore[union-attr]
            ),
            "group": self._ce(
                group_logits,
                batch["group_labels"].to(self.device),  # type: ignore[union-attr]
            ),
            "type": self._ce(
                type_logits,
                batch["type_labels"].to(self.device),  # type: ignore[union-attr]
            ),
        }
        losses["total"] = (
            self.cfg.mlm_weight * losses["mlm"]
            + self.cfg.position_weight * losses["position"]
            + self.cfg.group_weight * losses["group"]
            + self.cfg.type_weight * losses["type"]
        )
        return losses

    def _evaluate_loss(self, loader: DataLoader) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("D5模型尚未建立")
        self.model.eval()
        totals = {key: 0.0 for key in ("mlm", "position", "group", "type", "total")}
        batches = 0
        with torch.no_grad():
            for batch in loader:
                losses = self._losses(batch)
                for key in totals:
                    totals[key] += float(losses[key].item())
                batches += 1
        return {key: value / max(batches, 1) for key, value in totals.items()}

    def fit(
        self,
        train_documents: list[PreparedDocument],
        dev_documents: list[PreparedDocument],
        fold_path: Path,
    ) -> None:
        _set_seed(self.config.seed + self.fold)
        self.model = self._load_model()
        train_dataset, dev_dataset = self._datasets(train_documents, dev_documents)
        collate = lambda batch: collate_d5(
            batch, self.vocabulary[PAD_TOKEN]
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        head_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("encoder.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.cfg.encoder_learning_rate,
                },
                {"params": head_parameters, "lr": self.cfg.head_learning_rate},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        schedule, warmup_steps = _linear_schedule(
            self.cfg.max_steps, self.cfg.warmup_ratio
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        LOGGER.info(
            "[%d/%d][D5] 初始化=%s（stage=%s, step=%s）；训练/开发文献=%d/%d，训练块=%d，开发块=%d",
            self.fold,
            self.config.folds,
            self.checkpoint_path,
            self.parent_checkpoint.get("stage", "unknown"),
            self.parent_checkpoint.get("step", 0),
            len(train_documents),
            len(dev_documents),
            len(train_dataset),
            len(dev_dataset),
        )
        LOGGER.info(
            "[%d/%d][D5] max_steps=%d，batch=%d，encoder_lr=%g，head_lr=%g，warmup=%d；损失权重 MLM/位置/粗类/七类=%g/%g/%g/%g",
            self.fold,
            self.config.folds,
            self.cfg.max_steps,
            self.cfg.batch_size,
            self.cfg.encoder_learning_rate,
            self.cfg.head_learning_rate,
            warmup_steps,
            self.cfg.mlm_weight,
            self.cfg.position_weight,
            self.cfg.group_weight,
            self.cfg.type_weight,
        )
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        step = 0
        epoch = 0
        stopped = False
        interval_total = 0.0
        interval_batches = 0
        while step < self.cfg.max_steps and not stopped:
            epoch += 1
            train_dataset.set_epoch(epoch)
            loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                collate_fn=collate,
                generator=torch.Generator().manual_seed(
                    self.config.seed + self.fold * 1000 + epoch
                ),
            )
            for batch in loader:
                if step >= self.cfg.max_steps:
                    break
                self.model.train()
                optimizer.zero_grad()
                losses = self._losses(batch)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                step += 1
                interval_total += float(losses["total"].item())
                interval_batches += 1
                if step % self.cfg.log_interval == 0:
                    LOGGER.info(
                        "[%d/%d][D5] step %d/%d，epoch=%d，区间loss=%.6f，encoder_lr=%.2e",
                        self.fold,
                        self.config.folds,
                        step,
                        self.cfg.max_steps,
                        epoch,
                        interval_total / max(interval_batches, 1),
                        optimizer.param_groups[0]["lr"],
                    )
                    interval_total = 0.0
                    interval_batches = 0
                if step % self.cfg.eval_interval and step != self.cfg.max_steps:
                    continue
                metrics = self._evaluate_loss(dev_loader)
                self.history.append({"step": step, **metrics})
                LOGGER.info(
                    "[%d/%d][D5][开发集] step=%d，total=%.6f，MLM=%.6f，位置=%.6f，粗类=%.6f，七类=%.6f",
                    self.fold,
                    self.config.folds,
                    step,
                    metrics["total"],
                    metrics["mlm"],
                    metrics["position"],
                    metrics["group"],
                    metrics["type"],
                )
                if metrics["total"] < self.best_dev_loss - self.cfg.min_delta:
                    self.best_dev_loss = metrics["total"]
                    self.best_step = step
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    LOGGER.info(
                        "[%d/%d][D5] 早停计数=%d/%d（最佳step=%d）",
                        self.fold,
                        self.config.folds,
                        stale,
                        self.cfg.patience,
                        self.best_step,
                    )
                    if stale >= self.cfg.patience:
                        stopped = True
                        break
        self.model.load_state_dict(best_state)
        self.position_threshold = self._select_position_threshold(
            dev_documents
        )
        fold_path.mkdir(parents=True, exist_ok=True)
        self._save_checkpoints(fold_path, train_documents, dev_documents)

    def load_if_compatible(
        self,
        fold_path: Path,
        train_documents: list[PreparedDocument],
        dev_documents: list[PreparedDocument],
    ) -> bool:
        """复用同一折D5权重，使两个出口只改变下游处理。"""

        full_path = fold_path / "best_model.pt"
        encoder_path = fold_path / "best_encoder.pt"
        if not full_path.exists() or not encoder_path.exists():
            return False
        checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
        expected_train = [item.document_id for item in train_documents]
        expected_dev = [item.document_id for item in dev_documents]
        compatible = (
            checkpoint.get("format") == "d5_punctuation_model"
            and Path(str(checkpoint.get("parent_checkpoint", ""))).resolve()
            == self.checkpoint_path.resolve()
            and checkpoint.get("train_document_ids") == expected_train
            and checkpoint.get("validation_document_ids") == expected_dev
            and checkpoint.get("punctuation_pretraining_config") == asdict(self.cfg)
            and tuple(checkpoint.get("punctuation_labels", ())) == self.pause_labels
        )
        if not compatible:
            LOGGER.info(
                "[%d/%d][D5] 已有共享checkpoint与当前来源、配置或文献划分不一致，重新训练",
                self.fold,
                self.config.folds,
            )
            return False
        encoder, vocabulary, _ = load_tangut_encoder_checkpoint(encoder_path)
        self.vocabulary = vocabulary
        self.parent_checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.model = D5PunctuationModel(
            encoder,
            len(self.pause_labels),
            self.config.pretraining.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_step = int(checkpoint.get("step", 0))
        self.best_dev_loss = float(checkpoint.get("best_dev_loss", float("nan")))
        self.position_threshold = float(checkpoint.get("position_threshold", 0.5))
        self.history = list(checkpoint.get("history", []))
        LOGGER.info(
            "[%d/%d][D5] 复用共享checkpoint：%s（step=%d，位置阈值=%.2f）",
            self.fold,
            self.config.folds,
            full_path,
            self.best_step,
            self.position_threshold,
        )
        return True

    def _raw_predictions(
        self, documents: list[PreparedDocument]
    ) -> tuple[dict[str, list[float]], dict[str, list[int]], dict[str, list[list[float]]]]:
        if self.model is None:
            raise RuntimeError("D5模型尚未训练")
        chunks = make_chunks(documents, self.config.max_sequence_length)
        dataset = D5Dataset(
            chunks,
            self.vocabulary,
            self.config,
            dynamic=False,
            seed=self.config.seed + self.fold * 1000 + 900,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_d5(
                batch, self.vocabulary[PAD_TOKEN]
            ),
        )
        position_parts: list[list[float]] = []
        group_parts: list[list[int]] = []
        type_parts: list[list[list[float]]] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                original = batch["original_ids"].to(self.device)  # type: ignore[union-attr]
                padding = original.eq(self.vocabulary[PAD_TOKEN])
                position, group, punctuation = self.model.punctuation_logits(
                    original, padding
                )
                lengths = batch["lengths"].tolist()  # type: ignore[union-attr]
                for row, length in enumerate(lengths):
                    position_parts.append(
                        torch.softmax(position[row, :length], -1)[:, 1]
                        .cpu()
                        .tolist()
                    )
                    group_parts.append(group[row, :length].argmax(-1).cpu().tolist())
                    type_parts.append(punctuation[row, :length].cpu().tolist())
        def join(parts):
            return join_chunk_labels(chunks, parts)
        return join(position_parts), join(group_parts), join(type_parts)

    @staticmethod
    def _position_f1(gold: list[bool], predicted: list[bool]) -> tuple[float, float]:
        tp = sum(a and b for a, b in zip(gold, predicted))
        fp = sum(not a and b for a, b in zip(gold, predicted))
        fn = sum(a and not b for a, b in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, f1

    def _thresholds(self) -> list[float]:
        values = []
        current = self.cfg.position_threshold_start
        while current <= self.cfg.position_threshold_end + 1e-12:
            values.append(round(current, 10))
            current += self.cfg.position_threshold_step
        return values

    def _select_position_threshold(
        self, dev_documents: list[PreparedDocument]
    ) -> float:
        scores, _, _ = self._raw_predictions(dev_documents)
        _, gold_position = _gold_maps(dev_documents, self.config)
        gold = [
            label == POSITION_LABEL
            for document in dev_documents
            for label in gold_position[document.document_id]
        ]
        flat_scores = [
            score
            for document in dev_documents
            for score in scores[document.document_id]
        ]
        best = (-1.0, -1.0, 0.5)
        for threshold in self._thresholds():
            precision, f1 = self._position_f1(
                gold, [score >= threshold for score in flat_scores]
            )
            best = max(best, (f1, precision, threshold))
        LOGGER.info(
            "[%d/%d][D5] 开发集选择位置阈值=%.2f，位置F1=%.4f",
            self.fold,
            self.config.folds,
            best[2],
            best[0],
        )
        return best[2]

    def predict(self, documents: list[PreparedDocument]) -> dict[str, list[str]]:
        position, groups, types = self._raw_predictions(documents)
        predictions: dict[str, list[str]] = {}
        for document in documents:
            document_id = document.document_id
            values = []
            for probability, group, scores in zip(
                position[document_id], groups[document_id], types[document_id]
            ):
                if probability < self.position_threshold:
                    values.append(OUTSIDE)
                    continue
                candidates = self.intra_indices if group == 0 else self.sentence_indices
                selected = max(candidates, key=lambda index: scores[index])
                values.append(self.pause_labels[selected])
            predictions[document_id] = values
        return predictions

    def _save_checkpoints(
        self,
        fold_path: Path,
        train_documents: list[PreparedDocument],
        dev_documents: list[PreparedDocument],
    ) -> None:
        if self.model is None:
            raise RuntimeError("D5模型尚未训练")
        common = {
            "stage": "d5_punctuation_aware",
            "step": self.best_step,
            "parent_checkpoint": str(self.checkpoint_path),
            "parent_stage": self.parent_checkpoint.get("stage", "unknown"),
            "vocabulary": self.vocabulary,
            "model_config": self.parent_checkpoint["model_config"],
            "train_document_ids": [item.document_id for item in train_documents],
            "validation_document_ids": [item.document_id for item in dev_documents],
            "position_threshold": self.position_threshold,
            "best_dev_loss": self.best_dev_loss,
            "history": self.history,
            "punctuation_labels": self.pause_labels,
            "punctuation_pretraining_config": asdict(self.cfg),
        }
        torch.save(
            {
                "format": "tangut_encoder",
                "format_version": 3,
                "model_state_dict": _cpu_state_dict(self.model.encoder),
                **common,
            },
            fold_path / "best_encoder.pt",
        )
        torch.save(
            {
                "format": "d5_punctuation_model",
                "format_version": 1,
                "model_state_dict": _cpu_state_dict(self.model),
                **common,
            },
            fold_path / "best_model.pt",
        )

    def metadata(self) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("D5模型尚未训练")
        return {
            "神经编码器": "tangut_encoder_d5",
            "解码层": "层级Softmax（位置→粗类约束→七类）",
            "参数量": sum(p.numel() for p in self.model.parameters()),
            "最佳epoch": f"step {self.best_step}",
            "最佳开发集loss": self.best_dev_loss,
            "开发集位置阈值": self.position_threshold,
            "预训练checkpoint": str(self.checkpoint_path),
            "预训练阶段": self.parent_checkpoint.get("stage", "unknown"),
            "D5训练记录": self.history,
        }


class D5ExperimentRunner:
    """D5两个出口：保留头直接预测，或丢弃头后接D4式BiLSTM。"""

    def __init__(
        self,
        config: ExperimentConfig,
        mode: str,
        grade: int,
        initialization_source: str,
    ) -> None:
        if mode not in {"direct", "bilstm"}:
            raise ValueError("D5出口只能是direct或bilstm")
        self.config = config
        self.mode = mode
        self.grade = grade
        self.initialization_source = initialization_source

    def inspect(self) -> str:
        documents = read_main_documents(self.config)
        folds = VolumeCrossValidator(
            self.config.folds, self.config.dev_ratio, self.config.seed
        ).split(documents)
        rows = []
        for split in folds:
            rows.append(
                f"| {split.fold} | {len(split.train_ids)} | {len(split.dev_ids)} | {len(split.test_ids)} |"
            )
        return (
            "## D5运行前检查\n\n"
            f"- 初始化：{self.initialization_source.upper()}（{self.config.pretraining.checkpoint}）\n"
            f"- 出口：{'保留D5分类头直接预测' if self.mode == 'direct' else '丢弃D5分类头，接D4式BiLSTM'}\n"
            "- D5在每个外层折内独立训练；测试文献不参与D5训练与早停。\n\n"
            "| 折 | 训练文献 | 开发文献 | 测试文献 |\n| --- | ---: | ---: | ---: |\n"
            + "\n".join(rows)
        )

    def run(self, output_dir: Path) -> dict[str, object]:
        started = time.perf_counter()
        documents = read_main_documents(self.config)
        folds = VolumeCrossValidator(
            self.config.folds, self.config.dev_ratio, self.config.seed
        ).split(documents)
        checkpoint = self.config.pretraining.checkpoint
        if checkpoint is None:
            raise ValueError("D5需要D1或D2 checkpoint")
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "实验：D5（%s），初始化=%s，grade=%d",
            "保留头直接分类" if self.mode == "direct" else "丢弃头后接BiLSTM",
            self.initialization_source.upper(),
            self.grade,
        )
        fold_results = [
            self._run_fold(documents, split, checkpoint, output_dir)
            for split in folds
        ]
        result: dict[str, object] = {
            "experiment": f"d5_{self.mode}",
            "display_name": (
                "D5保留层级分类头直接预测"
                if self.mode == "direct"
                else "D5丢弃分类头＋BiLSTM"
            )
            + f"（从{self.initialization_source.upper()}初始化）",
            "model_name": f"d5_{self.mode}",
            "conditions": ["direct"] if self.mode == "direct" else ["oracle", "predicted"],
            "evaluation_grade": self.grade,
            "target_labels": list(
                self.config.punctuation.sentence_pause
                + self.config.punctuation.intra_sentence_pause
            ),
            "raw_target_labels": list(
                self.config.punctuation.sentence_pause
                + self.config.punctuation.intra_sentence_pause
            ),
            "stage_display_names": (
                {"d5": "D5层级标点直接输出"}
                if self.mode == "direct"
                else {"position": "停顿标点位置", "pause_type": "具体停顿标点"}
            ),
            "folds": fold_results,
        }
        (output_dir / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("D5五折完成，总耗时%.1f秒：%s", time.perf_counter() - started, output_dir)
        return result

    def _run_fold(
        self,
        documents: list[PreparedDocument],
        split: FoldSplit,
        checkpoint: Path,
        output_dir: Path,
    ) -> dict[str, object]:
        partitions = {
            "train": select_documents(documents, split.train_ids),
            "dev": select_documents(documents, split.dev_ids),
            "test": select_documents(documents, split.test_ids),
        }
        fold_path = output_dir / f"fold_{split.fold}"
        fold_path.mkdir(parents=True, exist_ok=True)
        mode_directory = output_dir.parent if output_dir.name == "grade_2" else output_dir
        shared_fold_path = (
            mode_directory.parent / "pretraining" / f"fold_{split.fold}"
        )
        trainer = D5FoldTrainer(self.config, checkpoint, split.fold)
        reused = trainer.load_if_compatible(
            shared_fold_path, partitions["train"], partitions["dev"]
        )
        if not reused:
            trainer.fit(
                partitions["train"], partitions["dev"], shared_fold_path
            )
        if self.mode == "direct":
            return self._direct_fold(trainer, partitions, split, fold_path)
        return self._bilstm_fold(
            trainer, partitions, split, fold_path, shared_fold_path
        )

    def _direct_fold(
        self,
        trainer: D5FoldTrainer,
        partitions: dict[str, list[PreparedDocument]],
        split: FoldSplit,
        fold_path: Path,
    ) -> dict[str, object]:
        gold, _ = _gold_maps(partitions["test"], self.config)
        predicted = trainer.predict(partitions["test"])
        final = evaluate_final_by_domain(
            gold, predicted, partitions["test"], self.config, self.grade
        )
        overall = final["overall"]
        LOGGER.info(
            "[%d/%d][D5直接输出][最终] 位置F1=%.4f，Micro-F1=%.4f，Macro-F1=%.4f，句界F1=%.4f，Accuracy=%.4f",
            split.fold,
            self.config.folds,
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
            "stage_metrics": {"direct": {"d5": final}},
            "final_metrics": {"direct": final},
            "model_metadata": {"d5": trainer.metadata()},
        }
        (fold_path / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def _bilstm_fold(
        self,
        trainer: D5FoldTrainer,
        partitions: dict[str, list[PreparedDocument]],
        split: FoldSplit,
        fold_path: Path,
        shared_fold_path: Path,
    ) -> dict[str, object]:
        encoder_checkpoint = shared_fold_path / "best_encoder.pt"
        all_documents = partitions["train"] + partitions["dev"] + partitions["test"]
        gold_pause, gold_position = _gold_maps(all_documents, self.config)
        position_model = TangutEncoderSequenceTagger(
            encoder_checkpoint,
            trainer.pause_labels,
            self.config.neural,
            self.config.pretraining,
            self.config.max_sequence_length,
            self.config.seed + split.fold,
            head_type="bilstm",
        )
        position_model.fit(
            make_chunks(_annotate(partitions["train"], gold_position), self.config.max_sequence_length),
            make_chunks(_annotate(partitions["dev"], gold_position), self.config.max_sequence_length),
        )
        test_position_chunks = make_chunks(
            _annotate(partitions["test"], gold_position), self.config.max_sequence_length
        )
        predicted_position = join_chunk_labels(
            test_position_chunks, position_model.predict(test_position_chunks)
        )
        position_model.save(fold_path / "position_bilstm.pt")

        type_model = TangutEncoderSequenceTagger(
            encoder_checkpoint,
            trainer.pause_labels,
            self.config.neural,
            self.config.pretraining,
            self.config.max_sequence_length,
            self.config.seed + split.fold,
            head_type="bilstm",
        )
        type_model.fit(
            make_chunks(
                _annotate(partitions["train"], gold_pause, gold_position),
                self.config.max_sequence_length,
            ),
            make_chunks(
                _annotate(partitions["dev"], gold_pause, gold_position),
                self.config.max_sequence_length,
            ),
        )
        type_model.save(fold_path / "pause_type_bilstm.pt")
        stage_metrics: dict[str, dict[str, object]] = {"oracle": {}, "predicted": {}}
        final_metrics: dict[str, object] = {}
        test_gold = {
            document.document_id: gold_pause[document.document_id]
            for document in partitions["test"]
        }
        position_gold_test = {
            document.document_id: gold_position[document.document_id]
            for document in partitions["test"]
        }
        position_metrics = evaluate_by_domain(
            position_gold_test,
            predicted_position,
            partitions["test"],
            class_labels=frozenset({POSITION_LABEL}),
            sentence_marks=frozenset(),
        )
        for condition, channel in (
            ("oracle", position_gold_test),
            ("predicted", predicted_position),
        ):
            chunks = make_chunks(
                _annotate(partitions["test"], test_gold, channel),
                self.config.max_sequence_length,
            )
            prediction = join_chunk_labels(chunks, type_model.predict(chunks))
            prediction = {
                document_id: [
                    label if mask != OUTSIDE else OUTSIDE
                    for label, mask in zip(labels, channel[document_id])
                ]
                for document_id, labels in prediction.items()
            }
            stage_metrics[condition]["position"] = position_metrics
            stage_metrics[condition]["pause_type"] = evaluate_by_domain(
                test_gold,
                prediction,
                partitions["test"],
                class_labels=frozenset(trainer.pause_labels),
                sentence_marks=frozenset(self.config.punctuation.sentence_pause),
            )
            final_metrics[condition] = evaluate_final_by_domain(
                test_gold,
                prediction,
                partitions["test"],
                self.config,
                self.grade,
            )
            overall = final_metrics[condition]["overall"]
            LOGGER.info(
                "[%d/%d][D5＋BiLSTM][%s] 位置F1=%.4f，Micro-F1=%.4f，Macro-F1=%.4f，句界F1=%.4f，Accuracy=%.4f",
                split.fold,
                self.config.folds,
                condition,
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
            "stage_metrics": stage_metrics,
            "final_metrics": final_metrics,
            "model_metadata": {
                "d5_pretraining": trainer.metadata(),
                "position": position_model.metadata(),
                "pause_type": type_model.metadata(),
            },
        }
        (fold_path / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
