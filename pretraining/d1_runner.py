from __future__ import annotations

import copy
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import ExperimentConfig, PretrainingConfig
from data.corpus import PreparedDocument
from experiments.runner import read_main_documents
from models.neural import NeuralSequenceTagger
from models.tangut_encoder import (
    PAD_TOKEN,
    build_tangut_encoder,
    checkpoint_model_config,
)
from reporting import markdown_table

from .data import (
    IGNORE_INDEX,
    MLMDataset,
    MLMSequence,
    build_vocabulary,
    collate_mlm,
    make_mlm_sequences,
    split_pretraining_documents,
)


LOGGER = logging.getLogger(__name__)
DOMAIN_NAMES = {"jingshu": "经书", "shisu": "世俗", "unknown": "未知"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(config: PretrainingConfig) -> tuple[torch.device, str]:
    """沿用B/C神经模型的空闲GPU检测与选择规则。"""

    value = config.device.lower()
    if value == "cpu":
        return torch.device("cpu"), "配置明确指定CPU"
    statuses = NeuralSequenceTagger._visible_cuda_devices()
    if not statuses:
        if value == "auto":
            reason = "PyTorch未检测到可用CUDA，自动回退CPU"
            LOGGER.warning("设备选择：%s", reason)
            return torch.device("cpu"), reason
        raise RuntimeError("配置要求CUDA，但当前PyTorch检测不到可用GPU")
    LOGGER.info(
        "CUDA候选状态：%s",
        "；".join(NeuralSequenceTagger._cuda_status_text(item) for item in statuses),
    )
    if value.startswith("cuda:"):
        index = int(value.split(":", 1)[1])
        selected = next((item for item in statuses if item.index == index), None)
        if selected is None:
            raise RuntimeError(f"指定的{value}不是当前PyTorch可见设备")
        reason = "手动指定 " + NeuralSequenceTagger._cuda_status_text(selected)
        return torch.device(value), reason
    if value == "cuda":
        selected = NeuralSequenceTagger._best_cuda_device(statuses)
        reason = (
            "明确要求GPU，选择当前空闲显存最多的 "
            + NeuralSequenceTagger._cuda_status_text(selected)
        )
        return torch.device(f"cuda:{selected.index}"), reason
    eligible = [
        item
        for item in statuses
        if item.free_memory_gb >= config.cuda_min_free_memory_gb
        and (
            item.utilization is None
            or item.utilization <= config.cuda_max_utilization
        )
    ]
    if not eligible:
        observed = "；".join(
            NeuralSequenceTagger._cuda_status_text(item) for item in statuses
        )
        raise RuntimeError(
            "检测到CUDA设备，但没有设备满足空闲条件："
            f"利用率≤{config.cuda_max_utilization}%且空闲显存≥"
            f"{config.cuda_min_free_memory_gb:.2f} GiB。当前状态：{observed}。"
        )
    selected = NeuralSequenceTagger._best_cuda_device(eligible)
    suffix = (
        "（利用率读取失败，本次仅按空闲显存判断）"
        if selected.utilization is None
        else ""
    )
    reason = (
        "自动选择空闲GPU "
        + NeuralSequenceTagger._cuda_status_text(selected)
        + suffix
    )
    return torch.device(f"cuda:{selected.index}"), reason


def _document_counts(documents: list[PreparedDocument]) -> Counter[str]:
    return Counter(document.domain for document in documents)


def _sequence_counts(sequences: list[MLMSequence]) -> Counter[str]:
    return Counter(sequence.domain for sequence in sequences)


def _character_counts(sequences: list[MLMSequence]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts[sequence.domain] += len(sequence.tokens)
    return counts


def _data_rows(
    train_documents: list[PreparedDocument],
    validation_documents: list[PreparedDocument],
    train_sequences: list[MLMSequence],
    validation_sequences: list[MLMSequence],
) -> list[list[object]]:
    train_docs = _document_counts(train_documents)
    valid_docs = _document_counts(validation_documents)
    train_chunks = _sequence_counts(train_sequences)
    valid_chunks = _sequence_counts(validation_sequences)
    train_chars = _character_counts(train_sequences)
    valid_chars = _character_counts(validation_sequences)
    rows: list[list[object]] = []
    for domain in ("jingshu", "shisu", "unknown"):
        if not train_docs[domain] and not valid_docs[domain]:
            continue
        rows.append(
            [
                DOMAIN_NAMES.get(domain, domain),
                train_docs[domain],
                valid_docs[domain],
                train_chunks[domain],
                valid_chunks[domain],
                train_chars[domain],
                valid_chars[domain],
            ]
        )
    rows.append(
        [
            "总体",
            len(train_documents),
            len(validation_documents),
            len(train_sequences),
            len(validation_sequences),
            sum(len(item.tokens) for item in train_sequences),
            sum(len(item.tokens) for item in validation_sequences),
        ]
    )
    return rows


def prepare_d1_data(config: ExperimentConfig) -> dict[str, object]:
    documents = read_main_documents(config)
    train_documents, validation_documents = split_pretraining_documents(
        documents, config.pretraining.validation_ratio, config.seed
    )
    vocabulary = build_vocabulary(documents)
    train_sequences = make_mlm_sequences(
        train_documents,
        config.pretraining.max_sequence_length,
        config.pretraining.min_sequence_length,
    )
    validation_sequences = make_mlm_sequences(
        validation_documents,
        config.pretraining.max_sequence_length,
        config.pretraining.min_sequence_length,
    )
    if not train_sequences or not validation_sequences:
        raise ValueError("过滤短序列后MLM训练集或开发集为空")
    return {
        "documents": documents,
        "train_documents": train_documents,
        "validation_documents": validation_documents,
        "vocabulary": vocabulary,
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
    }


def render_d1_inspection(config: ExperimentConfig, prepared: dict[str, object]) -> str:
    pretrain = config.pretraining
    train_documents = prepared["train_documents"]
    validation_documents = prepared["validation_documents"]
    train_sequences = prepared["train_sequences"]
    validation_sequences = prepared["validation_sequences"]
    vocabulary = prepared["vocabulary"]
    rows = _data_rows(
        train_documents,  # type: ignore[arg-type]
        validation_documents,  # type: ignore[arg-type]
        train_sequences,  # type: ignore[arg-type]
        validation_sequences,  # type: ignore[arg-type]
    )
    model_rows = [
        ["字符词表", len(vocabulary)],  # type: ignore[arg-type]
        ["Transformer层数", pretrain.layers],
        ["隐层维度", pretrain.embedding_dim],
        ["注意力头数", pretrain.heads],
        ["前馈层维度", pretrain.ff_dim],
        ["最大序列长度", pretrain.max_sequence_length],
        ["遮盖比例", f"{pretrain.mask_ratio:.1%}"],
        ["片段遮盖概率", f"{pretrain.span_mask_probability:.1%}"],
        ["片段长度", "、".join(map(str, pretrain.span_lengths))],
        ["最大训练步数", pretrain.max_steps],
        ["开发集评估间隔", f"每{pretrain.eval_interval}步"],
    ]
    return (
        "## 主实验D1：上下文MLM预训练检查\n\n"
        + markdown_table(
            [
                "领域",
                "训练文献",
                "开发文献",
                "训练块",
                "开发块",
                "训练字符",
                "开发字符",
            ],
            rows,
        )
        + "\n\n### 模型与遮盖配置\n\n"
        + markdown_table(["配置项", "值"], model_rows)
        + "\n\n> 输入由当前配置中的语料生成：七种停顿标点和结构标点均不作为token，"
        "`□`、`@`、`…`保留；TAB是不可跨越的硬边界。开发集按文献划分，只用于MLM早停。"
    )


def _metric_template() -> dict[str, float | int]:
    return {"loss_sum": 0.0, "masked": 0, "top1": 0, "top5": 0}


def _finalize_metric(values: dict[str, float | int]) -> dict[str, float | int]:
    masked = int(values["masked"])
    if masked == 0:
        return {
            "loss": float("inf"),
            "perplexity": float("inf"),
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "masked_tokens": 0,
        }
    loss = float(values["loss_sum"]) / masked
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "top1_accuracy": int(values["top1"]) / masked,
        "top5_accuracy": int(values["top5"]) / masked,
        "masked_tokens": masked,
    }


def evaluate_mlm(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    totals: dict[str, dict[str, float | int]] = defaultdict(_metric_template)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)  # type: ignore[union-attr]
            labels = batch["labels"].to(device)  # type: ignore[union-attr]
            padding_mask = batch["padding_mask"].to(device)  # type: ignore[union-attr]
            logits, _ = model(input_ids, padding_mask)
            domains: list[str] = batch["domains"]  # type: ignore[assignment]
            for row, domain in enumerate(domains):
                valid = labels[row] != IGNORE_INDEX
                count = int(valid.sum().item())
                if count == 0:
                    continue
                row_logits = logits[row, valid]
                row_labels = labels[row, valid]
                loss_sum = float(
                    F.cross_entropy(row_logits, row_labels, reduction="sum").item()
                )
                predictions = row_logits.argmax(dim=-1)
                top1 = int((predictions == row_labels).sum().item())
                top_k = min(5, row_logits.shape[-1])
                top_indices = row_logits.topk(top_k, dim=-1).indices
                top5 = int(
                    (top_indices == row_labels.unsqueeze(1)).any(dim=1).sum().item()
                )
                for key in ("overall", domain):
                    totals[key]["loss_sum"] = float(totals[key]["loss_sum"]) + loss_sum
                    totals[key]["masked"] = int(totals[key]["masked"]) + count
                    totals[key]["top1"] = int(totals[key]["top1"]) + top1
                    totals[key]["top5"] = int(totals[key]["top5"]) + top5
    return {key: _finalize_metric(value) for key, value in totals.items()}


def _linear_schedule(max_steps: int, warmup_ratio: float):
    warmup_steps = max(1, round(max_steps * warmup_ratio))

    def factor(step: int) -> float:
        current = step + 1
        if current <= warmup_steps:
            return current / warmup_steps
        remaining = max_steps - current
        return max(0.0, remaining / max(1, max_steps - warmup_steps))

    return factor, warmup_steps


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _serializable_pretraining_config(config: PretrainingConfig) -> dict[str, object]:
    values: dict[str, object] = asdict(config)
    if config.checkpoint is not None:
        values["checkpoint"] = str(config.checkpoint)
    return values


def _checkpoint(
    model: nn.Module,
    vocabulary: dict[str, int],
    config: ExperimentConfig,
    step: int,
    metrics: dict[str, dict[str, float | int]],
    train_documents: list[PreparedDocument],
    validation_documents: list[PreparedDocument],
) -> dict[str, object]:
    return {
        "format": "tangut_encoder",
        "format_version": 1,
        "stage": "d1_context_mlm",
        "model_state_dict": _cpu_state_dict(model),
        "vocabulary": vocabulary,
        "model_config": checkpoint_model_config(config.pretraining),
        "pretraining_config": asdict(config.pretraining),
        "seed": config.seed,
        "step": step,
        "validation_metrics": metrics,
        "train_document_ids": [item.document_id for item in train_documents],
        "validation_document_ids": [
            item.document_id for item in validation_documents
        ],
        "data_sources": [str(source.path) for source in config.data.sources],
    }


def render_d1_summary(result: dict[str, object]) -> str:
    metrics: dict[str, dict[str, float | int]] = result["validation_metrics"]  # type: ignore[assignment]
    rows = []
    for domain in ("overall", "jingshu", "shisu", "unknown"):
        if domain not in metrics:
            continue
        value = metrics[domain]
        rows.append(
            [
                "总体" if domain == "overall" else DOMAIN_NAMES.get(domain, domain),
                f"{float(value['loss']):.4f}",
                f"{float(value['perplexity']):.4f}",
                f"{float(value['top1_accuracy']):.4f}",
                f"{float(value['top5_accuracy']):.4f}",
                int(value["masked_tokens"]),
            ]
        )
    return (
        "## D1上下文MLM：最佳开发集结果\n\n"
        + markdown_table(
            ["评价集", "MLM Loss", "困惑度", "Top-1", "Top-5", "遮盖字符"],
            rows,
        )
        + f"\n\n最佳checkpoint：第{result['best_step']}步；参数量："
        f"{int(result['parameter_count']):,}。Top-1/Top-5只统计被选作预测目标的位置。"
    )


def render_d1_report(result: dict[str, object]) -> str:
    history: list[dict[str, object]] = result["history"]  # type: ignore[assignment]
    history_rows = [
        [
            item["step"],
            f"{float(item['train_loss']):.4f}",
            f"{float(item['validation_loss']):.4f}",
            f"{float(item['validation_top1']):.4f}",
            f"{float(item['validation_top5']):.4f}",
            f"{float(item['learning_rate']):.2e}",
        ]
        for item in history
    ]
    return (
        render_d1_summary(result)
        + "\n\n### 数据划分\n\n"
        + markdown_table(
            [
                "领域",
                "训练文献",
                "开发文献",
                "训练块",
                "开发块",
                "训练字符",
                "开发字符",
            ],
            result["data_rows"],  # type: ignore[arg-type]
        )
        + "\n\n### 开发集评估轨迹\n\n"
        + markdown_table(
            ["步数", "区间训练Loss", "开发Loss", "开发Top-1", "开发Top-5", "学习率"],
            history_rows,
        )
        + "\n\n> 开发集遮盖模式固定，因而不同训练步的Loss和准确率可以直接比较。"
        "普通未遮盖位置不进入MLM准确率。"
    )


class D1ContextMLMRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def inspect(self) -> str:
        return render_d1_inspection(self.config, prepare_d1_data(self.config))

    def run(self, output_directory: Path) -> dict[str, object]:
        _set_seed(self.config.seed)
        prepared = prepare_d1_data(self.config)
        train_documents: list[PreparedDocument] = prepared["train_documents"]  # type: ignore[assignment]
        validation_documents: list[PreparedDocument] = prepared["validation_documents"]  # type: ignore[assignment]
        vocabulary: dict[str, int] = prepared["vocabulary"]  # type: ignore[assignment]
        train_sequences: list[MLMSequence] = prepared["train_sequences"]  # type: ignore[assignment]
        validation_sequences: list[MLMSequence] = prepared["validation_sequences"]  # type: ignore[assignment]
        cfg = self.config.pretraining

        output_directory.mkdir(parents=True, exist_ok=True)
        LOGGER.info("实验：D1 TangutEncoder上下文MLM预训练")
        LOGGER.info(
            "语料：%d部文献，训练/开发=%d/%d；训练/开发字符=%d/%d",
            len(prepared["documents"]),  # type: ignore[arg-type]
            len(train_documents),
            len(validation_documents),
            sum(len(item.tokens) for item in train_sequences),
            sum(len(item.tokens) for item in validation_sequences),
        )
        LOGGER.debug(
            "D1开发文献：%s",
            ", ".join(item.document_id for item in validation_documents),
        )
        LOGGER.info(
            "遮盖：目标比例=%.1f%%，片段概率=%.1f%%，片段长度=%s，替换策略=%.0f/%.0f/%.0f",
            cfg.mask_ratio * 100,
            cfg.span_mask_probability * 100,
            "/".join(map(str, cfg.span_lengths)),
            cfg.mask_replace_probability * 100,
            cfg.random_replace_probability * 100,
            (1 - cfg.mask_replace_probability - cfg.random_replace_probability) * 100,
        )

        train_dataset = MLMDataset(
            train_sequences,
            vocabulary,
            cfg.mask_ratio,
            cfg.span_mask_probability,
            cfg.span_lengths,
            cfg.mask_replace_probability,
            cfg.random_replace_probability,
            self.config.seed,
            dynamic=True,
        )
        validation_dataset = MLMDataset(
            validation_sequences,
            vocabulary,
            cfg.mask_ratio,
            cfg.span_mask_probability,
            cfg.span_lengths,
            cfg.mask_replace_probability,
            cfg.random_replace_probability,
            self.config.seed + 10_000,
            dynamic=False,
        )
        collate = partial(collate_mlm, pad_id=vocabulary[PAD_TOKEN])
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        device, device_selection = _resolve_device(cfg)
        LOGGER.info("设备选择：%s", device_selection)
        model = build_tangut_encoder(
            len(vocabulary), cfg, vocabulary[PAD_TOKEN]
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        LOGGER.info(
            "TangutEncoder：词表=%d，参数=%s，层数=%d，hidden=%d，heads=%d，最大长度=%d",
            len(vocabulary),
            f"{parameter_count:,}",
            cfg.layers,
            cfg.embedding_dim,
            cfg.heads,
            cfg.max_sequence_length,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        schedule_factor, warmup_steps = _linear_schedule(
            cfg.max_steps, cfg.warmup_ratio
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_factor)
        LOGGER.info(
            "训练：batch=%d，max_steps=%d，lr=%g，warmup=%d步，eval=%d步，patience=%d",
            cfg.batch_size,
            cfg.max_steps,
            cfg.learning_rate,
            warmup_steps,
            cfg.eval_interval,
            cfg.patience,
        )

        best_loss = float("inf")
        best_step = 0
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, dict[str, float | int]] | None = None
        history: list[dict[str, object]] = []
        stale_evaluations = 0
        global_step = 0
        epoch = 0
        interval_loss_sum = 0.0
        interval_masked = 0
        stopped_early = False

        while global_step < cfg.max_steps and not stopped_early:
            epoch += 1
            train_dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(self.config.seed + epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                collate_fn=collate,
                generator=generator,
            )
            for batch in train_loader:
                if global_step >= cfg.max_steps:
                    break
                model.train()
                input_ids = batch["input_ids"].to(device)  # type: ignore[union-attr]
                labels = batch["labels"].to(device)  # type: ignore[union-attr]
                padding_mask = batch["padding_mask"].to(device)  # type: ignore[union-attr]
                optimizer.zero_grad()
                logits, _ = model(input_ids, padding_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1
                masked = int((labels != IGNORE_INDEX).sum().item())
                interval_loss_sum += float(loss.item()) * masked
                interval_masked += masked

                if global_step % cfg.log_interval == 0:
                    LOGGER.info(
                        "[D1] step %d/%d，epoch=%d，区间train_loss=%.6f，lr=%.2e",
                        global_step,
                        cfg.max_steps,
                        epoch,
                        interval_loss_sum / max(interval_masked, 1),
                        optimizer.param_groups[0]["lr"],
                    )

                should_evaluate = (
                    global_step % cfg.eval_interval == 0
                    or global_step == cfg.max_steps
                )
                if not should_evaluate:
                    continue
                train_interval_loss = interval_loss_sum / max(interval_masked, 1)
                metrics = evaluate_mlm(model, validation_loader, device)
                overall = metrics["overall"]
                history.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_interval_loss,
                        "validation_loss": overall["loss"],
                        "validation_perplexity": overall["perplexity"],
                        "validation_top1": overall["top1_accuracy"],
                        "validation_top5": overall["top5_accuracy"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                interval_loss_sum = 0.0
                interval_masked = 0
                LOGGER.info(
                    "[D1][开发集] step=%d，loss=%.6f，ppl=%.4f，Top-1=%.4f，Top-5=%.4f",
                    global_step,
                    float(overall["loss"]),
                    float(overall["perplexity"]),
                    float(overall["top1_accuracy"]),
                    float(overall["top5_accuracy"]),
                )
                if float(overall["loss"]) < best_loss - cfg.min_delta:
                    best_loss = float(overall["loss"])
                    best_step = global_step
                    best_state = _cpu_state_dict(model)
                    best_metrics = copy.deepcopy(metrics)
                    stale_evaluations = 0
                    torch.save(
                        _checkpoint(
                            model,
                            vocabulary,
                            self.config,
                            best_step,
                            metrics,
                            train_documents,
                            validation_documents,
                        ),
                        output_directory / "best_model.pt",
                    )
                    LOGGER.info("[D1] 开发集loss达到新最佳，保存best_model.pt")
                else:
                    stale_evaluations += 1
                    LOGGER.info(
                        "[D1] 开发集改善不足min_delta=%g，早停计数=%d/%d",
                        cfg.min_delta,
                        stale_evaluations,
                        cfg.patience,
                    )
                    if stale_evaluations >= cfg.patience:
                        stopped_early = True
                        LOGGER.info("[D1] 触发早停；最佳步数=%d", best_step)
                        break

        # max_steps不是eval_interval整数倍且未触发评估时，补一次固定开发集评测。
        if best_state is None or (global_step and global_step % cfg.eval_interval):
            metrics = evaluate_mlm(model, validation_loader, device)
            overall = metrics["overall"]
            if float(overall["loss"]) < best_loss - cfg.min_delta or best_state is None:
                best_loss = float(overall["loss"])
                best_step = global_step
                best_state = _cpu_state_dict(model)
                best_metrics = copy.deepcopy(metrics)
                torch.save(
                    _checkpoint(
                        model,
                        vocabulary,
                        self.config,
                        best_step,
                        metrics,
                        train_documents,
                        validation_documents,
                    ),
                    output_directory / "best_model.pt",
                )
        if best_state is None or best_metrics is None:
            raise RuntimeError("D1训练未产生有效checkpoint")

        torch.save(
            _checkpoint(
                model,
                vocabulary,
                self.config,
                global_step,
                evaluate_mlm(model, validation_loader, device),
                train_documents,
                validation_documents,
            ),
            output_directory / "last_model.pt",
        )
        model.load_state_dict(best_state)
        final_metrics = evaluate_mlm(model, validation_loader, device)
        result: dict[str, object] = {
            "experiment": "d1",
            "display_name": "D1 TangutEncoder上下文MLM预训练",
            "best_step": best_step,
            "finished_step": global_step,
            "stopped_early": stopped_early,
            "parameter_count": parameter_count,
            "vocabulary_size": len(vocabulary),
            "device": str(device),
            "device_selection": device_selection,
            "validation_metrics": final_metrics,
            "history": history,
            "data_rows": _data_rows(
                train_documents,
                validation_documents,
                train_sequences,
                validation_sequences,
            ),
            "train_document_ids": [item.document_id for item in train_documents],
            "validation_document_ids": [
                item.document_id for item in validation_documents
            ],
            "config": _serializable_pretraining_config(cfg),
        }
        (output_directory / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_directory / "results.md").write_text(
            render_d1_report(result) + "\n", encoding="utf-8"
        )
        ordered_vocabulary = sorted(vocabulary.items(), key=lambda item: item[1])
        (output_directory / "vocabulary.txt").write_text(
            "\n".join(f"{index}\t{token}" for token, index in ordered_vocabulary)
            + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "D1完成：最佳步数=%d，开发集loss=%.6f，checkpoint=%s",
            best_step,
            float(final_metrics["overall"]["loss"]),
            output_directory / "best_model.pt",
        )
        return result
