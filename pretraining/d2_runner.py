from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from data.corpus import PreparedDocument
from experiments.runner import read_main_documents
from models.tangut_encoder import PAD_TOKEN, load_tangut_encoder_checkpoint
from reporting import markdown_table

from .d1_runner import (
    DOMAIN_NAMES,
    _cpu_state_dict,
    _data_rows,
    _linear_schedule,
    _resolve_device,
    _set_seed,
    evaluate_mlm,
)
from .d2_data import WordAwareMLMDataset, collate_word_mlm
from .data import IGNORE_INDEX, MLMDataset, MLMSequence, collate_mlm, make_mlm_sequences, split_pretraining_documents
from .word_candidates import WordCandidate, build_word_candidates, find_candidate_occurrences


LOGGER = logging.getLogger(__name__)


class WordSpanScorer(nn.Module):
    """以首字、末字、均值、最大池化和长度表示判断候选片段凝固度。"""

    def __init__(self, hidden_dim: int, max_span_length: int, dropout: float) -> None:
        super().__init__()
        length_dim = min(32, max(8, hidden_dim // 12))
        self.length_embedding = nn.Embedding(max_span_length + 1, length_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 4 + length_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score(self, hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        scores: list[torch.Tensor] = []
        maximum_length = self.length_embedding.num_embeddings - 1
        for row in range(hidden.shape[0]):
            row_scores: list[torch.Tensor] = []
            for start_value, end_value in spans[row]:
                start = int(start_value.item())
                end = int(end_value.item())
                piece = hidden[row, start:end]
                length = max(1, min(end - start, maximum_length))
                vector = torch.cat(
                    (
                        piece[0],
                        piece[-1],
                        piece.mean(dim=0),
                        piece.max(dim=0).values,
                        self.length_embedding(
                            torch.tensor(length, device=hidden.device)
                        ),
                    )
                )
                row_scores.append(self.network(vector).squeeze(-1))
            scores.append(torch.stack(row_scores))
        return torch.stack(scores)

    def forward(
        self,
        hidden: torch.Tensor,
        positive_spans: torch.Tensor,
        negative_spans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = self.score(hidden, positive_spans.unsqueeze(1)).squeeze(1)
        negative = self.score(hidden, negative_spans)
        return positive, negative


def _ranking_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    if not bool(valid.any()):
        return positive.sum() * 0.0
    pair_loss = F.softplus(-(positive.unsqueeze(1) - negative)).mean(dim=1)
    weights = confidence[valid]
    return (pair_loss[valid] * weights).sum() / weights.sum().clamp_min(1e-8)


def _prepare_data(config: ExperimentConfig, mode: str) -> dict[str, object]:
    checkpoint_path = config.word_pretraining.initial_checkpoint
    if checkpoint_path is None:
        raise ValueError("D2需要word_pretraining.initial_checkpoint指向D1 best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到D1初始化checkpoint：{checkpoint_path}")
    _, vocabulary, parent_checkpoint = load_tangut_encoder_checkpoint(checkpoint_path)
    documents = read_main_documents(config)
    train_documents, validation_documents = split_pretraining_documents(
        documents, config.pretraining.validation_ratio, config.seed
    )
    expected_train = parent_checkpoint.get("train_document_ids")
    expected_validation = parent_checkpoint.get("validation_document_ids")
    actual_train = [item.document_id for item in train_documents]
    actual_validation = [item.document_id for item in validation_documents]
    if expected_train is not None and list(expected_train) != actual_train:
        raise ValueError("D2训练文献划分与D1 checkpoint不一致，请使用相同语料、比例和seed")
    if expected_validation is not None and list(expected_validation) != actual_validation:
        raise ValueError("D2开发文献划分与D1 checkpoint不一致，请使用相同语料、比例和seed")
    unknown = sorted(
        {token for document in documents for token in document.tokens if token not in vocabulary}
    )
    if unknown:
        raise ValueError(f"D2语料出现D1词表外字符，共{len(unknown)}种：{''.join(unknown[:20])}")
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
        raise ValueError("过滤短序列后D2训练集或开发集为空")

    candidates: dict[str, WordCandidate] = {}
    candidate_summary: dict[str, object] = {
        "candidate_types": 0,
        "candidate_occurrences": 0,
        "by_length": {},
        "by_source": {},
    }
    if mode != "control":
        candidates, candidate_summary = build_word_candidates(
            train_sequences, config.word_pretraining, mode
        )
        if not candidates:
            raise ValueError("D2未筛选出任何词语候选，请检查辞书路径和阈值")
    train_occurrences = find_candidate_occurrences(train_sequences, candidates)
    validation_occurrences = find_candidate_occurrences(validation_sequences, candidates)

    def coverage(
        sequences: list[MLMSequence], occurrences: list[tuple[object, ...]]
    ) -> tuple[float, float]:
        character_total = sum(len(item.tokens) for item in sequences)
        covered = 0
        overlapping = 0
        for sequence, sequence_occurrences in zip(sequences, occurrences):
            hit_counts = [0] * len(sequence.tokens)
            for occurrence in sequence_occurrences:
                for position in range(occurrence.start, occurrence.end):  # type: ignore[attr-defined]
                    hit_counts[position] += 1
            covered += sum(value > 0 for value in hit_counts)
            overlapping += sum(value > 1 for value in hit_counts)
        return (
            covered / max(character_total, 1),
            overlapping / max(character_total, 1),
        )

    train_coverage, train_overlap = coverage(train_sequences, train_occurrences)
    validation_coverage, validation_overlap = coverage(
        validation_sequences, validation_occurrences
    )
    candidate_summary["train_matched_occurrences"] = sum(map(len, train_occurrences))
    candidate_summary["validation_matched_occurrences"] = sum(
        map(len, validation_occurrences)
    )
    candidate_summary["train_sequences_with_candidate"] = sum(
        bool(items) for items in train_occurrences
    )
    candidate_summary["validation_sequences_with_candidate"] = sum(
        bool(items) for items in validation_occurrences
    )
    candidate_summary["train_character_coverage"] = train_coverage
    candidate_summary["validation_character_coverage"] = validation_coverage
    candidate_summary["train_overlap_character_rate"] = train_overlap
    candidate_summary["validation_overlap_character_rate"] = validation_overlap
    return {
        "documents": documents,
        "train_documents": train_documents,
        "validation_documents": validation_documents,
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "vocabulary": vocabulary,
        "parent_checkpoint": parent_checkpoint,
        "candidates": candidates,
        "candidate_summary": candidate_summary,
        "train_occurrences": train_occurrences,
        "validation_occurrences": validation_occurrences,
    }


def _make_word_dataset(
    sequences: list[MLMSequence],
    vocabulary: dict[str, int],
    config: ExperimentConfig,
    occurrences: list[tuple[object, ...]],
    candidates: dict[str, WordCandidate],
    whole_word_probability: float,
    dynamic: bool,
    seed: int,
    enable_ranking: bool,
) -> WordAwareMLMDataset:
    cfg = config.pretraining
    return WordAwareMLMDataset(
        sequences,
        vocabulary,
        cfg.mask_ratio,
        cfg.span_mask_probability,
        cfg.span_lengths,
        cfg.mask_replace_probability,
        cfg.random_replace_probability,
        seed,
        dynamic,
        occurrences=occurrences,  # type: ignore[arg-type]
        candidates=candidates,
        whole_word_probability=whole_word_probability,
        ranking_negatives=config.word_pretraining.ranking_negatives,
        enable_ranking=enable_ranking,
    )


def _evaluate_ranking(
    model: nn.Module,
    scorer: WordSpanScorer,
    loader: DataLoader[dict[str, object]],
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    scorer.eval()
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"loss_sum": 0.0, "pairs": 0, "correct": 0, "examples": 0}
    )
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)  # type: ignore[union-attr]
            padding_mask = batch["padding_mask"].to(device)  # type: ignore[union-attr]
            positive_spans = batch["positive_spans"].to(device)  # type: ignore[union-attr]
            negative_spans = batch["negative_spans"].to(device)  # type: ignore[union-attr]
            valid = batch["rank_valid"].to(device)  # type: ignore[union-attr]
            _, hidden = model(input_ids, padding_mask)
            positive, negative = scorer(hidden, positive_spans, negative_spans)
            domains: list[str] = batch["domains"]  # type: ignore[assignment]
            candidate_lengths = batch["candidate_lengths"]
            candidate_sources: list[str] = batch["candidate_sources"]  # type: ignore[assignment]
            for row, domain in enumerate(domains):
                if not bool(valid[row]):
                    continue
                losses = F.softplus(-(positive[row] - negative[row]))
                correct = int((positive[row] > negative[row]).sum().item())
                groups = (
                    "overall",
                    domain,
                    f"length_{int(candidate_lengths[row])}",
                    f"source_{candidate_sources[row]}",
                )
                for key in groups:
                    totals[key]["loss_sum"] = float(totals[key]["loss_sum"]) + float(losses.sum().item())
                    totals[key]["pairs"] = int(totals[key]["pairs"]) + negative.shape[1]
                    totals[key]["correct"] = int(totals[key]["correct"]) + correct
                    totals[key]["examples"] = int(totals[key]["examples"]) + 1
    result: dict[str, dict[str, float | int]] = {}
    for key, values in totals.items():
        pairs = int(values["pairs"])
        result[key] = {
            "loss": float(values["loss_sum"]) / max(pairs, 1),
            "accuracy": int(values["correct"]) / max(pairs, 1),
            "pairs": pairs,
            "examples": int(values["examples"]),
        }
    return result


def _composite_loss(
    ordinary: dict[str, dict[str, float | int]],
    word: dict[str, dict[str, float | int]] | None,
    ranking: dict[str, dict[str, float | int]] | None,
    ranking_weight: float,
) -> float:
    ordinary_loss = float(ordinary["overall"]["loss"])
    if word is None:
        return ordinary_loss
    mlm_loss = (ordinary_loss + float(word["overall"]["loss"])) / 2.0
    rank_loss = float((ranking or {}).get("overall", {}).get("loss", 0.0))
    return mlm_loss + ranking_weight * rank_loss


def _checkpoint(
    model: nn.Module,
    scorer: WordSpanScorer,
    vocabulary: dict[str, int],
    config: ExperimentConfig,
    mode: str,
    step: int,
    validation_metrics: dict[str, object],
    prepared: dict[str, object],
) -> dict[str, object]:
    parent = prepared["parent_checkpoint"]
    train_documents: list[PreparedDocument] = prepared["train_documents"]  # type: ignore[assignment]
    validation_documents: list[PreparedDocument] = prepared["validation_documents"]  # type: ignore[assignment]
    return {
        "format": "tangut_encoder",
        "format_version": 2,
        "stage": f"d2_{mode}",
        "model_state_dict": _cpu_state_dict(model),
        "word_head_state_dict": _cpu_state_dict(scorer),
        "vocabulary": vocabulary,
        # 编码器结构来自父checkpoint；即使用户后来调整配置文件，也不能把
        # 与实际权重不符的结构写进D2 checkpoint。
        "model_config": parent["model_config"],  # type: ignore[index]
        "pretraining_config": asdict(config.pretraining),
        "word_pretraining_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config.word_pretraining).items()
        },
        "seed": config.seed,
        "step": step,
        "parent_checkpoint": str(config.word_pretraining.initial_checkpoint),
        "parent_stage": parent.get("stage", "unknown"),  # type: ignore[union-attr]
        "validation_metrics": validation_metrics,
        "candidate_summary": prepared["candidate_summary"],
        "train_document_ids": [item.document_id for item in train_documents],
        "validation_document_ids": [item.document_id for item in validation_documents],
        "data_sources": [str(source.path) for source in config.data.sources],
    }


def _candidate_rows(summary: dict[str, object]) -> list[list[object]]:
    rows: list[list[object]] = []
    by_length: dict[str, int] = summary.get("by_length", {})  # type: ignore[assignment]
    for length, count in sorted(by_length.items(), key=lambda item: int(item[0])):
        rows.append([f"{length}字", count])
    rows.extend(
        [f"来源：{source}", count]
        for source, count in sorted((summary.get("by_source") or {}).items())  # type: ignore[union-attr]
    )
    return rows


def render_d2_inspection(config: ExperimentConfig, prepared: dict[str, object], mode: str) -> str:
    word = config.word_pretraining
    mode_name = {"control": "等步数普通MLM对照", "lexicon": "辞书候选", "fusion": "辞书＋统计融合"}[mode]
    rows = _data_rows(
        prepared["train_documents"],  # type: ignore[arg-type]
        prepared["validation_documents"],  # type: ignore[arg-type]
        prepared["train_sequences"],  # type: ignore[arg-type]
        prepared["validation_sequences"],  # type: ignore[arg-type]
    )
    summary: dict[str, object] = prepared["candidate_summary"]  # type: ignore[assignment]
    candidate_rows = _candidate_rows(summary) or [["候选（对照组不使用）", 0]]
    match_rows = [
        [
            "训练集",
            int(summary.get("train_matched_occurrences", 0)),
            f"{float(summary.get('train_character_coverage', 0.0)):.2%}",
            f"{float(summary.get('train_overlap_character_rate', 0.0)):.2%}",
        ],
        [
            "开发集",
            int(summary.get("validation_matched_occurrences", 0)),
            f"{float(summary.get('validation_character_coverage', 0.0)):.2%}",
            f"{float(summary.get('validation_overlap_character_rate', 0.0)):.2%}",
        ],
    ]
    return (
        f"## 主实验D2：{mode_name}检查\n\n"
        + markdown_table(["领域", "训练文献", "开发文献", "训练块", "开发块", "训练字符", "开发字符"], rows)
        + "\n\n### 候选词\n\n"
        + markdown_table(["候选分组", "类型数"], candidate_rows)
        + "\n\n### 候选匹配覆盖\n\n"
        + markdown_table(["数据集", "匹配次数", "字符覆盖率", "重叠字符率"], match_rows)
        + "\n\n### 训练配置\n\n"
        + markdown_table(
            ["配置项", "值"],
            [
                ["D1初始化", word.initial_checkpoint],
                ["普通/整词MLM", f"{1-word.whole_word_probability:.0%}/{word.whole_word_probability:.0%}" if mode != "control" else "100%/0%"],
                ["排序负例", word.ranking_negatives if mode != "control" else 0],
                ["排序损失权重", word.ranking_loss_weight if mode != "control" else 0],
                ["最大训练步数", word.max_steps],
                ["编码器/词语头学习率", f"{word.encoder_learning_rate:g}/{word.head_learning_rate:g}"],
            ],
        )
        + "\n\n> 候选频率、dPMI和左右熵只由训练文献计算；开发集只匹配训练阶段确定的候选。"
    )


def render_d2_summary(result: dict[str, object]) -> str:
    metrics: dict[str, object] = result["validation_metrics"]  # type: ignore[assignment]
    ordinary: dict[str, dict[str, float | int]] = metrics["ordinary_mlm"]  # type: ignore[assignment]
    word: dict[str, dict[str, float | int]] | None = metrics.get("word_mlm")  # type: ignore[assignment]
    ranking: dict[str, dict[str, float | int]] | None = metrics.get("ranking")  # type: ignore[assignment]
    rows: list[list[object]] = []
    for domain in ("overall", "jingshu", "shisu", "unknown"):
        if domain not in ordinary:
            continue
        label = "总体" if domain == "overall" else DOMAIN_NAMES.get(domain, domain)
        value = ordinary[domain]
        rows.append([label, "固定普通MLM", f"{float(value['loss']):.4f}", f"{float(value['perplexity']):.4f}", f"{float(value['top1_accuracy']):.4f}", f"{float(value['top5_accuracy']):.4f}", int(value["masked_tokens"])])
        if word is not None and domain in word:
            value = word[domain]
            rows.append([label, "整词MLM", f"{float(value['loss']):.4f}", f"{float(value['perplexity']):.4f}", f"{float(value['top1_accuracy']):.4f}", f"{float(value['top5_accuracy']):.4f}", int(value["masked_tokens"])])
    text = f"## {result['display_name']}：最佳开发集结果\n\n" + markdown_table(
        ["评价集", "目标", "Loss", "困惑度", "Top-1", "Top-5", "遮盖字符"], rows
    )
    if ranking:
        rank_rows = []
        rank_labels = {
            "overall": "总体",
            "jingshu": "经书",
            "shisu": "世俗",
            "unknown": "未知",
            "length_2": "2字候选",
            "length_3": "3字候选",
            "length_4": "4字候选",
            "source_lexicon_both": "双辞书候选",
            "source_lexicon_single": "单辞书候选",
            "source_statistics_only": "纯统计候选",
        }
        for key, label in rank_labels.items():
            if key not in ranking:
                continue
            value = ranking[key]
            rank_rows.append([label, f"{float(value['loss']):.4f}", f"{float(value['accuracy']):.4f}", int(value["examples"]), int(value["pairs"])])
        text += "\n\n### 词语候选排序\n\n" + markdown_table(
            ["评价集", "排序Loss", "成对准确率", "正例", "正负对"], rank_rows
        )
    return text + f"\n\n最佳checkpoint：第{result['best_step']}步；早停依据为开发集综合Loss={float(result['best_composite_loss']):.4f}。"


def render_d2_report(result: dict[str, object]) -> str:
    history: list[dict[str, object]] = result["history"]  # type: ignore[assignment]
    history_rows = [
        [item["step"], f"{float(item['train_loss']):.4f}", f"{float(item['ordinary_mlm_loss']):.4f}", "—" if item["word_mlm_loss"] is None else f"{float(item['word_mlm_loss']):.4f}", "—" if item["ranking_accuracy"] is None else f"{float(item['ranking_accuracy']):.4f}", f"{float(item['composite_loss']):.4f}"]
        for item in history
    ]
    initial: dict[str, object] = result["initial_validation_metrics"]  # type: ignore[assignment]
    final: dict[str, object] = result["validation_metrics"]  # type: ignore[assignment]
    initial_ordinary: dict[str, dict[str, float | int]] = initial["ordinary_mlm"]  # type: ignore[assignment]
    final_ordinary: dict[str, dict[str, float | int]] = final["ordinary_mlm"]  # type: ignore[assignment]
    comparison_rows: list[list[object]] = []
    for domain in ("overall", "jingshu", "shisu", "unknown"):
        if domain not in initial_ordinary or domain not in final_ordinary:
            continue
        before = initial_ordinary[domain]
        after = final_ordinary[domain]
        comparison_rows.append(
            [
                "总体" if domain == "overall" else DOMAIN_NAMES.get(domain, domain),
                f"{float(before['loss']):.4f}",
                f"{float(after['loss']):.4f}",
                f"{float(after['loss']) - float(before['loss']):+.4f}",
                f"{float(before['top1_accuracy']):.4f}",
                f"{float(after['top1_accuracy']):.4f}",
                f"{float(after['top1_accuracy']) - float(before['top1_accuracy']):+.4f}",
            ]
        )
    return (
        render_d2_summary(result)
        + "\n\n### D1初始化与D2最佳模型的固定MLM对比\n\n"
        + markdown_table(
            ["评价集", "训练前Loss", "训练后Loss", "Loss变化", "训练前Top-1", "训练后Top-1", "Top-1变化"],
            comparison_rows,
        )
        + "\n\n### 候选统计\n\n"
        + markdown_table(["统计项", "值"], [[key, json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value] for key, value in result["candidate_summary"].items()])  # type: ignore[union-attr]
        + "\n\n### 开发集评估轨迹\n\n"
        + markdown_table(["步数", "训练Loss", "普通MLM Loss", "整词MLM Loss", "排序准确率", "综合Loss"], history_rows)
        + "\n\n> 固定普通MLM使用与D1相同的开发集遮盖种子，适合直接比较；综合Loss只用于D2内部早停。"
    )


class D2WordPretrainingRunner:
    def __init__(self, config: ExperimentConfig, mode: str = "fusion") -> None:
        if mode not in {"control", "lexicon", "fusion"}:
            raise ValueError(f"未知D2模式：{mode}")
        self.config = config
        self.mode = mode

    def inspect(self) -> str:
        prepared = _prepare_data(self.config, self.mode)
        return render_d2_inspection(self.config, prepared, self.mode)

    def run(self, output_directory: Path) -> dict[str, object]:
        _set_seed(self.config.seed)
        prepared = _prepare_data(self.config, self.mode)
        train_documents: list[PreparedDocument] = prepared["train_documents"]  # type: ignore[assignment]
        validation_documents: list[PreparedDocument] = prepared["validation_documents"]  # type: ignore[assignment]
        train_sequences: list[MLMSequence] = prepared["train_sequences"]  # type: ignore[assignment]
        validation_sequences: list[MLMSequence] = prepared["validation_sequences"]  # type: ignore[assignment]
        vocabulary: dict[str, int] = prepared["vocabulary"]  # type: ignore[assignment]
        candidates: dict[str, WordCandidate] = prepared["candidates"]  # type: ignore[assignment]
        word_cfg = self.config.word_pretraining
        pre_cfg = self.config.pretraining
        enable_words = self.mode != "control"
        output_directory.mkdir(parents=True, exist_ok=True)
        LOGGER.info("实验：D2 %s", {"control": "等步数普通MLM对照", "lexicon": "辞书词语预训练", "fusion": "辞书＋统计融合词语预训练"}[self.mode])
        LOGGER.info("D1初始化：%s", word_cfg.initial_checkpoint)
        LOGGER.info("语料：训练/开发文献=%d/%d，训练/开发块=%d/%d", len(train_documents), len(validation_documents), len(train_sequences), len(validation_sequences))
        LOGGER.info("候选：类型=%d，训练匹配=%s，开发匹配=%s", len(candidates), f"{int(prepared['candidate_summary']['train_matched_occurrences']):,}", f"{int(prepared['candidate_summary']['validation_matched_occurrences']):,}")  # type: ignore[index]

        model, loaded_vocabulary, parent_checkpoint = load_tangut_encoder_checkpoint(word_cfg.initial_checkpoint)  # type: ignore[arg-type]
        if loaded_vocabulary != vocabulary:
            raise RuntimeError("D1 checkpoint词表在D2加载过程中发生变化")
        if model.max_sequence_length < pre_cfg.max_sequence_length:
            raise ValueError("D2序列上限超过D1位置嵌入上限")
        scorer = WordSpanScorer(model.output_dim, max(word_cfg.candidate_lengths), pre_cfg.dropout)
        device, device_selection = _resolve_device(pre_cfg)
        model = model.to(device)
        scorer = scorer.to(device)
        LOGGER.info("设备选择：%s", device_selection)

        train_dataset = _make_word_dataset(
            train_sequences, vocabulary, self.config,
            prepared["train_occurrences"], candidates,  # type: ignore[arg-type]
            word_cfg.whole_word_probability if enable_words else 0.0,
            True, self.config.seed, enable_words,
        )
        ordinary_validation_dataset = MLMDataset(
            validation_sequences, vocabulary, pre_cfg.mask_ratio,
            pre_cfg.span_mask_probability, pre_cfg.span_lengths,
            pre_cfg.mask_replace_probability, pre_cfg.random_replace_probability,
            self.config.seed + 10_000, False,
        )
        word_validation_dataset = _make_word_dataset(
            validation_sequences, vocabulary, self.config,
            prepared["validation_occurrences"], candidates,  # type: ignore[arg-type]
            1.0 if enable_words else 0.0, False,
            self.config.seed + 20_000, enable_words,
        )
        ordinary_loader = DataLoader(
            ordinary_validation_dataset, batch_size=word_cfg.batch_size,
            shuffle=False, collate_fn=partial(collate_mlm, pad_id=vocabulary[PAD_TOKEN]),
        )
        word_loader = DataLoader(
            word_validation_dataset, batch_size=word_cfg.batch_size,
            shuffle=False, collate_fn=partial(collate_word_mlm, pad_id=vocabulary[PAD_TOKEN]),
        )

        def validation_metrics() -> dict[str, object]:
            ordinary = evaluate_mlm(model, ordinary_loader, device)
            if not enable_words:
                return {"ordinary_mlm": ordinary}
            return {
                "ordinary_mlm": ordinary,
                "word_mlm": evaluate_mlm(model, word_loader, device),
                "ranking": _evaluate_ranking(model, scorer, word_loader, device),
            }

        initial_metrics = validation_metrics()
        initial_composite = _composite_loss(
            initial_metrics["ordinary_mlm"],  # type: ignore[arg-type]
            initial_metrics.get("word_mlm"),  # type: ignore[arg-type]
            initial_metrics.get("ranking"),  # type: ignore[arg-type]
            word_cfg.ranking_loss_weight if enable_words else 0.0,
        )
        LOGGER.info("[D2][训练前] 固定普通MLM loss=%.6f，综合loss=%.6f", float(initial_metrics["ordinary_mlm"]["overall"]["loss"]), initial_composite)  # type: ignore[index]

        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": word_cfg.encoder_learning_rate},
                {"params": scorer.parameters(), "lr": word_cfg.head_learning_rate},
            ],
            weight_decay=word_cfg.weight_decay,
        )
        factor, warmup_steps = _linear_schedule(word_cfg.max_steps, word_cfg.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        LOGGER.info("训练：batch=%d，max_steps=%d，encoder/head lr=%g/%g，warmup=%d步，eval=%d步", word_cfg.batch_size, word_cfg.max_steps, word_cfg.encoder_learning_rate, word_cfg.head_learning_rate, warmup_steps, word_cfg.eval_interval)

        best_composite = float("inf")
        best_step = 0
        best_model_state: dict[str, torch.Tensor] | None = None
        best_scorer_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, object] | None = None
        history: list[dict[str, object]] = []
        stale = 0
        global_step = 0
        epoch = 0
        interval_loss = 0.0
        interval_batches = 0
        stopped_early = False

        while global_step < word_cfg.max_steps and not stopped_early:
            epoch += 1
            train_dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(self.config.seed + epoch)
            train_loader = DataLoader(
                train_dataset, batch_size=word_cfg.batch_size, shuffle=True,
                collate_fn=partial(collate_word_mlm, pad_id=vocabulary[PAD_TOKEN]),
                generator=generator,
            )
            for batch in train_loader:
                if global_step >= word_cfg.max_steps:
                    break
                model.train()
                scorer.train()
                input_ids = batch["input_ids"].to(device)  # type: ignore[union-attr]
                labels = batch["labels"].to(device)  # type: ignore[union-attr]
                padding_mask = batch["padding_mask"].to(device)  # type: ignore[union-attr]
                optimizer.zero_grad()
                logits, hidden = model(input_ids, padding_mask)
                mlm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX)
                rank_loss = mlm_loss.new_zeros(())
                rank_weight = 0.0
                if enable_words:
                    positive, negative = scorer(
                        hidden,
                        batch["positive_spans"].to(device),  # type: ignore[union-attr]
                        batch["negative_spans"].to(device),  # type: ignore[union-attr]
                    )
                    rank_loss = _ranking_loss(
                        positive, negative,
                        batch["rank_valid"].to(device),  # type: ignore[union-attr]
                        batch["candidate_confidence"].to(device),  # type: ignore[union-attr]
                    )
                    warm = min(1.0, (global_step + 1) / max(1, word_cfg.ranking_warmup_steps)) if word_cfg.ranking_warmup_steps else 1.0
                    rank_weight = word_cfg.ranking_loss_weight * warm
                loss = mlm_loss + rank_weight * rank_loss
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(scorer.parameters()), word_cfg.gradient_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1
                interval_loss += float(loss.item())
                interval_batches += 1
                if global_step % word_cfg.log_interval == 0:
                    LOGGER.info("[D2] step %d/%d，epoch=%d，区间train_loss=%.6f，rank_weight=%.4f，encoder_lr=%.2e", global_step, word_cfg.max_steps, epoch, interval_loss / max(interval_batches, 1), rank_weight, optimizer.param_groups[0]["lr"])
                if global_step % word_cfg.eval_interval and global_step != word_cfg.max_steps:
                    continue
                metrics = validation_metrics()
                composite = _composite_loss(
                    metrics["ordinary_mlm"], metrics.get("word_mlm"), metrics.get("ranking"),  # type: ignore[arg-type]
                    word_cfg.ranking_loss_weight if enable_words else 0.0,
                )
                ordinary_loss = float(metrics["ordinary_mlm"]["overall"]["loss"])  # type: ignore[index]
                word_loss = float(metrics["word_mlm"]["overall"]["loss"]) if enable_words else None  # type: ignore[index]
                rank_accuracy = float(metrics["ranking"].get("overall", {}).get("accuracy", 0.0)) if enable_words else None  # type: ignore[union-attr]
                history.append({
                    "step": global_step, "epoch": epoch,
                    "train_loss": interval_loss / max(interval_batches, 1),
                    "ordinary_mlm_loss": ordinary_loss,
                    "word_mlm_loss": word_loss,
                    "ranking_accuracy": rank_accuracy,
                    "composite_loss": composite,
                    "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                    "head_learning_rate": optimizer.param_groups[1]["lr"],
                })
                interval_loss = 0.0
                interval_batches = 0
                LOGGER.info("[D2][开发集] step=%d，普通MLM=%.6f，整词MLM=%s，排序Acc=%s，综合Loss=%.6f", global_step, ordinary_loss, "—" if word_loss is None else f"{word_loss:.6f}", "—" if rank_accuracy is None else f"{rank_accuracy:.4f}", composite)
                if composite < best_composite - word_cfg.min_delta:
                    best_composite = composite
                    best_step = global_step
                    best_model_state = _cpu_state_dict(model)
                    best_scorer_state = _cpu_state_dict(scorer)
                    best_metrics = copy.deepcopy(metrics)
                    stale = 0
                    torch.save(_checkpoint(model, scorer, vocabulary, self.config, self.mode, best_step, metrics, prepared), output_directory / "best_model.pt")
                    LOGGER.info("[D2] 开发集综合Loss达到新最佳，保存best_model.pt")
                else:
                    stale += 1
                    LOGGER.info("[D2] 改善不足min_delta=%g，早停计数=%d/%d", word_cfg.min_delta, stale, word_cfg.patience)
                    if stale >= word_cfg.patience:
                        stopped_early = True
                        LOGGER.info("[D2] 触发早停；最佳步数=%d", best_step)
                        break

        if best_model_state is None:
            metrics = validation_metrics()
            best_composite = _composite_loss(metrics["ordinary_mlm"], metrics.get("word_mlm"), metrics.get("ranking"), word_cfg.ranking_loss_weight if enable_words else 0.0)  # type: ignore[arg-type]
            best_step = global_step
            best_model_state = _cpu_state_dict(model)
            best_scorer_state = _cpu_state_dict(scorer)
            best_metrics = copy.deepcopy(metrics)
            torch.save(_checkpoint(model, scorer, vocabulary, self.config, self.mode, best_step, metrics, prepared), output_directory / "best_model.pt")
        if best_scorer_state is None or best_metrics is None:
            raise RuntimeError("D2训练未产生有效checkpoint")
        torch.save(_checkpoint(model, scorer, vocabulary, self.config, self.mode, global_step, validation_metrics(), prepared), output_directory / "last_model.pt")
        model.load_state_dict(best_model_state)
        scorer.load_state_dict(best_scorer_state)
        final_metrics = validation_metrics()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        result: dict[str, object] = {
            "experiment": "d2" if self.mode == "fusion" else f"d2_{self.mode}",
            "mode": self.mode,
            "display_name": {"control": "D2-Control 等步数普通MLM", "lexicon": "D2-Lex 辞书词语预训练", "fusion": "D2 辞书＋统计融合词语预训练"}[self.mode],
            "best_step": best_step,
            "finished_step": global_step,
            "stopped_early": stopped_early,
            "best_composite_loss": best_composite,
            "parameter_count": parameter_count,
            "word_head_parameter_count": sum(parameter.numel() for parameter in scorer.parameters()),
            "device": str(device),
            "device_selection": device_selection,
            "initial_validation_metrics": initial_metrics,
            "validation_metrics": final_metrics,
            "candidate_summary": prepared["candidate_summary"],
            "history": history,
            "data_rows": _data_rows(train_documents, validation_documents, train_sequences, validation_sequences),
            "parent_stage": parent_checkpoint.get("stage", "unknown"),
            "parent_checkpoint": str(word_cfg.initial_checkpoint),
        }
        (output_directory / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_directory / "results.md").write_text(render_d2_report(result) + "\n", encoding="utf-8")
        if candidates:
            ordered = sorted(candidates.values(), key=lambda item: (-item.confidence, -item.frequency, item.term))
            (output_directory / "word_candidates.tsv").write_text(
                "词形\t长度\t频次\t来源\t置信度\t最小dPMI\t左熵\t右熵\n"
                + "\n".join(f"{item.term}\t{item.length}\t{item.frequency}\t{'+'.join(item.sources)}\t{item.confidence:.6f}\t{item.min_dpmi:.6f}\t{item.left_entropy:.6f}\t{item.right_entropy:.6f}" for item in ordered)
                + "\n", encoding="utf-8",
            )
        LOGGER.info("D2完成：最佳步数=%d，综合Loss=%.6f，checkpoint=%s", best_step, best_composite, output_directory / "best_model.pt")
        return result
