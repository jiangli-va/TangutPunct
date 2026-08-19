from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from data.corpus import PreparedDocument
from models.tangut_encoder import MASK_TOKEN, PAD_TOKEN, SPECIAL_TOKENS, UNK_TOKEN


IGNORE_INDEX = -100


@dataclass(frozen=True)
class MLMSequence:
    document_id: str
    domain: str
    tokens: tuple[str, ...]


def split_pretraining_documents(documents: list[PreparedDocument], validation_ratio: float, seed: int) -> tuple[list[PreparedDocument], list[PreparedDocument]]:
    """按领域分层、按文献划分MLM训练集和开发集。"""

    by_domain: dict[str, list[PreparedDocument]] = defaultdict(list)
    for document in documents:
        by_domain[document.domain].append(document)
    train: list[PreparedDocument] = []
    validation: list[PreparedDocument] = []
    for domain_index, domain in enumerate(sorted(by_domain)):
        group = sorted(by_domain[domain], key=lambda item: item.document_id)
        random.Random(seed + domain_index).shuffle(group)
        validation_size = round(len(group) * validation_ratio)
        if validation_size == 0 and len(group) > 1:
            validation_size = 1
        validation_size = min(validation_size, max(0, len(group) - 1))
        validation.extend(group[:validation_size])
        train.extend(group[validation_size:])
    if not train or not validation:
        raise ValueError("MLM文献划分后训练集或开发集为空")
    return train, validation


def build_vocabulary(documents: Iterable[PreparedDocument]) -> dict[str, int]:
    """从当前工作区配置的全部无标点正文构建字符词表。"""

    characters = sorted({token for document in documents for token in document.tokens})
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocabulary.update(
        {token: index + len(SPECIAL_TOKENS) for index, token in enumerate(characters)}
    )
    return vocabulary


def make_mlm_sequences(
    documents: Iterable[PreparedDocument],
    max_sequence_length: int,
    min_sequence_length: int,
) -> list[MLMSequence]:
    """沿用语料TAB硬边界切块，绝不跨删除片段或段落连接上下文。"""

    sequences: list[MLMSequence] = []
    for document in documents:
        for chunk in document.chunks(max_sequence_length):
            if len(chunk.tokens) < min_sequence_length:
                continue
            sequences.append(
                MLMSequence(document.document_id, document.domain, chunk.tokens)
            )
    return sequences


class MLMDataset(Dataset[dict[str, object]]):
    """动态混合遮盖数据集；开发集遮盖固定，保证早停指标可比较。"""

    def __init__(
        self,
        sequences: list[MLMSequence],
        vocabulary: dict[str, int],
        mask_ratio: float,
        span_mask_probability: float,
        span_lengths: tuple[int, ...],
        mask_replace_probability: float,
        random_replace_probability: float,
        seed: int,
        dynamic: bool,
    ) -> None:
        self.sequences = sequences
        self.vocabulary = vocabulary
        self.mask_ratio = mask_ratio
        self.span_mask_probability = span_mask_probability
        self.span_lengths = span_lengths
        self.mask_replace_probability = mask_replace_probability
        self.random_replace_probability = random_replace_probability
        self.seed = seed
        self.dynamic = dynamic
        self.epoch = 0
        self.pad_id = vocabulary[PAD_TOKEN]
        self.unk_id = vocabulary[UNK_TOKEN]
        self.mask_id = vocabulary[MASK_TOKEN]
        special_ids = {vocabulary[token] for token in SPECIAL_TOKENS}
        self.replacement_ids = tuple(
            index for index in range(len(vocabulary)) if index not in special_ids
        )
        self.encoded = [
            tuple(vocabulary.get(token, self.unk_id) for token in sequence.tokens)
            for sequence in sequences
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def _rng(self, index: int) -> random.Random:
        epoch = self.epoch if self.dynamic else 0
        return random.Random(self.seed + epoch * 1_000_003 + index * 97)

    def _masked_positions(
        self, length: int, target_count: int, rng: random.Random
    ) -> set[int]:
        selected: set[int] = set()
        while len(selected) < target_count:
            remaining = [index for index in range(length) if index not in selected]
            if not remaining:
                break
            use_span = rng.random() < self.span_mask_probability
            if use_span:
                span_length = rng.choice(self.span_lengths)
                starts = []
                for start in remaining:
                    stop = min(start + span_length, length)
                    positions = range(start, stop)
                    if stop - start >= 2 and all(pos not in selected for pos in positions):
                        starts.append(start)
                if starts:
                    start = rng.choice(starts)
                    for position in range(start, min(start + span_length, length)):
                        if len(selected) >= target_count:
                            break
                        selected.add(position)
                    continue
            selected.add(rng.choice(remaining))
        return selected

    def __getitem__(self, index: int) -> dict[str, object]:
        original = list(self.encoded[index])
        rng = self._rng(index)
        target_count = max(1, round(len(original) * self.mask_ratio))
        positions = self._masked_positions(len(original), target_count, rng)
        input_ids = original[:]
        labels = [IGNORE_INDEX] * len(original)
        for position in positions:
            labels[position] = original[position]
            draw = rng.random()
            if draw < self.mask_replace_probability:
                input_ids[position] = self.mask_id
            elif draw < (
                self.mask_replace_probability + self.random_replace_probability
            ):
                input_ids[position] = rng.choice(self.replacement_ids)
            # 剩余概率保持原字，仍然计算该位置的MLM损失。
        sequence = self.sequences[index]
        return {
            "input_ids": tuple(input_ids),
            "labels": tuple(labels),
            "document_id": sequence.document_id,
            "domain": sequence.domain,
        }


def collate_mlm(
    batch: list[dict[str, object]], pad_id: int
) -> dict[str, object]:
    lengths = [len(item["input_ids"]) for item in batch]  # type: ignore[arg-type]
    max_length = max(lengths)
    input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    labels = torch.full(
        (len(batch), max_length), IGNORE_INDEX, dtype=torch.long
    )
    for row, item in enumerate(batch):
        sequence_ids = item["input_ids"]
        sequence_labels = item["labels"]
        length = lengths[row]
        input_ids[row, :length] = torch.tensor(sequence_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(sequence_labels, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "padding_mask": input_ids.eq(pad_id),
        "document_ids": [str(item["document_id"]) for item in batch],
        "domains": [str(item["domain"]) for item in batch],
    }
