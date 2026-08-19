from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import Dataset

from config import ExperimentConfig
from data.corpus import SequenceChunk
from data.labels import select_punctuation
from models.tangut_encoder import PAD_TOKEN
from tasks import OUTSIDE

from .data import IGNORE_INDEX, MLMDataset, MLMSequence, collate_mlm


POSITION_LABELS = (OUTSIDE, "P")
GROUP_LABELS = ("INTRA", "SENTENCE")


class D5Dataset(Dataset[dict[str, object]]):
    """同一无标点字符序列的MLM视图和层级标点监督视图。"""

    def __init__(
        self,
        chunks: list[SequenceChunk],
        vocabulary: dict[str, int],
        config: ExperimentConfig,
        dynamic: bool,
        seed: int,
    ) -> None:
        self.chunks = chunks
        self.pause_labels = tuple(
            dict.fromkeys(
                config.punctuation.sentence_pause
                + config.punctuation.intra_sentence_pause
            )
        )
        self.pause_set = frozenset(self.pause_labels)
        self.intra = frozenset(config.punctuation.intra_sentence_pause)
        self.sentence = frozenset(config.punctuation.sentence_pause)
        self.type_to_id = {
            label: index for index, label in enumerate(self.pause_labels)
        }
        sequences = [
            MLMSequence(chunk.document_id, chunk.domain, chunk.tokens)
            for chunk in chunks
        ]
        cfg = config.pretraining
        self.mlm = MLMDataset(
            sequences,
            vocabulary,
            cfg.mask_ratio,
            cfg.span_mask_probability,
            cfg.span_lengths,
            cfg.mask_replace_probability,
            cfg.random_replace_probability,
            seed,
            dynamic,
        )

    def set_epoch(self, epoch: int) -> None:
        self.mlm.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.mlm[index])
        chunk = self.chunks[index]
        position_labels: list[int] = []
        group_labels: list[int] = []
        type_labels: list[int] = []
        for raw_label in chunk.labels:
            label = select_punctuation(raw_label, self.pause_set)
            if label == OUTSIDE:
                position_labels.append(0)
                group_labels.append(IGNORE_INDEX)
                type_labels.append(IGNORE_INDEX)
                continue
            if label not in self.type_to_id:
                raise ValueError(
                    f"{chunk.document_id}含D5无法处理的联合停顿标签：{label!r}"
                )
            position_labels.append(1)
            group_labels.append(0 if label in self.intra else 1)
            type_labels.append(self.type_to_id[label])
        item.update(
            {
                "original_ids": self.mlm.encoded[index],
                "position_labels": tuple(position_labels),
                "group_labels": tuple(group_labels),
                "type_labels": tuple(type_labels),
                "offset": chunk.offset,
            }
        )
        return item


def collate_d5(
    batch: list[dict[str, object]], pad_id: int
) -> dict[str, object]:
    mlm_batch = collate_mlm(batch, pad_id)
    lengths = [len(item["original_ids"]) for item in batch]  # type: ignore[arg-type]
    maximum = max(lengths)
    original_ids = torch.full(
        (len(batch), maximum), pad_id, dtype=torch.long
    )
    position_labels = torch.full(
        (len(batch), maximum), IGNORE_INDEX, dtype=torch.long
    )
    group_labels = torch.full(
        (len(batch), maximum), IGNORE_INDEX, dtype=torch.long
    )
    type_labels = torch.full(
        (len(batch), maximum), IGNORE_INDEX, dtype=torch.long
    )
    for row, item in enumerate(batch):
        length = lengths[row]
        original_ids[row, :length] = torch.tensor(item["original_ids"])
        position_labels[row, :length] = torch.tensor(item["position_labels"])
        group_labels[row, :length] = torch.tensor(item["group_labels"])
        type_labels[row, :length] = torch.tensor(item["type_labels"])
    mlm_batch.update(
        {
            "original_ids": original_ids,
            "position_labels": position_labels,
            "group_labels": group_labels,
            "type_labels": type_labels,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "offsets": [int(item["offset"]) for item in batch],
        }
    )
    return mlm_batch


def d5_collate(config: ExperimentConfig, vocabulary: dict[str, int]):
    del config
    return partial(collate_d5, pad_id=vocabulary[PAD_TOKEN])
