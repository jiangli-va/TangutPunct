from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from config import CascadeConfig, NeuralConfig
from data.corpus import SequenceChunk
from tasks import OUTSIDE

from .encoders import build_context_encoder
from .neural import IGNORE_LABEL, POSITION_LABEL, UNK_TOKEN, NeuralSequenceTagger


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CascadeSequence:
    token_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    upstream: tuple[float, ...]
    active: tuple[bool, ...]


@dataclass(frozen=True)
class _MultiTaskSequence:
    token_ids: tuple[int, ...]
    punctuation_ids: tuple[int, ...]
    position_ids: tuple[int, ...]


class _CascadeBiLSTMClassifier(nn.Module):
    """字符编码器加上游候选/概率通道的下游BiLSTM。"""

    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        config: NeuralConfig,
        mode: str,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.character_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=0
        )
        self.candidate_embedding = (
            nn.Embedding(2, config.embedding_dim) if mode == "candidate" else None
        )
        self.probability_projection = (
            nn.Linear(1, config.embedding_dim) if mode == "soft" else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.LSTM(
            config.embedding_dim,
            config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.bilstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(config.bilstm_hidden_dim * 2, label_count)
        nn.init.normal_(self.character_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.character_embedding.weight[0].zero_()

    def forward(
        self,
        token_ids: torch.Tensor,
        upstream: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.character_embedding(token_ids)
        if self.candidate_embedding is not None:
            embedded = embedded + self.candidate_embedding((upstream >= 0.5).long())
        elif self.probability_projection is not None:
            embedded = embedded + self.probability_projection(upstream.unsqueeze(-1))
        packed = pack_padded_sequence(
            self.dropout(embedded),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=token_ids.shape[1]
        )
        return self.classifier(self.dropout(encoded))


class _SharedBiLSTMMultiTaskClassifier(nn.Module):
    """一个共享BiLSTM和两个并行任务头。"""

    def __init__(
        self,
        vocab_size: int,
        punctuation_count: int,
        config: NeuralConfig,
        encoder_name: str = "bilstm",
    ) -> None:
        super().__init__()
        self.encoder = build_context_encoder(encoder_name, vocab_size, config)
        self.position_head = nn.Linear(self.encoder.output_dim, 2)
        self.punctuation_head = nn.Linear(
            self.encoder.output_dim, punctuation_count
        )

    def forward(
        self, token_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(token_ids, lengths)
        return self.position_head(encoded), self.punctuation_head(encoded)


def _loader(
    data: list[object], batch_size: int, seed: int, shuffle: bool, collate
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
    )


class CascadeBiLSTMTagger(NeuralSequenceTagger):
    """C2候选拒绝或C3连续概率软级联的下游模型。"""

    def __init__(
        self,
        mode: str,
        punctuation_labels: Iterable[str],
        neural_config: NeuralConfig,
        cascade_config: CascadeConfig,
        max_sequence_length: int,
        seed: int,
    ) -> None:
        if mode not in {"candidate", "soft"}:
            raise ValueError("级联下游mode只能是candidate或soft")
        super().__init__(
            "bilstm",
            punctuation_labels,
            neural_config,
            max_sequence_length,
            seed,
        )
        self.mode = mode
        self.cascade_config = cascade_config
        self.labels = (OUTSIDE,) + tuple(punctuation_labels)
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}
        self.selected_alpha = cascade_config.soft_train_alpha
        self.candidate_threshold: float | None = None

    @staticmethod
    def _collate(batch: list[_CascadeSequence]) -> dict[str, torch.Tensor]:
        lengths = torch.tensor([len(item.token_ids) for item in batch], dtype=torch.long)
        width = int(lengths.max().item())
        token_ids = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_LABEL, dtype=torch.long)
        upstream = torch.zeros((len(batch), width), dtype=torch.float)
        active = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, item in enumerate(batch):
            size = len(item.token_ids)
            token_ids[row, :size] = torch.tensor(item.token_ids)
            labels[row, :size] = torch.tensor(item.label_ids)
            upstream[row, :size] = torch.tensor(item.upstream)
            active[row, :size] = torch.tensor(item.active)
        return {
            "token_ids": token_ids,
            "labels": labels,
            "upstream": upstream,
            "active": active,
            "lengths": lengths,
        }

    def _encode_cascade(
        self,
        sequences: list[SequenceChunk],
        probabilities: list[list[float]],
        threshold: float | None,
    ) -> list[_CascadeSequence]:
        if len(sequences) != len(probabilities):
            raise ValueError("序列数量与上游概率数量不一致")
        encoded = []
        for chunk, scores in zip(sequences, probabilities):
            if len(chunk.tokens) != len(scores):
                raise ValueError(f"{chunk.document_id} 的上游概率长度不一致")
            token_ids = tuple(
                self.vocabulary.get(token, self.vocabulary[UNK_TOKEN])
                for token in chunk.tokens
            )
            if self.mode == "candidate":
                if threshold is None:
                    raise ValueError("C2编码必须提供候选阈值")
                active = tuple(score >= threshold for score in scores)
                upstream = tuple(float(value) for value in active)
            else:
                active = tuple(True for _ in scores)
                upstream = tuple(float(score) for score in scores)
            label_ids = tuple(
                self.label_to_id.get(label, self.label_to_id[OUTSIDE])
                if is_active
                else IGNORE_LABEL
                for label, is_active in zip(chunk.labels, active)
            )
            encoded.append(_CascadeSequence(token_ids, label_ids, upstream, active))
        return encoded

    def _fuse(self, logits: torch.Tensor, upstream: torch.Tensor, alpha: float) -> torch.Tensor:
        if self.mode != "soft":
            return logits
        epsilon = self.cascade_config.soft_epsilon
        probability = upstream.clamp(epsilon, 1.0 - epsilon)
        fused = logits.clone()
        fused[..., 0] += alpha * torch.log1p(-probability)
        fused[..., 1:] += alpha * torch.log(probability).unsqueeze(-1)
        return fused

    def _epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        if not isinstance(self.model, _CascadeBiLSTMClassifier):
            raise RuntimeError("级联下游模型尚未建立")
        training = optimizer is not None
        self.model.train(training)
        loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        total_loss = 0.0
        total_labels = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                labels = batch["labels"].to(self.device)
                valid = int((labels != IGNORE_LABEL).sum().item())
                if not valid:
                    continue
                if training:
                    optimizer.zero_grad()
                logits = self.model(
                    batch["token_ids"].to(self.device),
                    batch["upstream"].to(self.device),
                    batch["lengths"].to(self.device),
                )
                logits = self._fuse(
                    logits,
                    batch["upstream"].to(self.device),
                    self.cascade_config.soft_train_alpha,
                )
                loss = loss_function(logits.reshape(-1, len(self.labels)), labels.reshape(-1))
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                total_loss += float(loss.item()) * valid
                total_labels += valid
        return total_loss / total_labels if total_labels else float("inf")

    def fit_with_upstream(
        self,
        train: list[SequenceChunk],
        train_probabilities: list[list[float]],
        dev: list[SequenceChunk],
        dev_probabilities: list[list[float]],
        candidate_threshold: float | None = None,
    ) -> None:
        if not train:
            raise ValueError("级联下游训练序列不能为空")
        self._set_seed()
        self._build_vocabulary(train)
        self.candidate_threshold = candidate_threshold
        train_data = self._encode_cascade(train, train_probabilities, candidate_threshold)
        dev_data = self._encode_cascade(dev, dev_probabilities, candidate_threshold)
        if not any(any(label != IGNORE_LABEL for label in item.label_ids) for item in train_data):
            raise ValueError("级联下游没有有效训练标签")
        self.model = _CascadeBiLSTMClassifier(
            len(self.vocabulary), len(self.labels), self.config, self.mode
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = _loader(
            train_data, self.config.batch_size, self.seed, True, self._collate
        )
        dev_loader = (
            _loader(dev_data, self.config.batch_size, self.seed, False, self._collate)
            if dev_data
            else None
        )
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        LOGGER.info(
            "C%s下游BiLSTM：模式=%s，标签=%s，训练块=%d，开发块=%d",
            "2" if self.mode == "candidate" else "3",
            "高召回候选＋拒绝" if self.mode == "candidate" else "连续概率软融合",
            "/".join(self.labels),
            len(train_data),
            len(dev_data),
        )
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._epoch(train_loader, optimizer)
            dev_loss = self._epoch(dev_loader, None) if dev_loader is not None else train_loss
            self.training_history.append(
                {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
            )
            LOGGER.info(
                "[cascade-%s] epoch %d/%d：train_loss=%.6f，dev_loss=%.6f",
                self.mode,
                epoch,
                self.config.epochs,
                train_loss,
                dev_loss,
            )
            if dev_loss < self.best_dev_loss - self.config.min_delta:
                self.best_dev_loss = dev_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    LOGGER.info("[cascade-%s] 连续%d轮未改善，提前停止", self.mode, stale)
                    break
        self.model.load_state_dict(best_state)

    def _raw_predictions(
        self,
        sequences: list[SequenceChunk],
        probabilities: list[list[float]],
        threshold: float | None,
        alpha: float,
    ) -> list[list[str]]:
        if not isinstance(self.model, _CascadeBiLSTMClassifier):
            raise RuntimeError("级联下游模型尚未训练")
        data = self._encode_cascade(sequences, probabilities, threshold)
        loader = _loader(data, self.config.batch_size, self.seed, False, self._collate)
        self.model.eval()
        results = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                upstream = batch["upstream"].to(self.device)
                logits = self.model(
                    batch["token_ids"].to(self.device),
                    upstream,
                    batch["lengths"].to(self.device),
                )
                logits = self._fuse(logits, upstream, alpha).cpu()
                for row, length in enumerate(batch["lengths"].tolist()):
                    active = data[offset + row].active
                    ids = logits[row, :length].argmax(dim=-1).tolist()
                    results.append(
                        [
                            self.labels[label_id] if active[index] else OUTSIDE
                            for index, label_id in enumerate(ids)
                        ]
                    )
                offset += len(batch["lengths"])
        return results

    def predict_with_upstream(
        self,
        sequences: list[SequenceChunk],
        probabilities: list[list[float]],
        candidate_threshold: float | None = None,
        alpha: float | None = None,
    ) -> list[list[str]]:
        threshold = candidate_threshold if self.mode == "candidate" else None
        if self.mode == "candidate" and threshold is None:
            threshold = self.candidate_threshold
        return self._raw_predictions(
            sequences,
            probabilities,
            threshold,
            self.selected_alpha if alpha is None else alpha,
        )

    def metadata(self) -> dict[str, object]:
        parameter_count = (
            sum(parameter.numel() for parameter in self.model.parameters())
            if self.model is not None
            else 0
        )
        return {
            "神经编码器": "bilstm",
            "解码层": "候选拒绝Softmax" if self.mode == "candidate" else "连续概率软融合Softmax",
            "模型阶段": "C2候选拒绝" if self.mode == "candidate" else "C3软级联",
            "嵌入初始化": "随机初始化、训练中更新",
            "嵌入维度": self.config.embedding_dim,
            "词表大小": len(self.vocabulary),
            "参数量": parameter_count,
            "设备": str(self.device),
            "设备选择依据": self.device_selection,
            "最佳epoch": self.best_epoch,
            "最佳开发集loss": self.best_dev_loss,
            "候选阈值": self.candidate_threshold,
            "软融合alpha": self.selected_alpha if self.mode == "soft" else None,
            "训练记录": self.training_history,
        }

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("级联下游模型尚未训练")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "labels": self.labels,
                "vocabulary": self.vocabulary,
                "neural_config": asdict(self.config),
                "cascade_config": asdict(self.cascade_config),
                "candidate_threshold": self.candidate_threshold,
                "selected_alpha": self.selected_alpha,
                "state_dict": self.model.state_dict(),
            },
            path,
        )


class MultiTaskBiLSTMTagger(NeuralSequenceTagger):
    """C4：共享一个BiLSTM，位置头为辅助任务，完整标点头负责最终输出。"""

    def __init__(
        self,
        punctuation_labels: Iterable[str],
        neural_config: NeuralConfig,
        cascade_config: CascadeConfig,
        max_sequence_length: int,
        seed: int,
    ) -> None:
        super().__init__(
            "bilstm", punctuation_labels, neural_config, max_sequence_length, seed
        )
        self.cascade_config = cascade_config
        self.labels = (OUTSIDE,) + tuple(punctuation_labels)
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}

    @staticmethod
    def _collate_multitask(batch: list[_MultiTaskSequence]) -> dict[str, torch.Tensor]:
        lengths = torch.tensor([len(item.token_ids) for item in batch], dtype=torch.long)
        width = int(lengths.max().item())
        token_ids = torch.zeros((len(batch), width), dtype=torch.long)
        punctuation = torch.full((len(batch), width), IGNORE_LABEL, dtype=torch.long)
        position = torch.full((len(batch), width), IGNORE_LABEL, dtype=torch.long)
        for row, item in enumerate(batch):
            size = len(item.token_ids)
            token_ids[row, :size] = torch.tensor(item.token_ids)
            punctuation[row, :size] = torch.tensor(item.punctuation_ids)
            position[row, :size] = torch.tensor(item.position_ids)
        return {
            "token_ids": token_ids,
            "punctuation": punctuation,
            "position": position,
            "lengths": lengths,
        }

    def _encode_multitask(self, sequences: list[SequenceChunk]) -> list[_MultiTaskSequence]:
        result = []
        for chunk in sequences:
            token_ids = tuple(
                self.vocabulary.get(token, self.vocabulary[UNK_TOKEN])
                for token in chunk.tokens
            )
            punctuation_ids = tuple(
                self.label_to_id.get(label, self.label_to_id[OUTSIDE])
                for label in chunk.labels
            )
            position_ids = tuple(0 if label == OUTSIDE else 1 for label in chunk.labels)
            result.append(_MultiTaskSequence(token_ids, punctuation_ids, position_ids))
        return result

    def _multitask_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer | None
    ) -> float:
        if not isinstance(self.model, _SharedBiLSTMMultiTaskClassifier):
            raise RuntimeError("多任务模型尚未建立")
        training = optimizer is not None
        self.model.train(training)
        loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        total_loss = 0.0
        total_labels = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                punctuation = batch["punctuation"].to(self.device)
                position = batch["position"].to(self.device)
                valid = int((punctuation != IGNORE_LABEL).sum().item())
                if training:
                    optimizer.zero_grad()
                position_logits, punctuation_logits = self.model(
                    batch["token_ids"].to(self.device),
                    batch["lengths"].to(self.device),
                )
                punctuation_loss = loss_function(
                    punctuation_logits.reshape(-1, len(self.labels)), punctuation.reshape(-1)
                )
                position_loss = loss_function(
                    position_logits.reshape(-1, 2), position.reshape(-1)
                )
                loss = punctuation_loss + (
                    self.cascade_config.multitask_position_loss_weight * position_loss
                )
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                total_loss += float(loss.item()) * valid
                total_labels += valid
        return total_loss / total_labels if total_labels else float("inf")

    def fit(self, train: list[SequenceChunk], dev: list[SequenceChunk] | None = None) -> None:
        if not train:
            raise ValueError("多任务训练序列不能为空")
        self._set_seed()
        self._build_vocabulary(train)
        train_data = self._encode_multitask(train)
        dev_data = self._encode_multitask(dev or [])
        self.model = _SharedBiLSTMMultiTaskClassifier(
            len(self.vocabulary),
            len(self.labels),
            self.config,
            encoder_name=self.cascade_config.shared_encoder,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = _loader(
            train_data, self.config.batch_size, self.seed, True, self._collate_multitask
        )
        dev_loader = (
            _loader(dev_data, self.config.batch_size, self.seed, False, self._collate_multitask)
            if dev_data
            else None
        )
        best_state = copy.deepcopy(self.model.state_dict())
        stale = 0
        LOGGER.info(
            "C4共享BiLSTM：一个编码器、位置辅助头＋完整标点头；位置损失权重=%g",
            self.cascade_config.multitask_position_loss_weight,
        )
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._multitask_epoch(train_loader, optimizer)
            dev_loss = (
                self._multitask_epoch(dev_loader, None)
                if dev_loader is not None
                else train_loss
            )
            self.training_history.append(
                {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
            )
            LOGGER.info(
                "[multitask] epoch %d/%d：train_loss=%.6f，dev_loss=%.6f",
                epoch,
                self.config.epochs,
                train_loss,
                dev_loss,
            )
            if dev_loss < self.best_dev_loss - self.config.min_delta:
                self.best_dev_loss = dev_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    LOGGER.info("[multitask] 连续%d轮未改善，提前停止", stale)
                    break
        self.model.load_state_dict(best_state)

    def _predict_heads(
        self, sequences: list[SequenceChunk]
    ) -> tuple[list[list[str]], list[list[str]]]:
        if not isinstance(self.model, _SharedBiLSTMMultiTaskClassifier):
            raise RuntimeError("多任务模型尚未训练")
        data = self._encode_multitask(sequences)
        loader = _loader(
            data, self.config.batch_size, self.seed, False, self._collate_multitask
        )
        punctuation_predictions = []
        position_predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                position_logits, punctuation_logits = self.model(
                    batch["token_ids"].to(self.device),
                    batch["lengths"].to(self.device),
                )
                for row, length in enumerate(batch["lengths"].tolist()):
                    punctuation_predictions.append(
                        [
                            self.labels[index]
                            for index in punctuation_logits[row, :length].argmax(-1).tolist()
                        ]
                    )
                    position_predictions.append(
                        [
                            POSITION_LABEL if index == 1 else OUTSIDE
                            for index in position_logits[row, :length].argmax(-1).tolist()
                        ]
                    )
        return punctuation_predictions, position_predictions

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        return self._predict_heads(sequences)[0]

    def predict_position(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        return self._predict_heads(sequences)[1]

    def metadata(self) -> dict[str, object]:
        parameter_count = (
            sum(parameter.numel() for parameter in self.model.parameters())
            if self.model is not None
            else 0
        )
        return {
            "神经编码器": f"{self.cascade_config.shared_encoder}_shared",
            "解码层": "位置辅助头＋完整标点头",
            "模型阶段": "C4多任务联合",
            "嵌入初始化": "随机初始化、训练中更新",
            "嵌入维度": self.config.embedding_dim,
            "词表大小": len(self.vocabulary),
            "参数量": parameter_count,
            "设备": str(self.device),
            "设备选择依据": self.device_selection,
            "最佳epoch": self.best_epoch,
            "最佳开发集loss": self.best_dev_loss,
            "位置损失权重": self.cascade_config.multitask_position_loss_weight,
            "训练记录": self.training_history,
        }

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("多任务模型尚未训练")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_type": f"{self.cascade_config.shared_encoder}_shared",
                "labels": self.labels,
                "vocabulary": self.vocabulary,
                "neural_config": asdict(self.config),
                "cascade_config": asdict(self.cascade_config),
                "state_dict": self.model.state_dict(),
            },
            path,
        )
