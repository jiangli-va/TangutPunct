from __future__ import annotations

import copy
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from config import NeuralConfig
from data.corpus import SequenceChunk
from tasks import OUTSIDE

from .base import SequenceTagger
from .crf_layer import LinearChainCRF


LOGGER = logging.getLogger(__name__)
POSITION_LABEL = "P"
SENTENCE_GROUP_LABEL = "SENTENCE"
INTRA_GROUP_LABEL = "INTRA"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
IGNORE_LABEL = -100
STAGE_NONE = 0
STAGE_OUTSIDE = 1
STAGE_POSITION = 2
GROUP_NONE = 0
GROUP_SENTENCE = 1
GROUP_INTRA = 2


@dataclass(frozen=True)
class _EncodedSequence:
    token_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    stage_ids: tuple[int, ...]
    group_ids: tuple[int, ...]


@dataclass(frozen=True)
class _CudaDeviceStatus:
    """一块PyTorch可见GPU的即时状态；index是CUDA逻辑编号。"""

    index: int
    name: str
    free_memory_gb: float
    total_memory_gb: float
    utilization: int | None


class _BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        use_group_features: bool,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.character_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.stage_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.group_embedding = (
            nn.Embedding(3, embedding_dim, padding_idx=0)
            if use_group_features
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, label_count)
        self.crf = LinearChainCRF(label_count) if use_crf else None
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        nn.init.normal_(self.character_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.stage_embedding.weight, mean=0.0, std=0.02)
        if self.group_embedding is not None:
            nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.character_embedding.weight[0].zero_()
            self.stage_embedding.weight[0].zero_()
            if self.group_embedding is not None:
                self.group_embedding.weight[0].zero_()

    def forward(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        del padding_mask
        embedded = self.character_embedding(token_ids) + self.stage_embedding(
            stage_ids
        )
        if self.group_embedding is not None:
            embedded = embedded + self.group_embedding(group_ids)
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


class _RandomTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        embedding_dim: int,
        max_length: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        use_group_features: bool,
    ) -> None:
        super().__init__()
        self.character_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.stage_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.group_embedding = (
            nn.Embedding(3, embedding_dim, padding_idx=0)
            if use_group_features
            else None
        )
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, label_count)
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        for embedding in (
            self.character_embedding,
            self.stage_embedding,
            self.position_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        if self.group_embedding is not None:
            nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.character_embedding.weight[0].zero_()
            self.stage_embedding.weight[0].zero_()
            if self.group_embedding is not None:
                self.group_embedding.weight[0].zero_()

    def forward(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        del lengths
        positions = torch.arange(
            token_ids.shape[1], device=token_ids.device
        ).unsqueeze(0)
        embedded = (
            self.character_embedding(token_ids)
            + self.stage_embedding(stage_ids)
            + self.position_embedding(positions)
        )
        if self.group_embedding is not None:
            embedded = embedded + self.group_embedding(group_ids)
        encoded = self.encoder(
            self.dropout(embedded), src_key_padding_mask=padding_mask
        )
        return self.classifier(self.dropout(encoded))


class NeuralSequenceTagger(SequenceTagger):
    """B4/B5/B6共享训练器；编码器或解码层不同，其余流程一致。"""

    file_extension = ".pt"

    def __init__(
        self,
        encoder_type: str,
        punctuation_labels: Iterable[str],
        config: NeuralConfig,
        max_sequence_length: int,
        seed: int,
        decoder_type: str = "softmax",
    ) -> None:
        if encoder_type not in {"bilstm", "random_transformer"}:
            raise ValueError(f"未知神经编码器：{encoder_type}")
        if decoder_type not in {"softmax", "crf"}:
            raise ValueError(f"未知神经解码层：{decoder_type}")
        if decoder_type == "crf" and encoder_type != "bilstm":
            raise ValueError("当前线性链CRF解码层只接入BiLSTM编码器")
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.punctuation_labels = tuple(punctuation_labels)
        self.config = config
        self.max_sequence_length = max_sequence_length
        self.seed = seed
        self.task = "unfitted"
        self.vocabulary: dict[str, int] = {}
        self.labels: tuple[str, ...] = ()
        self.label_to_id: dict[str, int] = {}
        self.use_group_features = False
        self.model: nn.Module | None = None
        self.position_threshold: float | None = (
            0.5 if decoder_type == "softmax" else None
        )
        self.best_epoch = 0
        self.best_dev_loss = float("inf")
        self.training_history: list[dict[str, float | int]] = []
        self.device_selection = ""
        self.cuda_device_statuses: tuple[_CudaDeviceStatus, ...] = ()
        self.device = self._resolve_device(config)

    @staticmethod
    def _visible_cuda_devices() -> list[_CudaDeviceStatus]:
        if not torch.cuda.is_available():
            return []
        statuses = []
        for index in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                name = torch.cuda.get_device_name(index)
            except (RuntimeError, OSError) as error:
                LOGGER.warning("无法读取CUDA逻辑设备%d的显存状态：%s", index, error)
                continue
            utilization = None
            utilization_reader = getattr(torch.cuda, "utilization", None)
            if utilization_reader is not None:
                try:
                    utilization = int(utilization_reader(index))
                except Exception as error:  # NVML后端可能抛出厂商自定义异常。
                    # NVML不可用时仍可按空闲显存选卡，但日志会明确说明。
                    LOGGER.debug("无法读取cuda:%d的GPU利用率：%s", index, error)
                    utilization = None
            gibibyte = 1024**3
            statuses.append(
                _CudaDeviceStatus(
                    index=index,
                    name=name,
                    free_memory_gb=free_bytes / gibibyte,
                    total_memory_gb=total_bytes / gibibyte,
                    utilization=utilization,
                )
            )
        return statuses

    @staticmethod
    def _best_cuda_device(
        statuses: list[_CudaDeviceStatus],
    ) -> _CudaDeviceStatus:
        # 先比较空闲显存，再比较利用率，最后稳定选择较小逻辑编号。
        return max(
            statuses,
            key=lambda status: (
                status.free_memory_gb,
                -(status.utilization if status.utilization is not None else 101),
                -status.index,
            ),
        )

    @staticmethod
    def _cuda_status_text(status: _CudaDeviceStatus) -> str:
        utilization = (
            f"{status.utilization}%" if status.utilization is not None else "未知"
        )
        return (
            f"cuda:{status.index}（{status.name}，利用率={utilization}，"
            f"空闲显存={status.free_memory_gb:.2f}/{status.total_memory_gb:.2f} GiB）"
        )

    def _resolve_device(self, config: NeuralConfig) -> torch.device:
        value = config.device.lower()
        if value == "cpu":
            self.device_selection = "配置明确指定CPU"
            LOGGER.info("设备选择：%s", self.device_selection)
            return torch.device("cpu")

        statuses = self._visible_cuda_devices()
        self.cuda_device_statuses = tuple(statuses)
        if not statuses:
            if value == "auto":
                self.device_selection = "PyTorch未检测到可用CUDA，自动回退CPU"
                LOGGER.warning("设备选择：%s", self.device_selection)
                return torch.device("cpu")
            raise RuntimeError("配置要求CUDA，但当前PyTorch检测不到可用GPU")
        LOGGER.info(
            "CUDA候选状态：%s",
            "；".join(self._cuda_status_text(status) for status in statuses),
        )

        if value.startswith("cuda:"):
            requested_index = int(value.split(":", maxsplit=1)[1])
            selected = next(
                (status for status in statuses if status.index == requested_index), None
            )
            if selected is None:
                visible = "、".join(f"cuda:{status.index}" for status in statuses)
                raise RuntimeError(
                    f"指定的{value}不可用；当前PyTorch可见设备为：{visible}"
                )
            self.device_selection = "手动指定 " + self._cuda_status_text(selected)
            LOGGER.info("设备选择：%s", self.device_selection)
            return torch.device(value)

        if value == "cuda":
            selected = self._best_cuda_device(statuses)
            self.device_selection = "明确要求GPU，选择当前空闲显存最多的 " + self._cuda_status_text(selected)
            LOGGER.info("设备选择：%s", self.device_selection)
            return torch.device(f"cuda:{selected.index}")

        eligible = [
            status
            for status in statuses
            if status.free_memory_gb >= config.cuda_min_free_memory_gb
            and (
                status.utilization is None
                or status.utilization <= config.cuda_max_utilization
            )
        ]
        if not eligible:
            observed = "；".join(self._cuda_status_text(status) for status in statuses)
            raise RuntimeError(
                "检测到CUDA设备，但没有设备满足空闲条件："
                f"利用率≤{config.cuda_max_utilization}%且空闲显存≥"
                f"{config.cuda_min_free_memory_gb:.2f} GiB。当前状态：{observed}。"
                "请稍后重试、调整neural.cuda_*阈值，或用device: cuda:N手动指定。"
            )
        selected = self._best_cuda_device(eligible)
        unknown_utilization = selected.utilization is None
        reason = "（利用率读取失败，本次仅按空闲显存判断）" if unknown_utilization else ""
        self.device_selection = "自动选择空闲GPU " + self._cuda_status_text(selected) + reason
        LOGGER.info("设备选择：%s", self.device_selection)
        return torch.device(f"cuda:{selected.index}")

    def _set_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _infer_task(self, train: list[SequenceChunk]) -> None:
        observed = {
            label
            for chunk in train
            for label in chunk.labels
            if label != OUTSIDE
        }
        channel_names = {
            name
            for chunk in train
            for name, _ in chunk.document_feature_channels
        }
        self.use_group_features = "pause_group" in channel_names
        if observed == {POSITION_LABEL}:
            self.task = "position"
            self.labels = (OUTSIDE, POSITION_LABEL)
        elif observed and observed <= {
            SENTENCE_GROUP_LABEL,
            INTRA_GROUP_LABEL,
        }:
            # A3的组别阶段由金标准/预测位置门控，只在候选位置区分两类；
            # F1则直接对所有字符后的间隔学习O/句内/句间。二者必须由是否
            # 存在位置上游通道区分，否则直接训练会把大量O错误地忽略掉。
            if "position" in channel_names:
                self.task = "punctuation_group"
                self.labels = (SENTENCE_GROUP_LABEL, INTRA_GROUP_LABEL)
            else:
                self.task = "punctuation_group_joint"
                self.labels = (
                    OUTSIDE,
                    SENTENCE_GROUP_LABEL,
                    INTRA_GROUP_LABEL,
                )
        elif observed & set(self.punctuation_labels):
            has_upstream = bool(channel_names & {"position", "pause_group"})
            if has_upstream:
                self.task = "punctuation_type"
                self.labels = self.punctuation_labels
            else:
                self.task = "punctuation_joint"
                self.labels = (OUTSIDE,) + self.punctuation_labels
        else:
            raise ValueError(f"无法从训练标签识别神经模型阶段：{sorted(observed)}")
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}

    def _build_vocabulary(self, train: list[SequenceChunk]) -> None:
        characters = sorted({token for chunk in train for token in chunk.tokens})
        self.vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.vocabulary.update(
            {character: index + 2 for index, character in enumerate(characters)}
        )

    @staticmethod
    def _stage_values(
        chunk: SequenceChunk, absolute_index: int
    ) -> tuple[int, int]:
        channels = dict(chunk.document_feature_channels)
        position = channels.get("position")
        stage_value = STAGE_NONE
        if position is not None:
            stage_value = (
                STAGE_POSITION
                if position[absolute_index] == POSITION_LABEL
                else STAGE_OUTSIDE
            )
        group = channels.get("pause_group")
        if group is not None:
            if group[absolute_index] == SENTENCE_GROUP_LABEL:
                return stage_value, GROUP_SENTENCE
            if group[absolute_index] == INTRA_GROUP_LABEL:
                return stage_value, GROUP_INTRA
        return stage_value, GROUP_NONE

    def _encode_sequence(self, chunk: SequenceChunk) -> _EncodedSequence:
        unk_id = self.vocabulary.get(UNK_TOKEN)
        if unk_id is None:
            # D1的BERT式特殊token写作[UNK]；随机神经基线仍使用<UNK>。
            unk_id = self.vocabulary.get("[UNK]")
        if unk_id is None:
            raise RuntimeError("字符词表缺少UNK特殊token")
        token_ids = tuple(
            self.vocabulary.get(token, unk_id)
            for token in chunk.tokens
        )
        stage_ids = []
        group_ids = []
        label_ids = []
        for local_index, gold in enumerate(chunk.labels):
            absolute_index = chunk.offset + local_index
            stage_value, group_value = self._stage_values(chunk, absolute_index)
            stage_ids.append(stage_value)
            group_ids.append(group_value)

            if self.task == "position":
                label_ids.append(
                    self.label_to_id[
                        POSITION_LABEL if gold == POSITION_LABEL else OUTSIDE
                    ]
                )
            elif self.task in {"punctuation_group", "punctuation_type"}:
                # 严格级联阶段：损失只来自金标准上游激活的位置。
                label_ids.append(
                    self.label_to_id[gold] if gold in self.label_to_id else IGNORE_LABEL
                )
            elif gold in self.label_to_id:
                # A1联合模型同时学习O和七种停顿标点。
                label_ids.append(self.label_to_id[gold])
            else:
                label_ids.append(IGNORE_LABEL)
        return _EncodedSequence(
            token_ids, tuple(label_ids), tuple(stage_ids), tuple(group_ids)
        )

    def _encode(
        self, sequences: list[SequenceChunk], drop_empty_labels: bool
    ) -> list[_EncodedSequence]:
        encoded = [self._encode_sequence(sequence) for sequence in sequences]
        if drop_empty_labels:
            encoded = [
                sequence
                for sequence in encoded
                if any(label != IGNORE_LABEL for label in sequence.label_ids)
            ]
        return encoded

    @staticmethod
    def _collate(batch: list[_EncodedSequence]) -> dict[str, torch.Tensor]:
        lengths = torch.tensor([len(item.token_ids) for item in batch], dtype=torch.long)
        max_length = int(lengths.max().item())
        token_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
        stage_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
        group_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
        label_ids = torch.full(
            (len(batch), max_length), IGNORE_LABEL, dtype=torch.long
        )
        for row, item in enumerate(batch):
            length = len(item.token_ids)
            token_ids[row, :length] = torch.tensor(item.token_ids, dtype=torch.long)
            stage_ids[row, :length] = torch.tensor(item.stage_ids, dtype=torch.long)
            group_ids[row, :length] = torch.tensor(item.group_ids, dtype=torch.long)
            label_ids[row, :length] = torch.tensor(item.label_ids, dtype=torch.long)
        padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
        return {
            "token_ids": token_ids,
            "stage_ids": stage_ids,
            "group_ids": group_ids,
            "label_ids": label_ids,
            "lengths": lengths,
            "padding_mask": padding_mask,
        }

    def _loader(
        self, data: list[_EncodedSequence], shuffle: bool
    ) -> DataLoader[_EncodedSequence]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            generator=generator,
        )

    def _build_model(self) -> nn.Module:
        if self.encoder_type == "bilstm":
            return _BiLSTMClassifier(
                len(self.vocabulary),
                len(self.labels),
                self.config.embedding_dim,
                self.config.bilstm_hidden_dim,
                self.config.bilstm_layers,
                self.config.dropout,
                self.use_group_features,
                use_crf=self.decoder_type == "crf",
            )
        return _RandomTransformerClassifier(
            len(self.vocabulary),
            len(self.labels),
            self.config.embedding_dim,
            self.max_sequence_length,
            self.config.transformer_layers,
            self.config.transformer_heads,
            self.config.transformer_ff_dim,
            self.config.dropout,
            self.use_group_features,
        )

    def _crf(self) -> LinearChainCRF:
        if self.model is None:
            raise RuntimeError("神经模型尚未建立")
        crf = getattr(self.model, "crf", None)
        if not isinstance(crf, LinearChainCRF):
            raise RuntimeError("当前模型没有线性链CRF解码层")
        return crf

    def _crf_batch_loss(
        self, emissions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """逐条压紧有效位置，避免IGNORE位置参与第二阶段标签转移。"""
        total_nll = emissions.sum() * 0.0
        valid_count = 0
        crf = self._crf()
        for row in range(labels.shape[0]):
            valid = labels[row] != IGNORE_LABEL
            count = int(valid.sum().item())
            if count == 0:
                continue
            total_nll = total_nll + crf.neg_log_likelihood(
                emissions[row, valid], labels[row, valid]
            )
            valid_count += count
        if valid_count == 0:
            raise RuntimeError("当前批次没有可用于CRF训练的标签")
        return total_nll / valid_count

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("神经模型尚未建立")
        return self.model(
            batch["token_ids"].to(self.device),
            batch["stage_ids"].to(self.device),
            batch["group_ids"].to(self.device),
            batch["lengths"].to(self.device),
            batch["padding_mask"].to(self.device),
        )

    def _epoch_loss(
        self,
        loader: DataLoader[_EncodedSequence],
        loss_function: nn.CrossEntropyLoss | None,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        if self.model is None:
            raise RuntimeError("神经模型尚未建立")
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        total_labels = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                labels = batch["label_ids"].to(self.device)
                valid_labels = int((labels != IGNORE_LABEL).sum().item())
                if valid_labels == 0:
                    continue
                if training:
                    optimizer.zero_grad()
                logits = self._forward(batch)
                if self.decoder_type == "crf":
                    loss = self._crf_batch_loss(logits, labels)
                else:
                    if loss_function is None:
                        raise RuntimeError("Softmax模型缺少交叉熵损失函数")
                    loss = loss_function(
                        logits.reshape(-1, len(self.labels)), labels.reshape(-1)
                    )
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    optimizer.step()
                total_loss += float(loss.item()) * valid_labels
                total_labels += valid_labels
        return total_loss / total_labels if total_labels else float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError("神经模型尚未建立")
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _prepare_epoch(self, epoch: int) -> None:
        """预留给预训练编码器实现冻结/解冻等逐轮策略。"""

        return None

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        if not train:
            raise ValueError("神经模型训练序列不能为空")
        self._set_seed()
        self._infer_task(train)
        self._build_vocabulary(train)
        train_data = self._encode(train, drop_empty_labels=True)
        dev_sequences = dev or []
        dev_data = self._encode(dev_sequences, drop_empty_labels=True)
        if not train_data:
            raise ValueError("训练折没有有效神经模型标签")

        self.model = self._build_model().to(self.device)
        optimizer = self._build_optimizer()
        loss_function = (
            None
            if self.decoder_type == "crf"
            else nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        )
        train_loader = self._loader(train_data, shuffle=True)
        dev_loader = self._loader(dev_data, shuffle=False) if dev_data else None
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0

        encoder_name = getattr(self, "encoder_display_name", None) or {
            "bilstm": "BiLSTM",
            "random_transformer": "Random-Transformer",
            "tangut_encoder": "TangutEncoder",
        }.get(self.encoder_type, self.encoder_type)
        initialization = getattr(self, "initialization_description", None) or (
            "D1上下文MLM预训练"
            if self.encoder_type == "tangut_encoder"
            else "随机可训练字符嵌入"
        )
        LOGGER.info(
            "%s神经模型：阶段=%s，解码=%s，设备=%s，初始化=%s，词表=%d",
            encoder_name,
            self.task,
            "线性链CRF/Viterbi" if self.decoder_type == "crf" else "逐位置Softmax",
            self.device,
            initialization,
            len(self.vocabulary),
        )
        for epoch in range(1, self.config.epochs + 1):
            self._prepare_epoch(epoch)
            train_loss = self._epoch_loss(train_loader, loss_function, optimizer)
            dev_loss = (
                self._epoch_loss(dev_loader, loss_function, None)
                if dev_loader is not None
                else train_loss
            )
            self.training_history.append(
                {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
            )
            LOGGER.info(
                "[%s][%s] epoch %d/%d：train_loss=%.6f，dev_loss=%.6f",
                self.encoder_type,
                self.task,
                epoch,
                self.config.epochs,
                train_loss,
                dev_loss,
            )
            if dev_loss < self.best_dev_loss - self.config.min_delta:
                self.best_dev_loss = dev_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
                LOGGER.info(
                    "[%s][%s] 开发集loss达到新最佳，保存第%d轮参数",
                    self.encoder_type,
                    self.task,
                    epoch,
                )
            else:
                stale_epochs += 1
                LOGGER.info(
                    "[%s][%s] 开发集loss改善不足min_delta=%g，早停计数=%d/%d",
                    self.encoder_type,
                    self.task,
                    self.config.min_delta,
                    stale_epochs,
                    self.config.patience,
                )
                if stale_epochs >= self.config.patience:
                    LOGGER.info(
                        "[%s][%s] 开发集连续%d轮未改善，提前停止",
                        self.encoder_type,
                        self.task,
                        stale_epochs,
                    )
                    break
        self.model.load_state_dict(best_state)
        if (
            self.task == "position"
            and self.decoder_type == "softmax"
            and dev_sequences
        ):
            self.position_threshold = self._select_position_threshold(dev_sequences)
            LOGGER.info(
                "[%s][position] 最佳epoch=%d，开发集位置阈值=%.2f",
                self.encoder_type,
                self.best_epoch,
                self.position_threshold,
            )
        elif self.task == "position" and self.decoder_type == "crf":
            LOGGER.info(
                "[%s][position] 最佳epoch=%d，使用CRF的Viterbi路径解码，不调位置阈值",
                self.encoder_type,
                self.best_epoch,
            )

    def _predict_logits(
        self, sequences: list[SequenceChunk]
    ) -> list[tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]]:
        if self.model is None:
            raise RuntimeError("神经模型尚未训练")
        encoded = self._encode(sequences, drop_empty_labels=False)
        loader = self._loader(encoded, shuffle=False)
        self.model.eval()
        outputs: list[tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]] = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                logits = self._forward(batch).cpu()
                lengths = batch["lengths"].tolist()
                for row, length in enumerate(lengths):
                    stage_ids = encoded[offset + row].stage_ids
                    group_ids = encoded[offset + row].group_ids
                    outputs.append((logits[row, :length], stage_ids, group_ids))
                offset += len(lengths)
        return outputs

    @staticmethod
    def _prf(gold: list[bool], predicted: list[bool]) -> tuple[float, float]:
        tp = sum(expected and actual for expected, actual in zip(gold, predicted))
        fp = sum(not expected and actual for expected, actual in zip(gold, predicted))
        fn = sum(expected and not actual for expected, actual in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, f1

    def _threshold_values(self) -> list[float]:
        values = []
        value = self.config.position_threshold_start
        while value <= self.config.position_threshold_end + 1e-12:
            values.append(round(value, 10))
            value += self.config.position_threshold_step
        return values

    def _select_position_threshold(self, dev: list[SequenceChunk]) -> float:
        scores = [score for part in self.predict_position_probabilities(dev) for score in part]
        gold = [label == POSITION_LABEL for chunk in dev for label in chunk.labels]
        best = (-1.0, -1.0, 0.5)
        for threshold in self._threshold_values():
            precision, f1 = self._prf(
                gold, [score >= threshold for score in scores]
            )
            best = max(best, (f1, precision, threshold))
        return best[2]

    def predict_position_probabilities(
        self, sequences: list[SequenceChunk]
    ) -> list[list[float]]:
        """返回每个字符后的标点位置概率，供C2/C3折外级联使用。"""
        if self.task != "position" or self.decoder_type != "softmax":
            raise RuntimeError("只有Softmax位置模型能够输出连续位置概率")
        positive_id = self.label_to_id[POSITION_LABEL]
        return [
            torch.softmax(logits, dim=-1)[:, positive_id].tolist()
            for logits, _, _ in self._predict_logits(sequences)
        ]

    def predict_position_at_threshold(
        self, sequences: list[SequenceChunk], threshold: float
    ) -> list[list[str]]:
        if not 0 <= threshold <= 1:
            raise ValueError("位置阈值必须位于[0, 1]")
        return [
            [POSITION_LABEL if score >= threshold else OUTSIDE for score in part]
            for part in self.predict_position_probabilities(sequences)
        ]

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        outputs = self._predict_logits(sequences)
        predictions: list[list[str]] = []
        if self.task == "position":
            if self.decoder_type == "crf":
                crf = self._crf()
                for logits, _, _ in outputs:
                    predicted_ids = crf.decode(logits.to(self.device))
                    predictions.append(
                        [self.labels[label_id] for label_id in predicted_ids]
                    )
                return predictions
            positive_id = self.label_to_id[POSITION_LABEL]
            if self.position_threshold is None:
                raise RuntimeError("Softmax位置模型尚未设置预测阈值")
            for logits, _, _ in outputs:
                scores = torch.softmax(logits, dim=-1)[:, positive_id]
                predictions.append(
                    [
                        POSITION_LABEL
                        if float(score) >= self.position_threshold
                        else OUTSIDE
                        for score in scores
                    ]
                )
            return predictions

        for logits, stage_ids, group_ids in outputs:
            if self.decoder_type == "crf":
                active_indices = [
                    index
                    for index, (stage_id, group_id) in enumerate(
                        zip(stage_ids, group_ids)
                    )
                    if self.task in {"punctuation_joint", "punctuation_group_joint"}
                    or (
                        stage_id == STAGE_POSITION
                        and (
                            not self.use_group_features
                            or group_id in {GROUP_SENTENCE, GROUP_INTRA}
                        )
                    )
                ]
                active_emissions = logits[active_indices].to(self.device)
                active_predictions = self._crf().decode(active_emissions)
                values = [OUTSIDE] * logits.shape[0]
                for index, label_id in zip(active_indices, active_predictions):
                    values[index] = self.labels[label_id]
                predictions.append(values)
                continue
            predicted_ids = logits.argmax(dim=-1).tolist()
            if self.task in {"punctuation_joint", "punctuation_group_joint"}:
                predictions.append([self.labels[label_id] for label_id in predicted_ids])
                continue
            predictions.append(
                [
                    self.labels[label_id]
                    if stage_id == STAGE_POSITION
                    and (
                        not self.use_group_features
                        or group_id in {GROUP_SENTENCE, GROUP_INTRA}
                    )
                    else OUTSIDE
                    for label_id, stage_id, group_id in zip(
                        predicted_ids, stage_ids, group_ids
                    )
                ]
            )
        return predictions

    def metadata(self) -> dict[str, object]:
        parameter_count = (
            sum(parameter.numel() for parameter in self.model.parameters())
            if self.model is not None
            else 0
        )
        return {
            "神经编码器": self.encoder_type,
            "解码层": (
                "线性链CRF（Viterbi）"
                if self.decoder_type == "crf"
                else "逐位置Softmax"
            ),
            "模型阶段": {
                "position": "位置",
                "punctuation_group": "句间/句内类别",
                "punctuation_group_joint": "O与句内/句间两类停顿联合预测",
                "punctuation_type": "具体类别（级联）",
                "punctuation_joint": "O与七种标点联合预测",
            }.get(self.task, self.task),
            "嵌入初始化": "随机初始化、训练中更新",
            "嵌入维度": self.config.embedding_dim,
            "词表大小": len(self.vocabulary),
            "参数量": parameter_count,
            "设备": str(self.device),
            "设备选择依据": self.device_selection,
            "使用句间/句内上游特征": self.use_group_features,
            "最佳epoch": self.best_epoch,
            "最佳开发集loss": self.best_dev_loss,
            "开发集位置阈值": (
                self.position_threshold
                if self.task == "position" and self.decoder_type == "softmax"
                else None
            ),
            "训练记录": self.training_history,
        }

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("神经模型尚未训练")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_type": self.encoder_type,
                "decoder_type": self.decoder_type,
                "task": self.task,
                "punctuation_labels": self.punctuation_labels,
                "labels": self.labels,
                "vocabulary": self.vocabulary,
                "neural_config": asdict(self.config),
                "max_sequence_length": self.max_sequence_length,
                "seed": self.seed,
                "position_threshold": self.position_threshold,
                "best_epoch": self.best_epoch,
                "state_dict": self.model.state_dict(),
            },
            path,
        )
