from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import NeuralConfig


class ContextEncoder(nn.Module, ABC):
    """字符序列编码器统一接口；任务头只依赖output_dim和上下文输出。"""

    output_dim: int

    @abstractmethod
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """返回形状为[batch, length, output_dim]的上下文表示。"""
        raise NotImplementedError


class BiLSTMContextEncoder(ContextEncoder):
    def __init__(self, vocab_size: int, config: NeuralConfig) -> None:
        super().__init__()
        self.output_dim = config.bilstm_hidden_dim * 2
        self.character_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=0
        )
        self.dropout = nn.Dropout(config.dropout)
        self.sequence_encoder = nn.LSTM(
            config.embedding_dim,
            config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.bilstm_layers > 1 else 0.0,
        )
        nn.init.normal_(self.character_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.character_embedding.weight[0].zero_()

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.dropout(self.character_embedding(token_ids)),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.sequence_encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=token_ids.shape[1]
        )
        return self.dropout(encoded)


EncoderBuilder = Callable[[int, NeuralConfig], ContextEncoder]
ENCODER_REGISTRY: dict[str, EncoderBuilder] = {"bilstm": BiLSTMContextEncoder}


def register_context_encoder(name: str, builder: EncoderBuilder) -> None:
    """注册新编码器；TangutEncoder只需实现ContextEncoder并在此注册。"""
    if not name:
        raise ValueError("编码器名称不能为空")
    ENCODER_REGISTRY[name] = builder


def build_context_encoder(
    name: str, vocab_size: int, config: NeuralConfig
) -> ContextEncoder:
    try:
        return ENCODER_REGISTRY[name](vocab_size, config)
    except KeyError as error:
        raise ValueError(
            f"未知共享编码器{name!r}；可用编码器：{sorted(ENCODER_REGISTRY)}"
        ) from error
