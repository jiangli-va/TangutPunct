from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import PretrainingConfig


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, MASK_TOKEN)


class TangutEncoder(nn.Module):
    """小型字符级西夏文Transformer编码器。

    MLM输出层与字符嵌入共享权重；下游任务使用 ``encode`` 返回的逐字
    上下文表示，MLM预测头无需随下游模型加载。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 192,
        layers: int = 3,
        heads: int = 4,
        ff_dim: int = 768,
        max_sequence_length: int = 128,
        dropout: float = 0.15,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.pad_id = pad_id

        self.character_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlm_decoder = nn.Linear(embedding_dim, vocab_size, bias=True)
        self.mlm_decoder.weight = self.character_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.character_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlm_decoder.bias)
        with torch.no_grad():
            self.character_embedding.weight[self.pad_id].zero_()

    def encode(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("TangutEncoder输入必须是[batch, sequence]二维张量")
        sequence_length = input_ids.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"输入长度{sequence_length}超过模型上限{self.max_sequence_length}"
            )
        if padding_mask is None:
            padding_mask = input_ids.eq(self.pad_id)
        positions = torch.arange(
            sequence_length, device=input_ids.device
        ).unsqueeze(0)
        hidden = self.character_embedding(input_ids) + self.position_embedding(
            positions
        )
        hidden = self.dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.final_norm(hidden)

    def forward(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(input_ids, padding_mask)
        return self.mlm_decoder(hidden), hidden


def build_tangut_encoder(
    vocabulary_size: int, config: PretrainingConfig, pad_id: int = 0
) -> TangutEncoder:
    return TangutEncoder(
        vocab_size=vocabulary_size,
        embedding_dim=config.embedding_dim,
        layers=config.layers,
        heads=config.heads,
        ff_dim=config.ff_dim,
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout,
        pad_id=pad_id,
    )


def load_tangut_encoder_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TangutEncoder, dict[str, int], dict[str, Any]]:
    """加载D阶段checkpoint，供后续词语预训练和标点微调复用。"""

    checkpoint: dict[str, Any] = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    if checkpoint.get("format") != "tangut_encoder":
        raise ValueError(f"{path}不是TangutEncoder checkpoint")
    vocabulary = {
        str(token): int(index)
        for token, index in checkpoint["vocabulary"].items()
    }
    model_config = checkpoint["model_config"]
    model = TangutEncoder(
        vocab_size=len(vocabulary),
        embedding_dim=int(model_config["embedding_dim"]),
        layers=int(model_config["layers"]),
        heads=int(model_config["heads"]),
        ff_dim=int(model_config["ff_dim"]),
        max_sequence_length=int(model_config["max_sequence_length"]),
        dropout=float(model_config["dropout"]),
        pad_id=vocabulary[PAD_TOKEN],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocabulary, checkpoint


def checkpoint_model_config(config: PretrainingConfig) -> dict[str, Any]:
    """只保存重建编码器所需字段，避免后续阶段依赖整个运行配置。"""

    values = asdict(config)
    return {
        key: values[key]
        for key in (
            "embedding_dim",
            "layers",
            "heads",
            "ff_dim",
            "max_sequence_length",
            "dropout",
        )
    }
