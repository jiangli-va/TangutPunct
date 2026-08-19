from __future__ import annotations

import torch
from torch import nn

from .tangut_encoder import TangutEncoder


class D5PunctuationModel(nn.Module):
    """共享TangutEncoder的MLM、位置、粗类和七类标点任务头。"""

    def __init__(
        self,
        encoder: TangutEncoder,
        punctuation_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.position_head = nn.Linear(encoder.output_dim, 2)
        self.group_head = nn.Linear(encoder.output_dim, 2)
        self.type_head = nn.Linear(encoder.output_dim, punctuation_count)

    def punctuation_logits(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dropout(self.encoder.encode(input_ids, padding_mask))
        return (
            self.position_head(hidden),
            self.group_head(hidden),
            self.type_head(hidden),
        )

    def mlm_logits(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.encoder(input_ids, padding_mask)
        return logits
