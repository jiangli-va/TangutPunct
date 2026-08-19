from __future__ import annotations

import torch
from torch import nn


class LinearChainCRF(nn.Module):
    """可复用的一阶线性链CRF及Viterbi解码层。"""

    def __init__(self, label_count: int) -> None:
        super().__init__()
        if label_count < 2:
            raise ValueError("线性链CRF至少需要两个标签")
        self.label_count = label_count
        self.start_transitions = nn.Parameter(torch.empty(label_count))
        self.end_transitions = nn.Parameter(torch.empty(label_count))
        # transitions[previous, current]
        self.transitions = nn.Parameter(torch.empty(label_count, label_count))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def neg_log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor
    ) -> torch.Tensor:
        """返回一条非空标签序列的负条件对数似然。"""
        if emissions.ndim != 2 or emissions.shape[1] != self.label_count:
            raise ValueError("CRF发射分数形状应为[序列长度, 标签数]")
        if tags.ndim != 1 or tags.shape[0] != emissions.shape[0]:
            raise ValueError("CRF标签形状与发射分数不一致")
        if emissions.shape[0] == 0:
            raise ValueError("CRF不能计算空序列")

        gold_score = self.start_transitions[tags[0]] + emissions[0, tags[0]]
        for index in range(1, tags.shape[0]):
            gold_score = (
                gold_score
                + self.transitions[tags[index - 1], tags[index]]
                + emissions[index, tags[index]]
            )
        gold_score = gold_score + self.end_transitions[tags[-1]]

        forward_score = self.start_transitions + emissions[0]
        for index in range(1, emissions.shape[0]):
            candidate_scores = forward_score.unsqueeze(1) + self.transitions
            forward_score = torch.logsumexp(candidate_scores, dim=0) + emissions[index]
        log_partition = torch.logsumexp(
            forward_score + self.end_transitions, dim=0
        )
        return log_partition - gold_score

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor) -> list[int]:
        """用Viterbi返回最高分标签路径。"""
        if emissions.ndim != 2 or emissions.shape[1] != self.label_count:
            raise ValueError("CRF发射分数形状应为[序列长度, 标签数]")
        if emissions.shape[0] == 0:
            return []

        score = self.start_transitions + emissions[0]
        history: list[torch.Tensor] = []
        for index in range(1, emissions.shape[0]):
            candidate_scores = score.unsqueeze(1) + self.transitions
            best_score, best_previous = candidate_scores.max(dim=0)
            score = best_score + emissions[index]
            history.append(best_previous)

        current = int((score + self.end_transitions).argmax().item())
        path = [current]
        for best_previous in reversed(history):
            current = int(best_previous[current].item())
            path.append(current)
        path.reverse()
        return path
