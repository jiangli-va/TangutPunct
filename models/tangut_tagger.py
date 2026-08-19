from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (
    CascadeConfig,
    NeuralConfig,
    POSRelationKnowledgeConfig,
    PretrainingConfig,
)
from data.corpus import SequenceChunk
from knowledge import KnowledgeFeatureProvider
from knowledge.base import FeatureMatrix
from models.tangut_encoder import TangutEncoder, load_tangut_encoder_checkpoint
from tasks import OUTSIDE

from .crf_layer import LinearChainCRF
from .neural import IGNORE_LABEL, POSITION_LABEL, NeuralSequenceTagger


LOGGER = logging.getLogger(__name__)


class _TangutEncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder: TangutEncoder,
        label_count: int,
        dropout: float,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.output_dim, label_count)
        self.crf = LinearChainCRF(label_count) if use_crf else None

    def forward(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        del stage_ids, group_ids, lengths
        hidden = self.encoder.encode(token_ids, padding_mask)
        return self.classifier(self.dropout(hidden))


class _TangutEncoderBiLSTMClassifier(nn.Module):
    """以TangutEncoder表示代替随机字符嵌入，再复用B4最优BiLSTM头。"""

    def __init__(
        self,
        encoder: TangutEncoder,
        label_count: int,
        config: NeuralConfig,
        use_group_features: bool,
        knowledge_feature_dim: int = 0,
        residual_knowledge_dim: int = 0,
        projected_knowledge_dim: int = 0,
        projected_knowledge_output_dim: int = 0,
        projected_knowledge_dropout: float = 0.0,
        direct_knowledge_channel_dim: int = 0,
        direct_knowledge_channel_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # B4通过stage embedding读取上游O/P序列。D4保留同一机制，
        # 但将其与TangutEncoder上下文表示拼接，而不是替代字符表示。
        self.stage_embedding = nn.Embedding(
            3, config.embedding_dim, padding_idx=0
        )
        self.group_embedding = (
            nn.Embedding(3, config.embedding_dim, padding_idx=0)
            if use_group_features
            else None
        )
        if residual_knowledge_dim < 0 or projected_knowledge_dim < 0:
            raise ValueError("残差知识维度和投影知识维度不能为负")
        if residual_knowledge_dim + projected_knowledge_dim > knowledge_feature_dim:
            raise ValueError("残差知识维度与投影知识维度之和不能超过总知识维度")
        if projected_knowledge_dim:
            if projected_knowledge_output_dim <= 0:
                raise ValueError("启用投影知识时输出维度必须大于0")
            if not 0 <= projected_knowledge_dropout < 1:
                raise ValueError("投影知识整通道dropout必须位于[0,1)")
        elif projected_knowledge_output_dim or projected_knowledge_dropout:
            raise ValueError("未启用投影知识时不能配置输出维度或dropout")
        self.knowledge_feature_dim = knowledge_feature_dim
        self.residual_knowledge_dim = residual_knowledge_dim
        self.projected_knowledge_dim = projected_knowledge_dim
        self.projected_knowledge_output_dim = projected_knowledge_output_dim
        self.projected_knowledge_dropout = projected_knowledge_dropout
        self.direct_knowledge_dim = (
            knowledge_feature_dim
            - residual_knowledge_dim
            - projected_knowledge_dim
        )
        if not 0 <= direct_knowledge_channel_dim <= self.direct_knowledge_dim:
            raise ValueError("直接拼接知识子通道维度必须位于[0, 直接知识维度]")
        if direct_knowledge_channel_dim:
            if not 0 <= direct_knowledge_channel_dropout < 1:
                raise ValueError("直接拼接知识子通道dropout必须位于[0,1)")
        elif direct_knowledge_channel_dropout:
            raise ValueError("未指定直接知识子通道时不能配置其dropout")
        self.direct_knowledge_channel_dim = direct_knowledge_channel_dim
        self.direct_knowledge_channel_dropout = direct_knowledge_channel_dropout
        input_dim = encoder.output_dim + config.embedding_dim
        if self.group_embedding is not None:
            input_dim += config.embedding_dim
        input_dim += self.direct_knowledge_dim
        input_dim += self.projected_knowledge_output_dim
        self.dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(
            input_dim,
            config.bilstm_hidden_dim,
            num_layers=config.bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.bilstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(config.bilstm_hidden_dim * 2, label_count)
        self.crf = None
        # 在BiLSTM之后创建投影，保证基础BiLSTM的输入维度与E3完全一致。
        # tanh(0)=0，因此训练开始时分词知识不会覆盖已有效的E3表示。
        self.residual_projection = (
            nn.Linear(residual_knowledge_dim, encoder.output_dim, bias=False)
            if residual_knowledge_dim
            else None
        )
        self.residual_gate = (
            nn.Parameter(torch.zeros(())) if residual_knowledge_dim else None
        )
        self.projected_knowledge_layer = (
            nn.Sequential(
                nn.Linear(
                    projected_knowledge_dim,
                    projected_knowledge_output_dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.LayerNorm(projected_knowledge_output_dim),
            )
            if projected_knowledge_dim
            else None
        )
        self._initialize_stage_embeddings()

    def _initialize_stage_embeddings(self) -> None:
        nn.init.normal_(self.stage_embedding.weight, mean=0.0, std=0.02)
        if self.group_embedding is not None:
            nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.stage_embedding.weight[0].zero_()
            if self.group_embedding is not None:
                self.group_embedding.weight[0].zero_()

    def encode(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
        knowledge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encoder.encode(token_ids, padding_mask)
        direct_knowledge = knowledge_features
        projected_knowledge = None
        if self.knowledge_feature_dim:
            if knowledge_features is None:
                raise RuntimeError("知识增强模型缺少逐字符知识特征")
            if knowledge_features.shape[-1] != self.knowledge_feature_dim:
                raise ValueError("知识特征实际维度与模型配置不一致")
            direct_knowledge = knowledge_features[..., : self.direct_knowledge_dim]
            if (
                self.training
                and self.direct_knowledge_channel_dim
                and self.direct_knowledge_channel_dropout
            ):
                channel_start = (
                    self.direct_knowledge_dim - self.direct_knowledge_channel_dim
                )
                base_direct = direct_knowledge[..., :channel_start]
                channel = direct_knowledge[..., channel_start:]
                keep = (
                    torch.rand(
                        (channel.shape[0], 1, 1),
                        device=channel.device,
                    )
                    >= self.direct_knowledge_channel_dropout
                ).to(channel.dtype)
                channel = channel * keep / (
                    1.0 - self.direct_knowledge_channel_dropout
                )
                direct_knowledge = torch.cat((base_direct, channel), dim=-1)
            projected_end = self.direct_knowledge_dim + self.projected_knowledge_dim
            if self.projected_knowledge_dim:
                if self.projected_knowledge_layer is None:
                    raise RuntimeError("词性知识投影层尚未建立")
                raw_projected = knowledge_features[
                    ..., self.direct_knowledge_dim : projected_end
                ]
                # LayerNorm之后再次施加有效位掩码，保证缺字邻接和padding
                # 的全零POS行不会因可学习仿射参数变成伪特征。
                valid = raw_projected.abs().sum(dim=-1, keepdim=True).gt(0)
                projected_knowledge = self.projected_knowledge_layer(raw_projected)
                projected_knowledge = projected_knowledge * valid.to(
                    projected_knowledge.dtype
                )
                if self.training and self.projected_knowledge_dropout:
                    keep = (
                        torch.rand(
                            (projected_knowledge.shape[0], 1, 1),
                            device=projected_knowledge.device,
                        )
                        >= self.projected_knowledge_dropout
                    ).to(projected_knowledge.dtype)
                    projected_knowledge = projected_knowledge * keep / (
                        1.0 - self.projected_knowledge_dropout
                    )
            if self.residual_knowledge_dim:
                if self.residual_projection is None or self.residual_gate is None:
                    raise RuntimeError("分词知识门控残差层尚未建立")
                residual = knowledge_features[..., projected_end:]
                hidden = hidden + torch.tanh(self.residual_gate) * self.residual_projection(
                    residual
                )
        features = [hidden, self.stage_embedding(stage_ids)]
        if self.group_embedding is not None:
            features.append(self.group_embedding(group_ids))
        if self.direct_knowledge_dim:
            if direct_knowledge is None:
                raise RuntimeError("直接拼接知识特征缺失")
            features.append(direct_knowledge)
        if self.projected_knowledge_output_dim:
            if projected_knowledge is None:
                raise RuntimeError("投影知识特征缺失")
            features.append(projected_knowledge)
        packed = pack_padded_sequence(
            self.dropout(torch.cat(features, dim=-1)),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.bilstm(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=token_ids.shape[1]
        )
        return self.dropout(encoded)

    def forward(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
        knowledge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encode(
            token_ids,
            stage_ids,
            group_ids,
            lengths,
            padding_mask,
            knowledge_features,
        )
        return self.classifier(encoded)


class _TangutEncoderBiLSTMMultiTaskClassifier(_TangutEncoderBiLSTMClassifier):
    """E9：E3共享主干上的位置辅助头与完整标点主任务头。"""

    def __init__(
        self,
        encoder: TangutEncoder,
        punctuation_count: int,
        config: NeuralConfig,
        knowledge_feature_dim: int,
    ) -> None:
        super().__init__(
            encoder,
            punctuation_count,
            config,
            use_group_features=False,
            knowledge_feature_dim=knowledge_feature_dim,
        )
        # 删除父类的单任务分类器，避免保存无用参数；两个任务严格共享
        # TangutEncoder、E3知识输入和BiLSTM，只在最终线性层处分叉。
        del self.classifier
        output_dim = config.bilstm_hidden_dim * 2
        self.position_head = nn.Linear(output_dim, 2)
        self.punctuation_head = nn.Linear(output_dim, punctuation_count)

    def forward(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
        knowledge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(
            token_ids,
            stage_ids,
            group_ids,
            lengths,
            padding_mask,
            knowledge_features,
        )
        return self.position_head(encoded), self.punctuation_head(encoded)


class _TangutEncoderBiLSTMPOSResidualClassifier(_TangutEncoderBiLSTMClassifier):
    """E3基础logits＋细粒度左右词性关系的后期门控残差。"""

    def __init__(
        self,
        encoder: TangutEncoder,
        label_count: int,
        config: NeuralConfig,
        use_group_features: bool,
        base_knowledge_dim: int,
        relation_knowledge_dim: int,
        pos_tag_count: int,
        relation_config: POSRelationKnowledgeConfig,
    ) -> None:
        expected_relation_dim = pos_tag_count * 2 + 4
        if relation_knowledge_dim != expected_relation_dim:
            raise ValueError(
                "E7/E8词性关系维度不一致："
                f"期望{expected_relation_dim}，实际{relation_knowledge_dim}"
            )
        super().__init__(
            encoder,
            label_count,
            config,
            use_group_features,
            knowledge_feature_dim=base_knowledge_dim,
        )
        self.base_knowledge_dim = base_knowledge_dim
        self.relation_knowledge_dim = relation_knowledge_dim
        self.total_knowledge_dim = base_knowledge_dim + relation_knowledge_dim
        self.pos_tag_count = pos_tag_count
        self.relation_config = relation_config
        embedding_dim = relation_config.tag_embedding_dim
        relation_input_dim = embedding_dim * 4 + 4
        self.pos_tag_embedding = nn.Parameter(
            torch.empty(pos_tag_count, embedding_dim)
        )
        self.relation_projection = nn.Linear(
            relation_input_dim, relation_config.relation_hidden_dim
        )
        self.relation_activation = nn.GELU()
        self.relation_dropout = nn.Dropout(relation_config.relation_dropout)
        self.relation_classifier = nn.Linear(
            relation_config.relation_hidden_dim, label_count
        )
        self.relation_gate = nn.Linear(relation_input_dim, 1)
        self.register_buffer("fusion_weight", torch.tensor(0.0))
        self._initialize_relation_branch()

    def _initialize_relation_branch(self) -> None:
        nn.init.normal_(self.pos_tag_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.relation_projection.weight)
        nn.init.zeros_(self.relation_projection.bias)
        # 初始delta严格为0，第一阶段即使误开融合也等价于E3。
        nn.init.zeros_(self.relation_classifier.weight)
        nn.init.zeros_(self.relation_classifier.bias)
        nn.init.zeros_(self.relation_gate.weight)
        nn.init.constant_(self.relation_gate.bias, self.relation_config.gate_bias)

    def relation_parameters(self) -> list[nn.Parameter]:
        return [
            self.pos_tag_embedding,
            *self.relation_projection.parameters(),
            *self.relation_classifier.parameters(),
            *self.relation_gate.parameters(),
        ]

    def freeze_relation_branch(self) -> None:
        for parameter in self.relation_parameters():
            parameter.requires_grad = False
        self.set_fusion_weight(0.0)

    def freeze_base_and_enable_relation(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.relation_parameters():
            parameter.requires_grad = True

    def set_fusion_weight(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("E7/E8融合权重必须位于[0,1]")
        self.fusion_weight.fill_(float(value))

    def _relation_input(
        self, raw_relation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tag_count = self.pos_tag_count
        left = raw_relation[..., :tag_count]
        right = raw_relation[..., tag_count : tag_count * 2]
        scalars = raw_relation[..., tag_count * 2 :]
        left_embedding = left @ self.pos_tag_embedding
        right_embedding = right @ self.pos_tag_embedding
        relation = torch.cat(
            (
                left_embedding,
                right_embedding,
                left_embedding * right_embedding,
                (left_embedding - right_embedding).abs(),
                scalars,
            ),
            dim=-1,
        )
        valid = raw_relation.abs().sum(dim=-1, keepdim=True).gt(0)
        if self.training and self.relation_config.channel_dropout:
            keep = (
                torch.rand((relation.shape[0], 1, 1), device=relation.device)
                >= self.relation_config.channel_dropout
            ).to(relation.dtype)
            relation = relation * keep / (1.0 - self.relation_config.channel_dropout)
            valid = valid & keep.bool()
        return relation, valid

    def forward(
        self,
        token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        group_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
        knowledge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if knowledge_features is None:
            raise RuntimeError("E7/E8模型缺少E3基础知识与词性关系特征")
        if knowledge_features.shape[-1] != self.total_knowledge_dim:
            raise ValueError("E7/E8知识特征实际维度与模型配置不一致")
        base_logits = super().forward(
            token_ids,
            stage_ids,
            group_ids,
            lengths,
            padding_mask,
            knowledge_features[..., : self.base_knowledge_dim],
        )
        raw_relation = knowledge_features[..., self.base_knowledge_dim :]
        relation, valid = self._relation_input(raw_relation)
        relation_hidden = self.relation_dropout(
            self.relation_activation(self.relation_projection(relation))
        )
        delta_logits = self.relation_classifier(relation_hidden)
        gate = torch.sigmoid(self.relation_gate(relation))
        correction = gate * delta_logits * valid.to(delta_logits.dtype)
        return base_logits + self.fusion_weight * correction


class TangutEncoderSequenceTagger(NeuralSequenceTagger):
    """加载任一D阶段TangutEncoder checkpoint并进行自动标点微调。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        punctuation_labels: Iterable[str],
        neural_config: NeuralConfig,
        pretraining_config: PretrainingConfig,
        max_sequence_length: int,
        seed: int,
        head_type: str = "softmax",
    ) -> None:
        if head_type not in {"softmax", "crf", "bilstm"}:
            raise ValueError("TangutEncoder下游头只能是softmax、crf或bilstm")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到TangutEncoder checkpoint：{self.checkpoint_path}。请先完成预训练，"
                "或用--checkpoint指定best_model.pt。"
            )
        _, vocabulary, checkpoint = load_tangut_encoder_checkpoint(
            self.checkpoint_path
        )
        self.pretrained_vocabulary = vocabulary
        self.pretrained_step = int(checkpoint.get("step", 0))
        self.pretrained_stage = str(checkpoint.get("stage", "unknown"))
        self.pretrained_embedding_dim = int(
            checkpoint["model_config"]["embedding_dim"]
        )
        self.pretraining_config = pretraining_config
        self.head_type = head_type
        model_max_length = int(checkpoint["model_config"]["max_sequence_length"])
        if max_sequence_length > model_max_length:
            raise ValueError(
                f"下游max_sequence_length={max_sequence_length}超过预训练位置嵌入上限"
                f"{model_max_length}；请在配置中调小max_sequence_length"
            )
        # NeuralSequenceTagger的CRF通用训练/解码流程原先只允许BiLSTM。
        # 此处用BiLSTM完成构造期校验，真正模型始终由_build_model重建。
        bootstrap_encoder = "bilstm" if head_type == "crf" else "random_transformer"
        super().__init__(
            bootstrap_encoder,
            punctuation_labels,
            neural_config,
            max_sequence_length,
            seed,
            decoder_type="crf" if head_type == "crf" else "softmax",
        )
        self.encoder_type = "tangut_encoder"
        self.encoder_display_name = {
            "softmax": "TangutEncoder",
            "crf": "TangutEncoder＋CRF",
            "bilstm": "TangutEncoder＋BiLSTM",
        }[head_type]
        self.initialization_description = (
            "D1上下文MLM预训练"
            if self.pretrained_stage == "d1_context_mlm"
            else f"TangutEncoder预训练（{self.pretrained_stage}）"
        )
        self._encoder_is_frozen: bool | None = None

    def _build_vocabulary(self, train: list[SequenceChunk]) -> None:
        del train
        # 所有折和所有阶段严格复用预训练字符ID，不按当前折重建词表。
        self.vocabulary = dict(self.pretrained_vocabulary)

    def _build_model(self) -> nn.Module:
        encoder, vocabulary, _ = load_tangut_encoder_checkpoint(
            self.checkpoint_path
        )
        if vocabulary != self.vocabulary:
            raise RuntimeError("TangutEncoder checkpoint中的字符词表在加载过程中发生变化")
        if self.head_type == "bilstm":
            return _TangutEncoderBiLSTMClassifier(
                encoder,
                len(self.labels),
                self.config,
                self.use_group_features,
            )
        return _TangutEncoderClassifier(
            encoder,
            len(self.labels),
            self.config.dropout,
            use_crf=self.head_type == "crf",
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if not isinstance(
            self.model, (_TangutEncoderClassifier, _TangutEncoderBiLSTMClassifier)
        ):
            raise RuntimeError("TangutEncoder下游模型尚未建立")
        downstream_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("encoder.")
        ]
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.pretraining_config.downstream_encoder_learning_rate,
                },
                {
                    "params": downstream_parameters,
                    "lr": self.pretraining_config.downstream_head_learning_rate,
                },
            ],
            weight_decay=self.config.weight_decay,
        )

    def _prepare_epoch(self, epoch: int) -> None:
        if not isinstance(
            self.model, (_TangutEncoderClassifier, _TangutEncoderBiLSTMClassifier)
        ):
            raise RuntimeError("TangutEncoder下游模型尚未建立")
        frozen = epoch <= self.pretraining_config.downstream_freeze_epochs
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen != self._encoder_is_frozen:
            state = "冻结" if frozen else "解冻并微调"
            LOGGER.info(
                "[tangut_encoder][%s] epoch=%d：编码器%s",
                self.task,
                epoch,
                state,
            )
            self._encoder_is_frozen = frozen

    def metadata(self) -> dict[str, object]:
        values = super().metadata()
        initialization = (
            "D1上下文MLM预训练，随后下游微调"
            if self.pretrained_stage == "d1_context_mlm"
            else f"TangutEncoder预训练（{self.pretrained_stage}），随后下游微调"
        )
        values.update(
            {
                "神经编码器": "tangut_encoder",
                "下游结构": {
                    "softmax": "TangutEncoder → 逐位置Softmax",
                    "crf": "TangutEncoder → 线性层 → CRF/Viterbi",
                    "bilstm": "TangutEncoder → BiLSTM → 逐位置Softmax",
                }[self.head_type],
                "嵌入初始化": initialization,
                "嵌入维度": self.pretrained_embedding_dim,
                "TangutEncoder维度": self.pretrained_embedding_dim,
                "预训练checkpoint": str(self.checkpoint_path),
                "预训练最佳步数": self.pretrained_step,
                "预训练阶段": self.pretrained_stage,
                "编码器学习率": self.pretraining_config.downstream_encoder_learning_rate,
                "下游头学习率": self.pretraining_config.downstream_head_learning_rate,
                "编码器冻结epoch": self.pretraining_config.downstream_freeze_epochs,
            }
        )
        if self.head_type == "bilstm":
            values.update(
                {
                    "BiLSTM隐藏维度（每方向）": self.config.bilstm_hidden_dim,
                    "BiLSTM层数": self.config.bilstm_layers,
                    "BiLSTM上游特征维度": self.config.embedding_dim,
                }
            )
        # 保留D1旧报告字段，避免既有结果解析脚本失效。
        if self.pretrained_stage == "d1_context_mlm":
            values.update(
                {
                    "D1 checkpoint": str(self.checkpoint_path),
                    "D1最佳步数": self.pretrained_step,
                    "D1阶段": self.pretrained_stage,
                }
            )
        return values

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("神经模型尚未训练")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_type": self.encoder_type,
                "head_type": self.head_type,
                "decoder_type": self.decoder_type,
                "task": self.task,
                "punctuation_labels": self.punctuation_labels,
                "labels": self.labels,
                "vocabulary": self.vocabulary,
                "pretrained_checkpoint": str(self.checkpoint_path),
                "pretrained_stage": self.pretrained_stage,
                "neural_config": asdict(self.config),
                "pretraining_config": asdict(self.pretraining_config),
                "max_sequence_length": self.max_sequence_length,
                "seed": self.seed,
                "position_threshold": self.position_threshold,
                "best_epoch": self.best_epoch,
                "state_dict": self.model.state_dict(),
            },
            path,
        )


@dataclass(frozen=True)
class _KnowledgeEncodedSequence:
    token_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    stage_ids: tuple[int, ...]
    group_ids: tuple[int, ...]
    knowledge_features: FeatureMatrix


class KnowledgeEnhancedTangutEncoderTagger(TangutEncoderSequenceTagger):
    """D4主干＋可组合的逐字符显式知识通道。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        punctuation_labels: Iterable[str],
        neural_config: NeuralConfig,
        pretraining_config: PretrainingConfig,
        max_sequence_length: int,
        seed: int,
        knowledge_provider: KnowledgeFeatureProvider,
        residual_knowledge_dim: int = 0,
        projected_knowledge_dim: int = 0,
        projected_knowledge_output_dim: int = 0,
        projected_knowledge_dropout: float = 0.0,
        direct_knowledge_channel_dim: int = 0,
        direct_knowledge_channel_dropout: float = 0.0,
    ) -> None:
        self.knowledge_provider = knowledge_provider
        self.residual_knowledge_dim = residual_knowledge_dim
        self.projected_knowledge_dim = projected_knowledge_dim
        self.projected_knowledge_output_dim = projected_knowledge_output_dim
        self.projected_knowledge_dropout = projected_knowledge_dropout
        self.direct_knowledge_channel_dim = direct_knowledge_channel_dim
        self.direct_knowledge_channel_dropout = direct_knowledge_channel_dropout
        self._knowledge_cache: dict[tuple[str, int, int], FeatureMatrix] = {}
        super().__init__(
            checkpoint_path,
            punctuation_labels,
            neural_config,
            pretraining_config,
            max_sequence_length,
            seed,
            head_type="bilstm",
        )
        self.encoder_display_name = "TangutEncoder＋显式知识＋BiLSTM"

    @staticmethod
    def _knowledge_key(chunk: SequenceChunk) -> tuple[str, int, int]:
        return chunk.document_id, chunk.offset, len(chunk.tokens)

    def _cache_features(
        self,
        sequences: list[SequenceChunk],
        matrices: list[FeatureMatrix],
    ) -> None:
        if len(sequences) != len(matrices):
            raise ValueError("知识特征与序列数量不一致")
        for chunk, matrix in zip(sequences, matrices):
            if len(matrix) != len(chunk.tokens):
                raise ValueError(f"{chunk.document_id}的知识特征长度不一致")
            if any(len(row) != self.knowledge_provider.dimension for row in matrix):
                raise ValueError(f"{chunk.document_id}的知识特征维度不一致")
            self._knowledge_cache[self._knowledge_key(chunk)] = matrix

    def _prepare_knowledge_features(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk],
        fit_sequences: list[SequenceChunk] | None = None,
    ) -> None:
        """仅用指定的外折原始训练集拟合统计，再变换其余样本。"""

        self._knowledge_cache.clear()
        fitting = fit_sequences if fit_sequences is not None else train
        self._cache_features(
            fitting, self.knowledge_provider.fit_transform(fitting)
        )
        additional_train = [
            chunk
            for chunk in train
            if self._knowledge_key(chunk) not in self._knowledge_cache
        ]
        if additional_train:
            self._cache_features(
                additional_train,
                self.knowledge_provider.transform(additional_train),
            )
        if dev:
            self._cache_features(dev, self.knowledge_provider.transform(dev))

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        dev_sequences = dev or []
        self._prepare_knowledge_features(train, dev_sequences)
        relation_knowledge_dim = int(
            getattr(self, "relation_knowledge_dim", 0)
        )
        LOGGER.info(
            "显式知识融合：原始总计%d维（基础直接拼接%d维、单独投影%d→%d维、"
            "门控残差%d维、独立关系残差%d维；%s）",
            self.knowledge_provider.dimension,
            self.knowledge_provider.dimension
            - self.residual_knowledge_dim
            - self.projected_knowledge_dim
            - relation_knowledge_dim,
            self.projected_knowledge_dim,
            self.projected_knowledge_output_dim,
            self.residual_knowledge_dim,
            relation_knowledge_dim,
            "、".join(self.knowledge_provider.feature_names),
        )
        if self.direct_knowledge_channel_dim:
            LOGGER.info(
                "直接拼接知识子通道：末尾%d维保持原值，训练时整通道dropout=%.2f",
                self.direct_knowledge_channel_dim,
                self.direct_knowledge_channel_dropout,
            )
        super().fit(train, dev)

    def _encode_sequence(self, chunk: SequenceChunk) -> _KnowledgeEncodedSequence:
        base = super()._encode_sequence(chunk)
        matrix = self._knowledge_cache.get(self._knowledge_key(chunk))
        if matrix is None:
            matrix = self.knowledge_provider.transform([chunk])[0]
            self._knowledge_cache[self._knowledge_key(chunk)] = matrix
        return _KnowledgeEncodedSequence(
            base.token_ids,
            base.label_ids,
            base.stage_ids,
            base.group_ids,
            matrix,
        )

    def _collate(
        self, batch: list[_KnowledgeEncodedSequence]
    ) -> dict[str, torch.Tensor]:
        values = NeuralSequenceTagger._collate(batch)  # type: ignore[arg-type]
        max_length = values["token_ids"].shape[1]
        features = torch.zeros(
            (len(batch), max_length, self.knowledge_provider.dimension),
            dtype=torch.float,
        )
        for row, item in enumerate(batch):
            features[row, : len(item.knowledge_features)] = torch.tensor(
                item.knowledge_features, dtype=torch.float
            )
        values["knowledge_features"] = features
        return values

    def _build_model(self) -> nn.Module:
        encoder, vocabulary, _ = load_tangut_encoder_checkpoint(
            self.checkpoint_path
        )
        if vocabulary != self.vocabulary:
            raise RuntimeError("TangutEncoder checkpoint中的字符词表在加载过程中发生变化")
        return _TangutEncoderBiLSTMClassifier(
            encoder,
            len(self.labels),
            self.config,
            self.use_group_features,
            knowledge_feature_dim=self.knowledge_provider.dimension,
            residual_knowledge_dim=self.residual_knowledge_dim,
            projected_knowledge_dim=self.projected_knowledge_dim,
            projected_knowledge_output_dim=self.projected_knowledge_output_dim,
            projected_knowledge_dropout=self.projected_knowledge_dropout,
            direct_knowledge_channel_dim=self.direct_knowledge_channel_dim,
            direct_knowledge_channel_dropout=self.direct_knowledge_channel_dropout,
        )

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(self.model, _TangutEncoderBiLSTMClassifier):
            raise RuntimeError("知识增强TangutEncoder模型尚未建立")
        return self.model(
            batch["token_ids"].to(self.device),
            batch["stage_ids"].to(self.device),
            batch["group_ids"].to(self.device),
            batch["lengths"].to(self.device),
            batch["padding_mask"].to(self.device),
            batch["knowledge_features"].to(self.device),
        )

    def _prepare_prediction_features(self, sequences: list[SequenceChunk]) -> None:
        missing = [
            chunk
            for chunk in sequences
            if self._knowledge_key(chunk) not in self._knowledge_cache
        ]
        if missing:
            self._cache_features(missing, self.knowledge_provider.transform(missing))

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        self._prepare_prediction_features(sequences)
        return super().predict(sequences)

    def predict_position_probabilities(
        self, sequences: list[SequenceChunk]
    ) -> list[list[float]]:
        self._prepare_prediction_features(sequences)
        return super().predict_position_probabilities(sequences)

    def metadata(self) -> dict[str, object]:
        values = super().metadata()
        gate_value = None
        if (
            isinstance(self.model, _TangutEncoderBiLSTMClassifier)
            and self.model.residual_gate is not None
        ):
            gate_value = float(torch.tanh(self.model.residual_gate).detach().cpu())
        values.update(
            {
                "下游结构": (
                    "TangutEncoder → 基础知识拼接＋分词知识门控残差 → "
                    "BiLSTM → 逐位置Softmax"
                    if self.residual_knowledge_dim
                    else (
                        "TangutEncoder → 基础知识拼接＋词性知识独立投影 → "
                        "BiLSTM → 逐位置Softmax"
                        if self.projected_knowledge_dim
                        else "TangutEncoder → 显式知识拼接 → BiLSTM → 逐位置Softmax"
                    )
                ),
                "显式知识": self.knowledge_provider.metadata(),
                "显式知识维度": self.knowledge_provider.dimension,
                "门控残差知识维度": self.residual_knowledge_dim,
                "学习后分词知识门值": gate_value,
                "单独投影知识原始维度": self.projected_knowledge_dim,
                "单独投影知识输出维度": self.projected_knowledge_output_dim,
                "投影知识整通道dropout": self.projected_knowledge_dropout,
                "直接知识子通道维度": self.direct_knowledge_channel_dim,
                "直接知识子通道dropout": self.direct_knowledge_channel_dropout,
            }
        )
        return values

    def save(self, path: Path) -> None:
        super().save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["knowledge"] = self.knowledge_provider.metadata()
        payload["knowledge_state"] = self.knowledge_provider.state_dict()
        torch.save(payload, path)


class MultiTaskKnowledgeEnhancedTangutEncoderTagger(
    KnowledgeEnhancedTangutEncoderTagger
):
    """E9/F1：E3主干共享编码器的主分类任务＋位置辅助任务。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        punctuation_labels: Iterable[str],
        neural_config: NeuralConfig,
        pretraining_config: PretrainingConfig,
        cascade_config: CascadeConfig,
        max_sequence_length: int,
        seed: int,
        knowledge_provider: KnowledgeFeatureProvider,
        experiment_name: str = "E9",
        primary_label_description: str = "O/七类完整标点",
    ) -> None:
        self.cascade_config = cascade_config
        self.augmentation_metadata: dict[str, object] | None = None
        self.experiment_name = experiment_name
        self.primary_label_description = primary_label_description
        super().__init__(
            checkpoint_path,
            punctuation_labels,
            neural_config,
            pretraining_config,
            max_sequence_length,
            seed,
            knowledge_provider,
        )
        self.encoder_display_name = "TangutEncoder＋E3知识＋共享BiLSTM"

    def _build_model(self) -> nn.Module:
        encoder, vocabulary, _ = load_tangut_encoder_checkpoint(
            self.checkpoint_path
        )
        if vocabulary != self.vocabulary:
            raise RuntimeError("TangutEncoder checkpoint中的字符词表在加载过程中发生变化")
        return _TangutEncoderBiLSTMMultiTaskClassifier(
            encoder,
            len(self.labels),
            self.config,
            self.knowledge_provider.dimension,
        )

    def _forward_heads(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(
            self.model, _TangutEncoderBiLSTMMultiTaskClassifier
        ):
            raise RuntimeError(f"{self.experiment_name}多任务TangutEncoder模型尚未建立")
        return self.model(
            batch["token_ids"].to(self.device),
            batch["stage_ids"].to(self.device),
            batch["group_ids"].to(self.device),
            batch["lengths"].to(self.device),
            batch["padding_mask"].to(self.device),
            batch["knowledge_features"].to(self.device),
        )

    def _multitask_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        if self.model is None:
            raise RuntimeError(f"{self.experiment_name}多任务模型尚未建立")
        training = optimizer is not None
        self.model.train(training)
        loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        outside_id = self.label_to_id[OUTSIDE]
        total_loss = 0.0
        total_labels = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                punctuation = batch["label_ids"].to(self.device)
                valid = punctuation != IGNORE_LABEL
                valid_count = int(valid.sum().item())
                if valid_count == 0:
                    continue
                position = torch.full_like(punctuation, IGNORE_LABEL)
                position[valid] = (punctuation[valid] != outside_id).long()
                if training:
                    optimizer.zero_grad()
                position_logits, punctuation_logits = self._forward_heads(batch)
                punctuation_loss = loss_function(
                    punctuation_logits.reshape(-1, len(self.labels)),
                    punctuation.reshape(-1),
                )
                position_loss = loss_function(
                    position_logits.reshape(-1, 2), position.reshape(-1)
                )
                loss = punctuation_loss + (
                    self.cascade_config.multitask_position_loss_weight
                    * position_loss
                )
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    optimizer.step()
                total_loss += float(loss.item()) * valid_count
                total_labels += valid_count
        return total_loss / total_labels if total_labels else float("inf")

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        self.augmentation_metadata = None
        self._fit_multitask(train, dev or [], train)

    def fit_augmented(
        self,
        original_train: list[SequenceChunk],
        augmented_train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        """训练原始＋合成块，但E3统计量严格只在原始训练块拟合。"""

        if not original_train:
            raise ValueError("E9-Aug原始训练序列不能为空")
        if not augmented_train:
            raise ValueError("E9-Aug没有生成任何合成训练序列")
        self.augmentation_metadata = {
            "原始训练块": len(original_train),
            "增强训练块": len(augmented_train),
            "知识统计拟合范围": "仅原始外层训练折",
        }
        self._fit_multitask(
            original_train + augmented_train,
            dev or [],
            original_train,
        )

    def _fit_multitask(
        self,
        train: list[SequenceChunk],
        dev_sequences: list[SequenceChunk],
        knowledge_fit_sequences: list[SequenceChunk],
    ) -> None:
        if not train:
            raise ValueError(f"{self.experiment_name}多任务训练序列不能为空")
        self._set_seed()
        self._infer_task(train)
        if self.task not in {"punctuation_joint", "punctuation_group_joint"}:
            raise ValueError(
                f"{self.experiment_name}必须使用包含O的直接分类标签训练"
            )
        self._build_vocabulary(train)
        self._prepare_knowledge_features(
            train, dev_sequences, fit_sequences=knowledge_fit_sequences
        )
        train_data = self._encode(train, drop_empty_labels=True)
        dev_data = self._encode(dev_sequences, drop_empty_labels=True)
        if not train_data:
            raise ValueError(f"{self.experiment_name}训练折没有有效标签")

        self.model = self._build_model().to(self.device)
        optimizer = self._build_optimizer()
        train_loader = self._loader(train_data, shuffle=True)
        dev_loader = self._loader(dev_data, shuffle=False) if dev_data else None
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0
        experiment_name = (
            f"{self.experiment_name}-Aug"
            if self.augmentation_metadata
            else self.experiment_name
        )
        LOGGER.info(
            "%s共享多任务：TangutEncoder＋%d维E3知识＋BiLSTM仅训练一次；"
            "L=%s CE＋%g×位置CE",
            experiment_name,
            self.knowledge_provider.dimension,
            self.primary_label_description,
            self.cascade_config.multitask_position_loss_weight,
        )
        LOGGER.info(
            "%s知识特征：%s",
            experiment_name,
            "、".join(self.knowledge_provider.feature_names),
        )
        for epoch in range(1, self.config.epochs + 1):
            self._prepare_epoch(epoch)
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
                "[%s][multitask] epoch %d/%d：train_loss=%.6f，dev_loss=%.6f",
                experiment_name,
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
                    "[%s][multitask] 开发集loss达到新最佳",
                    experiment_name,
                )
            else:
                stale_epochs += 1
                LOGGER.info(
                    "[%s][multitask] 改善不足，早停计数=%d/%d",
                    experiment_name,
                    stale_epochs,
                    self.config.patience,
                )
                if stale_epochs >= self.config.patience:
                    LOGGER.info(
                        "[%s][multitask] 连续%d轮未改善，提前停止",
                        experiment_name,
                        stale_epochs,
                    )
                    break
        self.model.load_state_dict(best_state)

    def _predict_heads(
        self, sequences: list[SequenceChunk]
    ) -> tuple[list[list[str]], list[list[str]], list[list[float]]]:
        if not isinstance(
            self.model, _TangutEncoderBiLSTMMultiTaskClassifier
        ):
            raise RuntimeError(f"{self.experiment_name}多任务模型尚未训练")
        self._prepare_prediction_features(sequences)
        data = self._encode(sequences, drop_empty_labels=False)
        loader = self._loader(data, shuffle=False)
        punctuation_predictions: list[list[str]] = []
        position_predictions: list[list[str]] = []
        position_probabilities: list[list[float]] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                position_logits, punctuation_logits = self._forward_heads(batch)
                position_scores = torch.softmax(position_logits, dim=-1)[..., 1]
                for row, length in enumerate(batch["lengths"].tolist()):
                    punctuation_predictions.append(
                        [
                            self.labels[index]
                            for index in punctuation_logits[
                                row, :length
                            ].argmax(-1).tolist()
                        ]
                    )
                    position_predictions.append(
                        [
                            POSITION_LABEL if index == 1 else OUTSIDE
                            for index in position_logits[
                                row, :length
                            ].argmax(-1).tolist()
                        ]
                    )
                    position_probabilities.append(
                        position_scores[row, :length].tolist()
                    )
        return (
            punctuation_predictions,
            position_predictions,
            position_probabilities,
        )

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        return self._predict_heads(sequences)[0]

    def predict_position(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        return self._predict_heads(sequences)[1]

    def predict_position_probabilities(
        self, sequences: list[SequenceChunk]
    ) -> list[list[float]]:
        return self._predict_heads(sequences)[2]

    def metadata(self) -> dict[str, object]:
        values = super().metadata()
        values.update(
            {
                "下游结构": (
                    "TangutEncoder＋E3知识 → 共享BiLSTM → "
                    f"位置辅助头＋{self.primary_label_description}主分类头"
                ),
                "模型阶段": f"{self.experiment_name}多任务联合",
                "位置损失权重": (
                    self.cascade_config.multitask_position_loss_weight
                ),
                "最终输出来源": "主分类头（位置辅助头不参与硬门控）",
                "数据增强": self.augmentation_metadata,
            }
        )
        return values

    def save(self, path: Path) -> None:
        super().save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["multitask"] = {
            "position_loss_weight": (
                self.cascade_config.multitask_position_loss_weight
            ),
            "final_output": "punctuation_head",
            "experiment_name": self.experiment_name,
            "primary_label_description": self.primary_label_description,
            "data_augmentation": self.augmentation_metadata,
        }
        payload["cascade_config"] = asdict(self.cascade_config)
        torch.save(payload, path)


class POSRelationResidualTangutEncoderTagger(KnowledgeEnhancedTangutEncoderTagger):
    """E8：先拟合E3阶段模型，再冻结主体训练细词性关系残差头。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        punctuation_labels: Iterable[str],
        neural_config: NeuralConfig,
        pretraining_config: PretrainingConfig,
        max_sequence_length: int,
        seed: int,
        knowledge_provider: KnowledgeFeatureProvider,
        base_knowledge_dim: int,
        relation_knowledge_dim: int,
        pos_tag_count: int,
        relation_config: POSRelationKnowledgeConfig,
    ) -> None:
        self.base_knowledge_dim = base_knowledge_dim
        self.relation_knowledge_dim = relation_knowledge_dim
        self.pos_tag_count = pos_tag_count
        self.relation_config = relation_config
        self.base_best_epoch = 0
        self.base_best_dev_loss = float("inf")
        self.relation_best_epoch = 0
        self.relation_best_dev_loss = float("inf")
        self.relation_training_history: list[dict[str, float | int | str]] = []
        self.fusion_weight_scores: dict[str, float] = {}
        self.fusion_weight_thresholds: dict[str, float] = {}
        self.selected_fusion_weight = 0.0
        super().__init__(
            checkpoint_path,
            punctuation_labels,
            neural_config,
            pretraining_config,
            max_sequence_length,
            seed,
            knowledge_provider,
        )
        self.encoder_display_name = "TangutEncoder＋E3＋细粒度词性关系残差"

    def _build_model(self) -> nn.Module:
        encoder, vocabulary, _ = load_tangut_encoder_checkpoint(
            self.checkpoint_path
        )
        if vocabulary != self.vocabulary:
            raise RuntimeError("TangutEncoder checkpoint中的字符词表在加载过程中发生变化")
        model = _TangutEncoderBiLSTMPOSResidualClassifier(
            encoder,
            len(self.labels),
            self.config,
            self.use_group_features,
            self.base_knowledge_dim,
            self.relation_knowledge_dim,
            self.pos_tag_count,
            self.relation_config,
        )
        model.freeze_relation_branch()
        return model

    def fit(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk] | None = None,
    ) -> None:
        # 基础拟合时fusion_weight=0且关系分支冻结，因此数值结构等价于E3。
        super().fit(train, dev)
        if self.task not in {"position", "punctuation_type"}:
            raise RuntimeError("细粒度词性残差只支持标点位置或具体标点类别阶段")
        self.base_best_epoch = self.best_epoch
        self.base_best_dev_loss = self.best_dev_loss
        self._fit_relation_branch(train, dev or [])

    def _fit_relation_branch(
        self,
        train: list[SequenceChunk],
        dev: list[SequenceChunk],
    ) -> None:
        if not isinstance(self.model, _TangutEncoderBiLSTMPOSResidualClassifier):
            raise RuntimeError("E7/E8细粒度词性关系模型尚未建立")
        self.model.freeze_base_and_enable_relation()
        self.model.set_fusion_weight(1.0)
        relation_parameters = self.model.relation_parameters()
        optimizer = torch.optim.AdamW(
            relation_parameters,
            lr=self.relation_config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        train_data = self._encode(train, drop_empty_labels=True)
        dev_data = self._encode(dev, drop_empty_labels=True)
        train_loader = self._loader(train_data, shuffle=True)
        dev_loader = self._loader(dev_data, shuffle=False) if dev_data else None
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0
        LOGGER.info(
            "[细粒度词性残差][%s] 冻结TangutEncoder、E3知识BiLSTM与基础分类器；"
            "仅训练词性嵌入/关系MLP/门控，epochs=%d，lr=%g，patience=%d",
            self.task,
            self.relation_config.epochs,
            self.relation_config.learning_rate,
            self.relation_config.patience,
        )
        for epoch in range(1, self.relation_config.epochs + 1):
            train_loss = self._epoch_loss(train_loader, loss_function, optimizer)
            dev_loss = (
                self._epoch_loss(dev_loader, loss_function, None)
                if dev_loader is not None
                else train_loss
            )
            self.relation_training_history.append(
                {
                    "phase": "pos_relation_residual",
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                }
            )
            LOGGER.info(
                "[细粒度词性残差][%s] epoch %d/%d：train_loss=%.6f，dev_loss=%.6f",
                self.task,
                epoch,
                self.relation_config.epochs,
                train_loss,
                dev_loss,
            )
            if dev_loss < self.relation_best_dev_loss - self.relation_config.min_delta:
                self.relation_best_dev_loss = dev_loss
                self.relation_best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                LOGGER.info(
                    "[细粒度词性残差][%s] 改善不足min_delta=%g，早停计数=%d/%d",
                    self.task,
                    self.relation_config.min_delta,
                    stale_epochs,
                    self.relation_config.patience,
                )
                if stale_epochs >= self.relation_config.patience:
                    LOGGER.info("[细粒度词性残差][%s] 提前停止", self.task)
                    break
        self.model.load_state_dict(best_state)
        self._select_fusion_weight(dev)

    def _select_fusion_weight(self, dev: list[SequenceChunk]) -> None:
        if not isinstance(self.model, _TangutEncoderBiLSTMPOSResidualClassifier):
            raise RuntimeError("E7/E8细粒度词性关系模型尚未建立")
        candidates = sorted(set(self.relation_config.fusion_weight_candidates))
        if not dev:
            selected = max(candidates)
            self.model.set_fusion_weight(selected)
            self.selected_fusion_weight = selected
            return
        best_score = -1.0
        best_precision = -1.0
        selected = 0.0
        scores: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        for weight in candidates:
            self.model.set_fusion_weight(weight)
            if self.task == "position":
                probabilities = [
                    value
                    for part in self.predict_position_probabilities(dev)
                    for value in part
                ]
                gold = [
                    label == POSITION_LABEL
                    for chunk in dev
                    for label in chunk.labels
                ]
                best_at_weight = (-1.0, -1.0, 0.5)
                for threshold in self._threshold_values():
                    precision, f1 = self._prf(
                        gold,
                        [value >= threshold for value in probabilities],
                    )
                    best_at_weight = max(
                        best_at_weight,
                        (f1, precision, threshold),
                    )
                score, precision, threshold = best_at_weight
                thresholds[f"{weight:g}"] = threshold
            else:
                predictions = self.predict(dev)
                correct = 0
                total = 0
                for chunk, predicted in zip(dev, predictions):
                    for gold, actual in zip(chunk.labels, predicted):
                        if gold not in self.label_to_id:
                            continue
                        total += 1
                        correct += int(gold == actual)
                # 候选位置上是单标签七分类，此时Accuracy等于类别Micro-F1。
                score = correct / total if total else 0.0
                precision = score
            scores[f"{weight:g}"] = score
            # candidates升序；F1和精确率均相同时保留更小权重，优先E3。
            if (score, precision) > (best_score, best_precision):
                best_score = score
                best_precision = precision
                selected = weight
        self.model.set_fusion_weight(selected)
        if self.task == "position":
            self.position_threshold = thresholds[f"{selected:g}"]
        self.selected_fusion_weight = selected
        self.fusion_weight_scores = scores
        self.fusion_weight_thresholds = thresholds
        if self.task == "position":
            LOGGER.info(
                "[细粒度词性残差][position] 开发集λ/阈值位置F1：%s；选择λ=%g、阈值=%.2f",
                "、".join(
                    f"{weight}@{thresholds[weight]:.2f}→{score:.4f}"
                    for weight, score in scores.items()
                ),
                selected,
                self.position_threshold,
            )
        else:
            LOGGER.info(
                "[细粒度词性残差][punctuation_type] 开发集融合权重类别Micro-F1：%s；选择λ=%g",
                "、".join(
                    f"{weight}→{score:.4f}" for weight, score in scores.items()
                ),
                selected,
            )

    def metadata(self) -> dict[str, object]:
        values = super().metadata()
        relation_parameter_count = 0
        gate_bias = None
        if isinstance(self.model, _TangutEncoderBiLSTMPOSResidualClassifier):
            relation_parameter_count = sum(
                parameter.numel() for parameter in self.model.relation_parameters()
            )
            gate_bias = float(self.model.relation_gate.bias.detach().cpu().item())
        values.update(
            {
                "下游结构": (
                    "E3 TangutEncoder＋23维词典/上下文＋BiLSTM基础logits；"
                    "BIES×36类左右词性关系经独立MLP与置信门生成后期残差logits"
                ),
                "E3基础知识维度": self.base_knowledge_dim,
                "细粒度词性关系原始维度": self.relation_knowledge_dim,
                "细粒度词性数": self.pos_tag_count,
                "词性嵌入维度": self.relation_config.tag_embedding_dim,
                "词性关系隐藏维度": self.relation_config.relation_hidden_dim,
                "词性关系参数量": relation_parameter_count,
                "词性关系整通道dropout": self.relation_config.channel_dropout,
                "E3基础最佳epoch": self.base_best_epoch,
                "E3基础最佳开发集loss": self.base_best_dev_loss,
                "词性残差最佳epoch": self.relation_best_epoch,
                "词性残差最佳开发集loss": self.relation_best_dev_loss,
                "词性残差训练记录": self.relation_training_history,
                "开发集融合权重得分": self.fusion_weight_scores,
                "开发集融合权重对应位置阈值": self.fusion_weight_thresholds,
                "最终融合权重": self.selected_fusion_weight,
                "学习后门控偏置": gate_bias,
            }
        )
        return values

    def save(self, path: Path) -> None:
        super().save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["pos_relation_residual"] = {
            "config": asdict(self.relation_config),
            "base_knowledge_dim": self.base_knowledge_dim,
            "relation_knowledge_dim": self.relation_knowledge_dim,
            "pos_tag_count": self.pos_tag_count,
            "selected_fusion_weight": self.selected_fusion_weight,
            "fusion_weight_scores": self.fusion_weight_scores,
            "fusion_weight_thresholds": self.fusion_weight_thresholds,
            "base_best_epoch": self.base_best_epoch,
            "base_best_dev_loss": self.base_best_dev_loss,
            "relation_best_epoch": self.relation_best_epoch,
            "relation_best_dev_loss": self.relation_best_dev_loss,
        }
        torch.save(payload, path)
