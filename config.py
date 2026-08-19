from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CorpusSource:
    path: Path
    domain: str


@dataclass(frozen=True)
class DataConfig:
    sources: tuple[CorpusSource, ...]
    missing_volume_numbers: tuple[int, ...] = (38,)
    missing_characters: tuple[str, ...] = ("□", "@", "…")
    ignored_editorial_symbols: tuple[str, ...] = ("√", "×", "△")
    boundary_punctuation: tuple[str, ...] = ("，", "。", "、", "！", "？", "；", "：")

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(source.path for source in self.sources)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(source.domain for source in self.sources)


@dataclass(frozen=True)
class FeatureConfig:
    window: int = 3
    use_ngrams: bool = True
    use_missing: bool = True
    use_document_edges: bool = True
    use_stage_features: bool = True
    stage_feature_window: int = 1


@dataclass(frozen=True)
class PunctuationConfig:
    sentence_pause: tuple[str, ...] = ("。", "？", "！")
    intra_sentence_pause: tuple[str, ...] = ("，", "、", "；", "：")
    structural: tuple[str, ...] = ("“", "”", "‘", "’", "《", "》")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "crf"
    c1: float = 1.0
    c2: float = 0.001
    max_iterations: int = 200
    all_possible_transitions: bool = True


@dataclass(frozen=True)
class RuleConfig:
    """B0/B1 规则基线参数；阈值固定，不使用测试折调参。"""

    cue_min_support: int = 8
    cue_min_confidence: float = 0.70
    structure_min_support: int = 8
    structure_min_confidence: float = 0.55
    cue_max_length: int = 2
    force_document_final_period: bool = True


@dataclass(frozen=True)
class NGramConfig:
    """B2双向字符n-gram参数。"""

    n: int = 3
    min_support: int = 5
    alpha: float = 0.5
    backoff_k: float = 10.0
    threshold_start: float = 0.05
    threshold_end: float = 0.95
    threshold_step: float = 0.05
    use_left: bool = True
    use_right: bool = True
    use_cross_gap: bool = True


@dataclass(frozen=True)
class NeuralConfig:
    """B4/B5共享的随机初始化神经编码器参数。"""

    embedding_dim: int = 128
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    patience: int = 5
    min_delta: float = 0.0001
    gradient_clip: float = 5.0
    device: str = "auto"
    cuda_max_utilization: int = 20
    cuda_min_free_memory_gb: float = 4.0
    position_threshold_start: float = 0.05
    position_threshold_end: float = 0.95
    position_threshold_step: float = 0.05
    bilstm_hidden_dim: int = 128
    bilstm_layers: int = 2
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 256


@dataclass(frozen=True)
class PretrainingConfig:
    """主实验D：TangutEncoder逐阶段预训练的共享参数。"""

    stage: str = "context_mlm"
    validation_ratio: float = 0.1
    min_sequence_length: int = 4
    max_sequence_length: int = 128
    mask_ratio: float = 0.15
    span_mask_probability: float = 0.5
    span_lengths: tuple[int, ...] = (2, 3, 4)
    mask_replace_probability: float = 0.8
    random_replace_probability: float = 0.1
    embedding_dim: int = 192
    layers: int = 3
    heads: int = 4
    ff_dim: int = 768
    dropout: float = 0.15
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_interval: int = 200
    log_interval: int = 50
    patience: int = 5
    min_delta: float = 0.0001
    gradient_clip: float = 1.0
    device: str = "auto"
    cuda_max_utilization: int = 20
    cuda_min_free_memory_gb: float = 4.0
    checkpoint: Path | None = None
    downstream_encoder_learning_rate: float = 0.00005
    downstream_head_learning_rate: float = 0.001
    downstream_freeze_epochs: int = 1


@dataclass(frozen=True)
class WordPretrainingConfig:
    """D2：在D1 checkpoint上继续进行词语感知预训练。"""

    initial_checkpoint: Path | None = None
    checkpoint: Path | None = None
    dictionary_path: Path | None = None
    tongyin_path: Path | None = None
    candidate_lengths: tuple[int, ...] = (2, 3, 4)
    dictionary_min_frequency: int = 2
    intersection_min_frequency: int = 1
    statistical_min_frequencies: tuple[int, ...] = (5, 4, 3)
    statistical_min_dpmi: float = 1.0
    statistical_min_entropy: float = 0.0
    whole_word_probability: float = 0.5
    ranking_negatives: int = 5
    ranking_loss_weight: float = 0.3
    ranking_warmup_steps: int = 500
    encoder_learning_rate: float = 0.00005
    head_learning_rate: float = 0.0003
    batch_size: int = 32
    max_steps: int = 5000
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_interval: int = 200
    log_interval: int = 50
    patience: int = 5
    min_delta: float = 0.0001
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class DomainKnowledgeConfig:
    """E1：由训练语料估计的字符级局部词汇领域分布。"""

    enabled: bool = False
    dimension: int = 2
    inner_folds: int = 5
    smoothing: float = 1.0
    shrinkage: float = 2.0
    candidate_mode: str = "fusion"


@dataclass(frozen=True)
class LexiconKnowledgeConfig:
    """E2：固定辞书产生的字符后间隔软词典格网。"""

    enabled: bool = False
    dictionary_path: Path | None = None
    tongyin_path: Path | None = None
    candidate_lengths: tuple[int, ...] = (2, 3, 4)
    representation: str = "gap_lattice"
    use_source_features: bool = True
    include_unseen_terms: bool = True


@dataclass(frozen=True)
class ContextKnowledgeConfig:
    """E3：由未截长训练文献估计的间隔上下文统计。"""

    enabled: bool = False
    # (当前标点语料文件stem, 对应未截长clean语料路径)
    source_mapping: tuple[tuple[str, Path], ...] = ()
    inner_folds: int = 5
    association: str = "dpmi"
    dpmi_discount: float = 0.5
    clipping_percentile: float = 99.0
    dimension: int = 8


@dataclass(frozen=True)
class SegmentationKnowledgeConfig:
    """E4：由冻结西夏文CRF分词器提供的间隔级软分词知识。"""

    enabled: bool = False
    model_path: Path | None = None
    lexicon_path: Path | None = None
    gap_path: Path | None = None
    # 分词模型人工标注训练语料；仅用于审计和屏蔽精确重叠片段。
    annotation_paths: tuple[Path, ...] = ()
    # mask_exact=屏蔽重叠；error=发现即停止；allow=仅供污染上界对照。
    overlap_policy: str = "mask_exact"
    min_overlap_length: int = 4
    max_word_length: int = 5
    # compact将原始8维BIES/词长特征压缩为：软词界概率、硬词界、置信度。
    # full保留旧实验使用的8维表示，便于复现原始E4结果。
    representation: str = "compact"
    # gated_residual以零初始化门控注入TangutEncoder表示；direct为旧式直接拼接。
    fusion: str = "gated_residual"
    dimension: int = 3


@dataclass(frozen=True)
class POSKnowledgeConfig:
    """E5：冻结CRF-Joint-full提供的字符级软词性知识。"""

    enabled: bool = False
    model_path: Path | None = None
    lexicon_state_path: Path | None = None
    manifest_path: Path | None = None
    # 留空时从manifest的“资源/监督语料”读取；显式配置便于迁移工程。
    annotation_paths: tuple[Path, ...] = ()
    # coarse_soft_character=当前字7组概率＋自身置信度；
    # coarse_soft_gap保留旧17维间隔表示，仅用于复现第一次E5。
    representation: str = "coarse_soft_character"
    group_scheme: str = "seven"
    raw_dimension: int = 8
    # direct=8维原值直接拼接；projected=经Linear/GELU/LayerNorm压缩后拼接。
    fusion: str = "direct"
    projection_dimension: int = 4
    channel_dropout: float = 0.20
    # error=发现监督重叠即停止；mask_exact=屏蔽；allow仅用于污染上界。
    overlap_policy: str = "error"
    min_overlap_length: int = 4


@dataclass(frozen=True)
class POSRelationKnowledgeConfig:
    """E7/E8细粒度左右词性关系；其中残差校正参数只供E8使用。"""

    enabled: bool = False
    tag_embedding_dim: int = 8
    relation_hidden_dim: int = 16
    relation_dropout: float = 0.20
    channel_dropout: float = 0.20
    gate_bias: float = -2.0
    learning_rate: float = 0.001
    epochs: int = 10
    patience: int = 3
    min_delta: float = 0.0001
    # 必须包含0；开发集可据此选择完全退回E3，不强迫使用无效词性。
    fusion_weight_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class KnowledgeConfig:
    """主实验E的显式知识入口；后续知识通道在此并列扩展。"""

    domain: DomainKnowledgeConfig = field(default_factory=DomainKnowledgeConfig)
    lexicon: LexiconKnowledgeConfig = field(default_factory=LexiconKnowledgeConfig)
    context: ContextKnowledgeConfig = field(default_factory=ContextKnowledgeConfig)
    segmentation: SegmentationKnowledgeConfig = field(
        default_factory=SegmentationKnowledgeConfig
    )
    pos: POSKnowledgeConfig = field(default_factory=POSKnowledgeConfig)
    pos_relation: POSRelationKnowledgeConfig = field(
        default_factory=POSRelationKnowledgeConfig
    )


@dataclass(frozen=True)
class DDownstreamConfig:
    """D3/D4下游结构实验的预训练初始化来源。"""

    initialization_source: str = "d1"


@dataclass(frozen=True)
class PunctuationPretrainingConfig:
    """D5：折内断句与标点感知任务适应预训练。"""

    batch_size: int = 32
    max_steps: int = 10000
    encoder_learning_rate: float = 0.00002
    head_learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_interval: int = 200
    log_interval: int = 50
    patience: int = 5
    min_delta: float = 0.0001
    gradient_clip: float = 1.0
    mlm_weight: float = 0.25
    position_weight: float = 1.0
    group_weight: float = 0.5
    type_weight: float = 1.0
    position_threshold_start: float = 0.05
    position_threshold_end: float = 0.95
    position_threshold_step: float = 0.05


@dataclass(frozen=True)
class CascadeConfig:
    """主实验C的误差传播控制参数。"""

    # C2/C3在外层训练集内部生成折外上游预测，避免下游看到自拟合概率。
    inner_folds: int = 3
    # C2在开发集上选取满足该召回率的最高精确率候选阈值。
    candidate_recall_target: float = 0.95
    # C3训练时使用的软先验强度；最终alpha仍在开发集候选值中选择。
    soft_train_alpha: float = 1.0
    soft_alpha_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    soft_epsilon: float = 1.0e-8
    # C4中辅助位置损失相对于完整标点损失的权重。
    multitask_position_loss_weight: float = 0.5
    # C4共享编码器注册名；未来可直接切换为tangut_encoder。
    shared_encoder: str = "bilstm"


@dataclass(frozen=True)
class DataAugmentationConfig:
    """E9-Aug：训练折内的完整句子拼接增强。"""

    enabled: bool = False
    method: str = "sentence_concatenation"
    # 合成训练块数 / 原始训练块数；原始训练块始终全部保留。
    ratio: float = 1.0
    min_sentences: int = 2
    max_sentences: int = 4
    min_characters: int = 64
    max_characters: int = 128
    # 当前只实现同一文献内抽样，并按原文次序拼接。
    scope: str = "same_document"
    preserve_order: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    folds: int = 5
    # 外层每折 20% 测试；剩余 80% 中抽 12.5% 为开发集，即总体约 7:1:2。
    dev_ratio: float = 0.125
    max_sequence_length: int = 500
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            (CorpusSource(Path("corpus/dabaojijing.txt"), "unknown"),)
        )
    )
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    ngram: NGramConfig = field(default_factory=NGramConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    word_pretraining: WordPretrainingConfig = field(
        default_factory=WordPretrainingConfig
    )
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    d_downstream: DDownstreamConfig = field(default_factory=DDownstreamConfig)
    punctuation_pretraining: PunctuationPretrainingConfig = field(
        default_factory=PunctuationPretrainingConfig
    )
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    data_augmentation: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    punctuation: PunctuationConfig = field(default_factory=PunctuationConfig)


def _resolve_data_path(config_path: Path, value: str | Path) -> Path:
    data_path = Path(value)
    if not data_path.is_absolute():
        # 配置放在 configs/ 时，相对路径以工程根目录为基准。
        data_path = config_path.parent.parent / data_path
    return data_path.resolve()


def _infer_domain(path: Path) -> str:
    name = path.stem.lower()
    if "jingshu" in name:
        return "jingshu"
    if "shisu" in name:
        return "shisu"
    return "unknown"


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {})
    feature = raw.get("features", {})
    model = raw.get("model", {})
    rules = raw.get("rules", {})
    ngram = raw.get("ngram", {})
    neural = raw.get("neural", {})
    pretraining = raw.get("pretraining", {})
    word_pretraining = raw.get("word_pretraining", {})
    knowledge = raw.get("knowledge", {})
    context_knowledge = knowledge.get("context", {})
    segmentation_knowledge = knowledge.get("segmentation", {})
    pos_knowledge = knowledge.get("pos", {})
    pos_relation_knowledge = knowledge.get("pos_relation", {})
    d_downstream = raw.get("d_downstream", {})
    punctuation_pretraining = raw.get("punctuation_pretraining", {})
    cascade = raw.get("cascade", {})
    data_augmentation = raw.get("data_augmentation", {})
    punctuation = raw.get("punctuation", {})
    configured_sources = data.get("sources")
    sources: list[CorpusSource] = []
    if configured_sources is not None:
        if not isinstance(configured_sources, list) or not configured_sources:
            raise ValueError("data.sources 必须是非空列表")
        for item in configured_sources:
            if not isinstance(item, dict) or "path" not in item or "domain" not in item:
                raise ValueError("data.sources 每项必须包含 path 和 domain")
            sources.append(
                CorpusSource(_resolve_data_path(path, item["path"]), str(item["domain"]))
            )
    else:
        configured_paths = data.get("path", "corpus/dabaojijing.txt")
        if isinstance(configured_paths, (str, Path)):
            configured_paths = [configured_paths]
        if not isinstance(configured_paths, list) or not configured_paths:
            raise ValueError("data.path 必须是非空路径或非空路径列表")
        for configured_path in configured_paths:
            resolved = _resolve_data_path(path, configured_path)
            sources.append(CorpusSource(resolved, _infer_domain(resolved)))
    config = ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        folds=int(raw.get("folds", 5)),
        dev_ratio=float(raw.get("dev_ratio", 0.125)),
        max_sequence_length=int(raw.get("max_sequence_length", 500)),
        data=DataConfig(
            sources=tuple(sources),
            missing_volume_numbers=tuple(data.get("missing_volume_numbers", [38])),
            missing_characters=tuple(data.get("missing_characters", ["□", "@", "…"])),
            ignored_editorial_symbols=tuple(
                data.get("ignored_editorial_symbols", ["√", "×", "△"])
            ),
            boundary_punctuation=tuple(data.get("boundary_punctuation", "，。、！？；：")),
        ),
        features=FeatureConfig(**feature),
        model=ModelConfig(**model),
        rules=RuleConfig(**rules),
        ngram=NGramConfig(**ngram),
        neural=NeuralConfig(**neural),
        pretraining=PretrainingConfig(
            **{
                **pretraining,
                "span_lengths": tuple(pretraining.get("span_lengths", (2, 3, 4))),
                "checkpoint": (
                    _resolve_data_path(path, pretraining["checkpoint"])
                    if pretraining.get("checkpoint")
                    else None
                ),
            }
        ),
        word_pretraining=WordPretrainingConfig(
            **{
                **word_pretraining,
                "initial_checkpoint": (
                    _resolve_data_path(path, word_pretraining["initial_checkpoint"])
                    if word_pretraining.get("initial_checkpoint")
                    else None
                ),
                "checkpoint": (
                    _resolve_data_path(path, word_pretraining["checkpoint"])
                    if word_pretraining.get("checkpoint")
                    else None
                ),
                "dictionary_path": (
                    _resolve_data_path(path, word_pretraining["dictionary_path"])
                    if word_pretraining.get("dictionary_path")
                    else None
                ),
                "tongyin_path": (
                    _resolve_data_path(path, word_pretraining["tongyin_path"])
                    if word_pretraining.get("tongyin_path")
                    else None
                ),
                "candidate_lengths": tuple(
                    word_pretraining.get("candidate_lengths", (2, 3, 4))
                ),
                "statistical_min_frequencies": tuple(
                    word_pretraining.get("statistical_min_frequencies", (5, 4, 3))
                ),
            }
        ),
        knowledge=KnowledgeConfig(
            domain=DomainKnowledgeConfig(**knowledge.get("domain", {})),
            lexicon=LexiconKnowledgeConfig(
                **{
                    **knowledge.get("lexicon", {}),
                    "dictionary_path": (
                        _resolve_data_path(
                            path, knowledge["lexicon"]["dictionary_path"]
                        )
                        if knowledge.get("lexicon", {}).get("dictionary_path")
                        else None
                    ),
                    "tongyin_path": (
                        _resolve_data_path(path, knowledge["lexicon"]["tongyin_path"])
                        if knowledge.get("lexicon", {}).get("tongyin_path")
                        else None
                    ),
                    "candidate_lengths": tuple(
                        knowledge.get("lexicon", {}).get(
                            "candidate_lengths", (2, 3, 4)
                        )
                    ),
                }
            ),
            context=ContextKnowledgeConfig(
                **{
                    **context_knowledge,
                    "source_mapping": tuple(
                        (
                            str(source_stem),
                            _resolve_data_path(path, source_path),
                        )
                        for source_stem, source_path in context_knowledge.get(
                            "source_mapping", {}
                        ).items()
                    ),
                }
            ),
            segmentation=SegmentationKnowledgeConfig(
                **{
                    **segmentation_knowledge,
                    "model_path": (
                        _resolve_data_path(path, segmentation_knowledge["model_path"])
                        if segmentation_knowledge.get("model_path")
                        else None
                    ),
                    "lexicon_path": (
                        _resolve_data_path(path, segmentation_knowledge["lexicon_path"])
                        if segmentation_knowledge.get("lexicon_path")
                        else None
                    ),
                    "gap_path": (
                        _resolve_data_path(path, segmentation_knowledge["gap_path"])
                        if segmentation_knowledge.get("gap_path")
                        else None
                    ),
                    "annotation_paths": tuple(
                        _resolve_data_path(path, value)
                        for value in segmentation_knowledge.get(
                            "annotation_paths", ()
                        )
                    ),
                }
            ),
            pos=POSKnowledgeConfig(
                **{
                    **pos_knowledge,
                    "model_path": (
                        _resolve_data_path(path, pos_knowledge["model_path"])
                        if pos_knowledge.get("model_path")
                        else None
                    ),
                    "lexicon_state_path": (
                        _resolve_data_path(path, pos_knowledge["lexicon_state_path"])
                        if pos_knowledge.get("lexicon_state_path")
                        else None
                    ),
                    "manifest_path": (
                        _resolve_data_path(path, pos_knowledge["manifest_path"])
                        if pos_knowledge.get("manifest_path")
                        else None
                    ),
                    "annotation_paths": tuple(
                        _resolve_data_path(path, value)
                        for value in pos_knowledge.get("annotation_paths", ())
                    ),
                }
            ),
            pos_relation=POSRelationKnowledgeConfig(
                **{
                    **pos_relation_knowledge,
                    "fusion_weight_candidates": tuple(
                        float(value)
                        for value in pos_relation_knowledge.get(
                            "fusion_weight_candidates",
                            (0.0, 0.25, 0.5, 0.75, 1.0),
                        )
                    ),
                }
            ),
        ),
        d_downstream=DDownstreamConfig(
            initialization_source=str(
                d_downstream.get("initialization_source", "d1")
            ).lower()
        ),
        punctuation_pretraining=PunctuationPretrainingConfig(
            **punctuation_pretraining
        ),
        cascade=CascadeConfig(
            inner_folds=int(cascade.get("inner_folds", 3)),
            candidate_recall_target=float(
                cascade.get("candidate_recall_target", 0.95)
            ),
            soft_train_alpha=float(cascade.get("soft_train_alpha", 1.0)),
            soft_alpha_candidates=tuple(
                float(value)
                for value in cascade.get(
                    "soft_alpha_candidates", (0.25, 0.5, 1.0, 2.0)
                )
            ),
            soft_epsilon=float(cascade.get("soft_epsilon", 1.0e-8)),
            multitask_position_loss_weight=float(
                cascade.get("multitask_position_loss_weight", 0.5)
            ),
            shared_encoder=str(cascade.get("shared_encoder", "bilstm")),
        ),
        data_augmentation=DataAugmentationConfig(**data_augmentation),
        punctuation=PunctuationConfig(
            sentence_pause=tuple(punctuation.get("sentence_pause", "。？！")),
            intra_sentence_pause=tuple(
                punctuation.get("intra_sentence_pause", "，、；：")
            ),
            structural=tuple(punctuation.get("structural", "“”‘’《》")),
        ),
    )
    if config.folds < 2:
        raise ValueError("folds 至少为 2")
    if not 0 < config.dev_ratio < 1:
        raise ValueError("dev_ratio 必须在 (0, 1) 内")
    if config.max_sequence_length <= 0:
        raise ValueError("max_sequence_length 必须大于 0")
    if config.features.window < 0 or config.features.stage_feature_window < 0:
        raise ValueError("特征窗口不能为负数")
    if config.rules.cue_min_support <= 0 or config.rules.structure_min_support <= 0:
        raise ValueError("规则最小支持数必须大于 0")
    if not 0 < config.rules.cue_min_confidence <= 1:
        raise ValueError("cue_min_confidence 必须在 (0, 1] 内")
    if not 0 < config.rules.structure_min_confidence <= 1:
        raise ValueError("structure_min_confidence 必须在 (0, 1] 内")
    if config.rules.cue_max_length not in (1, 2):
        raise ValueError("cue_max_length 目前只支持 1 或 2")
    if not 1 <= config.ngram.n <= 6:
        raise ValueError("ngram.n 必须在 1 到 6 之间")
    if config.ngram.min_support <= 0:
        raise ValueError("ngram.min_support 必须大于 0")
    if config.ngram.alpha <= 0 or config.ngram.backoff_k <= 0:
        raise ValueError("ngram平滑参数必须大于 0")
    if not 0 < config.ngram.threshold_start <= config.ngram.threshold_end < 1:
        raise ValueError("ngram阈值范围必须位于 (0, 1) 且起点不大于终点")
    if config.ngram.threshold_step <= 0:
        raise ValueError("ngram.threshold_step 必须大于 0")
    if not any(
        (config.ngram.use_left, config.ngram.use_right, config.ngram.use_cross_gap)
    ):
        raise ValueError("ngram至少要启用一种上下文")
    if config.neural.embedding_dim <= 0 or config.neural.batch_size <= 0:
        raise ValueError("神经模型embedding_dim和batch_size必须大于 0")
    if config.neural.epochs <= 0 or config.neural.patience <= 0:
        raise ValueError("神经模型epochs和patience必须大于 0")
    if not 0 <= config.neural.dropout < 1:
        raise ValueError("神经模型dropout必须在 [0, 1) 内")
    if config.neural.learning_rate <= 0 or config.neural.weight_decay < 0:
        raise ValueError("神经模型学习率必须大于 0，weight_decay不能为负数")
    if config.neural.gradient_clip <= 0 or config.neural.min_delta < 0:
        raise ValueError("gradient_clip必须大于 0，min_delta不能为负数")
    if config.neural.bilstm_hidden_dim <= 0 or config.neural.bilstm_layers <= 0:
        raise ValueError("BiLSTM隐藏维度和层数必须大于 0")
    if (
        config.neural.transformer_layers <= 0
        or config.neural.transformer_heads <= 0
        or config.neural.transformer_ff_dim <= 0
    ):
        raise ValueError("Transformer层数、注意力头数和前馈维度必须大于 0")
    if config.neural.embedding_dim % config.neural.transformer_heads:
        raise ValueError("embedding_dim必须能被transformer_heads整除")
    device = config.neural.device
    indexed_cuda = device.startswith("cuda:") and device[5:].isdigit()
    if device not in {"auto", "cpu", "cuda"} and not indexed_cuda:
        raise ValueError("neural.device只能是 auto、cpu、cuda 或 cuda:N")
    if not 0 <= config.neural.cuda_max_utilization <= 100:
        raise ValueError("cuda_max_utilization必须在 [0, 100] 内")
    if config.neural.cuda_min_free_memory_gb < 0:
        raise ValueError("cuda_min_free_memory_gb不能为负数")
    if not (
        0
        < config.neural.position_threshold_start
        <= config.neural.position_threshold_end
        < 1
    ):
        raise ValueError("神经模型位置阈值范围必须位于 (0, 1)")
    if config.neural.position_threshold_step <= 0:
        raise ValueError("position_threshold_step必须大于 0")
    if config.pretraining.stage != "context_mlm":
        raise ValueError("当前主实验D只实现pretraining.stage=context_mlm")
    if not 0 < config.pretraining.validation_ratio < 1:
        raise ValueError("pretraining.validation_ratio必须位于 (0, 1)")
    if (
        config.pretraining.min_sequence_length <= 0
        or config.pretraining.max_sequence_length <= 0
        or config.pretraining.min_sequence_length
        > config.pretraining.max_sequence_length
    ):
        raise ValueError("预训练序列长度范围无效")
    if not 0 < config.pretraining.mask_ratio < 1:
        raise ValueError("pretraining.mask_ratio必须位于 (0, 1)")
    if not 0 <= config.pretraining.span_mask_probability <= 1:
        raise ValueError("pretraining.span_mask_probability必须位于 [0, 1]")
    if not config.pretraining.span_lengths or any(
        length < 2 for length in config.pretraining.span_lengths
    ):
        raise ValueError("pretraining.span_lengths必须是长度至少为2的非空列表")
    replace_total = (
        config.pretraining.mask_replace_probability
        + config.pretraining.random_replace_probability
    )
    if (
        config.pretraining.mask_replace_probability < 0
        or config.pretraining.random_replace_probability < 0
        or replace_total > 1
    ):
        raise ValueError("MLM替换概率必须非负且两者之和不能超过1")
    if (
        config.pretraining.embedding_dim <= 0
        or config.pretraining.layers <= 0
        or config.pretraining.heads <= 0
        or config.pretraining.ff_dim <= 0
    ):
        raise ValueError("TangutEncoder模型维度、层数和注意力头数必须大于0")
    if config.pretraining.embedding_dim % config.pretraining.heads:
        raise ValueError("pretraining.embedding_dim必须能被heads整除")
    if not 0 <= config.pretraining.dropout < 1:
        raise ValueError("pretraining.dropout必须位于 [0, 1)")
    if config.pretraining.batch_size <= 0 or config.pretraining.max_steps <= 0:
        raise ValueError("预训练batch_size和max_steps必须大于0")
    if config.pretraining.learning_rate <= 0 or config.pretraining.weight_decay < 0:
        raise ValueError("预训练学习率必须大于0，weight_decay不能为负数")
    if not 0 <= config.pretraining.warmup_ratio < 1:
        raise ValueError("pretraining.warmup_ratio必须位于 [0, 1)")
    if config.pretraining.eval_interval <= 0 or config.pretraining.log_interval <= 0:
        raise ValueError("预训练eval_interval和log_interval必须大于0")
    if config.pretraining.patience <= 0 or config.pretraining.min_delta < 0:
        raise ValueError("预训练patience必须大于0，min_delta不能为负数")
    if config.pretraining.gradient_clip <= 0:
        raise ValueError("pretraining.gradient_clip必须大于0")
    pretrain_device = config.pretraining.device
    indexed_pretrain_cuda = (
        pretrain_device.startswith("cuda:") and pretrain_device[5:].isdigit()
    )
    if pretrain_device not in {"auto", "cpu", "cuda"} and not indexed_pretrain_cuda:
        raise ValueError("pretraining.device只能是 auto、cpu、cuda 或 cuda:N")
    if not 0 <= config.pretraining.cuda_max_utilization <= 100:
        raise ValueError("pretraining.cuda_max_utilization必须位于 [0, 100]")
    if config.pretraining.cuda_min_free_memory_gb < 0:
        raise ValueError("pretraining.cuda_min_free_memory_gb不能为负数")
    if (
        config.pretraining.downstream_encoder_learning_rate <= 0
        or config.pretraining.downstream_head_learning_rate <= 0
    ):
        raise ValueError("TangutEncoder下游编码器和分类头学习率必须大于0")
    if config.pretraining.downstream_freeze_epochs < 0:
        raise ValueError("pretraining.downstream_freeze_epochs不能小于0")
    if config.d_downstream.initialization_source not in {"d1", "d2"}:
        raise ValueError("d_downstream.initialization_source只能是d1或d2")
    d5 = config.punctuation_pretraining
    if d5.batch_size <= 0 or d5.max_steps <= 0:
        raise ValueError("D5 batch_size和max_steps必须大于0")
    if d5.encoder_learning_rate <= 0 or d5.head_learning_rate <= 0:
        raise ValueError("D5编码器和任务头学习率必须大于0")
    if d5.weight_decay < 0 or not 0 <= d5.warmup_ratio < 1:
        raise ValueError("D5 weight_decay不能为负，warmup_ratio必须位于[0,1)")
    if d5.eval_interval <= 0 or d5.log_interval <= 0 or d5.patience <= 0:
        raise ValueError("D5评估、日志间隔和patience必须大于0")
    if d5.min_delta < 0 or d5.gradient_clip <= 0:
        raise ValueError("D5 min_delta不能为负，gradient_clip必须大于0")
    if any(
        weight < 0
        for weight in (
            d5.mlm_weight,
            d5.position_weight,
            d5.group_weight,
            d5.type_weight,
        )
    ) or not any(
        (
            d5.mlm_weight,
            d5.position_weight,
            d5.group_weight,
            d5.type_weight,
        )
    ):
        raise ValueError("D5损失权重必须非负且至少一项大于0")
    if not (
        0
        < d5.position_threshold_start
        <= d5.position_threshold_end
        < 1
    ) or d5.position_threshold_step <= 0:
        raise ValueError("D5位置阈值搜索范围无效")
    word = config.word_pretraining
    if not word.candidate_lengths or any(length < 2 for length in word.candidate_lengths):
        raise ValueError("word_pretraining.candidate_lengths必须是长度至少为2的非空列表")
    if len(word.statistical_min_frequencies) != len(word.candidate_lengths):
        raise ValueError(
            "word_pretraining.statistical_min_frequencies必须与candidate_lengths等长"
        )
    if any(value <= 0 for value in word.statistical_min_frequencies):
        raise ValueError("D2统计候选的最小词频必须大于0")
    if word.dictionary_min_frequency <= 0 or word.intersection_min_frequency <= 0:
        raise ValueError("D2辞书候选的最小词频必须大于0")
    if not 0 <= word.whole_word_probability <= 1:
        raise ValueError("word_pretraining.whole_word_probability必须位于[0, 1]")
    if word.ranking_negatives <= 0 or word.ranking_loss_weight < 0:
        raise ValueError("D2排序负例数必须大于0，排序损失权重不能为负")
    if word.ranking_warmup_steps < 0:
        raise ValueError("word_pretraining.ranking_warmup_steps不能为负")
    if word.encoder_learning_rate <= 0 or word.head_learning_rate <= 0:
        raise ValueError("D2编码器和词语头学习率必须大于0")
    if word.batch_size <= 0 or word.max_steps <= 0:
        raise ValueError("D2 batch_size和max_steps必须大于0")
    if word.weight_decay < 0 or not 0 <= word.warmup_ratio < 1:
        raise ValueError("D2 weight_decay不能为负，warmup_ratio必须位于[0, 1)")
    if word.eval_interval <= 0 or word.log_interval <= 0 or word.patience <= 0:
        raise ValueError("D2评估间隔、日志间隔和patience必须大于0")
    if word.min_delta < 0 or word.gradient_clip <= 0:
        raise ValueError("D2 min_delta不能为负，gradient_clip必须大于0")
    domain = config.knowledge.domain
    if domain.dimension != 2:
        raise ValueError("E1局部领域分布固定为2维：[经书倾向, 世俗倾向]")
    if domain.inner_folds < 2:
        raise ValueError("knowledge.domain.inner_folds至少为2")
    if domain.smoothing <= 0 or domain.shrinkage < 0:
        raise ValueError("E1领域平滑必须大于0，收缩强度不能为负")
    if domain.candidate_mode not in {"lexicon", "fusion"}:
        raise ValueError("knowledge.domain.candidate_mode只能是lexicon或fusion")
    lexicon = config.knowledge.lexicon
    if not lexicon.candidate_lengths or any(
        length < 2 for length in lexicon.candidate_lengths
    ):
        raise ValueError("knowledge.lexicon.candidate_lengths必须是长度至少为2的非空列表")
    if len(set(lexicon.candidate_lengths)) != len(lexicon.candidate_lengths):
        raise ValueError("knowledge.lexicon.candidate_lengths不能包含重复值")
    if lexicon.representation != "gap_lattice":
        raise ValueError("knowledge.lexicon.representation目前只支持gap_lattice")
    if lexicon.enabled and (
        lexicon.dictionary_path is None or lexicon.tongyin_path is None
    ):
        raise ValueError("E2需要配置knowledge.lexicon的dictionary_path和tongyin_path")
    context = config.knowledge.context
    if context.dimension != 8:
        raise ValueError("E3上下文统计固定为左右间隔各4维，共8维")
    if context.inner_folds < 2:
        raise ValueError("knowledge.context.inner_folds至少为2")
    if context.association not in {"dpmi", "t_score", "dice"}:
        raise ValueError(
            "knowledge.context.association只能是dpmi、t_score或dice"
        )
    if context.dpmi_discount < 0:
        raise ValueError("knowledge.context.dpmi_discount不能为负")
    if not 50.0 <= context.clipping_percentile <= 100.0:
        raise ValueError("knowledge.context.clipping_percentile必须位于[50,100]")
    if context.enabled:
        if not context.source_mapping:
            raise ValueError("E3需要配置knowledge.context.source_mapping")
        stems = [stem for stem, _ in context.source_mapping]
        if len(stems) != len(set(stems)):
            raise ValueError("knowledge.context.source_mapping不能包含重复stem")
        missing_paths = [str(path) for _, path in context.source_mapping if not path.exists()]
        if missing_paths:
            raise FileNotFoundError("找不到E3未截长语料：" + "、".join(missing_paths))
    segmentation = config.knowledge.segmentation
    expected_segmentation_dimension = {
        "compact": 3,
        "full": 8,
    }.get(segmentation.representation)
    if expected_segmentation_dimension is None:
        raise ValueError(
            "knowledge.segmentation.representation只能是compact或full"
        )
    if segmentation.dimension != expected_segmentation_dimension:
        raise ValueError(
            "knowledge.segmentation.dimension与representation不一致："
            f"{segmentation.representation}应为{expected_segmentation_dimension}维"
        )
    if segmentation.fusion not in {"direct", "gated_residual"}:
        raise ValueError(
            "knowledge.segmentation.fusion只能是direct或gated_residual"
        )
    if segmentation.overlap_policy not in {"mask_exact", "error", "allow"}:
        raise ValueError(
            "knowledge.segmentation.overlap_policy只能是mask_exact、error或allow"
        )
    if segmentation.min_overlap_length < 2:
        raise ValueError("knowledge.segmentation.min_overlap_length至少为2")
    if segmentation.max_word_length <= 0:
        raise ValueError("knowledge.segmentation.max_word_length必须大于0")
    if segmentation.enabled:
        resources = (
            segmentation.model_path,
            segmentation.lexicon_path,
            segmentation.gap_path,
        )
        if any(resource is None for resource in resources):
            raise ValueError("E4需要配置CRF model、lexicon和gap三个分词资源")
        missing_resources = [
            str(resource)
            for resource in (*resources, *segmentation.annotation_paths)
            if resource is not None and not resource.exists()
        ]
        if missing_resources:
            raise FileNotFoundError("找不到E4分词资源：" + "、".join(missing_resources))
        if (
            segmentation.overlap_policy in {"mask_exact", "error"}
            and not segmentation.annotation_paths
        ):
            raise ValueError("E4重叠保护需要配置annotation_paths")
    pos = config.knowledge.pos
    expected_pos_dimension = {
        "coarse_soft_character": 8,
        "coarse_soft_gap": 17,
    }.get(pos.representation)
    if expected_pos_dimension is None:
        raise ValueError(
            "knowledge.pos.representation只能是coarse_soft_character或"
            "coarse_soft_gap"
        )
    if pos.group_scheme != "seven":
        raise ValueError("knowledge.pos.group_scheme目前只支持seven")
    if pos.raw_dimension != expected_pos_dimension:
        raise ValueError(
            "knowledge.pos.raw_dimension与representation不一致："
            f"{pos.representation}应为{expected_pos_dimension}维"
        )
    if pos.fusion not in {"direct", "projected"}:
        raise ValueError("knowledge.pos.fusion只能是direct或projected")
    if pos.projection_dimension <= 0:
        raise ValueError("knowledge.pos.projection_dimension必须大于0")
    if not 0 <= pos.channel_dropout < 1:
        raise ValueError("knowledge.pos.channel_dropout必须位于[0,1)")
    if pos.overlap_policy not in {"mask_exact", "error", "allow"}:
        raise ValueError(
            "knowledge.pos.overlap_policy只能是mask_exact、error或allow"
        )
    if pos.min_overlap_length < 2:
        raise ValueError("knowledge.pos.min_overlap_length至少为2")
    if pos.enabled:
        resources = (pos.model_path, pos.lexicon_state_path, pos.manifest_path)
        if any(resource is None for resource in resources):
            raise ValueError("E5需要配置POS model、lexicon_state和manifest三个资源")
        missing_resources = [
            str(resource)
            for resource in (*resources, *pos.annotation_paths)
            if resource is not None and not resource.exists()
        ]
        if missing_resources:
            raise FileNotFoundError("找不到E5词性资源：" + "、".join(missing_resources))
    pos_relation = config.knowledge.pos_relation
    if pos_relation.tag_embedding_dim <= 0:
        raise ValueError("knowledge.pos_relation.tag_embedding_dim必须大于0")
    if pos_relation.relation_hidden_dim <= 0:
        raise ValueError("knowledge.pos_relation.relation_hidden_dim必须大于0")
    if not 0 <= pos_relation.relation_dropout < 1:
        raise ValueError("knowledge.pos_relation.relation_dropout必须位于[0,1)")
    if not 0 <= pos_relation.channel_dropout < 1:
        raise ValueError("knowledge.pos_relation.channel_dropout必须位于[0,1)")
    if pos_relation.learning_rate <= 0:
        raise ValueError("knowledge.pos_relation.learning_rate必须大于0")
    if pos_relation.epochs <= 0 or pos_relation.patience <= 0:
        raise ValueError("knowledge.pos_relation.epochs和patience必须大于0")
    if pos_relation.min_delta < 0:
        raise ValueError("knowledge.pos_relation.min_delta不能为负数")
    candidates = pos_relation.fusion_weight_candidates
    if not candidates or any(not 0 <= value <= 1 for value in candidates):
        raise ValueError(
            "knowledge.pos_relation.fusion_weight_candidates必须是[0,1]内的非空列表"
        )
    if 0.0 not in candidates:
        raise ValueError(
            "knowledge.pos_relation.fusion_weight_candidates必须包含0以允许退回E3"
        )
    if pos_relation.enabled and not pos.enabled:
        raise ValueError("E7/E8复用knowledge.pos冻结资源，必须同时启用knowledge.pos")
    if config.cascade.inner_folds < 2:
        raise ValueError("cascade.inner_folds至少为 2")
    if not 0 < config.cascade.candidate_recall_target <= 1:
        raise ValueError("candidate_recall_target必须位于 (0, 1]")
    if config.cascade.soft_train_alpha < 0:
        raise ValueError("soft_train_alpha不能为负数")
    if not config.cascade.soft_alpha_candidates or any(
        value < 0 for value in config.cascade.soft_alpha_candidates
    ):
        raise ValueError("soft_alpha_candidates必须是非空的非负数列表")
    if config.cascade.soft_epsilon <= 0:
        raise ValueError("soft_epsilon必须大于 0")
    if config.cascade.multitask_position_loss_weight < 0:
        raise ValueError("multitask_position_loss_weight不能为负数")
    if not config.cascade.shared_encoder:
        raise ValueError("cascade.shared_encoder不能为空")
    augmentation = config.data_augmentation
    if augmentation.method != "sentence_concatenation":
        raise ValueError(
            "data_augmentation.method目前只支持sentence_concatenation"
        )
    if augmentation.ratio < 0:
        raise ValueError("data_augmentation.ratio不能为负数")
    if augmentation.min_sentences < 2:
        raise ValueError("data_augmentation.min_sentences至少为2")
    if augmentation.max_sentences < augmentation.min_sentences:
        raise ValueError(
            "data_augmentation.max_sentences不能小于min_sentences"
        )
    if augmentation.min_characters <= 0:
        raise ValueError("data_augmentation.min_characters必须大于0")
    if augmentation.max_characters < augmentation.min_characters:
        raise ValueError(
            "data_augmentation.max_characters不能小于min_characters"
        )
    if (
        augmentation.enabled
        and augmentation.max_characters > config.max_sequence_length
    ):
        raise ValueError(
            "data_augmentation.max_characters不能超过max_sequence_length，"
            "否则合成段落会再次被截断"
        )
    if augmentation.scope != "same_document":
        raise ValueError("data_augmentation.scope目前只支持same_document")
    if not augmentation.preserve_order:
        raise ValueError("E9-Aug必须保持句子在原文中的先后次序")
    punctuation_groups = (
        set(config.punctuation.sentence_pause),
        set(config.punctuation.intra_sentence_pause),
        set(config.punctuation.structural),
    )
    if any(
        punctuation_groups[left] & punctuation_groups[right]
        for left in range(len(punctuation_groups))
        for right in range(left + 1, len(punctuation_groups))
    ):
        raise ValueError("句间、句内和结构标点分组必须互不重叠")
    return config
