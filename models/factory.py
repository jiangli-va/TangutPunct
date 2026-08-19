from __future__ import annotations

from collections.abc import Callable

from config import ExperimentConfig
from features.basic import BasicCharacterFeatures
from knowledge import (
    CompositeKnowledgeProvider,
    ContextStatisticsProvider,
    LexiconGapLatticeProvider,
    LocalDomainDistributionProvider,
    POSKnowledgeProvider,
    POSRelationKnowledgeProvider,
    SegmentationKnowledgeProvider,
)
from knowledge.pos import FINE_POS_LABELS
from .base import SequenceTagger
from .crf import CRFTagger
from .cascade import CascadeBiLSTMTagger, MultiTaskBiLSTMTagger
from .ngram import BackoffNGramTagger
from .neural import (
    INTRA_GROUP_LABEL,
    SENTENCE_GROUP_LABEL,
    NeuralSequenceTagger,
)
from .rules import JointRuleTagger, LengthMajorityRuleTagger
from .tangut_tagger import (
    KnowledgeEnhancedTangutEncoderTagger,
    MultiTaskKnowledgeEnhancedTangutEncoderTagger,
    POSRelationResidualTangutEncoderTagger,
    TangutEncoderSequenceTagger,
)


ModelBuilder = Callable[[ExperimentConfig], SequenceTagger]
MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


@register_model("crf")
def build_crf(config: ExperimentConfig) -> SequenceTagger:
    feature = BasicCharacterFeatures(
        missing_characters=frozenset(config.data.missing_characters),
        window=config.features.window,
        use_ngrams=config.features.use_ngrams,
        use_missing=config.features.use_missing,
        use_document_edges=config.features.use_document_edges,
        use_stage_features=config.features.use_stage_features,
        stage_feature_window=config.features.stage_feature_window,
    )
    return CRFTagger(
        feature,
        c1=config.model.c1,
        c2=config.model.c2,
        max_iterations=config.model.max_iterations,
        all_possible_transitions=config.model.all_possible_transitions,
    )


def _pause_labels(config: ExperimentConfig) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            config.punctuation.sentence_pause
            + config.punctuation.intra_sentence_pause
        )
    )


def _pause_group_labels() -> tuple[str, ...]:
    """F1直接分类的两个正类；O由神经标签器自动加入。"""

    return (SENTENCE_GROUP_LABEL, INTRA_GROUP_LABEL)


@register_model("length_majority_rule")
def build_length_majority_rule(config: ExperimentConfig) -> SequenceTagger:
    return LengthMajorityRuleTagger(
        _pause_labels(config),
        force_document_final_period=config.rules.force_document_final_period,
    )


@register_model("joint_rule")
def build_joint_rule(config: ExperimentConfig) -> SequenceTagger:
    return JointRuleTagger(
        _pause_labels(config),
        cue_min_support=config.rules.cue_min_support,
        cue_min_confidence=config.rules.cue_min_confidence,
        structure_min_support=config.rules.structure_min_support,
        structure_min_confidence=config.rules.structure_min_confidence,
        cue_max_length=config.rules.cue_max_length,
        force_document_final_period=config.rules.force_document_final_period,
    )


@register_model("ngram")
def build_ngram(config: ExperimentConfig) -> SequenceTagger:
    return BackoffNGramTagger(
        _pause_labels(config),
        n=config.ngram.n,
        min_support=config.ngram.min_support,
        alpha=config.ngram.alpha,
        backoff_k=config.ngram.backoff_k,
        threshold_start=config.ngram.threshold_start,
        threshold_end=config.ngram.threshold_end,
        threshold_step=config.ngram.threshold_step,
        use_left=config.ngram.use_left,
        use_right=config.ngram.use_right,
        use_cross_gap=config.ngram.use_cross_gap,
    )


def _build_neural(
    config: ExperimentConfig,
    encoder_type: str,
    decoder_type: str = "softmax",
) -> SequenceTagger:
    return NeuralSequenceTagger(
        encoder_type,
        _pause_labels(config),
        config.neural,
        config.max_sequence_length,
        config.seed,
        decoder_type=decoder_type,
    )


@register_model("bilstm")
def build_bilstm(config: ExperimentConfig) -> SequenceTagger:
    return _build_neural(config, "bilstm")


@register_model("bilstm_crf")
def build_bilstm_crf(config: ExperimentConfig) -> SequenceTagger:
    return _build_neural(config, "bilstm", decoder_type="crf")


@register_model("random_transformer")
def build_random_transformer(config: ExperimentConfig) -> SequenceTagger:
    return _build_neural(config, "random_transformer")


@register_model("tangut_encoder")
def build_pretrained_tangut_encoder(config: ExperimentConfig) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError(
            "TangutEncoder自动标点评测需要pretraining.checkpoint，"
            "或在命令行使用--checkpoint指定best_model.pt"
        )
    return TangutEncoderSequenceTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
    )


def _build_pretrained_tangut_head(
    config: ExperimentConfig, head_type: str
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError(
            "D3/D4需要TangutEncoder checkpoint；请设置初始化来源，"
            "或在命令行使用--checkpoint指定best_model.pt"
        )
    return TangutEncoderSequenceTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        head_type=head_type,
    )


@register_model("tangut_encoder_crf")
def build_pretrained_tangut_encoder_crf(config: ExperimentConfig) -> SequenceTagger:
    return _build_pretrained_tangut_head(config, "crf")


@register_model("tangut_encoder_bilstm")
def build_pretrained_tangut_encoder_bilstm(config: ExperimentConfig) -> SequenceTagger:
    return _build_pretrained_tangut_head(config, "bilstm")


def _build_knowledge_provider(
    config: ExperimentConfig, channels: tuple[str, ...]
) -> CompositeKnowledgeProvider:
    """按实验显式选择知识通道，避免单项消融意外累加其他特征。"""

    providers = []
    if "domain" in channels:
        if not config.knowledge.domain.enabled:
            raise ValueError("E1需要启用knowledge.domain")
        providers.append(
            LocalDomainDistributionProvider(
                config.knowledge.domain,
                config.word_pretraining,
                config.seed,
            )
        )
    if "lexicon" in channels:
        if not config.knowledge.lexicon.enabled:
            raise ValueError("E2需要启用knowledge.lexicon")
        providers.append(LexiconGapLatticeProvider(config.knowledge.lexicon))
    if "context" in channels:
        if not config.knowledge.context.enabled:
            raise ValueError("E3需要启用knowledge.context")
        providers.append(
            ContextStatisticsProvider(
                config.knowledge.context,
                config.data.missing_characters,
                config.seed,
            )
        )
    if "segmentation" in channels:
        if not config.knowledge.segmentation.enabled:
            raise ValueError("E4需要启用knowledge.segmentation")
        providers.append(
            SegmentationKnowledgeProvider(config.knowledge.segmentation)
        )
    if "pos" in channels:
        if not config.knowledge.pos.enabled:
            raise ValueError("E5需要启用knowledge.pos")
        providers.append(POSKnowledgeProvider(config.knowledge.pos))
    if "pos_relation" in channels:
        if not config.knowledge.pos_relation.enabled:
            raise ValueError("E7/E8需要启用knowledge.pos_relation")
        if not config.knowledge.pos.enabled:
            raise ValueError("E7/E8需要启用knowledge.pos以加载冻结词性资源")
        providers.append(
            POSRelationKnowledgeProvider(
                config.knowledge.pos,
                config.knowledge.pos_relation,
            )
        )
    if not providers:
        raise ValueError("知识增强模型至少要在knowledge配置中启用一个通道")
    return CompositeKnowledgeProvider(providers)


@register_model("tangut_encoder_bilstm_knowledge")
def build_knowledge_enhanced_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    return KnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        _build_knowledge_provider(config, ("domain",)),
    )


@register_model("tangut_encoder_bilstm_lexicon")
def build_lexicon_enhanced_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    return KnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        _build_knowledge_provider(config, ("lexicon",)),
    )


@register_model("tangut_encoder_bilstm_lexicon_context")
def build_lexicon_context_enhanced_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    return KnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        _build_knowledge_provider(config, ("lexicon", "context")),
    )


@register_model("tangut_encoder_bilstm_lexicon_context_multitask")
def build_lexicon_context_multitask_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    """E9：在E3知识增强主干上复用C4共享编码器多任务协议。"""

    if config.pretraining.checkpoint is None:
        raise ValueError("E9需要D4使用的TangutEncoder checkpoint")
    return MultiTaskKnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.cascade,
        config.max_sequence_length,
        config.seed,
        _build_knowledge_provider(config, ("lexicon", "context")),
    )


@register_model("tangut_encoder_bilstm_lexicon_context_group_multitask")
def build_lexicon_context_group_multitask_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    """F1：严格复用E9主干，直接学习O/句内/句间。"""

    if config.pretraining.checkpoint is None:
        raise ValueError("F1需要E9使用的D2 TangutEncoder checkpoint")
    return MultiTaskKnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_group_labels(),
        config.neural,
        config.pretraining,
        config.cascade,
        config.max_sequence_length,
        config.seed,
        _build_knowledge_provider(config, ("lexicon", "context")),
        experiment_name="F1",
        primary_label_description="O/句内/句间",
    )


@register_model("tangut_encoder_bilstm_lexicon_context_segmentation")
def build_lexicon_context_segmentation_enhanced_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    provider = _build_knowledge_provider(
        config, ("lexicon", "context", "segmentation")
    )
    residual_dimension = (
        config.knowledge.segmentation.dimension
        if config.knowledge.segmentation.fusion == "gated_residual"
        else 0
    )
    return KnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        provider,
        residual_knowledge_dim=residual_dimension,
    )


@register_model("tangut_encoder_bilstm_lexicon_context_pos")
def build_lexicon_context_pos_enhanced_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    provider = _build_knowledge_provider(config, ("lexicon", "context", "pos"))
    projected = config.knowledge.pos.fusion == "projected"
    return KnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        provider,
        projected_knowledge_dim=(
            config.knowledge.pos.raw_dimension if projected else 0
        ),
        projected_knowledge_output_dim=(
            config.knowledge.pos.projection_dimension if projected else 0
        ),
        projected_knowledge_dropout=(
            config.knowledge.pos.channel_dropout if projected else 0.0
        ),
        direct_knowledge_channel_dim=(
            0 if projected else config.knowledge.pos.raw_dimension
        ),
        direct_knowledge_channel_dropout=(
            0.0 if projected else config.knowledge.pos.channel_dropout
        ),
    )


@register_model("tangut_encoder_bilstm_lexicon_context_pos_relation")
def build_lexicon_context_pos_relation_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    provider = _build_knowledge_provider(
        config, ("lexicon", "context", "pos_relation")
    )
    relation_dimension = provider.providers[-1].dimension
    base_dimension = provider.dimension - relation_dimension
    return POSRelationResidualTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        provider,
        base_knowledge_dim=base_dimension,
        relation_knowledge_dim=relation_dimension,
        pos_tag_count=len(FINE_POS_LABELS),
        relation_config=config.knowledge.pos_relation,
    )


@register_model("tangut_encoder_bilstm_lexicon_context_pos_relation_direct")
def build_direct_lexicon_context_pos_relation_tangut_encoder(
    config: ExperimentConfig,
) -> SequenceTagger:
    """E7：将23维E3知识与76维细粒度词性关系直接拼接进BiLSTM。"""

    if config.pretraining.checkpoint is None:
        raise ValueError("主实验E需要D4使用的TangutEncoder checkpoint")
    provider = _build_knowledge_provider(
        config, ("lexicon", "context", "pos_relation")
    )
    relation_dimension = provider.providers[-1].dimension
    return KnowledgeEnhancedTangutEncoderTagger(
        config.pretraining.checkpoint,
        _pause_labels(config),
        config.neural,
        config.pretraining,
        config.max_sequence_length,
        config.seed,
        provider,
        direct_knowledge_channel_dim=relation_dimension,
        # E7是强制直接注入对照：不再随机关闭整条词性关系通道。
        direct_knowledge_channel_dropout=0.0,
    )


@register_model("bilstm_candidate_reject")
def build_bilstm_candidate_reject(config: ExperimentConfig) -> SequenceTagger:
    return CascadeBiLSTMTagger(
        "candidate",
        _pause_labels(config),
        config.neural,
        config.cascade,
        config.max_sequence_length,
        config.seed,
    )


@register_model("bilstm_soft_cascade")
def build_bilstm_soft_cascade(config: ExperimentConfig) -> SequenceTagger:
    return CascadeBiLSTMTagger(
        "soft",
        _pause_labels(config),
        config.neural,
        config.cascade,
        config.max_sequence_length,
        config.seed,
    )


@register_model("bilstm_multitask")
def build_bilstm_multitask(config: ExperimentConfig) -> SequenceTagger:
    return MultiTaskBiLSTMTagger(
        _pause_labels(config),
        config.neural,
        config.cascade,
        config.max_sequence_length,
        config.seed,
    )


def build_model(config: ExperimentConfig, model_name: str | None = None) -> SequenceTagger:
    selected = model_name or config.model.name
    try:
        return MODEL_REGISTRY[selected](config)
    except KeyError as error:
        raise ValueError(f"未知模型 {selected!r}；可用模型：{sorted(MODEL_REGISTRY)}") from error
