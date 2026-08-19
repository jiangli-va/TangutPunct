from __future__ import annotations

from dataclasses import dataclass

from config import ExperimentConfig


POSITION_LABEL = "P"
SENTENCE_GROUP_LABEL = "SENTENCE"
INTRA_GROUP_LABEL = "INTRA"


@dataclass(frozen=True)
class StageSpec:
    name: str
    display_name: str
    target: str
    punctuation: frozenset[str] = frozenset()
    upstream: tuple[str, ...] = ()
    mask_upstream: str | None = None
    group_upstream: str | None = None
    # 留空时沿用实验级后端；用于同一流水线的不同阶段选择不同知识通道。
    model_name: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    display_name: str
    stages: tuple[StageSpec, ...]
    model_name: str | None = None
    strategy: str = "standard"
    augmentation: str | None = None

    @property
    def conditions(self) -> tuple[str, ...]:
        if self.strategy == "multitask":
            return ("direct",)
        return ("direct",) if len(self.stages) == 1 else ("oracle", "predicted")


def build_experiments(config: ExperimentConfig) -> dict[str, ExperimentSpec]:
    sentence = frozenset(config.punctuation.sentence_pause)
    intra = frozenset(config.punctuation.intra_sentence_pause)
    pause = sentence | intra
    return {
        "a1": ExperimentSpec(
            "a1",
            "A1 直接预测七种停顿标点",
            (StageSpec("pause_type", "具体停顿标点", "pause_type", pause),),
        ),
        "a2": ExperimentSpec(
            "a2",
            "A2 标点位置 → 具体停顿标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
        ),
        "a3": ExperimentSpec(
            "a3",
            "A3 标点位置 → 句间/句内 → 具体停顿标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_group",
                    "句间/句内类别",
                    "pause_group",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position", "pause_group"),
                    mask_upstream="position",
                    group_upstream="pause_group",
                ),
            ),
        ),
        "b0": ExperimentSpec(
            "b0",
            "B0 长度—多数类规则基线",
            (StageSpec("pause_type", "联合规则预测", "pause_type", pause),),
            model_name="length_majority_rule",
        ),
        "b1": ExperimentSpec(
            "b1",
            "B1 长度＋特征字词＋结构联合规则",
            (StageSpec("pause_type", "联合规则预测", "pause_type", pause),),
            model_name="joint_rule",
        ),
        "b2": ExperimentSpec(
            "b2",
            "B2 双向字符n-gram：位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="ngram",
        ),
        "b4": ExperimentSpec(
            "b4",
            "B4 随机字符嵌入BiLSTM：位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="bilstm",
        ),
        "b5": ExperimentSpec(
            "b5",
            "B5 随机初始化Transformer：位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="random_transformer",
        ),
        "b6": ExperimentSpec(
            "b6",
            "B6 随机字符嵌入BiLSTM＋CRF：位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="bilstm_crf",
        ),
        "c1": ExperimentSpec(
            "c1",
            "C1 BiLSTM硬级联：标点位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点（硬门控）",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="bilstm",
            strategy="hard_cascade",
        ),
        "c2": ExperimentSpec(
            "c2",
            "C2 BiLSTM高召回候选＋拒绝类",
            (
                StageSpec("position", "高召回标点候选", "pause_position", pause),
                StageSpec(
                    "candidate_reject",
                    "候选拒绝与具体标点",
                    "pause_type",
                    pause,
                    ("position",),
                ),
            ),
            model_name="bilstm_candidate_reject",
            strategy="candidate_reject",
        ),
        "c3": ExperimentSpec(
            "c3",
            "C3 BiLSTM连续概率软级联",
            (
                StageSpec("position", "标点位置概率", "pause_position", pause),
                StageSpec(
                    "soft_cascade",
                    "概率软融合具体标点",
                    "pause_type",
                    pause,
                    ("position",),
                ),
            ),
            model_name="bilstm_soft_cascade",
            strategy="soft_cascade",
        ),
        "c4": ExperimentSpec(
            "c4",
            "C4 共享BiLSTM编码器多任务联合模型",
            (
                StageSpec(
                    "multitask",
                    "共享编码器：位置辅助头＋完整标点头",
                    "pause_type",
                    pause,
                ),
            ),
            model_name="bilstm_multitask",
            strategy="multitask",
        ),
        "d1_eval": ExperimentSpec(
            "d1_eval",
            "D1 MLM预训练TangutEncoder：标点位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder",
            strategy="standard",
        ),
        "d2_eval": ExperimentSpec(
            "d2_eval",
            "D2词语感知预训练TangutEncoder：标点位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder",
            strategy="standard",
        ),
        "d3": ExperimentSpec(
            "d3",
            "D3 TangutEncoder＋CRF：标点位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_crf",
            strategy="standard",
        ),
        "d4": ExperimentSpec(
            "d4",
            "D4 TangutEncoder＋BiLSTM：标点位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm",
            strategy="standard",
        ),
        "d5_direct": ExperimentSpec(
            "d5_direct",
            "D5保留层级分类头直接预测",
            (StageSpec("d5", "D5层级标点直接输出", "pause_type", pause),),
            model_name="d5_direct",
            strategy="d5_direct",
        ),
        "d5_bilstm": ExperimentSpec(
            "d5_bilstm",
            "D5丢弃分类头＋BiLSTM：标点位置 → 具体标点",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="d5_bilstm",
            strategy="d5_bilstm",
        ),
        "e0": ExperimentSpec(
            "e0",
            "E0 D4强基线：TangutEncoder＋BiLSTM",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm",
            strategy="standard",
        ),
        "e1": ExperimentSpec(
            "e1",
            "E1 D4＋字符级局部词汇领域分布",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm_knowledge",
            strategy="standard",
        ),
        "e2": ExperimentSpec(
            "e2",
            "E2 D4＋间隔中心软词典格网",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon",
            strategy="standard",
        ),
        "e3": ExperimentSpec(
            "e3",
            "E3 D4＋软词典格网＋上下文统计",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context",
            strategy="standard",
        ),
        "e4": ExperimentSpec(
            "e4",
            "E4 D4＋软词典格网＋上下文统计＋软分词知识",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context_segmentation",
            strategy="standard",
        ),
        "e5": ExperimentSpec(
            "e5",
            "E5 D4＋软词典格网＋上下文统计＋七组软词性知识",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context_pos",
            strategy="standard",
        ),
        "e6": ExperimentSpec(
            "e6",
            "E6 E3位置阶段＋仅类别阶段融入七组软词性知识",
            (
                StageSpec("position", "停顿标点位置", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点（融入词性知识）",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                    model_name="tangut_encoder_bilstm_lexicon_context_pos",
                ),
            ),
            # 第一阶段严格沿用E3；第二阶段通过StageSpec.model_name单独覆盖。
            model_name="tangut_encoder_bilstm_lexicon_context",
            strategy="standard",
        ),
        "e7": ExperimentSpec(
            "e7",
            "E7 两阶段均直接注入细粒度左右词性关系",
            (
                StageSpec(
                    "position",
                    "停顿标点位置（细粒度词性关系直接注入）",
                    "pause_position",
                    pause,
                ),
                StageSpec(
                    "pause_type",
                    "具体停顿标点（细粒度词性关系直接注入）",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name=(
                "tangut_encoder_bilstm_lexicon_context_pos_relation_direct"
            ),
            strategy="standard",
        ),
        "e8": ExperimentSpec(
            "e8",
            "E8 两阶段均融入细粒度左右词性关系后期残差",
            (
                StageSpec("position", "停顿标点位置（细粒度词性关系残差）", "pause_position", pause),
                StageSpec(
                    "pause_type",
                    "具体停顿标点（细粒度词性关系残差）",
                    "pause_type",
                    pause,
                    ("position",),
                    mask_upstream="position",
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context_pos_relation",
            strategy="standard",
        ),
        "e9": ExperimentSpec(
            "e9",
            "E9 TangutEncoder＋BiLSTM＋E3共享编码器多任务联合模型",
            (
                StageSpec(
                    "multitask",
                    "共享编码器：位置辅助头＋O/七类完整标点头",
                    "pause_type",
                    pause,
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context_multitask",
            strategy="multitask",
        ),
        "e9_aug": ExperimentSpec(
            "e9_aug",
            "E9-Aug E9＋同文献完整句子拼接增强",
            (
                StageSpec(
                    "multitask",
                    "共享编码器：位置辅助头＋O/七类完整标点头",
                    "pause_type",
                    pause,
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context_multitask",
            strategy="multitask",
            augmentation="sentence_concatenation",
        ),
        "f1": ExperimentSpec(
            "f1",
            "F1 当前最优模型直接预测句内/句间停顿",
            (
                StageSpec(
                    "multitask",
                    "共享编码器：位置辅助头＋O/句内/句间主分类头",
                    "pause_group",
                    pause,
                ),
            ),
            model_name="tangut_encoder_bilstm_lexicon_context_group_multitask",
            strategy="multitask",
        ),
    }


# 供命令行分组和文档使用；真正的标点集合由配置构建。
A_EXPERIMENTS = ("a1", "a2", "a3")
B_EXPERIMENTS = ("b0", "b1", "b2", "b4", "b5", "b6")
C_EXPERIMENTS = ("c1", "c2", "c3", "c4")
D_EVALUATION_EXPERIMENTS = (
    "d1_eval",
    "d2_eval",
    "d3",
    "d4",
    "d5_direct",
    "d5_bilstm",
)
E_EXPERIMENTS = (
    "e0",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "e6",
    "e7",
    "e8",
    "e9",
    "e9_aug",
)
F_EXPERIMENTS = ("f1",)
EXPERIMENTS = (
    A_EXPERIMENTS
    + B_EXPERIMENTS
    + C_EXPERIMENTS
    + D_EVALUATION_EXPERIMENTS
    + E_EXPERIMENTS
    + F_EXPERIMENTS
)


def get_experiment(name: str, config: ExperimentConfig) -> ExperimentSpec:
    try:
        return build_experiments(config)[name.lower()]
    except KeyError as error:
        raise ValueError(f"未知实验 {name!r}；可用实验：{', '.join(EXPERIMENTS)}") from error
