# TangutPunct：西夏文自动标点研究

西夏文（Tangut Scripts）是西夏王朝（1038–1227）创制的表意文字，是研究西夏历史、语言与文化的一手材料。西夏语属汉藏语系的羌语支，西夏人的语言现已灭绝，学术界认为其与现代的羌语和嘉绒语关系最密切。西夏文字形与汉字相仿，目前总计约6000余字，行体方整，笔划繁复。独体字较少，由2个字甚至3、4个字合成一字者居多数。

流传至今的西夏文献通常没有断句和标点，给西夏文古籍的整理、断句、翻译和数字化带来障碍。本项目使用标注语料、无标注语料、词典以及此前的分词和词性模型（见`https://github.com/jiangli-va/TangutSeg`），在极低资源条件下构建自动标点系统。

由于西夏文数据稀缺且宝贵，标注语料成本高昂，本项目目前没有公开全部训练数据的计划。但是，我们将开源所有训练代码，未来将开源自动标点模型，供学界使用。


## 语料与任务

- **语料**：67 部西夏文文献（经书 57 部、世俗文献 10 部），取自《大宝积经》、六本书、四行对译等材料，共约 **51.5 万字符**（其中西夏字约 44 万，标点约 7.4 万）。
- **任务**：对每个字符预测其后是否应出现停顿标点及其类别；缺字占位符（`□`、`@`、`…`）映射为特殊 token，残缺长片段删除并以 `<TAB>` 作为不可跨越的边界。
- **评价**：文献级五折交叉验证（约 7:1:2 训练/开发/测试划分），报告位置 P/R/F1、类别 Micro/Macro-F1、句界 F1 与含 `O` 的 Accuracy，经书与世俗文献分开统计。

## 方法路线

| 阶段 | 内容 |
| --- | --- |
| A | CRF 基线（直接预测 / 位置→类别两阶段） |
| B | 规则、n-gram、BiLSTM、Transformer 等基线对比 |
| C | 级联与多任务建模（候选生成、软级联、共享编码器多任务） |
| D | **TangutEncoder** 自监督预训练（MLM、词感知预训练），再以 BiLSTM 微调 |
| E | 融入外部知识：领域分布、词典软格网、t-score 上下文统计、分词与词性特征（复用姊妹项目 `xixia_seg` 的模型） |
| F | 直接训练粗粒度句内/句末两类停顿 |

**当前最优方案（E9）**：D2 预训练的 TangutEncoder + 词典格网/t-score 上下文特征 + 共享 BiLSTM + 位置辅助头与完整标点头多任务联合训练，配合同文献完整句子拼接数据增强（E9-Aug）。

## 实验结果（五折交叉验证，总体测试集）

下表列出各方法路线的代表性结果（五折均值；grade=1 为七种具体停顿标点，grade=2 为句内/句末两类合并评测，grade=2 数值来自同一模型按两类投影评价）：

| 模型 / 方法 | 位置 F1 | 类别 Micro-F1（grade=1） | 类别 Micro-F1（grade=2） |
| --- | ---: | ---: | ---: |
| n-gram（B2 基线） | 0.673 | 0.433 | — |
| CRF 两阶段（A2，基线） | 0.719 | 0.504 | — |
| Random-BiLSTM（B4） | 0.728 | 0.515 | — |
| Random-Transformer（B5） | 0.534 | 0.336 | — |
| 多任务 BiLSTM（C4，$\lambda$=0.2） | 0.749 | 0.552 | — |
| TangutEncoder 预训练 + BiLSTM（D2→D4） | 0.781 | 0.574 | 0.636 |
| + 词典格网 / t-score 上下文（E3） | 0.790 | 0.584 | 0.645 |
| + 词性知识（E7） | 0.791 | 0.584 | 0.644 |
| **多任务联合（E9，最优）** | **0.789** | **0.593** | **0.645** |

- 位置 F1 反映"停顿点找得准不准"，类别 Micro-F1 反映"七类标点分得对不对"（grade=2 仅区分句内/句末）。
- E9 相比 CRF 基线，位置 F1 提升约 7 个百分点，类别 Micro-F1（grade=1）提升约 9 个百分点；位置辅助任务 + 共享编码器消除了硬级联中"上游漏标则下游无法纠正"的问题。
- 经书文献表现显著优于世俗文献（如 E9 经书位置 F1 ≈ 0.80，世俗 ≈ 0.65），与世俗文献数据量少、题材多样、风格差异大有关。

## 文件结构

```text
xixia_senseg/
├── run.py                 # 命令行入口（inspect / train）
├── config.py              # 数据、标点、特征、模型配置的数据类
├── experiment.py          # 五折交叉验证实验框架
├── tasks.py               # 任务定义（边界检测 / 标点分类）
├── reporting.py           # 报告渲染
├── reporting_main.py      # 主实验报告（结果表、逐类指标）
├── requirements.txt
├── configs/               # YAML 实验配置（experiment_{a,b,c,d,e,f}.yaml）
├── corpus/                # 训练语料（暂不公开）
├── data/                  # 语料读取、标签、五折划分、数据增强
├── models/                # CRF、n-gram、BiLSTM、Transformer、TangutEncoder 等模型
├── pretraining/           # D1 上下文 MLM、D2 词感知预训练、D5 预训练标点
├── features/              # 基础特征构建
├── knowledge/             # 外部知识特征（领域/词典/上下文/分词/词性）
├── evaluation/            # 位置、类别、句界等指标计算
├── experiments/           # 实验规格（A–F 各实验定义）与各阶段 runner
└── README.md
```

## 运行方式

### 安装依赖

```bash
pip install -r requirements.txt
# sklearn-crfsuite、joblib、PyYAML、torch>=2.0
```

### 查看实验（不做训练）

先检查实验配置与五折划分是否正确：

```bash
python run.py inspect --experiment e9 \
  --config configs/experiment_e.yaml
```

### 训练与评测

```bash
# 运行 E9 实验，结果写入 outputs_e/（grade=1：七类停顿标点）
python run.py train --experiment e9 \
  --config configs/experiment_e.yaml \
  --output outputs_e --grade 1 --debug

# grade=2：同一模型按句内/句末两类投影评测
python run.py train --experiment e9 \
  --config configs/experiment_e.yaml \
  --output outputs_e --grade 2 --debug
```

结果（Markdown 报告、逐折指标、混淆矩阵）写入 `<output>/<EXPERIMENT>/<model>/`，其中 `results.md` 为报告，`results.json` 为完整指标。命令行参数：

| 参数 | 说明 |
| --- | --- |
| `command` | `inspect`（只检查）或 `train`（训练并评测） |
| `--experiment` | 实验名（`a1`–`a3`, `b0`–`b6`, `c1`–`c4`, `d1`/`d2`, `d3`–`d5`, `e0`–`e9`, `e9_aug`, `f1`）或分组 `all`/`b`/`c`/`d`/`e`/`f` |
| `--config` | YAML 配置文件（默认 `configs/experiment_a.yaml`） |
| `--model` | 覆盖 A 组后端：`crf`/`bilstm`/`random_transformer` |
| `--output` | 结果输出目录（默认 `outputs_pause`） |
| `--grade` | `1` = 七种具体停顿标点（默认）；`2` = 句内/句末两类 |
| `--checkpoint` | D 组下游评测所用 `best_model.pt` |
| `--init-from` | D3/D4/D5 的初始化来源：`d1`/`d2` |
| `--debug` | 显示调试信息 |
