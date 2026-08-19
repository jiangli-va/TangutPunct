from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from config import load_config
from data.splits import VolumeCrossValidator
from experiments import (
    A_EXPERIMENTS,
    B_EXPERIMENTS,
    C_EXPERIMENTS,
    E_EXPERIMENTS,
    F_EXPERIMENTS,
    CExperimentRunner,
    EXPERIMENTS,
    MainExperimentRunner,
    get_experiment,
)
from experiments.runner import read_main_documents
from reporting_main import (
    render_main_inspection,
    render_main_report,
    render_main_summary,
)
from pretraining import D1ContextMLMRunner, D2WordPretrainingRunner, D5ExperimentRunner
from pretraining.d1_runner import render_d1_summary
from pretraining.d2_runner import render_d2_summary


D_EXPERIMENTS = ("d1", "d2_control", "d2_lex", "d2")
D2_MODES = {"d2_control": "control", "d2_lex": "lexicon", "d2": "fusion"}


def _selected_experiments(value: str) -> tuple[str, ...]:
    # 保留既有 `all`=A1/A2/A3 的含义，避免旧命令突然额外运行B组。
    if value == "all":
        return A_EXPERIMENTS
    if value == "b":
        return B_EXPERIMENTS
    if value == "c":
        return C_EXPERIMENTS
    if value == "d":
        return D_EXPERIMENTS
    if value == "e":
        return E_EXPERIMENTS
    if value == "f":
        return F_EXPERIMENTS
    return (value,)


def _experiment_spec(name: str, config, model_name: str | None):
    spec = get_experiment(name, config)
    if model_name is None:
        return spec
    if name not in A_EXPERIMENTS:
        raise ValueError("--model只用于覆盖A1/A2/A3的基础模型；B/C组后端由实验定义")
    display_names = {
        "crf": "CRF",
        "bilstm": "BiLSTM",
        "random_transformer": "Random-Transformer",
    }
    display_name = f"{spec.display_name}（{display_names[model_name]}复核）"
    return replace(spec, model_name=model_name, display_name=display_name)


def _evaluation_checkpoint_config(
    config,
    name: str,
    checkpoint: Path | None,
    init_from: str | None = None,
):
    """解析D阶段下游初始化；显式checkpoint始终具有最高优先级。"""

    if init_from is not None and name not in {"d3", "d4", "d5_direct", "d5_bilstm"}:
        raise ValueError("--init-from只用于D3/D4/D5；D1/D2评测请使用--checkpoint")
    selected = checkpoint.resolve() if checkpoint is not None else None
    source: str | None = None
    if name == "d1_eval":
        source = "d1"
    elif name == "d2_eval":
        source = "d2"
        if selected is None:
            selected = config.word_pretraining.checkpoint
    elif name in {"d3", "d4", "d5_direct", "d5_bilstm"}:
        source = init_from or config.d_downstream.initialization_source
        if selected is None:
            selected = (
                config.pretraining.checkpoint
                if source == "d1"
                else config.word_pretraining.checkpoint
            )
        elif init_from is None:
            source = "checkpoint"
    if selected is None:
        return config, source
    return (
        replace(
            config,
            pretraining=replace(config.pretraining, checkpoint=selected),
        ),
        source,
    )


def _with_initialization_name(spec, source: str | None):
    if spec.name not in {"d3", "d4", "d5_direct", "d5_bilstm"} or source is None:
        return spec
    label = {"d1": "D1", "d2": "D2", "checkpoint": "自定义checkpoint"}[source]
    return replace(spec, display_name=f"{spec.display_name}（从{label}初始化）")


def inspect(
    config_path: Path,
    experiment: str,
    model_name: str | None,
    grade: int = 1,
    checkpoint: Path | None = None,
    init_from: str | None = None,
) -> None:
    config = load_config(config_path)
    selected = _selected_experiments(experiment)
    if init_from is not None and any(name not in {"d3", "d4", "d5_direct", "d5_bilstm"} for name in selected):
        raise ValueError("--init-from只用于D3/D4/D5")
    if any(name in D_EXPERIMENTS for name in selected):
        if model_name is not None:
            raise ValueError("主实验D的TangutEncoder结构由pretraining配置控制，不使用--model")
        for name in selected:
            runner = (
                D1ContextMLMRunner(config)
                if name == "d1"
                else D2WordPretrainingRunner(config, D2_MODES[name])
            )
            print(runner.inspect())
        return
    initialization_source = None
    # d2_eval默认读取word_pretraining.checkpoint；--checkpoint始终具有最高优先级。
    if len(selected) == 1:
        config, initialization_source = _evaluation_checkpoint_config(
            config, selected[0], checkpoint, init_from
        )
    elif checkpoint is not None:
        config, initialization_source = _evaluation_checkpoint_config(
            config, selected[0], checkpoint, init_from
        )
    documents = read_main_documents(config)
    folds = VolumeCrossValidator(config.folds, config.dev_ratio, config.seed).split(
        documents
    )
    specs = [
        _with_initialization_name(
            _experiment_spec(name, config, model_name), initialization_source
        )
        for name in _selected_experiments(experiment)
    ]
    if grade != 2 and any(spec.stages[-1].target == "pause_group" for spec in specs):
        raise ValueError(
            "F1直接训练句内/句间两类，只支持--grade 2；"
            "请使用 python run.py inspect --experiment f1 "
            "--config configs/experiment_f.yaml --grade 2"
        )
    if len(specs) == 1 and specs[0].strategy in {"d5_direct", "d5_bilstm"}:
        print(
            D5ExperimentRunner(
                config,
                "direct" if specs[0].strategy == "d5_direct" else "bilstm",
                grade,
                initialization_source or config.d_downstream.initialization_source,
            ).inspect()
        )
        return
    print(render_main_inspection(config, specs, documents, folds, grade=grade))


def train(
    config_path: Path,
    experiment: str,
    output: Path,
    model_name: str | None,
    grade: int = 1,
    checkpoint: Path | None = None,
    init_from: str | None = None,
) -> None:
    config = load_config(config_path)
    selected = _selected_experiments(experiment)
    if init_from is not None and any(name not in {"d3", "d4", "d5_direct", "d5_bilstm"} for name in selected):
        raise ValueError("--init-from只用于D3/D4/D5")
    for name in selected:
        if name in D_EXPERIMENTS:
            if model_name is not None:
                raise ValueError("主实验D的TangutEncoder结构由pretraining配置控制，不使用--model")
            if name == "d1":
                destination = output / "D1" / "tangut_encoder_mlm"
                result = D1ContextMLMRunner(config).run(destination)
                summary = render_d1_summary(result)
            else:
                destination = output / name.upper() / "tangut_encoder_word"
                result = D2WordPretrainingRunner(config, D2_MODES[name]).run(destination)
                summary = render_d2_summary(result)
            print("\n" + summary + "\n")
            continue
        experiment_config, initialization_source = _evaluation_checkpoint_config(
            config, name, checkpoint, init_from
        )
        spec = _with_initialization_name(
            _experiment_spec(name, experiment_config, model_name),
            initialization_source,
        )
        backend = spec.model_name or experiment_config.model.name
        if spec.strategy in {"d5_direct", "d5_bilstm"}:
            destination = (
                output
                / "D5"
                / f"from_{initialization_source or 'unknown'}"
                / ("direct" if spec.strategy == "d5_direct" else "bilstm")
            )
            if grade == 2:
                destination = destination / "grade_2"
            result = D5ExperimentRunner(
                experiment_config,
                "direct" if spec.strategy == "d5_direct" else "bilstm",
                grade,
                initialization_source or experiment_config.d_downstream.initialization_source,
            ).run(destination)
            report = render_main_report(result)
            (destination / "results.md").write_text(report + "\n", encoding="utf-8")
            print("\n" + render_main_summary(result) + "\n")
            continue
        destination = output / name.upper() / backend
        if name in {"d3", "d4"}:
            destination = (
                output / name.upper() / f"from_{initialization_source}" / backend
            )
        # grade=1 保持旧目录兼容；grade=2 单独存放，避免覆盖七类结果。
        if grade == 2:
            destination = destination / "grade_2"
        runner = (
            CExperimentRunner(experiment_config, spec, grade=grade)
            if spec.strategy in {"candidate_reject", "soft_cascade", "multitask"}
            else MainExperimentRunner(experiment_config, spec, grade=grade)
        )
        result = runner.run(destination)
        report = render_main_report(result)
        (destination / "results.md").write_text(report + "\n", encoding="utf-8")
        # 命令行只显示紧凑的 Markdown 总表；阶段细表和完整指标分别在
        # results.md / results.json 中，不在终端倾倒 JSON。
        print("\n" + render_main_summary(result) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="西夏文自动标点与TangutEncoder预训练实验（主实验A、B、C、D、E、F）"
    )
    parser.add_argument("command", choices=("inspect", "train"))
    parser.add_argument(
        "--experiment",
        choices=(*EXPERIMENTS, *D_EXPERIMENTS, "all", "b", "c", "d", "e", "f"),
        default="all",
        help="运行单项实验；all运行A1-A3（默认），b/c/d/e/f分别运行对应实验组",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment_a.yaml")
    )
    parser.add_argument(
        "--model",
        choices=("crf", "bilstm", "random_transformer"),
        default=None,
        help="覆盖A1/A2/A3的模型后端；例如 --model bilstm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_pause"),
        help="结果目录（默认 outputs_pause，避免覆盖旧版完整标点预实验）",
    )
    parser.add_argument(
        "--grade",
        type=int,
        choices=(1, 2),
        default=1,
        help="评价体系：1=七种具体停顿标点（默认），2=句内/句间两类；均另报含O的Accuracy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="D阶段下游评测所用best_model.pt；显式指定时覆盖D1/D2来源对应的配置路径",
    )
    parser.add_argument(
        "--init-from",
        choices=("d1", "d2"),
        default=None,
        help="D3/D4/D5的TangutEncoder初始化来源；默认读取配置中的d_downstream.initialization_source",
    )
    parser.add_argument(
        "--debug", action="store_true", help="显示测试文献 ID、特征构建等调试信息"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.command == "inspect":
        inspect(
            args.config,
            args.experiment,
            args.model,
            args.grade,
            args.checkpoint,
            args.init_from,
        )
    else:
        train(
            args.config,
            args.experiment,
            args.output,
            args.model,
            args.grade,
            args.checkpoint,
            args.init_from,
        )


if __name__ == "__main__":
    main()
