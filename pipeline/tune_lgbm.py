"""LightGBM 超参数调参（Optuna）。

用法：
    pip install optuna
    python pipeline/tune_lgbm.py --products RB --n-trials 50
    python pipeline/tune_lgbm.py --products RB CU AG --n-trials 80 --metric val_sharpe

说明：
    - 在指定品种集合上跑 Optuna，objective = 各品种验证集指标的均值（全局规则，不做
      per-product 调参）。默认 metric=val_ic（Spearman rank IC），速度最快；
      也可选 val_sharpe（多走一遍轻量 backtest，慢但更贴近实盘）。
    - 每个品种的 prepare_data 只在 study 开始前调一次，特征矩阵缓存在内存中；
      objective 只重训 LightGBM，避免反复读盘 / 重算因子。
    - 调用 train_single_regime_model（不是 train_dual_regime_models），所以不会
      生成绘图、不会持久化模型，trial 期间不污染 results/models 与 results/cache。
    - 只调 model.common_params 里的核心超参；regime override 仍读 config.yaml。
    - 结果输出到 results/optuna/<run_id>/{study.db, best_params.yaml, trials.csv}。
      best_params.yaml 是可直接复制粘贴到 config.yaml 的 YAML 块。

注意：scripts/ 必须 self-contained（CLAUDE.md），所以本文件放在 pipeline/。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
CURRENT_DIR_STR = str(CURRENT_DIR)
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if CURRENT_DIR_STR not in sys.path:
    sys.path.insert(0, CURRENT_DIR_STR)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from config_utils import load_project_config
from dataset import PreparedData, REGIME_NAME_MAP, prepare_data
from modeling import (
    build_model_settings,
    calc_prediction_metrics,
    predict_single_regime,
    train_single_regime_model,
)
from train_products import build_product_config_override


@dataclass
class ProductBundle:
    product_id: str
    prepared: PreparedData
    config_override: dict[str, Any]


def _load_product_registry(config_path: str | None) -> list[dict[str, Any]]:
    config, config_dir = load_project_config(config_path)
    registry_path = config.get("paths", {}).get("product_registry", "data/product_registry.json")
    full_path = (Path(config_dir) / registry_path).resolve() if not Path(registry_path).is_absolute() else Path(registry_path)
    with full_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_products(
    registry: list[dict[str, Any]],
    product_ids: list[str],
) -> list[dict[str, Any]]:
    wanted = {pid.upper() for pid in product_ids}
    selected = [m for m in registry if str(m.get("product_id", "")).upper() in wanted]
    found = {str(m.get("product_id", "")).upper() for m in selected}
    missing = wanted - found
    if missing:
        raise ValueError(f"未在 product_registry 中找到品种：{sorted(missing)}")
    return selected


def _build_product_bundles(
    product_metas: list[dict[str, Any]],
    config_path: str | None,
    work_dir: Path,
    force_rebuild: bool,
) -> list[ProductBundle]:
    bundles: list[ProductBundle] = []
    for product_meta in product_metas:
        product_id = str(product_meta["product_id"]).upper()
        # 用 work_dir 下的子目录作为 paths 占位（不会真的写文件，因为 persist_models=False
        # 且我们绕开了 train_dual_regime_models 的绘图调用）
        product_dir = work_dir / product_id
        config_override = build_product_config_override(
            product_meta, product_dir, persist_to_shared_dir=False
        )
        # 强制不持久化（调参阶段不应污染 results/models/）
        config_override.setdefault("model", {})["persist_models"] = False
        print(f"[prepare] {product_id}: 准备特征数据 ...", flush=True)
        prepared = prepare_data(
            config_path=config_path,
            force_rebuild=force_rebuild,
            config_override=config_override,
        )
        train_n = len(prepared.train_data)
        val_n = len(prepared.val_data)
        test_n = len(prepared.test_data)
        feat_n = len(prepared.feature_cols)
        print(
            f"[prepare] {product_id}: 完成 features={feat_n} "
            f"train={train_n} val={val_n} test={test_n}",
            flush=True,
        )
        bundles.append(
            ProductBundle(
                product_id=product_id,
                prepared=prepared,
                config_override=config_override,
            )
        )
    return bundles


def _suggest_params(trial, base_params: dict[str, Any]) -> dict[str, Any]:
    """在 base_params 之上覆盖待调超参；其它字段（objective, seed, n_jobs 等）保留。"""
    params = dict(base_params)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
    params["num_leaves"] = trial.suggest_int("num_leaves", 15, 255, log=True)
    params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    params["min_child_samples"] = trial.suggest_int("min_child_samples", 30, 400, log=True)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 1.0)
    params["bagging_freq"] = trial.suggest_int("bagging_freq", 0, 10)
    params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True)
    params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True)
    params["min_gain_to_split"] = trial.suggest_float("min_gain_to_split", 1e-8, 1.0, log=True)
    return params


def _evaluate_combined_val(
    bundle: ProductBundle,
    common_params: dict[str, Any],
    config_path: str | None,
    metric: str,
) -> dict[str, float]:
    """单品种：训两个 regime，输出 combined validation 指标。

    返回 dict 含：spearman_ic, pearson_ic, sharpe(可选), trade_count(可选)。
    """
    settings = build_model_settings(config_path, config_override=bundle.config_override)
    target_col = bundle.prepared.target_col
    feature_cols = bundle.prepared.feature_cols

    val_outputs: list[pd.DataFrame] = []
    for regime_label in sorted(REGIME_NAME_MAP):
        regime_name = REGIME_NAME_MAP[regime_label]
        train_df = bundle.prepared.train_data.loc[
            bundle.prepared.train_data["REGIME_LABEL"] == regime_label
        ].copy()
        val_df = bundle.prepared.val_data.loc[
            bundle.prepared.val_data["REGIME_LABEL"] == regime_label
        ].copy()
        test_df = bundle.prepared.test_data.loc[
            bundle.prepared.test_data["REGIME_LABEL"] == regime_label
        ].copy()
        if train_df.empty or val_df.empty:
            return {"spearman_ic": float("nan"), "pearson_ic": float("nan"), "sharpe": float("nan")}

        # 应用 regime override（low_vol_overrides / high_vol_overrides 仍读 config.yaml）
        params = dict(common_params)
        if regime_name == "low_vol":
            params.update(settings["low_vol_overrides"])
        else:
            params.update(settings["high_vol_overrides"])

        artifact = train_single_regime_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            scale_method=settings["scale_method"],
            params=params,
            num_boost_round=settings["num_boost_round"],
            early_stopping_rounds=settings["early_stopping_rounds"],
            regime_label=regime_label,
        )
        val_pred = predict_single_regime(
            df=val_df,
            feature_cols=feature_cols,
            target_col=target_col,
            booster=artifact.booster,
            scaler=artifact.scaler,
        )
        val_outputs.append(val_pred)

    combined = pd.concat(val_outputs, axis=0).sort_values("TDATE").reset_index(drop=True)
    metrics = calc_prediction_metrics(combined["future_return"], combined["pred_return"])
    result = {
        "spearman_ic": float(metrics.get("spearman_ic", 0.0)),
        "pearson_ic": float(metrics.get("pearson_ic", 0.0)),
    }
    if metric == "val_sharpe":
        result.update(_quick_val_sharpe(combined, bundle.config_override, config_path))
    return result


def _quick_val_sharpe(
    combined_val: pd.DataFrame,
    config_override: dict[str, Any],
    config_path: str | None,
) -> dict[str, float]:
    """轻量版验证集 backtest：复用 pipeline.backtest 的 build_signal_rule_map / generate_positions / calc_pnl。"""
    from backtest import (
        build_backtest_settings,
        build_signal_rule_map,
        calc_pnl,
        generate_positions,
        performance_summary,
    )

    settings = build_backtest_settings(config_path, config_override=config_override)
    rule_map = build_signal_rule_map(combined_val, settings)
    positions = generate_positions(combined_val, rule_map, settings)
    pnl, daily = calc_pnl(
        position_df=positions,
        commission_rate=settings["commission_rate"],
        slippage_rate=settings["slippage_rate"],
        hold_to_next_bar=settings["hold_to_next_bar"],
    )
    perf = performance_summary(daily, pnl, settings["annualization_days"])
    net = perf.get("net", {}) or {}
    return {
        "sharpe": float(net.get("sharpe", 0.0) or 0.0),
        "annual_return": float(net.get("annual_return", 0.0) or 0.0),
        "trade_count": float(perf.get("trade_count", 0) or 0),
    }


def _objective_value(per_product: list[dict[str, float]], metric: str) -> float:
    """根据 metric 把 per-product 指标聚合成单个 scalar（取算术平均）。"""
    if metric == "val_ic":
        values = [m["spearman_ic"] for m in per_product if not np.isnan(m["spearman_ic"])]
    elif metric == "val_sharpe":
        values = [m.get("sharpe", float("nan")) for m in per_product if not np.isnan(m.get("sharpe", float("nan")))]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    if not values:
        return float("nan")
    return float(np.mean(values))


def run_study(
    products: list[str],
    n_trials: int,
    metric: str,
    timeout: float | None,
    config_path: str | None,
    output_dir: Path,
    force_rebuild: bool,
    seed: int,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna 未安装。请先运行：pip install optuna"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    registry = _load_product_registry(config_path)
    product_metas = _select_products(registry, products)

    bundles = _build_product_bundles(
        product_metas=product_metas,
        config_path=config_path,
        work_dir=work_dir,
        force_rebuild=force_rebuild,
    )

    # 用第一个 bundle 拿基础 common_params（同一份 config.yaml，所有 bundle 一致）
    base_settings = build_model_settings(config_path, config_override=bundles[0].config_override)
    base_common_params = dict(base_settings["common_params"])

    storage_path = output_dir / "study.db"
    study_name = f"cta_lgbm_{metric}"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        common_params = _suggest_params(trial, base_common_params)
        per_product: list[dict[str, float]] = []
        for idx, bundle in enumerate(bundles):
            try:
                product_metrics = _evaluate_combined_val(
                    bundle=bundle,
                    common_params=common_params,
                    config_path=config_path,
                    metric=metric,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [trial {trial.number}] {bundle.product_id} 训练失败：{type(exc).__name__}: {exc}")
                return float("nan")
            per_product.append(product_metrics)
            partial = _objective_value(per_product, metric)
            trial.report(partial, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = _objective_value(per_product, metric)
        # 把 per-product 详情存到 trial.user_attrs，便于事后导出
        for bundle, m in zip(bundles, per_product):
            for k, v in m.items():
                trial.set_user_attr(f"{bundle.product_id}__{k}", float(v))
        return score

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    return _export_results(study=study, output_dir=output_dir, base_common_params=base_common_params, metric=metric, products=products)


def _export_results(
    study,
    output_dir: Path,
    base_common_params: dict[str, Any],
    metric: str,
    products: list[str],
) -> dict[str, Any]:
    import optuna  # noqa: F401  (确认 optuna 还在)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs", "duration"))
    trials_csv = output_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    if study.best_trial is None or study.best_trial.value is None or np.isnan(study.best_trial.value):
        print("WARNING: 没有有效 trial（全部失败或被剪枝）。")
        return {"best_value": None, "best_params": None}

    best = study.best_trial
    merged = dict(base_common_params)
    merged.update(best.params)

    # 输出 yaml 块（直接可粘贴到 config.yaml 的 model.common_params）
    yaml_block_lines = [
        f"# best params from optuna study (metric={metric}, products={','.join(products)})",
        f"# best value = {best.value:.6f} | trial #{best.number}",
        "common_params:",
    ]
    for key, value in merged.items():
        if isinstance(value, float):
            yaml_block_lines.append(f"  {key}: {value:.8g}")
        else:
            yaml_block_lines.append(f"  {key}: {value}")
    yaml_text = "\n".join(yaml_block_lines) + "\n"
    (output_dir / "best_params.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "metric": metric,
        "products": products,
        "best_value": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": best.params,
        "merged_common_params": merged,
        "n_trials_total": len(study.trials),
        "n_trials_complete": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
        "n_trials_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        "user_attrs_best": dict(best.user_attrs),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n========== Optuna 调参完成 ==========")
    print(f"metric:        {metric}")
    print(f"products:      {products}")
    print(f"best value:    {best.value:.6f}  (trial #{best.number})")
    print(f"trials total:  {len(study.trials)}  complete={summary['n_trials_complete']}  pruned={summary['n_trials_pruned']}")
    print(f"输出目录:      {output_dir}")
    print(f"  - best_params.yaml  （可直接粘贴到 config.yaml 的 model.common_params）")
    print(f"  - trials.csv         （每个 trial 的 params/value/per-product 指标）")
    print(f"  - summary.json       （结构化摘要）")
    print(f"  - study.db           （Optuna SQLite，可继续 --resume 续跑）")
    print("\n--- best_params.yaml ---")
    print(yaml_text)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LightGBM hyperparameter tuning with Optuna for CTA_vol dual-regime models.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["RB"],
        help="要参与调参的品种 ID 列表（取均值作为 objective），默认 RB。",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trial 数量，默认 50。")
    parser.add_argument(
        "--metric",
        choices=["val_ic", "val_sharpe"],
        default="val_ic",
        help="优化目标：val_ic（Spearman rank IC，快）/ val_sharpe（验证集回测 sharpe，慢）。",
    )
    parser.add_argument("--timeout", type=float, default=None, help="study 总时长上限（秒）。")
    parser.add_argument("--config", default=None, help="config.yaml 路径。")
    parser.add_argument("--force-rebuild", action="store_true", help="重建特征 cache（一次性，不影响 trial）。")
    parser.add_argument("--seed", type=int, default=42, help="TPESampler seed。")
    parser.add_argument(
        "--run-id",
        default=None,
        help="自定义 run id；默认 timestamp。结果落盘到 results/optuna/<run_id>/。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (PROJECT_ROOT / "results" / "optuna" / run_id).resolve()
    print(f"[tune] run_id={run_id}  metric={args.metric}  products={args.products}  n_trials={args.n_trials}")
    print(f"[tune] output_dir={output_dir}")
    run_study(
        products=args.products,
        n_trials=args.n_trials,
        metric=args.metric,
        timeout=args.timeout,
        config_path=args.config,
        output_dir=output_dir,
        force_rebuild=args.force_rebuild,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
