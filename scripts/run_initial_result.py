"""
Train a truly undivided LightGBM baseline on RB micro features.

Unlike the dual-regime micro model:
- Training target is raw future_return (NO vol normalization)
- Single LightGBM on all train rows (NO regime split)
- No prediction de-normalization

This produces the "naive" baseline that reveals the fundamental difficulty:
training on a target with mean≈0 and high heteroskedastic variance makes
the model unable to confidently predict high-return time points. The cost
filter (0.048% per trade) then filters out virtually all weak signals,
resulting in flat or negative performance — the problem that vol-regime
splitting is designed to address.

Outputs to results/runs/initial_result/RB/, mirroring micro_result/RB structure.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import RobustScaler

from pipeline.config_utils import get_section, load_project_config
from pipeline.dataset import prepare_data, REGIME_NAME_MAP
from pipeline.backtest import (
    build_backtest_settings,
    calc_pnl,
    generate_positions,
    performance_summary,
    build_benchmark_positions,
)
from pipeline.modeling import calc_prediction_metrics


OUT_DIR = PROJECT_ROOT / "results" / "runs" / "initial_result" / "RB"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def _safe_calmar(annual_return: float, max_drawdown: float) -> float:
    if abs(max_drawdown) < 1e-9:
        return float("inf") if annual_return > 0 else 0.0
    return annual_return / abs(max_drawdown)


def main() -> None:
    config_path = str(PROJECT_ROOT / "config.yaml")

    config_override = {
        "data": {"use_mid_weekly": False},
        "product": {
            "product_id": "RB",
            "instrument_code": "RBZL.SHF",
            "raw_data_file": "RBZL.SHF.csv",
            "mid_weekly_files": [],
        },
        "paths": {
            "run_root": str(PROJECT_ROOT / "results" / "runs"),
        },
    }

    print("Preparing data ...")
    prepared = prepare_data(config_path=config_path, config_override=config_override)
    feature_cols = prepared.feature_cols

    # KEY: use raw future_return as target — no vol normalization, no de-normalization.
    # This is the "naive" approach: train directly on a distribution with mean≈0 and
    # high heteroskedastic variance, so the model cannot learn regime-specific patterns.
    target_col = "future_return"

    train_df = prepared.train_data.copy()
    val_df = prepared.val_data.copy()
    test_df = prepared.test_data.copy()

    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, features={len(feature_cols)}")
    print(f"  Target '{target_col}': mean={train_df[target_col].mean():.2e}, "
          f"std={train_df[target_col].std():.4%}, "
          f"p1/p99={train_df[target_col].quantile(0.01):.4%}/{train_df[target_col].quantile(0.99):.4%}")

    config, _ = load_project_config(config_path, config_override=config_override)
    model_cfg = get_section(config, "model")
    params = dict(model_cfg.get("common_params", {}))
    num_boost_round = int(model_cfg.get("num_boost_round", 400))
    early_stopping_rounds = int(model_cfg.get("early_stopping_rounds", 50))

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype("float32"))
    X_val = scaler.transform(val_df[feature_cols].values.astype("float32"))
    X_test = scaler.transform(test_df[feature_cols].values.astype("float32"))

    y_train = train_df[target_col].values.astype("float32")
    y_val = val_df[target_col].values.astype("float32")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

    evals_result: dict = {}
    print("Training single undivided LightGBM on raw future_return ...")
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=50),
        ],
    )
    best_iter = int(booster.best_iteration or num_boost_round)
    print(f"  Best iteration: {best_iter}")

    # Predict — raw predictions are already in actual return units (no de-normalization needed)
    def predict_and_attach(df_in: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        out = df_in.copy()
        raw_pred = np.asarray(booster.predict(X), dtype="float64")
        out["pred_return"] = raw_pred
        out["pred_direction"] = np.sign(raw_pred).astype(np.int8)
        out["true_direction"] = np.sign(out["future_return"]).astype(np.int8)
        return out

    val_pred = predict_and_attach(val_df, X_val)
    test_pred = predict_and_attach(test_df, X_test)

    val_metrics = calc_prediction_metrics(val_pred["future_return"].values, val_pred["pred_return"].values)
    test_metrics = calc_prediction_metrics(test_pred["future_return"].values, test_pred["pred_return"].values)
    print(f"  Val  IC: Pearson={val_metrics['pearson_ic']:.4f}, Spearman={val_metrics['spearman_ic']:.4f}")
    print(f"  Test IC: Pearson={test_metrics['pearson_ic']:.4f}, Spearman={test_metrics['spearman_ic']:.4f}")
    print(f"  Test pred: mean={test_pred['pred_return'].mean():.2e}, std={test_pred['pred_return'].std():.2e}")

    # Per-regime IC breakdown
    regime_metrics: dict[str, dict] = {}
    for regime_label, regime_name in REGIME_NAME_MAP.items():
        grp = test_pred[test_pred["REGIME_LABEL"] == regime_label]
        if not grp.empty:
            regime_metrics[regime_name] = calc_prediction_metrics(
                grp["future_return"].values, grp["pred_return"].values
            )
            print(f"  Test IC [{regime_name}]: "
                  f"Pearson={regime_metrics[regime_name]['pearson_ic']:.4f}, "
                  f"Spearman={regime_metrics[regime_name]['spearman_ic']:.4f}")

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    # --- Signal rules (same for both regimes, thresholds from val predictions) ---
    abs_val_pred = val_pred["pred_return"].abs()
    nonzero_pred = abs_val_pred[abs_val_pred > 0]
    entry_q = float(np.quantile(nonzero_pred, 0.88)) if len(nonzero_pred) else 0.0
    exit_q = float(np.quantile(nonzero_pred, 0.45)) if len(nonzero_pred) else 0.0
    cost_filter = float(0.0002 * 2.0 * 1.2)  # (commission+slippage) * round_trip * multiple
    effective_entry = max(entry_q, cost_filter)

    print(f"  Entry 88th pct of |pred|: {entry_q:.6f}  Cost filter: {cost_filter:.6f}  Effective: {effective_entry:.6f}")
    pct_above_entry = float((abs_val_pred >= effective_entry).mean())
    print(f"  Val bars with |pred| >= effective_entry: {pct_above_entry:.2%}")

    rule_map: dict[int, dict] = {}
    for regime_label in sorted(REGIME_NAME_MAP.keys()):
        rule_map[regime_label] = {
            "regime_name": "unsegmented",
            "entry_threshold": entry_q,
            "exit_threshold": exit_q,
            "cost_filter_threshold": cost_filter,
            "effective_entry_threshold": effective_entry,
            "confirmation_bars": 2,
            "min_hold_bars": 4,
            "cooldown_bars": 2,
            "allow_direct_flip": False,
            "flip_to_flat_first": True,
        }

    backtest_settings = build_backtest_settings(config_path=config_path, config_override=config_override)

    print("Running backtest ...")
    val_pos = generate_positions(val_pred, rule_map, backtest_settings)
    val_pnl, val_daily = calc_pnl(
        val_pos,
        commission_rate=backtest_settings["commission_rate"],
        slippage_rate=backtest_settings["slippage_rate"],
        hold_to_next_bar=backtest_settings["hold_to_next_bar"],
    )
    val_summary = performance_summary(val_daily, val_pnl, backtest_settings["annualization_days"])

    test_pos = generate_positions(test_pred, rule_map, backtest_settings)
    test_pnl, test_daily = calc_pnl(
        test_pos,
        commission_rate=backtest_settings["commission_rate"],
        slippage_rate=backtest_settings["slippage_rate"],
        hold_to_next_bar=backtest_settings["hold_to_next_bar"],
    )
    test_summary = performance_summary(test_daily, test_pnl, backtest_settings["annualization_days"])

    test_net = test_summary["net"]
    calmar = _safe_calmar(test_net.get("annual_return", 0.0), test_net.get("max_drawdown", 0.0))
    print(f"  Test net Sharpe:  {test_net['sharpe']:.3f}")
    print(f"  Test net MaxDD:   {test_net['max_drawdown']:.4f}")
    print(f"  Test net Return:  {test_net['total_return']:.4f}")
    print(f"  Test net Annual:  {test_net.get('annual_return', 0.0):.4f}")
    print(f"  Test Calmar:      {calmar:.2f}x")
    print(f"  Test trades:      {test_summary['trade_count']}")
    print(f"  Test win rate:    {test_summary.get('trade_win_rate', 0.0):.4f}")

    val_bench_long = build_benchmark_positions(val_pred, 1, backtest_settings["flatten_at_day_end"])
    _, val_bench_long_daily = calc_pnl(
        val_bench_long, backtest_settings["commission_rate"], backtest_settings["slippage_rate"], True
    )
    test_bench_long = build_benchmark_positions(test_pred, 1, backtest_settings["flatten_at_day_end"])
    _, test_bench_long_daily = calc_pnl(
        test_bench_long, backtest_settings["commission_rate"], backtest_settings["slippage_rate"], True
    )

    # --- Save JSON summaries ---
    training_summary = {
        "model_type": "single_undivided_raw_return",
        "target_col": target_col,
        "best_iteration": best_iter,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_cols),
        "target_stats": {
            "mean": float(train_df[target_col].mean()),
            "std": float(train_df[target_col].std()),
            "p1": float(train_df[target_col].quantile(0.01)),
            "p99": float(train_df[target_col].quantile(0.99)),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "per_regime_test_metrics": {k: _to_native(v) for k, v in regime_metrics.items()},
        "top_features": fi.head(20).to_dict(orient="records"),
    }
    with (OUT_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_to_native(training_summary), f, indent=2, ensure_ascii=False)

    backtest_summary_out = {
        "model_type": "initial_undivided_raw_return",
        "target_col": target_col,
        "signal_rules": {
            "entry_threshold_88th_pct": entry_q,
            "exit_threshold_45th_pct": exit_q,
            "cost_filter_threshold": cost_filter,
            "effective_entry_threshold": effective_entry,
            "val_bars_above_entry_pct": pct_above_entry,
        },
        "validation": val_summary,
        "test": test_summary,
        "test_calmar": calmar,
    }
    with (OUT_DIR / "backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_to_native(backtest_summary_out), f, indent=2, ensure_ascii=False)

    # --- Save CSVs ---
    test_daily_out = test_daily[["TRADE_DATE", "gross_ret", "net_ret", "nav_gross", "nav_net"]].copy()
    test_daily_out.to_csv(OUT_DIR / "daily_returns.csv", index=False)

    nav_out = test_daily[["TRADE_DATE", "nav_net"]].copy()
    nav_out.to_csv(OUT_DIR / "nav_curve.csv", index=False)

    pred_cols = ["TDATE", "TRADE_DATE", "REGIME_LABEL", "future_return", "pred_return"]
    test_pred_out = test_pred[[c for c in pred_cols if c in test_pred.columns]].copy()
    test_pred_out.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # --- Plots ---
    print("Plotting ...")
    _plot_backtest_curve(
        test_daily=test_daily,
        test_bench_daily=test_bench_long_daily,
        test_pred=test_pred,
        out_path=OUT_DIR / "backtest_curve.png",
        test_summary=test_summary,
        val_daily=val_daily,
        val_summary=val_summary,
    )
    _plot_training_diagnostics(
        evals_result=evals_result,
        fi=fi,
        test_pred=test_pred,
        best_iter=best_iter,
        out_path=OUT_DIR / "training_diagnostics.png",
    )
    _plot_pred_actual_diagnostics(
        test_pred=test_pred,
        test_metrics=test_metrics,
        regime_metrics=regime_metrics,
        effective_entry=effective_entry,
        out_path=OUT_DIR / "pred_actual_diagnostics.png",
    )
    print(f"Saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_backtest_curve(
    test_daily: pd.DataFrame,
    test_bench_daily: pd.DataFrame,
    test_pred: pd.DataFrame,
    out_path: Path,
    test_summary: dict,
    val_daily: pd.DataFrame,
    val_summary: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Undivided Baseline (raw future_return target) — RB Test Set", fontsize=13, fontweight="bold")

    # Test NAV
    ax = axes[0, 0]
    ax.plot(test_daily["TRADE_DATE"], test_daily["nav_net"], color="#1f77b4", linewidth=1.5, label="Strategy Net NAV")
    ax.plot(test_bench_daily["TRADE_DATE"], test_bench_daily["nav_net"],
            color="#7f7f7f", linewidth=1.0, linestyle="--", label="Benchmark Long-only")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set_title("Test Set NAV (net of cost)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    s = test_summary["net"]
    ax.text(0.02, 0.97,
            f"Return={s['total_return']:.2%}  Sharpe={s['sharpe']:.2f}  MaxDD={s['max_drawdown']:.2%}\n"
            f"Trades={test_summary['trade_count']}  WinRate={test_summary.get('trade_win_rate', 0.0):.1%}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#aaa"})

    # Val NAV
    ax = axes[0, 1]
    ax.plot(val_daily["TRADE_DATE"], val_daily["nav_net"], color="#ff7f0e", linewidth=1.5, label="Val Net NAV")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set_title("Validation Set NAV (net of cost)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    sv = val_summary["net"]
    ax.text(0.02, 0.97,
            f"Return={sv['total_return']:.2%}  Sharpe={sv['sharpe']:.2f}  MaxDD={sv['max_drawdown']:.2%}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#aaa"})

    # Test drawdown
    ax = axes[1, 0]
    ax.fill_between(test_daily["TRADE_DATE"], test_daily["net_drawdown"], 0.0, color="#c44e52", alpha=0.35)
    ax.plot(test_daily["TRADE_DATE"], test_daily["net_drawdown"], color="#8c2d04", linewidth=1.0)
    ax.set_title("Test Set Net Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)

    # Sampled predicted vs actual return (key chart: shows model can't distinguish)
    ax = axes[1, 1]
    sampled_step = max(len(test_pred) // 3000, 1)
    sampled = test_pred.iloc[::sampled_step]
    tdate_col = "TDATE" if "TDATE" in sampled.columns else "TRADE_DATE"
    ax.plot(sampled[tdate_col], sampled["future_return"], color="#aaaaaa", linewidth=0.6, alpha=0.8, label="Actual future return")
    ax.plot(sampled[tdate_col], sampled["pred_return"], color="#1f77b4", linewidth=0.8, alpha=0.9, label="Predicted return")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Sampled: Predicted vs Actual Return\n(flat prediction = model cannot distinguish high-return bars)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    pred_std = test_pred["pred_return"].std()
    actual_std = test_pred["future_return"].std()
    ax.text(0.02, 0.97,
            f"Pred std={pred_std:.2e}  Actual std={actual_std:.2e}\n"
            f"Pred/Actual std ratio={pred_std/actual_std:.3f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#aaa"})

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_training_diagnostics(
    evals_result: dict,
    fi: pd.DataFrame,
    test_pred: pd.DataFrame,
    best_iter: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Undivided Baseline — Training Diagnostics", fontsize=13, fontweight="bold")

    # Loss curves
    ax = axes[0, 0]
    metric_key = next(iter(next(iter(evals_result.values())))) if evals_result else "rmse"
    if "train" in evals_result and metric_key in evals_result["train"]:
        rounds = list(range(1, len(evals_result["train"][metric_key]) + 1))
        ax.plot(rounds, evals_result["train"][metric_key], color="#1f77b4", linewidth=1.2, label=f"Train {metric_key}")
        ax.plot(rounds, evals_result["val"][metric_key], color="#ff7f0e", linewidth=1.2, label=f"Val {metric_key}")
        ax.axvline(best_iter, color="#2ca02c", linestyle="--", linewidth=1.0, label=f"Best iter={best_iter}")
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Boosting Round")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Feature importance (gain) top-20
    ax = axes[0, 1]
    top_fi = fi.head(20)
    y_pos = np.arange(len(top_fi))
    ax.barh(y_pos, top_fi["importance_gain"].values, color="#4c78a8", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_fi["feature"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Top 20 Features by Gain")
    ax.set_xlabel("Gain")
    ax.grid(alpha=0.25, axis="x")

    # Prediction distribution vs actual return distribution
    ax = axes[1, 0]
    actual_vals = test_pred["future_return"].values
    pred_vals = test_pred["pred_return"].values
    ax.hist(actual_vals, bins=80, color="#7f7f7f", alpha=0.55, label="Actual future_return", density=True)
    ax.hist(pred_vals, bins=80, color="#1f77b4", alpha=0.7, label="Predicted return (undivided)", density=True)
    ax.set_title("Prediction vs Actual Distribution (Test Set)")
    ax.set_xlabel("Return value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.text(0.97, 0.97,
            f"Actual std={actual_vals.std():.3e}\nPred  std={pred_vals.std():.3e}\n"
            f"Shrinkage={pred_vals.std()/actual_vals.std():.4f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#aaa"})

    # |pred| distribution vs cost filter
    ax = axes[1, 1]
    abs_pred = np.abs(pred_vals)
    ax.hist(abs_pred, bins=80, color="#d62728", alpha=0.7, density=True)
    cost_filter = 0.0002 * 2.0 * 1.2
    ax.axvline(cost_filter, color="black", linewidth=1.5, linestyle="--", label=f"Cost filter={cost_filter:.4%}")
    pct_above = (abs_pred >= cost_filter).mean()
    ax.set_title(f"|Predicted return| vs Cost Filter\n{pct_above:.2%} of bars clear the entry threshold")
    ax.set_xlabel("|Predicted return|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.text(0.97, 0.97,
            f"Cost filter: {cost_filter:.4%}\nBars above: {pct_above:.2%}\n"
            f"=> very few/no trades generated",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#aaa"})

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_pred_actual_diagnostics(
    test_pred: pd.DataFrame,
    test_metrics: dict,
    regime_metrics: dict[str, dict],
    effective_entry: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Undivided Baseline — Prediction Quality & Regime Breakdown", fontsize=13, fontweight="bold")

    # Decile analysis: predicted return decile → mean actual future_return
    ax = axes[0, 0]
    df_tmp = test_pred[["pred_return", "future_return"]].dropna().copy()
    try:
        df_tmp["decile"] = pd.qcut(df_tmp["pred_return"], q=10, labels=False, duplicates="drop")
        decile_means = df_tmp.groupby("decile")["future_return"].mean()
        n_dec = len(decile_means)
        colors_dec = ["#d62728" if v < 0 else "#2ca02c" for v in decile_means.values]
        ax.bar(range(1, n_dec + 1), decile_means.values * 1e4, color=colors_dec, alpha=0.8, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(1, n_dec + 1))
        ax.set_xticklabels([f"D{i}" for i in range(1, n_dec + 1)], fontsize=8)
        spread = float(decile_means.iloc[-1] - decile_means.iloc[0]) * 1e4
        ax.set_title(f"Predicted Return Decile → Mean Actual Return\nD10−D1 spread = {spread:.3f} bps")
    except Exception:
        ax.text(0.5, 0.5, "Decile analysis failed\n(all predictions near zero)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Predicted Return Decile → Mean Actual Return")
    ax.set_ylabel("Mean actual return (×10⁻⁴)")
    ax.set_xlabel("Predicted return decile (D1=lowest, D10=highest)")
    ax.grid(alpha=0.25, axis="y")

    # Per-regime IC comparison (Pearson IC for this undivided model in each regime)
    ax = axes[0, 1]
    regime_names = list(regime_metrics.keys())
    pearson_ics = [regime_metrics[r]["pearson_ic"] for r in regime_names]
    spearman_ics = [regime_metrics[r]["spearman_ic"] for r in regime_names]
    x_pos = np.arange(len(regime_names))
    width = 0.35
    b1 = ax.bar(x_pos - width / 2, pearson_ics, width, label="Pearson IC", color="#4c78a8", alpha=0.8)
    b2 = ax.bar(x_pos + width / 2, spearman_ics, width, label="Spearman IC", color="#f58518", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    overall_pearson = test_metrics["pearson_ic"]
    overall_spearman = test_metrics["spearman_ic"]
    ax.axhline(overall_pearson, color="#4c78a8", linewidth=1.0, linestyle="--", alpha=0.7, label=f"Overall Pearson={overall_pearson:.4f}")
    ax.axhline(overall_spearman, color="#f58518", linewidth=1.0, linestyle="--", alpha=0.7, label=f"Overall Spearman={overall_spearman:.4f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_names, fontsize=10)
    ax.set_title("Per-Regime IC (Undivided Model)\nOne model applied to both regimes separately")
    ax.set_ylabel("IC value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)

    # Scatter: predicted vs actual (sampled)
    ax = axes[1, 0]
    n_sample = min(5000, len(test_pred))
    sample_idx = np.random.default_rng(42).choice(len(test_pred), size=n_sample, replace=False)
    x_scatter = test_pred["pred_return"].values[sample_idx]
    y_scatter = test_pred["future_return"].values[sample_idx]
    ax.scatter(x_scatter, y_scatter, s=2, alpha=0.3, color="#4c78a8", rasterized=True)
    if len(x_scatter) > 2:
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(x_scatter, y_scatter)
        x_line = np.linspace(x_scatter.min(), x_scatter.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=1.5,
                label=f"OLS fit (r={r_val:.4f})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_title(f"Predicted vs Actual Return (Sampled {n_sample})\nPearson IC={test_metrics['pearson_ic']:.4f}")
    ax.set_xlabel("Predicted return")
    ax.set_ylabel("Actual future_return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Rolling prediction vs actual (240-bar window)
    ax = axes[1, 1]
    roll_w = 240
    pred_roll = test_pred["pred_return"].rolling(roll_w, min_periods=1).mean()
    actual_roll = test_pred["future_return"].rolling(roll_w, min_periods=1).mean()
    tdate_col = "TDATE" if "TDATE" in test_pred.columns else "TRADE_DATE"
    ax.plot(test_pred[tdate_col], actual_roll, color="#aaaaaa", linewidth=1.2, label=f"Actual (rolling {roll_w})")
    ax.plot(test_pred[tdate_col], pred_roll, color="#1f77b4", linewidth=1.2, label=f"Predicted (rolling {roll_w})")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"Rolling {roll_w}-bar Average: Predicted vs Actual\n(flat blue line = model produces near-zero signal)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
