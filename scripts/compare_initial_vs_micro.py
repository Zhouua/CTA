"""
Compare initial (undivided) vs micro (dual-regime) model on RB rebar futures.
Outputs comparison charts to results/comparison/.
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

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

MICRO_DIR = PROJECT_ROOT / "results" / "runs" / "micro_result" / "RB"
INIT_DIR = PROJECT_ROOT / "results" / "runs" / "initial_result" / "RB"
COMPARISON_METRICS = MICRO_DIR / "regime_vs_unsegmented_metrics.json"
OUT_DIR = PROJECT_ROOT / "results" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MICRO_LABEL = "Dual-Regime (micro)"
INIT_LABEL = "Undivided Baseline (initial)"
MICRO_COLOR = "#1a73e8"
INIT_COLOR = "#f57c00"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _get_test_metrics(summary: dict) -> tuple[dict, dict]:
    """Extract test net metrics and trade stats — handles both summary formats."""
    # Format A (initial_result): {"test": {"net": {...}, "trade_count": ...}}
    if "test" in summary and isinstance(summary["test"], dict) and "net" in summary["test"]:
        blk = summary["test"]
        return blk["net"], {
            "trade_count": blk["trade_count"],
            "trade_win_rate": blk["trade_win_rate"],
            "avg_trade_net_return": blk.get("avg_trade_net_return", 0.0),
        }
    # Format B (micro_result backtest.py output): {"test_backtest": {"net": {...}, ...}}
    if "test_backtest" in summary:
        blk = summary["test_backtest"]
        return blk["net"], {
            "trade_count": blk["trade_count"],
            "trade_win_rate": blk["trade_win_rate"],
            "avg_trade_net_return": blk.get("avg_trade_net_return", 0.0),
        }
    raise KeyError("Cannot find test metrics in summary dict")


MICRO_PRED_PATH = MICRO_DIR / "test_predictions.csv"


def _compute_threshold_metrics(
    pred_df: pd.DataFrame,
    quantile_levels: list[float],
) -> list[dict]:
    """Compute precision/recall at each future_return quantile threshold."""
    results = []
    future_ret = pred_df["future_return"].values
    pred_ret = pred_df["pred_return"].values
    n = len(pred_df)
    for q in quantile_levels:
        threshold = float(np.quantile(future_ret, q))
        actual_high = future_ret >= threshold
        pred_high = pred_ret >= threshold
        hit = actual_high & pred_high
        hit_count = int(hit.sum())
        actual_high_count = int(actual_high.sum())
        pred_high_count = int(pred_high.sum())
        precision = hit_count / pred_high_count if pred_high_count > 0 else 0.0
        recall = hit_count / actual_high_count if actual_high_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({
            "q": q,
            "threshold": threshold,
            "n": n,
            "actual_high_count": actual_high_count,
            "pred_high_count": pred_high_count,
            "hit_count": hit_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "lift": (hit_count / pred_high_count) / (actual_high_count / n) if pred_high_count > 0 else 0.0,
        })
    return results


def main() -> None:
    micro_summary = load_json(MICRO_DIR / "backtest_summary.json")
    init_summary = load_json(INIT_DIR / "backtest_summary.json")

    # Load NAV curves (test period only)
    micro_nav = pd.read_csv(MICRO_DIR / "nav_curve.csv", parse_dates=["TRADE_DATE"])
    init_nav = pd.read_csv(INIT_DIR / "nav_curve.csv", parse_dates=["TRADE_DATE"])

    # Load per-bar test predictions for both models
    init_pred = pd.read_csv(INIT_DIR / "test_predictions.csv", parse_dates=["TDATE"])
    micro_pred_exists = MICRO_PRED_PATH.exists()
    if micro_pred_exists:
        micro_pred = pd.read_csv(MICRO_PRED_PATH, parse_dates=["TDATE"])
    else:
        micro_pred = None
        print(f"  [warning] Micro test_predictions.csv not found at {MICRO_PRED_PATH}")
        print(f"  Run scripts/generate_micro_test_predictions.py first for precision/recall charts.")

    # Load micro decile data from existing comparison JSON (dual_regime entries)
    comp = load_json(COMPARISON_METRICS)
    micro_decile_raw = [r for r in comp["predicted_decile_returns"]
                        if r["split"] == "test" and r["model"] == "dual_regime"]
    micro_decile = pd.DataFrame(micro_decile_raw).sort_values("pred_decile").reset_index(drop=True)

    # Compute prediction decile returns for initial model from per-bar predictions
    init_pred_sorted = init_pred.copy()
    init_pred_sorted["pred_decile"] = pd.qcut(
        init_pred_sorted["pred_return"], 10, labels=False
    ) + 1
    init_decile = (
        init_pred_sorted.groupby("pred_decile")
        .agg(mean_future_return=("future_return", "mean"),
             mean_pred_return=("pred_return", "mean"))
        .reset_index()
    )

    m_net, m_stats = _get_test_metrics(micro_summary)
    i_net, i_stats = _get_test_metrics(init_summary)
    calmar_m = m_net["annual_return"] / abs(m_net["max_drawdown"])
    calmar_i = i_net["annual_return"] / abs(i_net["max_drawdown"])

    # Print summary to console
    print("\n=== Key Metrics Comparison (Test Set) ===")
    print(f"{'Metric':<28} {INIT_LABEL:<30} {MICRO_LABEL}")
    print("-" * 82)
    rows = [
        ("Net Total Return", f"{i_net['total_return']:.2%}", f"{m_net['total_return']:.2%}"),
        ("Net Annual Return", f"{i_net['annual_return']:.2%}", f"{m_net['annual_return']:.2%}"),
        ("Net Sharpe Ratio", f"{i_net['sharpe']:.3f}", f"{m_net['sharpe']:.3f}"),
        ("Net Max Drawdown", f"{i_net['max_drawdown']:.2%}", f"{m_net['max_drawdown']:.2%}"),
        ("Calmar Ratio", f"{calmar_i:.1f}x", f"{calmar_m:.1f}x"),
        ("Trade Count", f"{i_stats['trade_count']}", f"{m_stats['trade_count']}"),
        ("Trade Win Rate", f"{i_stats['trade_win_rate']:.2%}", f"{m_stats['trade_win_rate']:.2%}"),
    ]
    for name, iv, mv in rows:
        print(f"  {name:<26} {iv:<30} {mv}")

    # Compute precision/recall metrics if micro predictions are available
    quantile_levels = [0.6, 0.7, 0.8, 0.9]
    init_pr = _compute_threshold_metrics(init_pred, quantile_levels)
    micro_pr = _compute_threshold_metrics(micro_pred, quantile_levels) if micro_pred is not None else None

    if init_pr and micro_pr:
        print("\n=== Precision / Recall at Future Return Quantile Thresholds (Test) ===")
        print(f"{'Quantile':<10} {'Model':<20} {'Threshold':>12} {'Hits':>8} {'Precision':>12} {'Recall':>10}")
        print("-" * 76)
        for q_data, label in [(init_pr, "Undivided"), (micro_pr, "Dual-Regime")]:
            for r in q_data:
                print(f"  p{int(r['q']*100):<8} {label:<20} {r['threshold']:>12.5f} "
                      f"{r['hit_count']:>8} {r['precision']:>12.3f} {r['recall']:>10.3f}")

    # Generate all charts
    plot_nav_comparison(micro_nav, init_nav, m_net, i_net, calmar_m, calmar_i)
    plot_metrics_bars(m_net, i_net, calmar_m, calmar_i, m_stats, i_stats)
    plot_decile_quality(micro_decile, init_decile)
    plot_regime_ic_comparison(init_pred_sorted, micro_decile)
    plot_comprehensive(micro_nav, init_nav, m_net, i_net, micro_decile, init_decile,
                       calmar_m, calmar_i, init_pred_sorted)
    if micro_pr is not None:
        plot_precision_recall_threshold(init_pr, micro_pr, quantile_levels)

    print(f"\nAll comparison charts saved to {OUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_nav_comparison(micro_nav, init_nav, m_net, i_net, calmar_m, calmar_i):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(micro_nav["TRADE_DATE"], micro_nav["nav_net"],
            color=MICRO_COLOR, linewidth=2.0,
            label=f"{MICRO_LABEL}  Sharpe={m_net['sharpe']:.2f}  MaxDD={m_net['max_drawdown']:.2%}")
    ax.plot(init_nav["TRADE_DATE"], init_nav["nav_net"],
            color=INIT_COLOR, linewidth=1.8, linestyle="--",
            label=f"{INIT_LABEL}  Sharpe={i_net['sharpe']:.2f}  MaxDD={i_net['max_drawdown']:.2%}")
    ax.axhline(1.0, color="#777", linewidth=0.7, linestyle=":")

    # Shade the area where dual-regime beats initial
    if len(micro_nav) == len(init_nav):
        dates = micro_nav["TRADE_DATE"].values
        m_vals = micro_nav["nav_net"].values
        i_vals = init_nav["nav_net"].values
        ax.fill_between(dates, m_vals, i_vals, where=m_vals >= i_vals,
                        alpha=0.12, color=MICRO_COLOR, interpolate=True, label="Dual-regime outperforms")

    ax.set_title("RB (Rebar) Test-Set NAV Comparison: Dual-Regime vs Undivided Baseline\n"
                 "螺纹钢测试集净值曲线对比 (样本外 2025-04 至 2026-03)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Net NAV")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper left")

    stats_txt = (
        f"Dual-Regime:  Total Return {m_net['total_return']:.2%}  MaxDD {m_net['max_drawdown']:.2%}  "
        f"Calmar {calmar_m:.1f}x\n"
        f"Undiv.Baseline: Total Return {i_net['total_return']:.2%}  MaxDD {i_net['max_drawdown']:.2%}  "
        f"Calmar {calmar_i:.1f}x"
    )
    ax.text(0.01, 0.04, stats_txt, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ccc"})
    ax.text(0.99, 0.01, "Source: CICC Research", transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right", color="#888")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "initial_vs_micro_nav.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: initial_vs_micro_nav.png")


def plot_metrics_bars(m_net, i_net, calmar_m, calmar_i, m_stats, i_stats):
    labels = ["Sharpe Ratio", "Annual\nReturn (%)", "Max Drawdown\n(abs, %)", "Calmar\nRatio"]
    micro_vals = [m_net["sharpe"], m_net["annual_return"] * 100, abs(m_net["max_drawdown"]) * 100, calmar_m]
    init_vals  = [i_net["sharpe"], i_net["annual_return"] * 100, abs(i_net["max_drawdown"]) * 100, calmar_i]
    higher_better = [True, True, False, True]

    x = np.arange(len(labels))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 1.4]})

    ax = axes[0]
    ax.bar(x - w / 2, init_vals, w, color=INIT_COLOR, alpha=0.82, label=INIT_LABEL)
    ax.bar(x + w / 2, micro_vals, w, color=MICRO_COLOR, alpha=0.82, label=MICRO_LABEL)

    for i, (iv, mv, hb) in enumerate(zip(init_vals, micro_vals, higher_better)):
        delta = mv - iv
        better = (delta > 0) if hb else (delta < 0)
        color = MICRO_COLOR if better else "#d32f2f"
        fmt = ".2f" if i in (0, 3) else ".1f"
        ax.text(x[i] - w / 2, iv + 0.08, f"{iv:{fmt}}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w / 2, mv + 0.08, f"{mv:{fmt}}", ha="center", va="bottom",
                fontsize=9, color=MICRO_COLOR, fontweight="bold")
        sign = "+" if delta > 0 else ""
        ax.text(x[i], max(iv, mv) + 1.2, f"{sign}{delta:{fmt}}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color)

    ax.set_title("Key Risk-Return Metrics Comparison (Test Set)\n关键风险收益指标对比", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, 0.01, "Note: Max Drawdown — lower abs value is better",
            transform=ax.transAxes, fontsize=8.5, va="bottom", color="#555")

    # Right panel: trade statistics
    ax2 = axes[1]
    trade_labels = ["Trade\nCount", "Win\nRate (%)", "Avg Trade\nReturn (bps)"]
    mv2 = [m_stats["trade_count"], m_stats["trade_win_rate"] * 100,
            m_stats["avg_trade_net_return"] * 10000]
    iv2 = [i_stats["trade_count"], i_stats["trade_win_rate"] * 100,
            i_stats["avg_trade_net_return"] * 10000]

    x2 = np.arange(len(trade_labels))
    ax2.bar(x2 - w / 2, iv2, w, color=INIT_COLOR, alpha=0.82, label=INIT_LABEL)
    ax2.bar(x2 + w / 2, mv2, w, color=MICRO_COLOR, alpha=0.82, label=MICRO_LABEL)
    for i, (iv, mv) in enumerate(zip(iv2, mv2)):
        fmt = ".0f" if i == 0 else ".1f"
        ax2.text(x2[i] - w / 2, iv + 0.5, f"{iv:{fmt}}", ha="center", va="bottom", fontsize=9)
        ax2.text(x2[i] + w / 2, mv + 0.5, f"{mv:{fmt}}", ha="center", va="bottom", fontsize=9, color=MICRO_COLOR)

    ax2.set_title("Trade Statistics", fontsize=11, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(trade_labels, fontsize=9.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.text(0.99, 0.01, "Source: CICC Research", ha="right", fontsize=8.5, color="#888")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "initial_vs_micro_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: initial_vs_micro_metrics.png")


def plot_decile_quality(micro_decile, init_decile):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: line chart of all 10 deciles
    ax = axes[0]
    ax.plot(micro_decile["pred_decile"], micro_decile["mean_future_return"] * 1e4,
            "o-", color=MICRO_COLOR, linewidth=2.2, markersize=8, label=MICRO_LABEL, zorder=5)
    ax.plot(init_decile["pred_decile"], init_decile["mean_future_return"] * 1e4,
            "s--", color=INIT_COLOR, linewidth=2.0, markersize=8, label=INIT_LABEL, zorder=4)
    ax.fill_between(micro_decile["pred_decile"], micro_decile["mean_future_return"] * 1e4, 0,
                    alpha=0.08, color=MICRO_COLOR)
    ax.axhline(0.0, color="#555", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Prediction Score Decile (1=lowest, 10=highest)", fontsize=10)
    ax.set_ylabel("Mean Actual Future Return (×10⁻⁴)", fontsize=10)
    ax.set_title("Prediction Ranking Quality — Actual Return by Score Decile\n"
                 "预测排序能力：按预测分位分组后的实际收益", fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.25)

    m_spread = (micro_decile["mean_future_return"].iloc[-1] - micro_decile["mean_future_return"].iloc[0]) * 1e4
    i_spread = (init_decile["mean_future_return"].iloc[-1] - init_decile["mean_future_return"].iloc[0]) * 1e4
    ax.text(0.03, 0.97,
            f"D10−D1 spread:\n  Dual-Regime: {m_spread:.2f}×10⁻⁴\n  Undivided: {i_spread:.2f}×10⁻⁴",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ccc"})

    # Right: bar chart for extreme deciles only
    ax2 = axes[1]
    deciles_show = [1, 2, 3, 8, 9, 10]
    xlabels = [f"D{d}" for d in deciles_show]

    def _get_decile_val(df, d):
        row = df[df["pred_decile"] == d]
        return float(row["mean_future_return"].values[0] * 1e4) if len(row) else 0.0

    m_vals = [_get_decile_val(micro_decile, d) for d in deciles_show]
    i_vals = [_get_decile_val(init_decile, d) for d in deciles_show]

    x = np.arange(len(xlabels))
    w = 0.35
    ax2.bar(x - w / 2, i_vals, w, color=INIT_COLOR, alpha=0.82, label=INIT_LABEL)
    ax2.bar(x + w / 2, m_vals, w, color=MICRO_COLOR, alpha=0.82, label=MICRO_LABEL)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Mean Actual Future Return (×10⁻⁴)", fontsize=10)
    ax2.set_title("Bottom / Top Prediction Decile Actual Returns\n底部/顶部预测分位实际收益对比",
                  fontsize=10.5, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    # Highlight D10 advantage for dual-regime
    d10_m = _get_decile_val(micro_decile, 10)
    d10_i = _get_decile_val(init_decile, 10)
    ax2.text(0.98, 0.98,
             f"D10: Dual-Regime  {d10_m:.2f}×10⁻⁴\n"
             f"D10: Undivided    {d10_i:.2f}×10⁻⁴",
             transform=ax2.transAxes, fontsize=9, va="top", ha="right",
             bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef6ff", "alpha": 0.95, "edgecolor": MICRO_COLOR})

    fig.text(0.99, 0.01, "Source: CICC Research", ha="right", fontsize=8.5, color="#888")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "initial_vs_micro_decile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: initial_vs_micro_decile.png")


def plot_regime_ic_comparison(init_pred: pd.DataFrame, micro_decile: pd.DataFrame):
    """Show IC and decile spread for undivided model within each vol regime."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors_regime = {-1: "#4caf50", 1: "#e53935"}
    names_regime = {-1: "Low-Vol Regime (low_vol)", 1: "High-Vol Regime (high_vol)"}

    for ax, (regime_label, rname, rcolor) in zip(
        axes,
        [(-1, names_regime[-1], colors_regime[-1]), (1, names_regime[1], colors_regime[1])]
    ):
        grp = init_pred[init_pred["REGIME_LABEL"] == regime_label].copy()
        if grp.empty:
            continue
        grp = grp.sort_values("pred_return")
        grp["pd_dec"] = pd.qcut(grp["pred_return"], 10, labels=False) + 1
        dec_df = grp.groupby("pd_dec")["future_return"].mean().reset_index()
        dec_df.columns = ["pred_decile", "mean_actual"]

        ax.bar(dec_df["pred_decile"], dec_df["mean_actual"] * 1e4,
               color=rcolor, alpha=0.78, edgecolor="white")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Undivided Baseline — {rname}\n单域模型在{rname}中的预测能力",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Prediction Decile", fontsize=10)
        ax.set_ylabel("Mean Actual Future Return (×10⁻⁴)", fontsize=10)
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.25)

        ic = float(pd.Series(grp["future_return"].values).corr(pd.Series(grp["pred_return"].values)))
        spread = float((dec_df["mean_actual"].iloc[-1] - dec_df["mean_actual"].iloc[0]) * 1e4)
        n = len(grp)
        ax.text(0.03, 0.97,
                f"n = {n:,}\nPearson IC = {ic:.4f}\nD10−D1 spread = {spread:.2f}×10⁻⁴",
                transform=ax.transAxes, fontsize=9.5, va="top",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ccc"})

    fig.suptitle("Undivided Baseline Prediction Quality Within Each Volatility Regime\n"
                 "单域基线在两个波动率域内的预测能力（揭示分域建模的必要性）",
                 fontsize=11, fontweight="bold", y=1.02)
    axes[1].text(0.99, 0.01, "Source: CICC Research", transform=axes[1].transAxes,
                 fontsize=8, va="bottom", ha="right", color="#888")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "initial_regime_decile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: initial_regime_decile.png")


def plot_comprehensive(micro_nav, init_nav, m_net, i_net, micro_decile, init_decile,
                       calmar_m, calmar_i, init_pred):
    """Master 4-panel figure for the report."""
    fig = plt.figure(figsize=(16, 12))
    gs0 = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32, figure=fig)

    # ── Panel A: NAV Curves ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs0[0, 0])
    ax0.plot(micro_nav["TRADE_DATE"], micro_nav["nav_net"],
             color=MICRO_COLOR, linewidth=2.2, label=f"{MICRO_LABEL}  Sharpe={m_net['sharpe']:.2f}")
    ax0.plot(init_nav["TRADE_DATE"], init_nav["nav_net"],
             color=INIT_COLOR, linewidth=1.8, linestyle="--",
             label=f"{INIT_LABEL}  Sharpe={i_net['sharpe']:.2f}")
    ax0.axhline(1.0, color="#aaa", linewidth=0.6, linestyle=":")
    ax0.set_title("A.  Test-Set NAV (net of cost)", fontsize=11, fontweight="bold")
    ax0.legend(fontsize=9, loc="upper left")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax0.grid(alpha=0.25)
    ax0.text(0.02, 0.04,
             f"Dual-Regime:  {m_net['total_return']:.2%} total  MaxDD {m_net['max_drawdown']:.2%}  Calmar {calmar_m:.1f}x\n"
             f"Undivided:    {i_net['total_return']:.2%} total  MaxDD {i_net['max_drawdown']:.2%}  Calmar {calmar_i:.1f}x",
             transform=ax0.transAxes, fontsize=8.5, va="bottom",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ccc"})

    # ── Panel B: Metrics Bars ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs0[0, 1])
    met_labels = ["Sharpe", "Ann. Return\n(%)", "|Max DD|\n(%)", "Calmar\nRatio"]
    mv_b = [m_net["sharpe"], m_net["annual_return"] * 100, abs(m_net["max_drawdown"]) * 100, calmar_m]
    iv_b = [i_net["sharpe"], i_net["annual_return"] * 100, abs(i_net["max_drawdown"]) * 100, calmar_i]
    hb_b = [True, True, False, True]
    x_b = np.arange(len(met_labels))
    w_b = 0.35
    ax1.bar(x_b - w_b / 2, iv_b, w_b, color=INIT_COLOR, alpha=0.82, label="Undivided")
    ax1.bar(x_b + w_b / 2, mv_b, w_b, color=MICRO_COLOR, alpha=0.82, label="Dual-Regime")
    for i, (iv, mv, hb) in enumerate(zip(iv_b, mv_b, hb_b)):
        delta = mv - iv
        better = (delta > 0) if hb else (delta < 0)
        col = MICRO_COLOR if better else "#d32f2f"
        fmt = ".2f" if i in (0, 3) else ".1f"
        ax1.text(x_b[i] - w_b / 2, iv + 0.05, f"{iv:{fmt}}", ha="center", va="bottom", fontsize=8.5)
        ax1.text(x_b[i] + w_b / 2, mv + 0.05, f"{mv:{fmt}}", ha="center", va="bottom",
                 fontsize=8.5, color=MICRO_COLOR, fontweight="bold")
        sign = "+" if delta > 0 else ""
        ax1.text(x_b[i], max(iv, mv) + 1.0, f"{sign}{delta:{fmt}}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)
    ax1.set_title("B.  Key Risk-Return Metrics Comparison", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_b)
    ax1.set_xticklabels(met_labels, fontsize=9.5)
    ax1.legend(fontsize=9.5)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.01, 0.01, "Note: |Max DD| lower is better",
             transform=ax1.transAxes, fontsize=8, va="bottom", color="#666")

    # ── Panel C: Decile Quality ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs0[1, 0])
    ax2.plot(micro_decile["pred_decile"], micro_decile["mean_future_return"] * 1e4,
             "o-", color=MICRO_COLOR, linewidth=2.2, markersize=7, label="Dual-Regime", zorder=5)
    ax2.plot(init_decile["pred_decile"], init_decile["mean_future_return"] * 1e4,
             "s--", color=INIT_COLOR, linewidth=2.0, markersize=7, label="Undivided", zorder=4)
    ax2.fill_between(micro_decile["pred_decile"], micro_decile["mean_future_return"] * 1e4, 0,
                     alpha=0.08, color=MICRO_COLOR)
    ax2.axhline(0.0, color="#555", linewidth=0.7, linestyle=":")
    ax2.set_xlabel("Prediction Score Decile (1=lowest → 10=highest)", fontsize=9.5)
    ax2.set_ylabel("Mean Actual Future Return (×10⁻⁴)", fontsize=9.5)
    ax2.set_title("C.  Prediction Ranking Quality — Actual Return by Decile", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(1, 11))
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.25)
    m_sp = (micro_decile["mean_future_return"].iloc[-1] - micro_decile["mean_future_return"].iloc[0]) * 1e4
    i_sp = (init_decile["mean_future_return"].iloc[-1] - init_decile["mean_future_return"].iloc[0]) * 1e4
    ax2.text(0.03, 0.97,
             f"D10−D1 spread:\n  Dual-Regime:  {m_sp:.2f}×10⁻⁴\n  Undivided:    {i_sp:.2f}×10⁻⁴",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ccc"})

    # ── Panel D: Regime-specific IC for undivided model ───────────────────────
    ax3 = fig.add_subplot(gs0[1, 1])
    regime_info = {}
    for rl, rname, rcol in [(-1, "Low-Vol", "#4caf50"), (1, "High-Vol", "#e53935")]:
        grp = init_pred[init_pred["REGIME_LABEL"] == rl].copy()
        if grp.empty:
            continue
        grp["pd_dec"] = pd.qcut(grp["pred_return"], 10, labels=False) + 1
        dec_m = grp.groupby("pd_dec")["future_return"].mean()
        ic = float(pd.Series(grp["future_return"].values).corr(pd.Series(grp["pred_return"].values)))
        spread = float((dec_m.iloc[-1] - dec_m.iloc[0]) * 1e4) if len(dec_m) >= 10 else 0.0
        regime_info[rname] = {"ic": ic, "spread": spread, "color": rcol, "n": len(grp)}

    r_keys = list(regime_info.keys())
    bar_met = ["Pearson IC", "D10−D1 Spread\n(×10⁻⁴)"]
    x_d = np.arange(len(bar_met))
    w_d = 0.3
    for j, rk in enumerate(r_keys):
        vals = [regime_info[rk]["ic"], regime_info[rk]["spread"]]
        bars = ax3.bar(x_d + (j - 0.5) * w_d, vals, w_d,
                       color=regime_info[rk]["color"], alpha=0.82,
                       label=f"{rk} (n={regime_info[rk]['n']:,})")
        for xi, val in zip(x_d, vals):
            ax3.text(xi + (j - 0.5) * w_d, val + 0.001 if xi == 0 else val + 0.02,
                     f"{val:.4f}" if xi == 0 else f"{val:.2f}",
                     ha="center", va="bottom", fontsize=9)

    ax3.axhline(0.0, color="black", linewidth=0.7)
    ax3.set_xticks(x_d)
    ax3.set_xticklabels(bar_met, fontsize=10)
    ax3.set_title("D.  Undivided Baseline Prediction Quality\n    Within Each Vol Regime",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9.5)
    ax3.grid(axis="y", alpha=0.25)
    ax3.text(0.96, 0.97,
             "Dual-Regime trains\nseparate models\nper regime → higher IC",
             transform=ax3.transAxes, fontsize=8.5, va="top", ha="right",
             bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef6ff", "alpha": 0.92,
                   "edgecolor": MICRO_COLOR})

    fig.suptitle("RB Rebar — Dual-Regime vs Undivided Baseline: Comprehensive Comparison\n"
                 "螺纹钢  双波动率域模型 vs 单域基线  全面对比（样本外测试集 2025-04 至 2026-03）",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.99, -0.005, "Source: CICC Research", ha="right", fontsize=9, color="#888")
    fig.savefig(OUT_DIR / "initial_vs_micro_comprehensive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: initial_vs_micro_comprehensive.png")


def plot_precision_recall_threshold(
    init_pr: list[dict],
    micro_pr: list[dict],
    quantile_levels: list[float],
) -> None:
    """
    Show how well each model identifies top-quantile future-return bars.

    Threshold definition: for each q in [p60, p70, p80, p90],
      threshold τ_q = quantile(actual_future_return, q)
      'predicted high': pred_return >= τ_q
      'actual high':    future_return >= τ_q
      hit (TP): pred >= τ_q AND actual >= τ_q
      precision = TP / predicted_high
      recall    = TP / actual_high
    """
    q_labels = [f"p{int(q * 100)}" for q in quantile_levels]
    x = np.arange(len(quantile_levels))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "RB — High-Return Bar Identification: Precision, Recall, and Hit Count\n"
        "at Future Return Quantile Thresholds (Test Set 2025-04 to 2026-03)",
        fontsize=12, fontweight="bold"
    )

    # ── Panel 1: Hit Count (# of bars correctly identified as high-return) ─────
    ax = axes[0]
    i_hits = [r["hit_count"] for r in init_pr]
    m_hits = [r["hit_count"] for r in micro_pr]
    b1 = ax.bar(x - w / 2, i_hits, w, color=INIT_COLOR, alpha=0.82, label=INIT_LABEL)
    b2 = ax.bar(x + w / 2, m_hits, w, color=MICRO_COLOR, alpha=0.82, label=MICRO_LABEL)
    for bi, iv, mv in zip(x, i_hits, m_hits):
        ax.text(bi - w / 2, iv + 1, str(iv), ha="center", va="bottom", fontsize=10)
        ax.text(bi + w / 2, mv + 1, str(mv), ha="center", va="bottom", fontsize=10,
                color=MICRO_COLOR, fontweight="bold")
        delta = mv - iv
        ax.text(bi, max(iv, mv) + 8, f"+{delta}" if delta > 0 else str(delta),
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color=MICRO_COLOR if delta > 0 else "#d32f2f")
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=11)
    ax.set_title("Hit Count (Correctly Identified High-Return Bars)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of bars (TP = pred >= τ_q AND actual >= τ_q)")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.97,
            "How many top-return bars\ndoes each model successfully flag?",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#ccc"})

    # ── Panel 2: Precision ─────────────────────────────────────────────────────
    ax = axes[1]
    i_prec = [r["precision"] for r in init_pr]
    m_prec = [r["precision"] for r in micro_pr]
    ax.bar(x - w / 2, i_prec, w, color=INIT_COLOR, alpha=0.82, label=INIT_LABEL)
    ax.bar(x + w / 2, m_prec, w, color=MICRO_COLOR, alpha=0.82, label=MICRO_LABEL)
    for bi, iv, mv in zip(x, i_prec, m_prec):
        ax.text(bi - w / 2, iv + 0.005, f"{iv:.3f}", ha="center", va="bottom", fontsize=10)
        ax.text(bi + w / 2, mv + 0.005, f"{mv:.3f}", ha="center", va="bottom", fontsize=10,
                color=MICRO_COLOR, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Precision at Each Threshold\nP(actual high | predicted high)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.5, color="#777", linewidth=0.9, linestyle="--", label="50% baseline")
    ax.text(0.02, 0.97,
            "Of all flagged bars, what\nfraction actually delivered\nhigh return?",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#ccc"})

    # ── Panel 3: Recall ────────────────────────────────────────────────────────
    ax = axes[2]
    i_rec = [r["recall"] for r in init_pr]
    m_rec = [r["recall"] for r in micro_pr]
    i_actual = [r["actual_high_count"] for r in init_pr]
    b1 = ax.bar(x - w / 2, i_rec, w, color=INIT_COLOR, alpha=0.82, label=INIT_LABEL)
    b2 = ax.bar(x + w / 2, m_rec, w, color=MICRO_COLOR, alpha=0.82, label=MICRO_LABEL)
    for bi, iv, mv, act in zip(x, i_rec, m_rec, i_actual):
        ax.text(bi - w / 2, iv + 0.002, f"{iv:.3f}", ha="center", va="bottom", fontsize=10)
        ax.text(bi + w / 2, mv + 0.002, f"{mv:.3f}", ha="center", va="bottom", fontsize=10,
                color=MICRO_COLOR, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=11)
    ax.set_ylim(0, max(max(i_rec), max(m_rec)) * 1.35 + 0.01)
    ax.set_title("Recall at Each Threshold\nP(predicted high | actual high)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recall")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.97,
            "Of all bars that actually\ndelivered high return, what\nfraction did each model flag?",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#ccc"})

    # Annotate actual_high_count below x-axis
    for bi, act in zip(x, i_actual):
        axes[2].text(bi, -0.015, f"actual={act:,}", ha="center", va="top",
                     fontsize=8, color="#555", transform=axes[2].get_xaxis_transform())

    fig.text(0.99, 0.01, "Source: CICC Research", ha="right", fontsize=9, color="#888")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "initial_vs_micro_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: initial_vs_micro_precision_recall.png")


if __name__ == "__main__":
    main()
