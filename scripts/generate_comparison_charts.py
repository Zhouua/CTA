"""
Generate 中金研报-style comparison charts for report section 1.7.

Outputs (all to results/comparison/):
  cmp_nav.png               — net NAV curve comparison
  cmp_metrics.png           — key risk-return metrics bar comparison
  cmp_decile.png            — prediction decile quality (ranking ability)
  cmp_precision_recall.png  — high-return bar hit count / precision / recall
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

MICRO_DIR = PROJECT_ROOT / "results" / "runs" / "micro_result" / "RB"
INIT_DIR  = PROJECT_ROOT / "results" / "runs" / "initial_result" / "RB"
OUT_DIR   = PROJECT_ROOT / "results" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────
CICC_BLUE   = "#1A5276"
CICC_RED    = "#C0392B"
CICC_GRAY   = "#85929E"
CICC_LBLUE  = "#AED6F1"
CICC_LRED   = "#F1948A"
CICC_ORANGE = "#D35400"

DUAL_COLOR   = CICC_BLUE
SINGLE_COLOR = CICC_ORANGE
FIG_SIZE     = (9, 4.8)

FONT_TITLE = dict(fontsize=11, fontweight="bold", color=CICC_BLUE)
FONT_AXIS  = dict(fontsize=9, color="#333333")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})


def _source(ax, y=-0.12):
    ax.annotate("资料来源：中金公司研究部", xy=(0, y), xycoords="axes fraction",
                fontsize=7.5, color=CICC_GRAY, ha="left")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _test_block(summary: dict) -> dict:
    if "test_backtest" in summary:
        return summary["test_backtest"]
    return summary["test"]


# ── chart 1: net NAV curves ────────────────────────────────────────────
def plot_nav():
    micro_nav = pd.read_csv(MICRO_DIR / "nav_curve.csv", parse_dates=["TRADE_DATE"])
    init_nav  = pd.read_csv(INIT_DIR  / "nav_curve.csv", parse_dates=["TRADE_DATE"])
    m = _test_block(_load_json(MICRO_DIR / "backtest_summary.json"))["net"]
    i = _test_block(_load_json(INIT_DIR  / "backtest_summary.json"))["net"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(micro_nav["TRADE_DATE"], micro_nav["nav_net"],
            color=DUAL_COLOR, linewidth=2.0,
            label=f"双域模型  夏普{m['sharpe']:.2f}  最大回撤{m['max_drawdown']:.2%}")
    ax.plot(init_nav["TRADE_DATE"], init_nav["nav_net"],
            color=SINGLE_COLOR, linewidth=1.8, linestyle="--",
            label=f"单域基线  夏普{i['sharpe']:.2f}  最大回撤{i['max_drawdown']:.2%}")
    ax.axhline(1.0, color=CICC_GRAY, linewidth=0.7, linestyle=":")

    # shade outperformance
    if len(micro_nav) == len(init_nav):
        dates = micro_nav["TRADE_DATE"].values
        mv    = micro_nav["nav_net"].values
        iv    = init_nav["nav_net"].values
        ax.fill_between(dates, mv, iv, where=mv >= iv,
                        alpha=0.10, color=DUAL_COLOR, interpolate=True)

    stats_text = (
        f"双域模型：总收益{m['total_return']:.2%}  年化{m['annual_return']:.2%}  "
        f"最大回撤{m['max_drawdown']:.2%}\n"
        f"单域基线：总收益{i['total_return']:.2%}  年化{i['annual_return']:.2%}  "
        f"最大回撤{i['max_drawdown']:.2%}"
    )
    ax.text(0.01, 0.04, stats_text, transform=ax.transAxes,
            fontsize=8, va="bottom", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=CICC_GRAY, alpha=0.88))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("净值（成本后）", **FONT_AXIS)
    ax.set_title(
        "双域模型 vs 单域基线：测试集净值曲线对比\n"
        "（双域模型蓝色阴影区域为领先段；测试集2025-04—2026-03）",
        **FONT_TITLE, pad=8,
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linestyle=":")
    _source(ax)

    fig.tight_layout()
    out = OUT_DIR / "cmp_nav.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 2: key metrics bar comparison ───────────────────────────────
def plot_metrics():
    m_sum = _load_json(MICRO_DIR / "backtest_summary.json")
    i_sum = _load_json(INIT_DIR  / "backtest_summary.json")
    m_blk = _test_block(m_sum)
    i_blk = _test_block(i_sum)
    mn = m_blk["net"]
    in_ = i_blk["net"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                              gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35})
    fig.patch.set_facecolor("white")

    # ── Left: risk-return metrics ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")

    labels    = ["夏普比率", "年化净收益率（%）", "最大回撤绝对值（%）"]
    mv        = [mn["sharpe"], mn["annual_return"] * 100, abs(mn["max_drawdown"]) * 100]
    iv        = [in_["sharpe"], in_["annual_return"] * 100, abs(in_["max_drawdown"]) * 100]
    higher_ok = [True, True, False]   # lower is better for drawdown

    x = np.arange(len(labels))
    w = 0.32
    bars_i = ax.bar(x - w/2, iv, w, color=SINGLE_COLOR, alpha=0.82, label="单域基线")
    bars_m = ax.bar(x + w/2, mv, w, color=DUAL_COLOR,   alpha=0.82, label="双域模型")

    for i, (iv_val, mv_val, hok) in enumerate(zip(iv, mv, higher_ok)):
        fmt = ".2f" if i == 0 else ".1f"
        ax.text(x[i] - w/2, iv_val + 0.1, f"{iv_val:{fmt}}",
                ha="center", va="bottom", fontsize=8.5, color=SINGLE_COLOR)
        ax.text(x[i] + w/2, mv_val + 0.1, f"{mv_val:{fmt}}",
                ha="center", va="bottom", fontsize=8.5, color=DUAL_COLOR, fontweight="bold")
        delta = mv_val - iv_val
        better = (delta > 0) if hok else (delta < 0)
        col  = DUAL_COLOR if better else CICC_RED
        sign = "+" if delta > 0 else ""
        ax.text(x[i], max(iv_val, mv_val) + 1.0, f"{sign}{delta:{fmt}}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("风险收益指标对比（净值口径，测试集）", **FONT_TITLE, pad=8)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linestyle=":")
    ax.text(0.01, 0.01, "注：最大回撤绝对值越小越好", transform=ax.transAxes,
            fontsize=7.5, va="bottom", color=CICC_GRAY)

    # ── Right: trade statistics ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    t_labels = ["交易次数", "交易胜率（%）", "均笔净收益（bps）"]
    mv2 = [m_blk["trade_count"],
           m_blk["trade_win_rate"] * 100,
           m_blk["avg_trade_net_return"] * 10000]
    iv2 = [i_blk["trade_count"],
           i_blk["trade_win_rate"] * 100,
           i_blk["avg_trade_net_return"] * 10000]

    x2 = np.arange(len(t_labels))
    ax2.bar(x2 - w/2, iv2, w, color=SINGLE_COLOR, alpha=0.82, label="单域基线")
    ax2.bar(x2 + w/2, mv2, w, color=DUAL_COLOR,   alpha=0.82, label="双域模型")

    for i, (iv_val, mv_val) in enumerate(zip(iv2, mv2)):
        fmt = ".0f" if i == 0 else ".1f"
        ax2.text(x2[i] - w/2, iv_val + 0.5, f"{iv_val:{fmt}}",
                 ha="center", va="bottom", fontsize=8.5, color=SINGLE_COLOR)
        ax2.text(x2[i] + w/2, mv_val + 0.5, f"{mv_val:{fmt}}",
                 ha="center", va="bottom", fontsize=8.5, color=DUAL_COLOR, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(t_labels, fontsize=8.5)
    ax2.set_title("交易统计对比（测试集）", **FONT_TITLE, pad=8)
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.grid(axis="y", alpha=0.18, linestyle=":")

    _source(axes[0])

    fig.tight_layout()
    out = OUT_DIR / "cmp_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 3: prediction decile quality ───────────────────────────────
def plot_decile():
    comp = _load_json(MICRO_DIR / "regime_vs_unsegmented_metrics.json")
    micro_dec_raw = [r for r in comp["predicted_decile_returns"]
                     if r["split"] == "test" and r["model"] == "dual_regime"]
    micro_dec = pd.DataFrame(micro_dec_raw).sort_values("pred_decile").reset_index(drop=True)

    init_pred = pd.read_csv(INIT_DIR / "test_predictions.csv", parse_dates=["TDATE"])
    init_pred["pred_decile"] = pd.qcut(init_pred["pred_return"], 10, labels=False) + 1
    init_dec = (init_pred.groupby("pred_decile")
                .agg(mean_future_return=("future_return", "mean"),
                     mean_pred_return=("pred_return", "mean"))
                .reset_index())

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(micro_dec["pred_decile"], micro_dec["mean_future_return"] * 1e4,
            "o-", color=DUAL_COLOR, linewidth=2.0, markersize=7, label="双域模型", zorder=5)
    ax.plot(init_dec["pred_decile"], init_dec["mean_future_return"] * 1e4,
            "s--", color=SINGLE_COLOR, linewidth=1.8, markersize=7, label="单域基线", zorder=4)
    ax.fill_between(micro_dec["pred_decile"], micro_dec["mean_future_return"] * 1e4, 0,
                    alpha=0.08, color=DUAL_COLOR)
    ax.axhline(0.0, color=CICC_GRAY, linewidth=0.8, linestyle=":")

    m_sp = (micro_dec["mean_future_return"].iloc[-1] - micro_dec["mean_future_return"].iloc[0]) * 1e4
    i_sp = (init_dec["mean_future_return"].iloc[-1]  - init_dec["mean_future_return"].iloc[0])  * 1e4
    ax.text(0.03, 0.97,
            f"D10−D1极差：\n  双域模型  {m_sp:.2f}×10⁻⁴\n  单域基线  {i_sp:.2f}×10⁻⁴",
            transform=ax.transAxes, fontsize=8.5, va="top", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=CICC_GRAY, alpha=0.88))

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{d}" for d in range(1, 11)], fontsize=8.5)
    ax.set_xlabel("预测收益率分位组（D1=最低，D10=最高）", **FONT_AXIS)
    ax.set_ylabel("组内平均实际收益率（×10⁻⁴）", **FONT_AXIS)
    ax.set_title(
        "预测排序能力：各十分位组实际收益均值（测试集）\n"
        "（双域模型D10−D1极差更大，说明信号方向性更集中、排序能力更强）",
        **FONT_TITLE, pad=8,
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linestyle=":")
    _source(ax)

    fig.tight_layout()
    out = OUT_DIR / "cmp_decile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 4: precision / recall at quantile thresholds ───────────────
def _threshold_metrics(pred_df: pd.DataFrame, quantiles: list[float]) -> list[dict]:
    fr = pred_df["future_return"].values
    pr = pred_df["pred_return"].values
    n  = len(pred_df)
    rows = []
    for q in quantiles:
        tau          = float(np.quantile(fr, q))
        actual_high  = fr >= tau
        pred_high    = pr >= tau
        hit          = actual_high & pred_high
        hit_n        = int(hit.sum())
        pred_high_n  = int(pred_high.sum())
        actual_high_n = int(actual_high.sum())
        precision    = hit_n / pred_high_n  if pred_high_n  > 0 else 0.0
        recall       = hit_n / actual_high_n if actual_high_n > 0 else 0.0
        rows.append({"q": q, "tau": tau, "hit": hit_n,
                     "precision": precision, "recall": recall,
                     "actual_n": actual_high_n})
    return rows


def _label_bars(ax, x, w, iv_list, mv_list, fmt, data_range):
    """Place bar-top value labels with a fixed offset above each bar."""
    bar_gap = data_range * 0.03

    for i, (iv, mv) in enumerate(zip(iv_list, mv_list)):
        ax.text(x[i] - w/2, iv + bar_gap, f"{iv:{fmt}}",
                ha="center", va="bottom", fontsize=8, color=SINGLE_COLOR)
        ax.text(x[i] + w/2, mv + bar_gap, f"{mv:{fmt}}",
                ha="center", va="bottom", fontsize=8, color=DUAL_COLOR, fontweight="bold")


def plot_precision_recall():
    init_pred  = pd.read_csv(INIT_DIR  / "test_predictions.csv", parse_dates=["TDATE"])
    micro_pred = pd.read_csv(MICRO_DIR / "test_predictions.csv", parse_dates=["TDATE"])

    quantiles = [0.60, 0.70, 0.80, 0.90]
    init_pr   = _threshold_metrics(init_pred,  quantiles)
    micro_pr  = _threshold_metrics(micro_pred, quantiles)
    q_labels  = [f"p{int(q*100)}" for q in quantiles]
    x = np.arange(len(quantiles))
    w = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    fig.patch.set_facecolor("white")

    panels = [
        # (title, ylabel, init_vals, micro_vals, fmt)
        ("命中数：双域模型识别高收益K线的数量更多",
         "命中数（K线根数）",
         [r["hit"] for r in init_pr],
         [r["hit"] for r in micro_pr],
         ".0f"),
        ("精确率：单域基线略高，但建立在极低命中数基础上",
         "精确率（%）",
         [r["precision"] * 100 for r in init_pr],
         [r["precision"] * 100 for r in micro_pr],
         ".1f"),
        ("召回率：双域模型在各阈值下均显著更高",
         "召回率（%）",
         [r["recall"] * 100 for r in init_pr],
         [r["recall"] * 100 for r in micro_pr],
         ".2f"),
    ]

    for ax, (title, ylabel, iv_list, mv_list, fmt) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.bar(x - w/2, iv_list, w, color=SINGLE_COLOR, alpha=0.82, label="单域基线")
        ax.bar(x + w/2, mv_list, w, color=DUAL_COLOR,   alpha=0.82, label="双域模型")

        all_vals    = iv_list + mv_list
        data_max    = max(all_vals)
        data_range  = data_max if data_max > 0 else 1.0
        _label_bars(ax, x, w, iv_list, mv_list, fmt, data_range)

        # ylim: leave 18% headroom above tallest bar for value labels only
        ax.set_ylim(0, data_max * 1.22)
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9, color="#333333")
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=CICC_BLUE, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.18, linestyle=":")
        ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

    # annotate actual_n count below x-tick labels on recall panel
    for i, r in enumerate(init_pr):
        axes[2].annotate(f"实际高收益={r['actual_n']:,}根",
                         xy=(x[i], 0), xycoords=("data", "axes fraction"),
                         xytext=(0, -28), textcoords="offset points",
                         ha="center", va="top", fontsize=7, color=CICC_GRAY)

    axes[0].annotate("资料来源：中金公司研究部",
                     xy=(0, -0.16), xycoords="axes fraction",
                     fontsize=7.5, color=CICC_GRAY, ha="left")

    fig.tight_layout()
    out = OUT_DIR / "cmp_precision_recall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating section 1.7 comparison charts ...")
    plot_nav()
    plot_metrics()
    plot_decile()
    plot_precision_recall()
    print("Done.")
