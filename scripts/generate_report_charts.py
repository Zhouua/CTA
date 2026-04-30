"""报告章节 1.3 / 1.7 图表生成脚本（中金研报样式，输出供 docs/report.md 引用）。

报告章节 1.2（数据分布）/ 1.3 前半部分（组别 IC、Top 20 IC、月度 IC）/ 1.6（模型
importance）的图已经迁移到 pipeline/diagnostics.py，由 train_products.py 在每次
训练后自动写入当次的 results/runs/<run_id>/<product>/，不再由本脚本生成。

本脚本生成两类图：

  章节 1.3 因子 ICIR Top 20（1 张，输出到 results/runs/micro_result/RB/）：
    - factor_top20_icir.png     — 训练期 walk-forward 月度 IC 跨期稳定性 Top 20

  章节 1.7 双域 vs 单域基线对比（4 张，输出到 results/comparison/）：
    - cmp_nav.png               — 双域 vs 单域 NAV 曲线对比
    - cmp_metrics.png           — 风险收益指标 + 交易统计柱状对比
    - cmp_decile.png            — 预测十分位组的实际收益均值（排序能力）
    - cmp_precision_recall.png  — 高收益 K 线命中数 / 精确率 / 召回率

辅助选项：
  --regenerate-predictions   先重训 dual-regime 模型并刷新
                             results/runs/micro_result/RB/test_predictions.csv
                             （decile / precision-recall 图依赖此文件）

调用：
  python scripts/generate_report_charts.py
  python scripts/generate_report_charts.py --regenerate-predictions
"""
from __future__ import annotations

import argparse
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


# ───────────────────────── 路径与常量 ─────────────────────────

MICRO_DIR    = PROJECT_ROOT / "results" / "runs" / "micro_result" / "RB"
INIT_DIR     = PROJECT_ROOT / "results" / "runs" / "initial_result" / "RB"
COMPARE_DIR  = PROJECT_ROOT / "results" / "comparison"
# factor_registry.json 现在是 per-product 文件，由训练时即时计算写入
# results/runs/<run_id>/<PID>/factor_registry.json，不再有全局静态版本。
# 通过 --registry CLI 参数显式传入，main() 把路径透传给 generate_section_13()。


# ───────────────────────── 中金研报样式（统一调色板） ─────────────────────────

CICC_BLUE   = "#1A5276"
CICC_RED    = "#C0392B"
CICC_GRAY   = "#85929E"
CICC_LBLUE  = "#AED6F1"
CICC_LRED   = "#F1948A"
CICC_GREEN  = "#1E8449"
CICC_ORANGE = "#D35400"
CICC_PURPLE = "#7D3C98"
CICC_TEAL   = "#148F77"

DUAL_COLOR   = CICC_BLUE
SINGLE_COLOR = CICC_ORANGE

FONT_TITLE = dict(fontsize=11, fontweight="bold", color=CICC_BLUE)
FONT_AXIS  = dict(fontsize=9, color="#333333")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})

# 35 个原始 ENG_* 工程化特征（合成因子扩展前），用于因子分组着色
ORIGINAL_ENG_FEATURES: frozenset[str] = frozenset({
    "ENG_RET_1", "ENG_LOG_RET_1", "ENG_RANGE_1", "ENG_BODY_1", "ENG_BODY_ABS",
    "ENG_CLOSE_TO_RANGE", "ENG_INTRABAR_VOL",
    "ENG_RET_5", "ENG_RET_20", "ENG_RET_60",
    "ENG_RV_5", "ENG_RV_20", "ENG_RV_60",
    "ENG_PRICE_TO_MA_5", "ENG_PRICE_TO_MA_20", "ENG_PRICE_TO_MA_60",
    "ENG_VOLUME_RATIO_5", "ENG_VOLUME_RATIO_20", "ENG_VOLUME_RATIO_60",
    "ENG_POSITION_RATIO_5", "ENG_POSITION_RATIO_20", "ENG_POSITION_RATIO_60",
    "ENG_AMOUNT_RATIO_5", "ENG_AMOUNT_RATIO_20", "ENG_AMOUNT_RATIO_60",
    "ENG_ATR_20", "ENG_ATR_60",
    "ENG_VOL_RATIO_5_20", "ENG_RET_DIFF_5_20",
    "ENG_PRICE_BREAKOUT_20", "ENG_PRICE_BREAKDOWN_20",
    "ENG_TOD_SIN", "ENG_TOD_COS", "ENG_WEEKDAY", "ENG_IS_DAY_SESSION",
})

FACTOR_GROUP_COLORS: dict[str, str] = {
    "量价因子":   CICC_BLUE,
    "工程化特征": CICC_GREEN,
    "合成因子":   CICC_ORANGE,
    "中观因子":   CICC_RED,
    "中微交互":   CICC_PURPLE,
}


def _assign_factor_group(feat: str) -> str:
    if feat.startswith("MIDxMICRO_"):
        return "中微交互"
    if feat.startswith("MID_"):
        return "中观因子"
    if feat.startswith("ENG_"):
        return "工程化特征" if feat in ORIGINAL_ENG_FEATURES else "合成因子"
    return "量价因子"


def _source(ax, y: float = -0.12) -> None:
    """统一的资料来源标注。"""
    ax.annotate("资料来源：中金公司研究部", xy=(0, y), xycoords="axes fraction",
                fontsize=7.5, color=CICC_GRAY, ha="left")


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ════════════════════════════════════════════════════════════════════
#   SECTION 1.3 — 因子 ICIR Top 20（1 张图）
# ════════════════════════════════════════════════════════════════════


def _s13_plot_top20_icir(registry_path: Path, top_n: int = 20) -> None:
    """读取指定的 per-product factor_registry.json，绘制按 |ICIR| 降序的 Top N 因子图。

    数据来源：训练时由 pipeline/factor_audit.py 在当前产品 train split 上按月切窗
    计算的 Spearman walk-forward IC（mean_ic / std_ic / icir = mean_ic/std_ic）。
    样式与 pipeline/diagnostics.py::_plot_top20_ic 保持一致，便于报告中并列对比。
    """
    if not registry_path.exists():
        print(f"  [skip] {registry_path} 不存在")
        return
    with registry_path.open(encoding="utf-8") as f:
        reg = json.load(f)
    factors = reg.get("train_factor", []) + reg.get("not_train_factor", [])
    factors = [f for f in factors if f.get("std_ic", 0) and f.get("n_windows", 0) >= 1]
    factors.sort(key=lambda f: abs(float(f.get("icir", 0.0))), reverse=True)
    top = factors[:top_n]
    if not top:
        print("  [skip] registry 中无可用因子")
        return

    names = [f["name"] for f in top]
    icir_abs = [abs(float(f["icir"])) for f in top]
    ic_abs = [abs(float(f["mean_ic"])) for f in top]
    colors = [FACTOR_GROUP_COLORS.get(_assign_factor_group(n), CICC_GRAY) for n in names]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(names))
    ax.barh(y, icir_abs, 0.65, color=colors, alpha=0.88, edgecolor="none")
    for i, (icv, icirv) in enumerate(zip(ic_abs, icir_abs)):
        ax.text(icirv + max(icir_abs) * 0.01, i,
                f"|ICIR|={icirv:.2f}  |IC|={icv:.4f}",
                va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("|ICIR|（训练期月度 Spearman IC 的均值/标准差）", **FONT_AXIS)
    ax.set_title(
        f"Top {top_n} 单因子 |ICIR|（训练集，walk-forward 月度 IC 跨期稳定性）",
        **FONT_TITLE, pad=8,
    )
    ax.set_xlim(0, max(icir_abs) * 1.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)

    seen, handles = {}, []
    for name, color in zip(names, colors):
        g = _assign_factor_group(name)
        if g not in seen:
            seen[g] = color
            handles.append(mpatches.Patch(facecolor=color, alpha=0.88, label=g))
    if handles:
        ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)
    _source(ax)
    fig.tight_layout()
    _save(fig, MICRO_DIR / "factor_top20_icir.png")


def generate_section_13(registry_path: Path) -> None:
    """Section 1.3：因子 ICIR Top 20（1 张图）。registry_path 必须显式传入
    指向 `results/runs/<run_id>/<PID>/factor_registry.json`。"""
    print("[section 1.3] 因子 ICIR Top 20")
    _s13_plot_top20_icir(registry_path)


# ════════════════════════════════════════════════════════════════════
#   SECTION 1.7 — 双域 vs 单域基线对比（4 张图）
# ════════════════════════════════════════════════════════════════════

S17_FIG_SIZE = (9, 4.8)


def _s17_load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _s17_test_block(summary: dict) -> dict:
    return summary.get("test_backtest", summary.get("test", {}))


def _s17_plot_nav() -> None:
    micro_nav = pd.read_csv(MICRO_DIR / "nav_curve.csv", parse_dates=["TRADE_DATE"])
    init_nav  = pd.read_csv(INIT_DIR  / "nav_curve.csv", parse_dates=["TRADE_DATE"])
    m = _s17_test_block(_s17_load_json(MICRO_DIR / "backtest_summary.json"))["net"]
    i = _s17_test_block(_s17_load_json(INIT_DIR  / "backtest_summary.json"))["net"]

    fig, ax = plt.subplots(figsize=S17_FIG_SIZE)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.plot(micro_nav["TRADE_DATE"], micro_nav["nav_net"], color=DUAL_COLOR, linewidth=2.0,
            label=f"双域模型  夏普{m['sharpe']:.2f}  最大回撤{m['max_drawdown']:.2%}")
    ax.plot(init_nav["TRADE_DATE"], init_nav["nav_net"], color=SINGLE_COLOR, linewidth=1.8, linestyle="--",
            label=f"单域基线  夏普{i['sharpe']:.2f}  最大回撤{i['max_drawdown']:.2%}")
    ax.axhline(1.0, color=CICC_GRAY, linewidth=0.7, linestyle=":")
    if len(micro_nav) == len(init_nav):
        dates = micro_nav["TRADE_DATE"].values
        mv, iv = micro_nav["nav_net"].values, init_nav["nav_net"].values
        ax.fill_between(dates, mv, iv, where=mv >= iv, alpha=0.10, color=DUAL_COLOR, interpolate=True)

    stats_text = (
        f"双域模型：总收益{m['total_return']:.2%}  年化{m['annual_return']:.2%}  最大回撤{m['max_drawdown']:.2%}\n"
        f"单域基线：总收益{i['total_return']:.2%}  年化{i['annual_return']:.2%}  最大回撤{i['max_drawdown']:.2%}"
    )
    ax.text(0.01, 0.04, stats_text, transform=ax.transAxes, fontsize=8, va="bottom", color="#333333",
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
    _save(fig, COMPARE_DIR / "cmp_nav.png")


def _s17_plot_metrics() -> None:
    m_sum = _s17_load_json(MICRO_DIR / "backtest_summary.json")
    i_sum = _s17_load_json(INIT_DIR  / "backtest_summary.json")
    m_blk, i_blk = _s17_test_block(m_sum), _s17_test_block(i_sum)
    mn, in_ = m_blk["net"], i_blk["net"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                              gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35})
    fig.patch.set_facecolor("white")

    ax = axes[0]; ax.set_facecolor("white")
    labels    = ["夏普比率", "年化净收益率（%）", "最大回撤绝对值（%）"]
    mv        = [mn["sharpe"], mn["annual_return"] * 100, abs(mn["max_drawdown"]) * 100]
    iv        = [in_["sharpe"], in_["annual_return"] * 100, abs(in_["max_drawdown"]) * 100]
    higher_ok = [True, True, False]   # 最大回撤越小越好
    x = np.arange(len(labels)); w = 0.32
    ax.bar(x - w/2, iv, w, color=SINGLE_COLOR, alpha=0.82, label="单域基线")
    ax.bar(x + w/2, mv, w, color=DUAL_COLOR,   alpha=0.82, label="双域模型")
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
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("风险收益指标对比（净值口径，测试集）", **FONT_TITLE, pad=8)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linestyle=":")
    ax.text(0.01, 0.01, "注：最大回撤绝对值越小越好", transform=ax.transAxes,
            fontsize=7.5, va="bottom", color=CICC_GRAY)

    ax2 = axes[1]; ax2.set_facecolor("white")
    t_labels = ["交易次数", "交易胜率（%）", "均笔净收益（bps）"]
    mv2 = [m_blk["trade_count"], m_blk["trade_win_rate"] * 100,
           m_blk["avg_trade_net_return"] * 10000]
    iv2 = [i_blk["trade_count"], i_blk["trade_win_rate"] * 100,
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
    ax2.set_xticks(x2); ax2.set_xticklabels(t_labels, fontsize=8.5)
    ax2.set_title("交易统计对比（测试集）", **FONT_TITLE, pad=8)
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.grid(axis="y", alpha=0.18, linestyle=":")
    _source(axes[0])
    fig.tight_layout()
    _save(fig, COMPARE_DIR / "cmp_metrics.png")


def _s17_plot_decile() -> None:
    comp_path = MICRO_DIR / "regime_vs_unsegmented_metrics.json"
    if not comp_path.exists():
        print(f"  [skip] {comp_path} 不存在")
        return
    comp = _s17_load_json(comp_path)
    micro_dec_raw = [r for r in comp["predicted_decile_returns"]
                     if r["split"] == "test" and r["model"] == "dual_regime"]
    micro_dec = pd.DataFrame(micro_dec_raw).sort_values("pred_decile").reset_index(drop=True)
    init_pred_path = INIT_DIR / "test_predictions.csv"
    if not init_pred_path.exists():
        print(f"  [skip] {init_pred_path} 不存在")
        return
    init_pred = pd.read_csv(init_pred_path, parse_dates=["TDATE"])
    init_pred["pred_decile"] = pd.qcut(init_pred["pred_return"], 10, labels=False) + 1
    init_dec = (init_pred.groupby("pred_decile")
                .agg(mean_future_return=("future_return", "mean"),
                     mean_pred_return=("pred_return", "mean"))
                .reset_index())

    fig, ax = plt.subplots(figsize=S17_FIG_SIZE)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.plot(micro_dec["pred_decile"], micro_dec["mean_future_return"] * 1e4,
            "o-", color=DUAL_COLOR, linewidth=2.0, markersize=7, label="双域模型", zorder=5)
    ax.plot(init_dec["pred_decile"], init_dec["mean_future_return"] * 1e4,
            "s--", color=SINGLE_COLOR, linewidth=1.8, markersize=7, label="单域基线", zorder=4)
    ax.fill_between(micro_dec["pred_decile"], micro_dec["mean_future_return"] * 1e4, 0,
                    alpha=0.08, color=DUAL_COLOR)
    ax.axhline(0.0, color=CICC_GRAY, linewidth=0.8, linestyle=":")
    m_sp = (micro_dec["mean_future_return"].iloc[-1] - micro_dec["mean_future_return"].iloc[0]) * 1e4
    i_sp = (init_dec["mean_future_return"].iloc[-1]  - init_dec["mean_future_return"].iloc[0])  * 1e4
    ax.text(0.03, 0.97, f"D10−D1极差：\n  双域模型  {m_sp:.2f}×10⁻⁴\n  单域基线  {i_sp:.2f}×10⁻⁴",
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
    _save(fig, COMPARE_DIR / "cmp_decile.png")


def _s17_threshold_metrics(pred_df: pd.DataFrame, quantiles: list[float]) -> list[dict]:
    fr = pred_df["future_return"].values
    pr = pred_df["pred_return"].values
    rows = []
    for q in quantiles:
        tau           = float(np.quantile(fr, q))
        actual_high   = fr >= tau
        pred_high     = pr >= tau
        hit           = actual_high & pred_high
        hit_n         = int(hit.sum())
        pred_high_n   = int(pred_high.sum())
        actual_high_n = int(actual_high.sum())
        precision = hit_n / pred_high_n   if pred_high_n   > 0 else 0.0
        recall    = hit_n / actual_high_n if actual_high_n > 0 else 0.0
        rows.append({"q": q, "tau": tau, "hit": hit_n,
                     "precision": precision, "recall": recall, "actual_n": actual_high_n})
    return rows


def _s17_label_bars(ax, x, w, iv_list, mv_list, fmt, data_range) -> None:
    bar_gap = data_range * 0.03
    for i, (iv, mv) in enumerate(zip(iv_list, mv_list)):
        ax.text(x[i] - w/2, iv + bar_gap, f"{iv:{fmt}}",
                ha="center", va="bottom", fontsize=8, color=SINGLE_COLOR)
        ax.text(x[i] + w/2, mv + bar_gap, f"{mv:{fmt}}",
                ha="center", va="bottom", fontsize=8, color=DUAL_COLOR, fontweight="bold")


def _s17_plot_precision_recall() -> None:
    init_pred_path  = INIT_DIR  / "test_predictions.csv"
    micro_pred_path = MICRO_DIR / "test_predictions.csv"
    if not init_pred_path.exists() or not micro_pred_path.exists():
        print(f"  [skip] 缺少 test_predictions.csv（init={init_pred_path.exists()}, "
              f"micro={micro_pred_path.exists()}）—— 用 --regenerate-predictions 生成")
        return
    init_pred  = pd.read_csv(init_pred_path,  parse_dates=["TDATE"])
    micro_pred = pd.read_csv(micro_pred_path, parse_dates=["TDATE"])

    quantiles = [0.60, 0.70, 0.80, 0.90]
    init_pr   = _s17_threshold_metrics(init_pred,  quantiles)
    micro_pr  = _s17_threshold_metrics(micro_pred, quantiles)
    q_labels  = [f"p{int(q*100)}" for q in quantiles]
    x = np.arange(len(quantiles)); w = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    fig.patch.set_facecolor("white")
    panels = [
        ("命中数：双域模型识别高收益K线的数量更多",
         "命中数（K线根数）",
         [r["hit"] for r in init_pr], [r["hit"] for r in micro_pr], ".0f"),
        ("精确率：单域基线略高，但建立在极低命中数基础上",
         "精确率（%）",
         [r["precision"] * 100 for r in init_pr], [r["precision"] * 100 for r in micro_pr], ".1f"),
        ("召回率：双域模型在各阈值下均显著更高",
         "召回率（%）",
         [r["recall"] * 100 for r in init_pr], [r["recall"] * 100 for r in micro_pr], ".2f"),
    ]
    for ax, (title, ylabel, iv_list, mv_list, fmt) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.bar(x - w/2, iv_list, w, color=SINGLE_COLOR, alpha=0.82, label="单域基线")
        ax.bar(x + w/2, mv_list, w, color=DUAL_COLOR,   alpha=0.82, label="双域模型")
        all_vals   = iv_list + mv_list
        data_max   = max(all_vals)
        data_range = data_max if data_max > 0 else 1.0
        _s17_label_bars(ax, x, w, iv_list, mv_list, fmt, data_range)
        ax.set_ylim(0, data_max * 1.22)
        ax.set_xticks(x); ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9, color="#333333")
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=CICC_BLUE, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.18, linestyle=":")
        ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

    for i, r in enumerate(init_pr):
        axes[2].annotate(f"实际高收益={r['actual_n']:,}根",
                         xy=(x[i], 0), xycoords=("data", "axes fraction"),
                         xytext=(0, -28), textcoords="offset points",
                         ha="center", va="top", fontsize=7, color=CICC_GRAY)
    axes[0].annotate("资料来源：中金公司研究部", xy=(0, -0.16), xycoords="axes fraction",
                     fontsize=7.5, color=CICC_GRAY, ha="left")
    fig.tight_layout()
    _save(fig, COMPARE_DIR / "cmp_precision_recall.png")


def regenerate_micro_test_predictions() -> None:
    """重训 dual-regime 模型并刷新 micro_result/RB/test_predictions.csv（section 1.7 依赖）。"""
    print("[regenerate] 重训 dual-regime 模型 → 刷新 test_predictions.csv")
    from pipeline.dataset import prepare_data
    from pipeline.modeling import train_dual_regime_models, predict_dual_regime

    config_path = str(PROJECT_ROOT / "config.yaml")
    config_override = {
        "data": {"use_mid_weekly": False},
        "product": {
            "product_id": "RB",
            "instrument_code": "RBZL.SHF",
            "raw_data_file": "RBZL.SHF.csv",
            "mid_weekly_files": [],
        },
        "paths": {"run_root": str(PROJECT_ROOT / "results" / "runs")},
    }
    prepared = prepare_data(config_path=config_path, config_override=config_override)
    print(f"  prepared: train={len(prepared.train_data)}, val={len(prepared.val_data)}, test={len(prepared.test_data)}")
    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared, config_path=config_path, config_override=config_override,
    )
    print(f"  trained {len(artifact_map)} regime models")
    test_pred = predict_dual_regime(
        df=prepared.test_data, feature_cols=prepared.feature_cols,
        target_col=prepared.target_col, artifact_map=artifact_map,
    )
    cols = ["TDATE", "TRADE_DATE", "REGIME_LABEL", "future_return", "pred_return"]
    out = test_pred[[c for c in cols if c in test_pred.columns]].copy()
    out_path = MICRO_DIR / "test_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  saved {len(out)} rows → {out_path}")


def generate_section_17() -> None:
    """Section 1.7：双域 vs 单域基线对比（共 4 张图）。"""
    print("[section 1.7] 双域 vs 单域基线对比")
    _s17_plot_nav()
    _s17_plot_metrics()
    _s17_plot_decile()
    _s17_plot_precision_recall()


# ════════════════════════════════════════════════════════════════════
#   主入口
# ════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成报告章节 1.3（因子 ICIR）+ 1.7（双域 vs 单域基线对比）图表。"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="per-product factor_registry.json 路径，例如 "
             "results/runs/<run_id>/RB/factor_registry.json。"
             "由 pipeline/train_products.py 在因子筛选阶段写入。",
    )
    parser.add_argument(
        "--regenerate-predictions",
        action="store_true",
        help="重训 dual-regime 模型刷新 micro_result/RB/test_predictions.csv "
             "（decile / precision-recall 图依赖此文件）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.regenerate_predictions:
        regenerate_micro_test_predictions()
    generate_section_13(args.registry)
    generate_section_17()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
