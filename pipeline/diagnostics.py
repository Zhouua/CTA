"""单品种训练后诊断图（中金研报样式）。

由 pipeline/train_products.py 在 backtest 完成后调用，输出以下 7 张图到当次训练
的 product_dir，覆盖 docs/report.md 中 单品种章节（1.2 / 1.3 / 1.6）所有引用图：

  章节 1.2 价量分布与风险画像（3 张）：
    - data_dist_single_bar.png          — 单根 K 线对数收益率分布（实际 vs 同均值同方差正态）
    - data_dist_future_return.png       — 预测目标 future_return 分布
    - data_dist_vol_regime.png          — 20 日日波动率分布 + 高低波 cutoff

  章节 1.3 因子体系（3 张）：
    - factor_group_ic.png               — 三类因子（量价 / 工程化 / 合成）组别 |IC|
    - factor_top20_ic.png               — Top 20 单因子 |IC|
    - factor_ic_monthly_train.png       — 训练期月度均值 |IC| 走势 + ICIR

  章节 1.6 单品种回测（2 张，按 regime 命名）：
    - factor_top20_model_importance_low_vol.png  — LightGBM low_vol 模型 Top 20 Gain
    - factor_top20_model_importance_high_vol.png — LightGBM high_vol 模型 Top 20 Gain

  说明：章节 1.6 backtest_curve.png 已由 pipeline/backtest.py 自动写入 backtest_plot。
        本模块不重复生成。

接口设计：直接接受内存里的 PreparedData + artifact_map，不依赖任何磁盘硬编码路径。
失败时只打印警告，不抛出异常 —— 训练 / 回测主流程不会因为画图失败而中断。
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 必须在 import matplotlib 之前设置 MPLCONFIGDIR，避免污染 ~/.config/matplotlib
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str((_PROJECT_ROOT / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm


# ───────────────────────── 中金研报样式（统一调色板） ─────────────────────────

CICC_BLUE = "#1A5276"
CICC_RED = "#C0392B"
CICC_GRAY = "#85929E"
CICC_LBLUE = "#AED6F1"
CICC_LRED = "#F1948A"
CICC_ORANGE = "#D35400"
CICC_GREEN = "#1E8449"
CICC_PURPLE = "#7D3C98"

FONT_TITLE = dict(fontsize=11, fontweight="bold", color=CICC_BLUE)
FONT_AXIS = dict(fontsize=9, color="#333333")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})

# 35 个原始 ENG_* 工程化特征（合成因子扩展前）；其余 ENG_* 一律视为合成因子
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

GROUPS: dict[str, dict] = {
    "量价因子":   {"color": CICC_BLUE},
    "工程化特征": {"color": CICC_GREEN},
    "合成因子":   {"color": CICC_ORANGE},
    "中观因子":   {"color": CICC_RED},      # MID_*
    "中微交互":   {"color": CICC_PURPLE},   # MIDxMICRO_*（A1.2 引入的 mid × micro 显式乘积）
}


def _assign_group(feat: str) -> str:
    """图例分组：MID_ → 中观因子；MIDxMICRO_ → 中微交互；ENG_ 中 35 个原始项 → 工程化特征，
    其余 ENG_* → 合成因子；其它（runtime 量价因子等） → 量价因子。"""
    if feat.startswith("MIDxMICRO_"):
        return "中微交互"
    if feat.startswith("MID_"):
        return "中观因子"
    if feat.startswith("ENG_"):
        return "工程化特征" if feat in ORIGINAL_ENG_FEATURES else "合成因子"
    return "量价因子"


def _source(ax, y: float = -0.12) -> None:
    ax.annotate("资料来源：中金公司研究部", xy=(0, y), xycoords="axes fraction",
                fontsize=7.5, color=CICC_GRAY, ha="left")


def _save(fig: plt.Figure, out_path: Path, label: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [diag]{(' ' + label) if label else ''} → {out_path.name}")


# ════════════════════════════════════════════════════════════════════
#   PART A — 数据分布（章节 1.2）
# ════════════════════════════════════════════════════════════════════

DIST_FIG_SIZE = (7.5, 4.2)
DIST_N_BINS = 200


def _stats_box(ax, r: np.ndarray, extra_lines: list[str] | None = None) -> None:
    lines = [
        f"均值     {r.mean()*100:.2e}%",
        f"标准差  {r.std():.3f}%",
        f"峰度     {pd.Series(r).kurt():.1f}",
        f"p1/p99  {np.percentile(r,1):.3f}% / {np.percentile(r,99):.3f}%",
    ]
    if extra_lines:
        lines += extra_lines
    ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=CICC_GRAY, alpha=0.85))


def _own_dist_ymax(series_pct: np.ndarray, clip: float) -> float:
    """单图的 y 轴上限：取实际 hist 峰值与正态曲线峰值的较大者 *1.10。
    每张图独立计算，不再共享 —— 避免 1-bar 分布峰值过高把 future_return 压扁。
    """
    counts, _ = np.histogram(np.clip(series_pct, -clip, clip), bins=DIST_N_BINS, density=True)
    pk_normal = norm.pdf(series_pct.mean(), series_pct.mean(), series_pct.std())
    peak = max(float(counts.max()), float(pk_normal))
    return peak * 1.10


def _plot_dist_distribution(
    *,
    series: np.ndarray,
    clip: float,
    y_max: float,
    title: str,
    xlabel: str,
    out_path: Path,
    extra_stats: list[str] | None = None,
) -> None:
    r = series * 100  # 转 %
    fig, ax = plt.subplots(figsize=DIST_FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    r_clip = np.clip(r, -clip, clip)
    ax.hist(r_clip, bins=DIST_N_BINS, density=True, color=CICC_LBLUE,
            edgecolor="none", alpha=0.88, label="实际分布")
    mu, sigma = r.mean(), r.std()
    xg = np.linspace(-clip, clip, 500)
    ax.plot(xg, norm.pdf(xg, mu, sigma), color=CICC_RED, linewidth=1.5, linestyle="--",
            label="正态分布（同均值/方差）")
    ax.axvline(0, color=CICC_GRAY, linewidth=0.8, linestyle=":")

    _stats_box(ax, r, extra_lines=extra_stats)
    _source(ax)

    ax.set_xlim(-clip, clip)
    ax.set_ylim(0, y_max)
    if clip <= 0.30:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel(xlabel, **FONT_AXIS)
    ax.set_ylabel("概率密度", **FONT_AXIS)
    ax.set_title(title, **FONT_TITLE, pad=8)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, out_path, label="dist")


def _plot_vol_regime_distribution(
    *,
    full_data: pd.DataFrame,
    cutoff: float,
    out_path: Path,
) -> None:
    # 每日 daily_vol_20（取每个交易日内的最后一行）
    if "daily_vol_20" not in full_data.columns:
        print("  [diag] dist_vol_regime: daily_vol_20 不在 full_data 中，跳过")
        return
    daily = full_data.groupby("TRADE_DATE", as_index=False)["daily_vol_20"].last()
    v = daily["daily_vol_20"].dropna().values
    if v.size == 0:
        print("  [diag] dist_vol_regime: daily_vol_20 全 NaN，跳过")
        return

    fig, ax = plt.subplots(figsize=DIST_FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    upper = float(min(v.max(), max(0.045, cutoff * 2.5)))
    bins = np.linspace(float(v.min()), upper, 120)
    counts, edges = np.histogram(v, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(np.diff(edges)[0]) if len(edges) > 1 else 0.0
    low_mask = centers <= cutoff
    high_mask = centers > cutoff

    low_pct = float((v <= cutoff).mean() * 100)
    high_pct = float((v > cutoff).mean() * 100)
    ax.bar(centers[low_mask], counts[low_mask], width=width,
           color=CICC_LBLUE, edgecolor="none", alpha=0.88,
           label=f"低波动域（≈{low_pct:.0f}%样本）")
    ax.bar(centers[high_mask], counts[high_mask], width=width,
           color=CICC_LRED, edgecolor="none", alpha=0.88,
           label=f"高波动域（≈{high_pct:.0f}%样本）")
    ax.axvline(cutoff, color=CICC_RED, linewidth=1.8, linestyle="--",
               label=f"分域阈值 {cutoff:.4f}")

    if counts.size > 0:
        max_count = float(counts.max())
        ax.text(cutoff * 0.4, max_count * 0.82, f"低波动域\n{low_pct:.0f}%",
                fontsize=9, color=CICC_BLUE, ha="center", fontweight="bold")
        ax.text(cutoff * 1.7, max_count * 0.52, f"高波动域\n{high_pct:.0f}%",
                fontsize=9, color=CICC_RED, ha="center", fontweight="bold")

    stats_text = (
        f"均值     {v.mean():.4f}\n"
        f"中位数  {np.median(v):.4f}\n"
        f"p75/p90 {np.percentile(v, 75):.4f}/{np.percentile(v, 90):.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=CICC_GRAY, alpha=0.85))
    _source(ax)

    ax.set_xlim(0, upper)
    ax.set_xlabel("20日日波动率", **FONT_AXIS)
    ax.set_ylabel("概率密度", **FONT_AXIS)
    ax.set_title("20日日波动率分布与高低波分域阈值", **FONT_TITLE, pad=8)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, out_path, label="dist")


def _infer_bar_minutes(full_data: pd.DataFrame) -> int | None:
    """从 TDATE 推断 bar 时间间隔（分钟）。取连续两根 bar 时间差的众数，
    避免日切 / 跨夜断点污染。返回 None 表示无法推断。
    """
    if "TDATE" not in full_data.columns:
        return None
    t = pd.to_datetime(full_data["TDATE"], errors="coerce").dropna()
    if len(t) < 100:
        return None
    # 取前 N 根的相邻差，避开跨日间隙
    diffs = t.iloc[:5000].sort_values().diff().dropna()
    diffs_min = (diffs.dt.total_seconds() / 60.0).round()
    diffs_min = diffs_min[diffs_min > 0]
    if diffs_min.empty:
        return None
    mode = int(diffs_min.mode().iloc[0])
    return mode if mode > 0 else None


def _format_horizon_label(horizon: int, bar_minutes: int | None) -> str:
    """把 horizon (单位: bar) 转成人类可读的时间标签。
    horizon=60, bar=5min → '60根 K线（5 小时）'
    horizon=12, bar=15min → '12根 K线（3 小时）'
    bar=None → '60根 K线'
    """
    if bar_minutes is None or bar_minutes <= 0:
        return f"{horizon} 根 K 线"
    total_min = horizon * bar_minutes
    if total_min < 60:
        return f"{horizon} 根 K 线（{total_min} 分钟）"
    hours = total_min / 60
    if abs(hours - round(hours)) < 1e-6:
        return f"{horizon} 根 K 线（{int(round(hours))} 小时）"
    return f"{horizon} 根 K 线（{hours:.1f} 小时）"


def plot_data_distribution(*, prepared: Any, output_dir: Path) -> None:
    """3 张数据分布图。
    - dist_single_bar_return.png：来自 full_data 的 1-bar log return（LOGRET1 优先，
      退回 RET1 / 用 CLOSE 对数差现算）
    - dist_future_return.png：来自 full_data 的 future_return（horizon-bar 后向收益）
    - dist_vol_regime.png：daily_vol_20 + cutoff
    标题中的 bar 时间间隔从 TDATE 自动推断（5min / 15min / ...），不写死。
    """
    full = prepared.full_data
    # 1-bar log return
    if "LOGRET1" in full.columns:
        r1 = pd.to_numeric(full["LOGRET1"], errors="coerce").dropna().to_numpy()
    elif "RET1" in full.columns:
        r1 = pd.to_numeric(full["RET1"], errors="coerce").dropna().to_numpy()
    else:
        # 退回：用 CLOSE 即时算 log return（不持久化到 full）
        c = pd.to_numeric(full["CLOSE"], errors="coerce")
        r1 = np.log(c / c.shift(1)).dropna().to_numpy()

    if "future_return" not in full.columns:
        print("  [diag] dist: full_data 中没有 future_return，跳过")
        return
    r5 = pd.to_numeric(full["future_return"], errors="coerce").dropna().to_numpy()

    # 自适应 clip：用 p99 *1.5 作为窗口，避免少数极端值压扁分布
    clip_single = max(0.05, float(np.percentile(np.abs(r1), 99) * 100 * 1.5))  # %
    clip_future = max(0.10, float(np.percentile(np.abs(r5), 99) * 100 * 1.5))
    # 独立 y 轴：1-bar 分布峰值很高，会把 future_return 压扁到看不清，所以分别计算上限
    y_max_single = _own_dist_ymax(r1 * 100, clip_single)
    y_max_future = _own_dist_ymax(r5 * 100, clip_future)

    # 从数据动态推断 bar 间隔（分钟）；同时从 metadata 读 target_horizon（不写死 5）
    bar_minutes = _infer_bar_minutes(full)
    horizon = (
        int(prepared.metadata.get("target_horizon", 0))
        if isinstance(prepared.metadata, dict)
        else 0
    )
    # 兼容：metadata 没存就退回到从 future_return 长度反推（最后只能 1）
    if horizon <= 0:
        horizon = 1

    bar_label = f"{bar_minutes} 分钟" if bar_minutes else "K 线周期"
    horizon_label = _format_horizon_label(horizon, bar_minutes)

    _plot_dist_distribution(
        series=r1, clip=clip_single, y_max=y_max_single,
        title=f"单根 {bar_label} K 线对数收益率分布",
        xlabel=f"单根 {bar_label} K 线对数收益率（%）",
        out_path=output_dir / "data_dist_single_bar.png",
    )
    _plot_dist_distribution(
        series=r5, clip=clip_future, y_max=y_max_future,
        title=f"预测目标分布：未来 {horizon_label} 累积收益率",
        xlabel=f"未来 {horizon_label} 累积收益率（%）",
        out_path=output_dir / "data_dist_future_return.png",
    )

    cutoff = float(prepared.metadata.get("regime_cutoff", 0.0)) if isinstance(prepared.metadata, dict) else 0.0
    if cutoff > 0:
        _plot_vol_regime_distribution(full_data=full, cutoff=cutoff, out_path=output_dir / "data_dist_vol_regime.png")
    else:
        print("  [diag] data_dist_vol_regime: 未取到 regime_cutoff，跳过")


# ════════════════════════════════════════════════════════════════════
#   PART B — 因子 IC 分析（章节 1.3）
# ════════════════════════════════════════════════════════════════════

IC_FIG_SIZE = (8, 4.8)


def _compute_train_ic(prepared: Any) -> dict[str, float]:
    """在 train_data 上计算每个 feature 与 future_return 的 Pearson IC。"""
    train = prepared.train_data
    if "future_return" not in train.columns:
        return {}
    y = pd.to_numeric(train["future_return"], errors="coerce").dropna()
    if y.empty:
        return {}
    ic: dict[str, float] = {}
    for feat in prepared.feature_cols:
        if feat not in train.columns:
            continue
        x = pd.to_numeric(train[feat], errors="coerce").dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 1000:
            continue
        xi, yi = x.loc[idx], y.loc[idx]
        if xi.std() == 0 or yi.std() == 0:
            continue
        v = float(xi.corr(yi))
        if not np.isnan(v):
            ic[feat] = v
    return ic


def _compute_monthly_ic(prepared: Any, ic_dict: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """train_data 月份内的所有 feature |IC| 均值，按月聚合。"""
    train = prepared.train_data
    if "future_return" not in train.columns or "TDATE" not in train.columns:
        return [], np.array([])
    df = train.copy()
    df["TDATE"] = pd.to_datetime(df["TDATE"])
    df["__month__"] = df["TDATE"].dt.to_period("M")
    feat_cols = list(ic_dict.keys())
    months_out, vals_out = [], []
    for m, grp in df.groupby("__month__"):
        y = pd.to_numeric(grp["future_return"], errors="coerce").dropna()
        ics: list[float] = []
        for col in feat_cols:
            if col not in grp.columns:
                continue
            x = pd.to_numeric(grp[col], errors="coerce").dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 200:
                continue
            xi, yi = x.loc[idx], y.loc[idx]
            if xi.std() == 0 or yi.std() == 0:
                continue
            v = float(xi.corr(yi))
            if not np.isnan(v):
                ics.append(abs(v))
        if ics:
            months_out.append(str(m))
            vals_out.append(float(np.mean(ics)))
    return months_out, np.array(vals_out)


def _plot_group_ic(ic_dict: dict[str, float], out_path: Path) -> None:
    group_ics: dict[str, list[float]] = {g: [] for g in GROUPS}
    group_ics["其他"] = []
    for feat, ic in ic_dict.items():
        group_ics.setdefault(_assign_group(feat), []).append(abs(ic))
    group_mean = {g: float(np.mean(v)) for g, v in group_ics.items() if v}
    if not group_mean:
        return
    order = sorted(group_mean, key=lambda k: group_mean[k], reverse=True)
    vals = [group_mean[g] for g in order]
    colors = [GROUPS.get(g, {}).get("color", CICC_GRAY) for g in order]

    fig, ax = plt.subplots(figsize=IC_FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(order))
    bars = ax.barh(y, vals, 0.60, color=colors, alpha=0.88, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("组内因子平均 |IC|（训练集 Pearson 相关）", **FONT_AXIS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title("因子组别平均信息系数 |IC|", **FONT_TITLE, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    _source(ax)
    fig.tight_layout()
    _save(fig, out_path, label="ic")


def _plot_top20_ic(ic_dict: dict[str, float], out_path: Path) -> None:
    if not ic_dict:
        return
    top20 = sorted(ic_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    names = [n for n, _ in top20]
    vals = [abs(v) for _, v in top20]
    colors = [GROUPS.get(_assign_group(n), {}).get("color", CICC_GRAY) for n in names]

    fig, ax = plt.subplots(figsize=IC_FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(names))
    ax.barh(y, vals, 0.65, color=colors, alpha=0.88, edgecolor="none")
    for i, val in enumerate(vals):
        ax.text(val + max(vals) * 0.01, i, f"{val:.4f}",
                va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("|IC|（训练集 Pearson 相关）", **FONT_AXIS)
    ax.set_title("Top 20 单因子信息系数 |IC|", **FONT_TITLE, pad=8)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)

    seen, handles = {}, []
    for name, color in zip(names, colors):
        g = _assign_group(name)
        if g not in seen:
            seen[g] = color
            handles.append(mpatches.Patch(facecolor=color, alpha=0.88, label=g))
    if handles:
        ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)
    _source(ax)
    fig.tight_layout()
    _save(fig, out_path, label="ic")


def _plot_ic_monthly(months: list[str], ic_vals: np.ndarray, out_path: Path) -> None:
    if len(months) == 0 or ic_vals.size == 0:
        return
    mean_ic = float(ic_vals.mean())
    std_ic = float(ic_vals.std()) if ic_vals.size > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    pos_pct = float((ic_vals > 0).mean() * 100)
    cum_ic = np.cumsum(ic_vals) / np.arange(1, len(ic_vals) + 1)

    fig, ax1 = plt.subplots(figsize=IC_FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    x = np.arange(len(months))
    bar_colors = [CICC_LBLUE if v >= mean_ic else CICC_LRED for v in ic_vals]
    ax1.bar(x, ic_vals, width=0.6, color=bar_colors, edgecolor="none", alpha=0.88, label="月度均值|IC|")
    ax1.axhline(mean_ic, color=CICC_BLUE, linewidth=1.4, linestyle="-",
                label=f"训练期均值|IC|={mean_ic:.4f}")
    ax2 = ax1.twinx()
    ax2.plot(x, cum_ic, color=CICC_ORANGE, linewidth=1.8, linestyle="-",
             marker="o", markersize=3, label="累计均值|IC|（滚动）")
    ax2.set_ylabel("累计均值|IC|", fontsize=9, color=CICC_ORANGE)
    ax2.tick_params(axis="y", labelcolor=CICC_ORANGE, labelsize=8)
    ax2.set_ylim(0, float(cum_ic.max()) * 1.6 if cum_ic.size else 1.0)
    ax2.spines[["top"]].set_visible(False)

    step = max(1, len(months) // 9)
    xtick_pos = list(range(0, len(months), step))
    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels([months[i] for i in xtick_pos], rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("月度均值|IC|", **FONT_AXIS)
    ax1.set_ylim(0, float(ic_vals.max()) * 1.45)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", labelsize=8)

    stats_text = (
        f"均值|IC|  {mean_ic:.4f}\n"
        f"ICIR       {icir:.2f}\n"
        f"正向月数 {int(pos_pct/100*len(ic_vals))}/{len(ic_vals)}"
    )
    ax1.text(0.02, 0.96, stats_text, transform=ax1.transAxes,
             fontsize=8, va="top", ha="left", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=CICC_GRAY, alpha=0.85))
    ax1.set_title(
        f"训练期各月因子均值 |IC| 走势（共 {len(months)} 个月）",
        **FONT_TITLE, pad=8,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right", framealpha=0.9)
    ax1.annotate("资料来源：中金公司研究部", xy=(0, -0.16), xycoords="axes fraction",
                 fontsize=7.5, color=CICC_GRAY, ha="left")
    fig.tight_layout()
    _save(fig, out_path, label="ic")


def _plot_top20_model_importance(
    *,
    artifact_map: dict,
    regime_label: int,
    regime_name: str,
    out_path: Path,
) -> None:
    artifact = artifact_map.get(regime_label)
    if artifact is None or not hasattr(artifact, "feature_importance"):
        print(f"  [diag] {regime_name} 模型 importance: artifact 缺失，跳过")
        return
    df_imp = artifact.feature_importance
    if df_imp is None or len(df_imp) == 0 or "importance_gain" not in df_imp.columns:
        print(f"  [diag] {regime_name} 模型 importance: 数据为空，跳过")
        return

    top20 = df_imp.sort_values("importance_gain", ascending=False).head(20)
    names = top20["feature"].tolist()
    gains = top20["importance_gain"].astype(float).tolist()
    splits = top20["importance_split"].astype(int).tolist() if "importance_split" in top20 else [0] * len(names)
    bar_colors = [GROUPS.get(_assign_group(n), {}).get("color", CICC_GRAY) for n in names]
    total_gain = float(df_imp["importance_gain"].sum())
    top20_share = sum(gains) / total_gain * 100 if total_gain > 0 else 0.0

    fig, ax = plt.subplots(figsize=IC_FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(names))
    ax.barh(y, gains, 0.65, color=bar_colors, alpha=0.88, edgecolor="none")
    for i, (g, sp) in enumerate(zip(gains, splits)):
        ax.text(g + max(gains) * 0.01, i, f"{int(g):,} (split={sp})",
                va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Gain增益（LightGBM 特征重要性，越高表示模型越依赖该特征）", **FONT_AXIS)
    ax.set_title(
        f"LightGBM {regime_name} 模型 Top 20 特征 Gain 重要性",
        **FONT_TITLE, pad=8,
    )
    ax.set_xlim(0, max(gains) * 1.30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    seen, handles = {}, []
    for name, color in zip(names, bar_colors):
        g = _assign_group(name)
        if g not in seen:
            seen[g] = color
            handles.append(mpatches.Patch(facecolor=color, alpha=0.88, label=g))
    if handles:
        ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)
    _source(ax)
    fig.tight_layout()
    _save(fig, out_path, label="importance")


def plot_factor_analysis(*, prepared: Any, artifact_map: dict, output_dir: Path) -> None:
    """4 张因子 / 模型分析图：组别 IC + Top 20 IC + 月度 IC + 两个 regime 模型 importance。"""
    ic_dict = _compute_train_ic(prepared)
    if ic_dict:
        _plot_group_ic(ic_dict, output_dir / "factor_group_ic.png")
        _plot_top20_ic(ic_dict, output_dir / "factor_top20_ic.png")
        months, monthly_ic = _compute_monthly_ic(prepared, ic_dict)
        _plot_ic_monthly(months, monthly_ic, output_dir / "factor_ic_monthly_train.png")
    else:
        print("  [diag] 因子 IC: train_data 中样本不足，跳过 IC 系列图")

    # 高低波两个 regime 的 importance 图各画一张，用 _low_vol / _high_vol 后缀区分。
    for label, name in [(-1, "low_vol"), (1, "high_vol")]:
        _plot_top20_model_importance(
            artifact_map=artifact_map,
            regime_label=label,
            regime_name=name,
            out_path=output_dir / f"factor_top20_model_importance_{name}.png",
        )


# ════════════════════════════════════════════════════════════════════
#   入口
# ════════════════════════════════════════════════════════════════════


def generate_diagnostic_charts(
    *,
    prepared: Any,
    artifact_map: dict,
    output_dir: Path | str,
) -> None:
    """生成全部诊断图（数据分布 3 张 + 因子 IC 3 张 + 模型 importance 2 张 = 共 8 张）。

    所有失败被吞下转为告警 —— 训练 / 回测主流程不应因画图失败而中断。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] generating diagnostic charts → {output_dir}")
    try:
        plot_data_distribution(prepared=prepared, output_dir=output_dir)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"[diag] data distribution failed: {type(exc).__name__}: {exc}",
                      RuntimeWarning, stacklevel=2)
    try:
        plot_factor_analysis(prepared=prepared, artifact_map=artifact_map, output_dir=output_dir)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"[diag] factor analysis failed: {type(exc).__name__}: {exc}",
                      RuntimeWarning, stacklevel=2)
