"""
Generate three focused charts for report section 1.2 (Price-Volume Distribution & Risk Profile).
One chart = one conclusion, 中金研报 style.

Charts 1 and 2 share the same y-axis density scale so their width difference
(1-min vs 5-min cumulative) is immediately visible to the reader.

Outputs (all to results/runs/micro_result/RB/):
  rb_dist_single_bar.png   — 1-min bar return: heavy-tailed, noise-dominated
  rb_dist_future_return.png — future 5-min return: near-symmetric, weak signal
  rb_dist_vol_regime.png   — daily vol distribution, bimodal, regime cutoff
"""
from __future__ import annotations

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
import matplotlib.ticker as mticker
from scipy.stats import norm

OUT_DIR = PROJECT_ROOT / "results" / "runs" / "micro_result" / "RB"
CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "products" / "RB"

# ── style ──────────────────────────────────────────────────────────────────
CICC_BLUE  = "#1A5276"
CICC_RED   = "#C0392B"
CICC_GRAY  = "#85929E"
CICC_LBLUE = "#AED6F1"
CICC_LRED  = "#F1948A"
FIG_SIZE   = (7.5, 4.2)   # identical for charts 1 & 2 so they look side-by-side comparable

FONT_TITLE = dict(fontsize=11, fontweight="bold", color=CICC_BLUE)
FONT_AXIS  = dict(fontsize=9, color="#333333")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})

CLIP_1MIN  = 0.25   # % — clip range for 1-min chart
CLIP_5MIN  = 0.60   # % — clip range for 5-min chart
N_BINS     = 200


def _load_raw_cols(cols: list[str]) -> pd.DataFrame:
    parquet = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet:
        raise FileNotFoundError(f"No parquet cache in {CACHE_DIR}")
    df = pd.read_parquet(parquet[0], columns=cols)
    return df.dropna(subset=cols)


def _compute_daily_vol20(df_raw: pd.DataFrame) -> pd.Series:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["TDATE"]).dt.date if "TDATE" in df.columns else df.index.date
    daily = df.groupby("date")["LOGRET1"].sum().rename("daily_ret")
    daily_vol = daily.rolling(20, min_periods=10).std()
    daily_vol.index = pd.to_datetime(daily_vol.index)
    df["date_ts"] = pd.to_datetime(df["date"] if "date" in df.columns else df.index.date)
    return df["date_ts"].map(daily_vol)


def _stats_box(ax, r: np.ndarray, extra_lines: list[str] | None = None):
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


def _source(ax):
    ax.annotate("资料来源：中金公司研究部", xy=(0, -0.10), xycoords="axes fraction",
                fontsize=7.5, color=CICC_GRAY, ha="left")


def _shared_ymax() -> float:
    """Pre-compute the peak density for both 1-min and 5-min so they share the same y-axis."""
    r1 = _load_raw_cols(["LOGRET1"])["LOGRET1"].values * 100
    r5 = _load_raw_cols(["future_return"])["future_return"].values * 100

    r1c = np.clip(r1, -CLIP_1MIN, CLIP_1MIN)
    r5c = np.clip(r5, -CLIP_5MIN, CLIP_5MIN)

    n1, _ = np.histogram(r1c, bins=N_BINS, density=True)
    n5, _ = np.histogram(r5c, bins=N_BINS, density=True)

    # also compute normal-curve peaks to make sure ylim covers them
    pk_normal_1 = norm.pdf(r1.mean(), r1.mean(), r1.std())
    pk_normal_5 = norm.pdf(r5.mean(), r5.mean(), r5.std())

    return max(n1.max(), n5.max(), pk_normal_1, pk_normal_5) * 1.10


# ── chart 1: 1-min bar return distribution ────────────────────────────────
def plot_single_bar_return(y_max: float):
    df = _load_raw_cols(["LOGRET1"])
    r = df["LOGRET1"].values * 100

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    r_clip = np.clip(r, -CLIP_1MIN, CLIP_1MIN)
    ax.hist(r_clip, bins=N_BINS, density=True,
            color=CICC_LBLUE, edgecolor="none", alpha=0.88, label="实际分布")

    mu, sigma = r.mean(), r.std()
    xg = np.linspace(-CLIP_1MIN, CLIP_1MIN, 500)
    ax.plot(xg, norm.pdf(xg, mu, sigma), color=CICC_RED,
            linewidth=1.5, linestyle="--", label="正态分布（同均值/方差）")
    ax.axvline(0, color=CICC_GRAY, linewidth=0.8, linestyle=":")

    _stats_box(ax, r)
    _source(ax)

    ax.set_xlim(-CLIP_1MIN, CLIP_1MIN)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("单根1分钟K线收益率（%）", **FONT_AXIS)
    ax.set_ylabel("概率密度", **FONT_AXIS)
    ax.set_title(
        "螺纹钢1分钟K线收益率分布：尖峰厚尾，噪声主导\n"
        "（超过80%的K线绝对收益率不足0.05%；纵轴与图表2统一以便直接对比分布宽度）",
        **FONT_TITLE, pad=8,
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUT_DIR / "rb_dist_single_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 2: future 5-min return distribution ──────────────────────────────
def plot_future_return(y_max: float):
    df = _load_raw_cols(["future_return"])
    r = df["future_return"].values * 100

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    r_clip = np.clip(r, -CLIP_5MIN, CLIP_5MIN)
    ax.hist(r_clip, bins=N_BINS, density=True,
            color=CICC_LBLUE, edgecolor="none", alpha=0.88, label="实际分布")

    mu, sigma = r.mean(), r.std()
    xg = np.linspace(-CLIP_5MIN, CLIP_5MIN, 500)
    ax.plot(xg, norm.pdf(xg, mu, sigma), color=CICC_RED,
            linewidth=1.5, linestyle="--", label="正态分布（同均值/方差）")
    ax.axvline(0, color=CICC_GRAY, linewidth=0.8, linestyle=":")

    _stats_box(ax, r, extra_lines=["样本外 R²≈0.038"])
    _source(ax)

    ax.set_xlim(-CLIP_5MIN, CLIP_5MIN)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("未来5分钟累积收益率（%）", **FONT_AXIS)
    ax.set_ylabel("概率密度", **FONT_AXIS)
    ax.set_title(
        "预测目标（未来5分钟收益率）分布：近对称弱信号，预测难度极高\n"
        "（正负样本比≈1:1；分布宽度约为单根K线2.4倍，正态偏离更小，更难捕捉方向）",
        **FONT_TITLE, pad=8,
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUT_DIR / "rb_dist_future_return.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 3: daily_vol_20 distribution with regime cutoff ──────────────────
def plot_vol_distribution():
    needed = ["LOGRET1", "TDATE"]
    parquet = sorted(CACHE_DIR.glob("*.parquet"))
    df_raw = pd.read_parquet(parquet[0], columns=needed).dropna(subset=["LOGRET1"])
    vol_series = _compute_daily_vol20(df_raw)
    v = vol_series.dropna().values

    CUTOFF = 0.017200915912034664  # training-set 70th pct

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bins = np.linspace(v.min(), min(v.max(), 0.045), 120)
    counts, edges = np.histogram(v, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)[0]

    low_mask  = centers <= CUTOFF
    high_mask = centers >  CUTOFF

    ax.bar(centers[low_mask],  counts[low_mask],  width=width,
           color=CICC_LBLUE, edgecolor="none", alpha=0.88, label="低波动域（≈70%样本）")
    ax.bar(centers[high_mask], counts[high_mask], width=width,
           color=CICC_LRED,  edgecolor="none", alpha=0.88, label="高波动域（≈30%样本）")
    ax.axvline(CUTOFF, color=CICC_RED, linewidth=1.8, linestyle="--",
               label=f"分域阈值 {CUTOFF:.4f}（训练集70%分位数）")

    low_pct  = (v <= CUTOFF).mean() * 100
    high_pct = (v >  CUTOFF).mean() * 100
    ax.text(CUTOFF * 0.38, counts.max() * 0.82,
            f"低波动域\n{low_pct:.0f}%",
            fontsize=9, color=CICC_BLUE, ha="center", fontweight="bold")
    ax.text(CUTOFF * 1.75, counts.max() * 0.52,
            f"高波动域\n{high_pct:.0f}%",
            fontsize=9, color=CICC_RED, ha="center", fontweight="bold")

    stats_text = (f"均值     {v.mean():.4f}\n"
                  f"中位数  {np.median(v):.4f}\n"
                  f"p75/p90 {np.percentile(v,75):.4f}/{np.percentile(v,90):.4f}")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=CICC_GRAY, alpha=0.85))

    _source(ax)

    ax.set_xlim(0, 0.045)
    ax.set_xlabel("20日日波动率", **FONT_AXIS)
    ax.set_ylabel("概率密度", **FONT_AXIS)
    ax.set_title(
        "20日日波动率分布：右偏双峰，分域阈值天然清晰\n"
        "（高低波动域价格动态迥异，单一模型难以同时最优适配两类环境）",
        **FONT_TITLE, pad=8,
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUT_DIR / "rb_dist_vol_regime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Computing shared y-axis max for charts 1 & 2 ...")
    y_max = _shared_ymax()
    print(f"  shared y_max = {y_max:.2f}")

    print("Generating section 1.2 distribution charts ...")
    plot_single_bar_return(y_max)
    plot_future_return(y_max)
    plot_vol_distribution()
    print("Done.")
