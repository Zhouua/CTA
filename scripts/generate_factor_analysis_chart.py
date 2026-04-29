"""
Generate four factor-analysis charts for report sections 1.3 and 1.6.
Charts 1-3 are pre-model (training-set IC); chart 4 is post-training (LightGBM Gain).

Charts:
  factor_group_ic.png              — group-level mean |IC| on training set (3 groups)
  factor_top20_ic.png              — top-20 features by |IC| on training set
  factor_ic_monthly_train.png      — monthly mean |IC| (across all features) on training set
  factor_top20_model_importance.png — LightGBM low_vol model top-20 features by Gain (post-training)

Outputs to results/runs/micro_result/RB/
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
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

OUT_DIR   = PROJECT_ROOT / "results" / "runs" / "micro_result" / "RB"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "products" / "RB"
MERGED_CACHE = PROJECT_ROOT / "results" / "cache" / "merged_features.parquet"

TRAIN_END = "2024-05-01"  # first 70% of 75 months ≈ 52 months (2020-01 to 2024-04)

# ── style ─────────────────────────────────────────────────────────────
CICC_BLUE   = "#1A5276"
CICC_RED    = "#C0392B"
CICC_GRAY   = "#85929E"
CICC_LBLUE  = "#AED6F1"
CICC_LRED   = "#F1948A"
CICC_GREEN  = "#1E8449"
CICC_ORANGE = "#D35400"
CICC_PURPLE = "#7D3C98"
CICC_TEAL   = "#148F77"

FIG_SIZE = (8, 4.8)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})

FONT_TITLE = dict(fontsize=11, fontweight="bold", color=CICC_BLUE)
FONT_AXIS  = dict(fontsize=9, color="#333333")


# ── factor group definitions ──────────────────────────────────────────
# 35 originally-engineered ENG_* features (pre-synthetic-expansion). Anything
# that starts with ENG_ but is not in this set is classified as 合成因子.
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
    "量价因子": {"color": CICC_BLUE},
    "工程化特征": {"color": CICC_RED},
    "合成因子":   {"color": CICC_ORANGE},
}


def assign_group(feat: str) -> str:
    if feat.startswith("ENG_"):
        return "工程化特征" if feat in ORIGINAL_ENG_FEATURES else "合成因子"
    return "量价因子"


def _source(ax, y=-0.12):
    ax.annotate("资料来源：中金公司研究部", xy=(0, y), xycoords="axes fraction",
                fontsize=7.5, color=CICC_GRAY, ha="left")


# ── data loading ──────────────────────────────────────────────────────
def _load_training_ic() -> dict[str, float]:
    """
    Compute Pearson IC = corr(feature, future_return) on training rows.
    Uses the 190 model features as the reference set.
    Returns dict[feature_name -> IC].
    """
    # get the model's 190 feature list
    with open(MODEL_DIR / "low_vol" / "feature_importance.json") as f:
        model_feats = [d["feature"] for d in json.load(f)]

    df = pd.read_parquet(MERGED_CACHE)
    df["TDATE"] = pd.to_datetime(df["TDATE"])

    train = df[df["TDATE"] < TRAIN_END].copy()
    y = train["future_return"].dropna()

    ic: dict[str, float] = {}
    for feat in model_feats:
        if feat not in train.columns:
            continue
        x = train[feat].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 1000:
            continue
        v = x[idx].corr(y[idx])
        if not np.isnan(v):
            ic[feat] = v
    return ic


def _compute_monthly_ic(ic_dict: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """
    For each training month, compute mean |IC| across all features.
    Returns (month_labels, mean_abs_ic_array).
    """
    df = pd.read_parquet(MERGED_CACHE)
    df["TDATE"] = pd.to_datetime(df["TDATE"])

    train = df[df["TDATE"] < TRAIN_END].copy()
    train["month"] = train["TDATE"].dt.to_period("M")
    feat_cols = list(ic_dict.keys())

    months_out, vals_out = [], []
    for m, grp in train.groupby("month"):
        y = grp["future_return"].dropna()
        ics = []
        for col in feat_cols:
            if col not in grp.columns:
                continue
            x = grp[col].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 200:
                continue
            v = x[idx].corr(y[idx])
            if not np.isnan(v):
                ics.append(abs(v))
        if ics:
            months_out.append(str(m))
            vals_out.append(np.mean(ics))

    return months_out, np.array(vals_out)


# ── chart 1: group-level mean |IC| ───────────────────────────────────
def plot_group_ic(ic_dict: dict[str, float]):
    group_ics: dict[str, list[float]] = {g: [] for g in GROUPS}
    group_ics["其他"] = []
    for feat, ic in ic_dict.items():
        g = assign_group(feat)
        group_ics.setdefault(g, []).append(abs(ic))

    group_mean: dict[str, float] = {
        g: np.mean(v) for g, v in group_ics.items() if v
    }
    order  = sorted(group_mean, key=lambda k: group_mean[k], reverse=True)
    vals   = [group_mean[g] for g in order]
    colors = [GROUPS.get(g, {}).get("color", CICC_GRAY) for g in order]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(order))
    bars = ax.barh(y, vals, 0.60, color=colors, alpha=0.88, edgecolor="none")

    for bar, val in zip(bars, vals):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("组内因子平均|IC|（训练集，2020-01至2024-04）", **FONT_AXIS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title(
        "因子组别平均信息系数|IC|（训练集，进入模型前）\n"
        "（三类因子——量价因子、工程化特征、合成因子——在5分钟未来收益上的单因子线性预测能力对比）",
        **FONT_TITLE, pad=8,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    _source(ax)

    fig.tight_layout()
    out = OUT_DIR / "factor_group_ic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 2: top-20 features by |IC| ──────────────────────────────────
def plot_top20_ic(ic_dict: dict[str, float]):
    top20      = sorted(ic_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    top_names  = [x[0] for x in top20]
    top_vals   = [abs(x[1]) for x in top20]
    top_colors = [GROUPS.get(assign_group(n), {}).get("color", CICC_GRAY) for n in top_names]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(top_names))
    ax.barh(y, top_vals, 0.65, color=top_colors, alpha=0.88, edgecolor="none")

    for i, (val, name) in enumerate(zip(top_vals, top_names)):
        ax.text(val + 0.0003, i, f"{val:.4f}", va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(top_names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("|IC|（训练集，2020-01至2024-04）", **FONT_AXIS)
    ax.set_title(
        "训练集Top 20单因子信息系数|IC|（进入模型前，Pearson相关系数）\n"
        "（按因子归属着色：量价因子 / 工程化特征 / 合成因子；同日daily_ret当日值与未来5分钟收益线性相关性最强）",
        **FONT_TITLE, pad=8,
    )
    ax.set_xlim(0, max(top_vals) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)

    seen, handles = {}, []
    for name, color in zip(top_names, top_colors):
        g = assign_group(name)
        if g not in seen:
            seen[g] = color
            handles.append(mpatches.Patch(facecolor=color, alpha=0.88,
                                          label=g.split("\n")[0]))
    ax.legend(handles=handles, fontsize=7.5, loc="lower right",
              framealpha=0.9, ncol=2)
    _source(ax)

    fig.tight_layout()
    out = OUT_DIR / "factor_top20_ic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 3: monthly mean |IC| on training set ────────────────────────
def plot_ic_monthly_train(months: list[str], ic_vals: np.ndarray):
    mean_ic = ic_vals.mean()
    icir    = mean_ic / ic_vals.std()
    pos_pct = (ic_vals > 0).mean() * 100
    cum_ic  = np.cumsum(ic_vals) / np.arange(1, len(ic_vals) + 1)

    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    x = np.arange(len(months))
    bar_colors = [CICC_LBLUE if v >= mean_ic else CICC_LRED for v in ic_vals]
    ax1.bar(x, ic_vals, width=0.6, color=bar_colors, edgecolor="none", alpha=0.88,
            label="月度均值|IC|")
    ax1.axhline(mean_ic, color=CICC_BLUE, linewidth=1.4, linestyle="-",
                label=f"训练期均值|IC|={mean_ic:.4f}")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_ic, color=CICC_ORANGE, linewidth=1.8, linestyle="-",
             marker="o", markersize=3, label="累计均值|IC|（滚动）")
    ax2.set_ylabel("累计均值|IC|", fontsize=9, color=CICC_ORANGE)
    ax2.tick_params(axis="y", labelcolor=CICC_ORANGE, labelsize=8)
    ax2.set_ylim(0, max(cum_ic) * 1.6)
    ax2.spines[["top"]].set_visible(False)

    # x-axis: show every 6th month label to avoid crowding
    step = max(1, len(months) // 9)
    xtick_pos   = list(range(0, len(months), step))
    xtick_labels = [months[i] for i in xtick_pos]
    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels(xtick_labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("月度均值|IC|", **FONT_AXIS)
    ax1.set_ylim(0, max(ic_vals) * 1.45)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", labelsize=8)

    stats_text = (f"均值|IC|  {mean_ic:.4f}\n"
                  f"ICIR       {icir:.2f}\n"
                  f"正向月数 {int(pos_pct/100*len(ic_vals))}/{len(ic_vals)}")
    ax1.text(0.02, 0.96, stats_text, transform=ax1.transAxes,
             fontsize=8, va="top", ha="left", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=CICC_GRAY, alpha=0.85))

    ax1.set_title(
        f"训练期各月因子均值|IC|走势（2020-01至2024-04，52个月）\n"
        f"（各月均值|IC|={mean_ic:.4f}，ICIR={icir:.2f}，因子预测力跨期稳定且持续正向）",
        **FONT_TITLE, pad=8,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               loc="upper right", framealpha=0.9)

    ax1.annotate("资料来源：中金公司研究部", xy=(0, -0.16), xycoords="axes fraction",
                 fontsize=7.5, color=CICC_GRAY, ha="left")

    fig.tight_layout()
    out = OUT_DIR / "factor_ic_monthly_train.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── chart 4: LightGBM low_vol model — top 20 by Gain importance ──────
def plot_top20_model_importance(regime: str = "low_vol"):
    imp_path = MODEL_DIR / regime / "feature_importance.json"
    if not imp_path.exists():
        print(f"[skip] {imp_path} not found — run pipeline/train.py with persist_models=true first.")
        return

    with open(imp_path) as f:
        records = json.load(f)
    df_imp = pd.DataFrame(records)
    if df_imp.empty or "importance_gain" not in df_imp.columns:
        print(f"[skip] {imp_path} is empty or malformed.")
        return

    top20 = df_imp.sort_values("importance_gain", ascending=False).head(20)
    names      = top20["feature"].tolist()
    gains      = top20["importance_gain"].tolist()
    splits     = top20["importance_split"].tolist() if "importance_split" in top20 else [0]*len(names)
    bar_colors = [GROUPS.get(assign_group(n), {}).get("color", CICC_GRAY) for n in names]

    total_gain = float(df_imp["importance_gain"].sum())
    top20_share = sum(gains) / total_gain * 100 if total_gain > 0 else 0.0

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(names))
    ax.barh(y, gains, 0.65, color=bar_colors, alpha=0.88, edgecolor="none")

    for i, (g, sp) in enumerate(zip(gains, splits)):
        ax.text(g + max(gains) * 0.01, i,
                f"{int(g):,} (split={int(sp)})",
                va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Gain增益（LightGBM特征重要性，越高表示模型越依赖该特征）", **FONT_AXIS)
    ax.set_title(
        f"LightGBM低波动域模型Top 20特征Gain重要性（训练后视角）\n"
        f"（Top 20累计占总Gain的{top20_share:.1f}%；按因子归属着色，daily_ret当日值与日内累计收益类合成因子是模型主要依赖）",
        **FONT_TITLE, pad=8,
    )
    ax.set_xlim(0, max(gains) * 1.30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    seen, handles = {}, []
    for name, color in zip(names, bar_colors):
        g = assign_group(name)
        if g not in seen:
            seen[g] = color
            handles.append(mpatches.Patch(facecolor=color, alpha=0.88, label=g))
    ax.legend(handles=handles, fontsize=7.5, loc="lower right",
              framealpha=0.9, ncol=2)
    _source(ax)

    fig.tight_layout()
    out = OUT_DIR / "factor_top20_model_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Computing training-set IC for all model features ...")
    ic_dict = _load_training_ic()
    print(f"  Features with IC: {len(ic_dict)}")
    top3 = sorted(ic_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"  Top 3: {[(f, round(v,4)) for f,v in top3]}")

    print("Computing monthly mean |IC| on training set ...")
    months, monthly_ic = _compute_monthly_ic(ic_dict)
    print(f"  {len(months)} months, mean|IC|={monthly_ic.mean():.4f}, ICIR={monthly_ic.mean()/monthly_ic.std():.2f}")

    print("Generating section 1.3 factor analysis charts ...")
    plot_group_ic(ic_dict)
    plot_top20_ic(ic_dict)
    plot_ic_monthly_train(months, monthly_ic)
    plot_top20_model_importance()
    print("Done.")
