"""
backtest_macro.py
=================
宏观过滤型回测脚本，与 backtest.py 完全隔离（不修改任何原有文件）。

思路：
    1. 复用已训练的双 regime LightGBM 模型，在测试集上生成原始仓位（Benchmark）
    2. 用 judge_macro.build_monthly_regime() 判断每个月宏观是否强、库存周期是否强
    3. 若某月两者均为 False → 当月所有 5min bar 强制空仓
       否则 → 保留原始 CTA 仓位（策略正常运行）
    4. 分别计算 Benchmark / Macro-Filtered 的 P&L，输出对比报告

运行：
    python code/backtest_macro.py [--config config.yaml]

输出目录：results/backtest_macro/
    macro_backtest_report.png   - 对比图报告
    macro_backtest_summary.json - 量化指标对比
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 路径初始化（与其他模块保持一致）───────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
for _p in [str(CURRENT_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 复用原有基础设施（数据加载、模型推断）────────────────────────────────────
from config_utils import get_section, load_project_config, resolve_paths
from dataset import prepare_data
from modeling import load_dual_regime_models, predict_dual_regime

# ── 复用原有回测工具函数（纯函数，不修改原文件）──────────────────────────────
from backtest import (
    build_signal_rule_map,
    generate_positions,
    calc_pnl,
    performance_summary,
    build_monthly_returns,
    build_backtest_settings,
)

# ── 宏观判断模块（本次新增）─────────────────────────────────────────────────
from judge_macro import build_monthly_regime


MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_native(value: Any) -> Any:
    """将 numpy/pandas 标量转换为 Python 原生类型，供 JSON 序列化。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def apply_macro_filter(
    position_df: pd.DataFrame,
    monthly_regime: pd.DataFrame,
) -> pd.DataFrame:
    """
    将月度宏观 regime 映射到 5min bar 级别，对不允许开仓的月份强制 position=0。

    Parameters
    ----------
    position_df    : 已有仓位序列（含 TRADE_DATE 列，每行为一个 5min bar）
    monthly_regime : judge_macro.build_monthly_regime() 的输出，月初日期为索引

    Returns
    -------
    position_df 的副本，新增 is_macro_strong / is_inventory_strong / open_position 列，
    并在 open_position=False 的月份将 position 覆写为 0。
    """
    df = position_df.copy()
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])

    # 将每个 bar 的 TRADE_DATE 映射到对应月初（用于 join）
    df["_month_key"] = df["TRADE_DATE"].dt.to_period("M").dt.to_timestamp()

    # 将 monthly_regime 索引也统一为月初
    regime_lookup = monthly_regime.copy()
    regime_lookup.index = pd.to_datetime(regime_lookup.index).to_period("M").to_timestamp()

    # 左连接：TRADE_DATE 所在月份 → regime 信号
    df = df.merge(
        regime_lookup[["is_macro_strong", "is_inventory_strong", "open_position"]],
        left_on="_month_key",
        right_index=True,
        how="left",
    )
    df.drop(columns=["_month_key"], inplace=True)

    # 找不到对应月份的（数据开头）默认视为不开仓，避免意外风险
    df["open_position"] = df["open_position"].fillna(False)
    df["is_macro_strong"] = df["is_macro_strong"].fillna(False)
    df["is_inventory_strong"] = df["is_inventory_strong"].fillna(False)

    # 核心过滤：open_position=False 时强制空仓
    df.loc[~df["open_position"], "position"] = 0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def _add_regime_background(ax, daily_df: pd.DataFrame) -> None:
    """
    在 daily_df（含 open_position 列）基础上，给 NAV 图添加背景色：
      - 绿色: open_position=True（策略运行中）
      - 红色: open_position=False（宏观过滤，空仓）
    连续区间合并为一个 axvspan 减少重叠绘制。
    """
    if "open_position" not in daily_df.columns or daily_df.empty:
        return

    dates = pd.to_datetime(daily_df["TRADE_DATE"])
    flags = daily_df["open_position"].values

    color_open = "#4daf4a"   # 绿
    color_flat = "#e41a1c"   # 红

    # 合并连续相同区间
    prev_flag = None
    seg_start = None
    for date, flag in zip(dates, flags):
        if prev_flag is None:
            prev_flag = flag
            seg_start = date
        elif flag != prev_flag:
            color = color_open if prev_flag else color_flat
            ax.axvspan(seg_start, date, color=color, alpha=0.08, linewidth=0)
            prev_flag = flag
            seg_start = date
    if seg_start is not None:
        color = color_open if prev_flag else color_flat
        ax.axvspan(seg_start, dates.iloc[-1] + pd.Timedelta(days=1), color=color, alpha=0.08, linewidth=0)


def _plot_nav(
    ax,
    benchmark_daily: pd.DataFrame,
    filtered_daily: pd.DataFrame,
    regime_daily: pd.DataFrame,
) -> None:
    """
    绘制 NAV 曲线：benchmark（灰色）vs macro-filtered（深蓝色），
    背景色标识开仓/空仓 regime。
    始终使用线性坐标。
    """
    import matplotlib.dates as mdates

    _add_regime_background(ax, regime_daily)

    ax.plot(
        benchmark_daily["TRADE_DATE"],
        benchmark_daily["nav_net"],
        color="#888888",
        linewidth=1.4,
        linestyle="--",
        label="Benchmark (model always on)",
        alpha=0.85,
    )
    ax.plot(
        filtered_daily["TRADE_DATE"],
        filtered_daily["nav_net"],
        color="#1a6faf",
        linewidth=1.6,
        label="Macro-Filtered Strategy",
    )

    ax.set_ylabel("Net NAV")

    ax.set_title("NAV Curve: Benchmark vs Macro-Filtered", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.22)

    from matplotlib.patches import Patch
    bg_handles = [
        Patch(facecolor="#4daf4a", alpha=0.25, label="Position open (macro OK)"),
        Patch(facecolor="#e41a1c", alpha=0.25, label="Force flat (macro weak)"),
    ]
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + bg_handles,
        labels=ax.get_legend_handles_labels()[1] + [h.get_label() for h in bg_handles],
        loc="upper left", fontsize=8.5,
    )


def _plot_macro_signal_bars(
    ax,
    monthly_regime: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> None:
    """
    每个月绘制两根小柱：
        左柱（蓝色）: is_macro_strong
        右柱（橙色）: is_inventory_strong
    柱高 = 1（信号激活）或 0（未激活，不画），y 轴 [0, 1.3]。
    仅显示测试期内的月份。
    """
    import matplotlib.dates as mdates

    regime = monthly_regime.copy()
    regime.index = pd.to_datetime(regime.index)
    # 只展示测试区间
    regime = regime.loc[(regime.index >= test_start) & (regime.index <= test_end)]

    if regime.empty:
        ax.set_axis_off()
        return

    bar_width = pd.Timedelta(days=8)   # 每根柱约 8 天宽
    offset = pd.Timedelta(days=5)      # 两柱之间偏移

    dates = regime.index
    macro_vals = regime["is_macro_strong"].astype(float).values
    inv_vals = regime["is_inventory_strong"].astype(float).values

    # 左柱：宏观强
    ax.bar(
        dates - offset,
        macro_vals,
        width=bar_width,
        color="#1a6faf",
        alpha=0.80,
        label="Macro strong (PPI>0 & PMI≥49.5)",
    )
    # 右柱：库存周期强
    ax.bar(
        dates + offset,
        inv_vals,
        width=bar_width,
        color="#e6891a",
        alpha=0.80,
        label="Inventory cycle strong",
    )

    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Off", "On"], fontsize=8)
    ax.set_title("Monthly Macro Signals", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", alpha=0.22)


def _plot_drawdown(
    ax,
    benchmark_daily: pd.DataFrame,
    filtered_daily: pd.DataFrame,
) -> None:
    """回撤对比：benchmark（灰色）vs filtered（蓝色填充）。"""
    import matplotlib.dates as mdates

    ax.fill_between(
        filtered_daily["TRADE_DATE"],
        filtered_daily["net_drawdown"],
        0.0,
        color="#1a6faf",
        alpha=0.25,
        label="Filtered drawdown",
    )
    ax.plot(
        filtered_daily["TRADE_DATE"],
        filtered_daily["net_drawdown"],
        color="#1a6faf",
        linewidth=1.1,
    )
    ax.plot(
        benchmark_daily["TRADE_DATE"],
        benchmark_daily["net_drawdown"],
        color="#888888",
        linewidth=1.1,
        linestyle="--",
        label="Benchmark drawdown",
        alpha=0.85,
    )

    ax.set_title("Drawdown Comparison", fontsize=11)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.22)


def _plot_monthly_returns(
    ax,
    benchmark_daily: pd.DataFrame,
    filtered_daily: pd.DataFrame,
) -> None:
    """
    每月净收益率柱状图对比：
        benchmark 半透明灰色柱 + filtered 蓝色柱，并排显示。
    """
    import matplotlib.dates as mdates

    def _monthly_ret(daily: pd.DataFrame) -> pd.DataFrame:
        return (
            daily.assign(m=daily["TRADE_DATE"].dt.to_period("M"))
            .groupby("m")["net_ret"]
            .apply(lambda x: float((1.0 + x).prod() - 1.0))
            .reset_index()
            .rename(columns={"m": "month"})
        )

    bm_m = _monthly_ret(benchmark_daily)
    ft_m = _monthly_ret(filtered_daily)
    merged = bm_m.merge(ft_m, on="month", suffixes=("_bm", "_ft"))
    merged["month_ts"] = merged["month"].dt.to_timestamp()

    bar_width = pd.Timedelta(days=10)
    offset = pd.Timedelta(days=6)

    ax.bar(
        merged["month_ts"] - offset,
        merged["net_ret_bm"],
        width=bar_width,
        color="#888888",
        alpha=0.55,
        label="Benchmark",
    )
    ax.bar(
        merged["month_ts"] + offset,
        merged["net_ret_ft"],
        width=bar_width,
        color="#1a6faf",
        alpha=0.75,
        label="Macro-Filtered",
    )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Monthly Net Return Comparison", fontsize=11)
    ax.set_ylabel("Monthly Return")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.22)


def _calc_avg_drawdown(daily: pd.DataFrame) -> float:
    """计算平均回撤：仅取处于回撤中（< 0）的交易日，求均值。"""
    dd = daily["net_drawdown"]
    underwater = dd[dd < 0]
    return float(underwater.mean()) if not underwater.empty else 0.0


def _calc_active_win_rate(daily: pd.DataFrame) -> float:
    """
    活跃日胜率：仅统计 net_ret != 0 的交易日（position 非 0 的日子）。
    强制空仓日 net_ret == 0，排除在外，避免虚假拉低胜率。
    """
    active = daily["net_ret"][daily["net_ret"] != 0]
    return float((active > 0).mean()) if not active.empty else 0.0


def _add_summary_text(ax, bm_summary: dict, ft_summary: dict,
                      bm_daily: pd.DataFrame, ft_daily: pd.DataFrame) -> None:
    """在空白 axes 上展示关键指标对比文字。"""
    bm_avg_dd = _calc_avg_drawdown(bm_daily)
    ft_avg_dd = _calc_avg_drawdown(ft_daily)
    bm_win = _calc_active_win_rate(bm_daily)
    ft_win = _calc_active_win_rate(ft_daily)

    ax.set_axis_off()
    lines = [
        "            Benchmark   Filtered",
        "─" * 36,
        f"Ann Return  {bm_summary['net']['annual_return']:>9.2%}   {ft_summary['net']['annual_return']:>9.2%}",
        f"Sharpe      {bm_summary['net']['sharpe']:>9.3f}   {ft_summary['net']['sharpe']:>9.3f}",
        f"Max DD      {bm_summary['net']['max_drawdown']:>9.2%}   {ft_summary['net']['max_drawdown']:>9.2%}",
        f"Avg DD      {bm_avg_dd:>9.2%}   {ft_avg_dd:>9.2%}",
        f"Ann Vol     {bm_summary['net']['annual_volatility']:>9.2%}   {ft_summary['net']['annual_volatility']:>9.2%}",
        f"Win Rate*   {bm_win:>9.2%}   {ft_win:>9.2%}",
        f"Trades      {bm_summary['trade_count']:>9d}   {ft_summary['trade_count']:>9d}",
    ]
    ax.text(
        0.08, 0.55, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        va="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )
    ax.set_title("Performance Summary", fontsize=11)


def _add_split_lines(axes: list, split_dates: dict) -> None:
    """
    在指定的多个 axes 上绘制训练/验证/测试集分割虚线。

    split_dates 格式：
        {"val_start": pd.Timestamp, "test_start": pd.Timestamp}
    """
    boundaries = [
        (split_dates["val_start"],  "Val start",  0.92),
        (split_dates["test_start"], "Test start", 0.92),
    ]
    for ax in axes:
        ylims = ax.get_ylim()
        for date, label, y_frac in boundaries:
            ax.axvline(date, color="#555555", linewidth=1.2, linestyle="--", alpha=0.75, zorder=5)
            y_pos = ylims[0] + (ylims[1] - ylims[0]) * y_frac
            ax.text(
                date, y_pos, f" {label}",
                fontsize=7.5, color="#333333", va="top", ha="left",
                rotation=90, zorder=6,
            )


def plot_macro_backtest_report(
    benchmark_pnl: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    filtered_pnl: pd.DataFrame,
    filtered_daily: pd.DataFrame,
    monthly_regime: pd.DataFrame,
    bm_summary: dict,
    ft_summary: dict,
    output_path: Path,
    title_suffix: str = "",
    split_dates: dict | None = None,
) -> None:
    """
    输出宏观过滤对比报告，包含：
        [0,:] NAV 曲线（全宽），背景色标识 regime + 月度信号柱
        [1,:] 月度宏观信号柱（全宽，shared-x）
        [2,0] 回撤对比
        [2,1] 月度收益柱状图
        [3,0] 绩效指标文字表
        [3,1] 空（留白或备用）

    split_dates（可选）：{"val_start": Timestamp, "test_start": Timestamp}
        若提供则在 NAV / 信号柱 / 回撤图上画训练/验证/测试分割虚线。
    """
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR.resolve()))
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[4, 1.2, 3, 2.5],
        hspace=0.45,
        wspace=0.30,
    )

    ax_nav = fig.add_subplot(gs[0, :])
    ax_sig = fig.add_subplot(gs[1, :], sharex=ax_nav)
    ax_dd = fig.add_subplot(gs[2, 0])
    ax_mon = fig.add_subplot(gs[2, 1])
    ax_txt = fig.add_subplot(gs[3, 0])
    ax_blank = fig.add_subplot(gs[3, 1])

    # 需要把 open_position 信息挂到 daily 上，供背景色使用
    # 取 filtered_pnl 聚合为日级 open_position（当天只要有开仓信号就为 True）
    daily_regime = (
        filtered_pnl.groupby("TRADE_DATE", sort=True)
        .agg(open_position=("open_position", "any"))
        .reset_index()
    )
    daily_regime["TRADE_DATE"] = pd.to_datetime(daily_regime["TRADE_DATE"])

    test_start = pd.to_datetime(benchmark_daily["TRADE_DATE"].min())
    test_end = pd.to_datetime(benchmark_daily["TRADE_DATE"].max())

    _plot_nav(ax_nav, benchmark_daily, filtered_daily, daily_regime)
    _plot_macro_signal_bars(ax_sig, monthly_regime, test_start, test_end)
    _plot_drawdown(ax_dd, benchmark_daily, filtered_daily)
    _plot_monthly_returns(ax_mon, benchmark_daily, filtered_daily)
    _add_summary_text(ax_txt, bm_summary, ft_summary, benchmark_daily, filtered_daily)

    # 在 NAV / 信号柱 / 回撤图上标注数据集分割线
    if split_dates is not None:
        _add_split_lines([ax_nav, ax_sig, ax_dd], split_dates)

    ax_blank.set_axis_off()

    title = "Macro-Filtered CTA Backtest Report"
    if title_suffix:
        title += f"  [{title_suffix}]"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.005)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 主回测流程
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_report(
    eval_pred: pd.DataFrame,
    rule_map: dict,
    settings: dict,
    monthly_regime: pd.DataFrame,
    report_path: Path,
    summary_path: Path,
    title_suffix: str,
    lag_months: int,
    split_dates: dict | None,
) -> dict[str, Any]:
    """单次回测流程：生成仓位 → 宏观过滤 → P&L → 绘图 → 保存 JSON。"""
    pos_bm = generate_positions(eval_pred, rule_map, settings)
    pos_ft = apply_macro_filter(pos_bm, monthly_regime)

    bm_pnl, bm_daily = calc_pnl(pos_bm, settings["commission_rate"], settings["slippage_rate"], settings["hold_to_next_bar"])
    ft_pnl, ft_daily = calc_pnl(pos_ft, settings["commission_rate"], settings["slippage_rate"], settings["hold_to_next_bar"])

    bm_summary = performance_summary(bm_daily, bm_pnl, settings["annualization_days"])
    ft_summary = performance_summary(ft_daily, ft_pnl, settings["annualization_days"])

    plot_macro_backtest_report(
        benchmark_pnl=bm_pnl,
        benchmark_daily=bm_daily,
        filtered_pnl=ft_pnl,
        filtered_daily=ft_daily,
        monthly_regime=monthly_regime,
        bm_summary=bm_summary,
        ft_summary=ft_summary,
        output_path=report_path,
        title_suffix=title_suffix,
        split_dates=split_dates,
    )
    print(f"[macro_backtest] Report saved → {report_path}")

    n_open = int(monthly_regime["open_position"].sum())
    n_total = len(monthly_regime)
    result = {
        "macro_filter": {
            "lag_months": lag_months,
            "months_open": n_open,
            "months_total": n_total,
            "macro_strong_months": int(monthly_regime["is_macro_strong"].sum()),
            "inventory_strong_months": int(monthly_regime["is_inventory_strong"].sum()),
        },
        "benchmark_backtest": bm_summary,
        "filtered_backtest": ft_summary,
        "benchmark_monthly_returns": build_monthly_returns(bm_daily).to_dict(orient="records"),
        "filtered_monthly_returns": build_monthly_returns(ft_daily).to_dict(orient="records"),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_native(result), f, indent=2, ensure_ascii=False)
    print(f"[macro_backtest] Summary saved → {summary_path}")
    return _to_native(result)


def run_macro_backtest(
    config_path: str | None = None,
    force_rebuild: bool | None = None,
    macro_csv_path: str | None = None,
    lag_months: int = 1,
) -> dict[str, Any]:
    """
    宏观过滤回测主函数。默认同时输出两份报告：
        - macro_backtest_report.png       : 仅测试集（样本外，模型无偏）
        - macro_backtest_full.png         : 全期（含训练/验证集，带数据集分割线；
                                            注意模型在训练集上 in-sample，收益偏高）

    Parameters
    ----------
    config_path   : 项目 config.yaml 路径（None 则自动搜索）
    force_rebuild : 是否强制重建数据缓存
    macro_csv_path: 月度宏观因子 CSV 路径
    lag_months    : 宏观信号滞后月数（默认 1，避免前视偏差）
    """
    # ── 加载配置 ────────────────────────────────────────────────────────────
    settings = build_backtest_settings(config_path)
    load_project_config(config_path)

    if macro_csv_path is None:
        macro_csv_path = str(
            PROJECT_ROOT / "data" / "macro" / "macro_monthly_features_core.csv"
        )
    output_dir = PROJECT_ROOT / "results" / "backtest_macro"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据准备 & 模型推断（共用）──────────────────────────────────────────
    print("[macro_backtest] Preparing data and running model inference ...")
    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild)
    artifact_map = load_dual_regime_models(config_path)
    val_pred = predict_dual_regime(
        prepared.val_data, prepared.feature_cols, prepared.target_col, artifact_map
    )
    rule_map = build_signal_rule_map(val_pred, settings)

    # ── 提取数据集分割日期（用于全期报告的虚线标注）────────────────────────
    split_dates = {
        "val_start":  pd.to_datetime(prepared.val_data["TRADE_DATE"].min()),
        "test_start": pd.to_datetime(prepared.test_data["TRADE_DATE"].min()),
    }
    print(
        f"[macro_backtest] Split dates — val_start: {split_dates['val_start'].date()}, "
        f"test_start: {split_dates['test_start'].date()}"
    )

    # ── 加载月度宏观 regime ─────────────────────────────────────────────────
    print(f"[macro_backtest] Loading macro regime from {macro_csv_path} ...")
    monthly_regime = build_monthly_regime(macro_csv_path, lag_months=lag_months)
    n_open = int(monthly_regime["open_position"].sum())
    n_total = len(monthly_regime)
    print(
        f"[macro_backtest] Macro regime: {n_open}/{n_total} months allow position "
        f"(macro_strong={int(monthly_regime['is_macro_strong'].sum())}, "
        f"inv_strong={int(monthly_regime['is_inventory_strong'].sum())})"
    )

    # ── 报告①：仅测试集 ─────────────────────────────────────────────────────
    print("[macro_backtest] === Report 1/2: Test set only ===")
    test_pred = predict_dual_regime(
        prepared.test_data, prepared.feature_cols, prepared.target_col, artifact_map
    )
    test_result = _run_single_report(
        eval_pred=test_pred,
        rule_map=rule_map,
        settings=settings,
        monthly_regime=monthly_regime,
        report_path=output_dir / "macro_backtest_report.png",
        summary_path=output_dir / "macro_backtest_summary.json",
        title_suffix="Test set only (out-of-sample)",
        lag_months=lag_months,
        split_dates=None,
    )

    # ── 报告②：全期（train + val + test），带分割虚线 ──────────────────────
    print("[macro_backtest] === Report 2/2: Full period (train + val + test) ===")
    all_data = pd.concat(
        [prepared.train_data, prepared.val_data, prepared.test_data],
        ignore_index=True,
    )
    all_data["TRADE_DATE"] = pd.to_datetime(all_data["TRADE_DATE"])
    full_pred = predict_dual_regime(
        all_data, prepared.feature_cols, prepared.target_col, artifact_map
    )
    print(
        f"[macro_backtest] Full-period: {len(full_pred)} bars "
        f"from {all_data['TRADE_DATE'].min().date()} to {all_data['TRADE_DATE'].max().date()}"
    )
    full_result = _run_single_report(
        eval_pred=full_pred,
        rule_map=rule_map,
        settings=settings,
        monthly_regime=monthly_regime,
        report_path=output_dir / "macro_backtest_full.png",
        summary_path=output_dir / "macro_backtest_summary_full.json",
        title_suffix="Full period [train+val in-sample; test out-of-sample]",
        lag_months=lag_months,
        split_dates=split_dates,
    )

    return {"test": test_result, "full": full_result}


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Macro-filtered CTA backtest. Always outputs two reports: test-only and full-period."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild merged dataset cache.")
    parser.add_argument("--macro-csv", default=None, help="Path to macro_monthly_features_core.csv")
    parser.add_argument("--lag-months", type=int, default=1, help="Macro signal lag in months (default=1).")
    return parser


def _print_summary(label: str, result: dict) -> None:
    bm = result["benchmark_backtest"]["net"]
    ft = result["filtered_backtest"]["net"]
    print(f"\n── {label} ──────────────────────────────")
    print(f"{'':20s} {'Benchmark':>12s}  {'Filtered':>12s}")
    print(f"{'Ann. Return':20s} {bm['annual_return']:>12.2%}  {ft['annual_return']:>12.2%}")
    print(f"{'Sharpe':20s} {bm['sharpe']:>12.3f}  {ft['sharpe']:>12.3f}")
    print(f"{'Max Drawdown':20s} {bm['max_drawdown']:>12.2%}  {ft['max_drawdown']:>12.2%}")
    print(f"{'Ann. Volatility':20s} {bm['annual_volatility']:>12.2%}  {ft['annual_volatility']:>12.2%}")
    print(f"{'Trade Count':20s} {result['benchmark_backtest']['trade_count']:>12d}  {result['filtered_backtest']['trade_count']:>12d}")


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_macro_backtest(
        config_path=args.config,
        force_rebuild=args.force_rebuild if args.force_rebuild else None,
        macro_csv_path=args.macro_csv,
        lag_months=args.lag_months,
    )
    _print_summary("Test set only", result["test"])
    _print_summary("Full period (includes in-sample)", result["full"])
