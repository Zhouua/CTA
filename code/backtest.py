from __future__ import annotations

import argparse
import json
import os
import sys
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

from config_utils import get_section, load_project_config, resolve_paths
from dataset import REGIME_NAME_MAP, prepare_data
from modeling import calc_prediction_metrics, load_dual_regime_models, predict_dual_regime


MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"


def _to_native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def build_backtest_settings(config_path: str | None = None) -> dict[str, Any]:
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(
        config_dir,
        get_section(config, "paths"),
        ["backtest_dir", "backtest_plot", "prediction_cache"],
    )
    signal_cfg = get_section(config, "signal")
    backtest_cfg = get_section(config, "backtest")
    return {
        "paths": paths,
        "signal_col": str(signal_cfg.get("signal_col", "pred_return")),
        "threshold_mode": str(signal_cfg.get("threshold_mode", "quantile")).lower(),
        "entry_quantile": float(signal_cfg.get("entry_quantile", 0.88)),
        "exit_quantile": float(signal_cfg.get("exit_quantile", 0.45)),
        "confirmation_bars": int(signal_cfg.get("confirmation_bars", 2)),
        "min_hold_bars": int(signal_cfg.get("min_hold_bars", 4)),
        "cooldown_bars": int(signal_cfg.get("cooldown_bars", 2)),
        "allow_direct_flip": bool(signal_cfg.get("allow_direct_flip", False)),
        "flip_to_flat_first": bool(signal_cfg.get("flip_to_flat_first", True)),
        "enforce_cost_filter": bool(signal_cfg.get("enforce_cost_filter", True)),
        "round_trip_turnover": float(signal_cfg.get("round_trip_turnover", 2.0)),
        "cost_filter_multiple": float(signal_cfg.get("cost_filter_multiple", 1.2)),
        "min_expected_edge": float(signal_cfg.get("min_expected_edge", 0.0)),
        "commission_rate": float(backtest_cfg.get("commission_rate", 0.0001)),
        "slippage_rate": float(backtest_cfg.get("slippage_rate", 0.0001)),
        "hold_to_next_bar": bool(backtest_cfg.get("hold_to_next_bar", True)),
        "annualization_days": int(backtest_cfg.get("annualization_days", 250)),
        "flatten_at_day_end": bool(backtest_cfg.get("flatten_at_day_end", True)),
        "save_prediction_table": bool(backtest_cfg.get("save_prediction_table", False)),
    }


def _resolve_thresholds(
    pred_series: pd.Series,
    threshold_mode: str,
    entry_quantile: float,
    exit_quantile: float,
) -> tuple[float, float]:
    abs_pred = pred_series.astype("float64").abs()
    abs_pred = abs_pred[np.isfinite(abs_pred)]
    abs_pred = abs_pred[abs_pred > 0]
    if abs_pred.empty:
        return 0.0, 0.0
    if threshold_mode != "quantile":
        raise ValueError("Only quantile threshold mode is supported in CTA_vol/backtest.py.")
    entry = float(np.quantile(abs_pred, entry_quantile))
    exit_ = float(np.quantile(abs_pred, exit_quantile))
    return max(entry, 0.0), max(min(exit_, entry), 0.0)


def build_signal_rule_map(validation_df: pd.DataFrame, settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    total_cost = settings["commission_rate"] + settings["slippage_rate"]
    round_trip_cost = total_cost * settings["round_trip_turnover"]
    rule_map: dict[int, dict[str, Any]] = {}

    for regime_label in sorted(REGIME_NAME_MAP):
        regime_df = validation_df.loc[validation_df["REGIME_LABEL"] == regime_label].copy()
        if regime_df.empty:
            continue
        entry_threshold, exit_threshold = _resolve_thresholds(
            pred_series=regime_df[settings["signal_col"]],
            threshold_mode=settings["threshold_mode"],
            entry_quantile=settings["entry_quantile"],
            exit_quantile=settings["exit_quantile"],
        )
        cost_filter_threshold = max(
            settings["min_expected_edge"],
            round_trip_cost * settings["cost_filter_multiple"],
        )
        effective_entry = max(entry_threshold, cost_filter_threshold) if settings["enforce_cost_filter"] else entry_threshold
        rule_map[regime_label] = {
            "regime_name": REGIME_NAME_MAP[regime_label],
            "entry_threshold": float(entry_threshold),
            "exit_threshold": float(exit_threshold),
            "cost_filter_threshold": float(cost_filter_threshold),
            "effective_entry_threshold": float(effective_entry),
            "confirmation_bars": int(settings["confirmation_bars"]),
            "min_hold_bars": int(settings["min_hold_bars"]),
            "cooldown_bars": int(settings["cooldown_bars"]),
            "allow_direct_flip": bool(settings["allow_direct_flip"]),
            "flip_to_flat_first": bool(settings["flip_to_flat_first"]),
        }
    return rule_map


def generate_positions(
    prediction_df: pd.DataFrame,
    rule_map: dict[int, dict[str, Any]],
    settings: dict[str, Any],
) -> pd.DataFrame:
    if prediction_df.empty:
        return prediction_df.copy()

    df = prediction_df.sort_values("TDATE").reset_index(drop=True).copy()
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df["is_day_end"] = df["TRADE_DATE"].ne(df["TRADE_DATE"].shift(-1)).fillna(True)

    positions: list[int] = []
    raw_signals: list[int] = []
    actions: list[str] = []

    current_position = 0
    hold_bars = 0
    cooldown = 0
    confirm_side = 0
    confirm_count = 0

    for row in df.itertuples(index=False):
        regime_label = int(row.REGIME_LABEL)
        rules = rule_map.get(regime_label)
        if rules is None:
            raise ValueError(f"Missing signal rules for regime_label={regime_label}")

        pred_value = float(getattr(row, settings["signal_col"]))
        entry_threshold = float(rules["effective_entry_threshold"])
        exit_threshold = float(rules["exit_threshold"])
        raw_signal = 0
        if pred_value >= entry_threshold:
            raw_signal = 1
        elif pred_value <= -entry_threshold:
            raw_signal = -1

        action = "hold_flat" if current_position == 0 else "hold_position"
        exit_signal = abs(pred_value) <= exit_threshold

        if current_position == 0:
            if cooldown > 0:
                cooldown -= 1
            if raw_signal != 0 and cooldown == 0:
                if confirm_side == raw_signal:
                    confirm_count += 1
                else:
                    confirm_side = raw_signal
                    confirm_count = 1
                if confirm_count >= int(rules["confirmation_bars"]):
                    current_position = raw_signal
                    hold_bars = 0
                    confirm_side = 0
                    confirm_count = 0
                    action = "open_long" if raw_signal > 0 else "open_short"
                else:
                    action = "wait_confirm"
            else:
                confirm_side = 0
                confirm_count = 0
        else:
            hold_bars += 1
            reverse_signal = raw_signal == -current_position and raw_signal != 0
            can_exit = hold_bars >= int(rules["min_hold_bars"])

            if can_exit and exit_signal:
                current_position = 0
                hold_bars = 0
                cooldown = int(rules["cooldown_bars"])
                confirm_side = 0
                confirm_count = 0
                action = "exit_to_flat"
            elif can_exit and reverse_signal:
                if bool(rules["allow_direct_flip"]):
                    current_position = raw_signal
                    hold_bars = 0
                    cooldown = 0
                    confirm_side = 0
                    confirm_count = 0
                    action = "flip"
                elif bool(rules["flip_to_flat_first"]):
                    current_position = 0
                    hold_bars = 0
                    cooldown = int(rules["cooldown_bars"])
                    confirm_side = raw_signal
                    confirm_count = 1
                    action = "flatten_before_flip"

        if bool(settings["flatten_at_day_end"]) and bool(row.is_day_end):
            if current_position != 0:
                current_position = 0
                hold_bars = 0
                cooldown = 0
                confirm_side = 0
                confirm_count = 0
                action = "day_end_flat"
            else:
                confirm_side = 0
                confirm_count = 0

        positions.append(int(current_position))
        raw_signals.append(int(raw_signal))
        actions.append(action)

    df["raw_signal"] = raw_signals
    df["position"] = positions
    df["action"] = actions
    return df


def calc_max_drawdown(nav: pd.Series) -> tuple[float, int, int]:
    if nav.empty:
        return 0.0, 0, 0
    roll_max = nav.cummax()
    drawdown = nav / roll_max - 1.0
    end_idx = int(drawdown.idxmin())
    start_idx = int(nav.loc[:end_idx].idxmax())
    return float(drawdown.min()), start_idx, end_idx


def calc_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    returns = returns.astype("float64")
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def annualize_return(nav: pd.Series, periods_per_year: int) -> float:
    if len(nav) < 2:
        return 0.0
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    return float((1.0 + total_return) ** (periods_per_year / len(nav)) - 1.0)


def calc_profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def calc_pnl(
    position_df: pd.DataFrame,
    commission_rate: float,
    slippage_rate: float,
    hold_to_next_bar: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = position_df.sort_values("TDATE").reset_index(drop=True).copy()
    total_cost = commission_rate + slippage_rate

    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df["raw_ret"] = df["CLOSE"].pct_change().fillna(0.0)
    day_start_mask = df["TRADE_DATE"].ne(df["TRADE_DATE"].shift()).fillna(True)
    df.loc[day_start_mask, "raw_ret"] = 0.0

    prev_position = df["position"].shift(1).fillna(0.0)
    effective_prev_position = prev_position if hold_to_next_bar else df["position"]

    df["turnover"] = (df["position"] - prev_position).abs()
    df["gross_ret"] = effective_prev_position * df["raw_ret"]
    df["cost_ret"] = df["turnover"] * total_cost
    df["net_ret"] = df["gross_ret"] - df["cost_ret"]

    daily = (
        df.groupby("TRADE_DATE", sort=True)
        .agg(
            gross_ret=("gross_ret", lambda x: float((1.0 + x).prod() - 1.0)),
            net_ret=("net_ret", lambda x: float((1.0 + x).prod() - 1.0)),
            turnover=("turnover", "sum"),
            rebalance_count=("turnover", lambda x: int((x > 0).sum())),
            long_bars=("position", lambda x: int((x == 1).sum())),
            short_bars=("position", lambda x: int((x == -1).sum())),
            flat_bars=("position", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )
    daily["nav_gross"] = (1.0 + daily["gross_ret"]).cumprod()
    daily["nav_net"] = (1.0 + daily["net_ret"]).cumprod()
    daily["net_drawdown"] = daily["nav_net"] / daily["nav_net"].cummax() - 1.0
    return df, daily


def extract_trade_log(position_df: pd.DataFrame) -> pd.DataFrame:
    if position_df.empty:
        return pd.DataFrame()

    trade_df = position_df.sort_values("TDATE").reset_index(drop=True).copy()
    start_mask = trade_df["position"].ne(0) & trade_df["position"].ne(trade_df["position"].shift().fillna(0))
    trade_df["trade_id"] = np.where(start_mask, np.arange(1, len(trade_df) + 1), np.nan)
    trade_df["trade_id"] = pd.Series(trade_df["trade_id"]).ffill()
    trade_df.loc[trade_df["position"] == 0, "trade_id"] = np.nan

    rows: list[dict[str, Any]] = []
    for trade_id, group in trade_df.dropna(subset=["trade_id"]).groupby("trade_id", sort=True):
        rows.append(
            {
                "trade_id": int(trade_id),
                "side": int(group["position"].iloc[0]),
                "entry_time": group["TDATE"].iloc[0],
                "exit_time": group["TDATE"].iloc[-1],
                "holding_bars": int(len(group)),
                "gross_ret": float((1.0 + group["gross_ret"]).prod() - 1.0),
                "net_ret": float((1.0 + group["net_ret"]).prod() - 1.0),
                "avg_pred_return": float(group["pred_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def performance_summary(
    daily: pd.DataFrame,
    pnl_df: pd.DataFrame,
    periods_per_year: int,
) -> dict[str, Any]:
    trade_log = extract_trade_log(pnl_df)
    summary: dict[str, Any] = {}

    for label, nav_col, ret_col in [("gross", "nav_gross", "gross_ret"), ("net", "nav_net", "net_ret")]:
        nav = daily[nav_col]
        ret = daily[ret_col]
        mdd, mdd_start, mdd_end = calc_max_drawdown(nav)
        summary[label] = {
            "total_return": float(nav.iloc[-1] - 1.0) if not nav.empty else 0.0,
            "annual_return": annualize_return(nav, periods_per_year),
            "sharpe": calc_sharpe(ret, periods_per_year),
            "max_drawdown": float(mdd),
            "daily_win_rate": float((ret > 0).mean()) if not ret.empty else 0.0,
            "annual_volatility": float(ret.std() * np.sqrt(periods_per_year)) if len(ret) > 1 else 0.0,
            "profit_factor": calc_profit_factor(ret),
            "max_drawdown_start": str(daily.loc[mdd_start, "TRADE_DATE"].date()) if not daily.empty else "",
            "max_drawdown_end": str(daily.loc[mdd_end, "TRADE_DATE"].date()) if not daily.empty else "",
        }

    summary["trade_count"] = int(len(trade_log))
    summary["avg_daily_turnover"] = float(daily["turnover"].mean()) if not daily.empty else 0.0
    summary["total_turnover"] = float(daily["turnover"].sum()) if not daily.empty else 0.0
    summary["rebalance_count"] = int(daily["rebalance_count"].sum()) if not daily.empty else 0
    summary["position_mix"] = {
        "long_ratio": float((pnl_df["position"] == 1).mean()) if not pnl_df.empty else 0.0,
        "short_ratio": float((pnl_df["position"] == -1).mean()) if not pnl_df.empty else 0.0,
        "flat_ratio": float((pnl_df["position"] == 0).mean()) if not pnl_df.empty else 0.0,
    }
    summary["trade_win_rate"] = float((trade_log["net_ret"] > 0).mean()) if not trade_log.empty else 0.0
    summary["avg_holding_bars"] = float(trade_log["holding_bars"].mean()) if not trade_log.empty else 0.0
    summary["avg_trade_net_return"] = float(trade_log["net_ret"].mean()) if not trade_log.empty else 0.0
    return summary


def build_benchmark_positions(base_df: pd.DataFrame, side: int, flatten_at_day_end: bool) -> pd.DataFrame:
    benchmark = base_df.sort_values("TDATE").reset_index(drop=True).copy()
    benchmark["TRADE_DATE"] = pd.to_datetime(benchmark["TRADE_DATE"])
    benchmark["position"] = side
    if flatten_at_day_end:
        is_day_end = benchmark["TRADE_DATE"].ne(benchmark["TRADE_DATE"].shift(-1)).fillna(True)
        benchmark.loc[is_day_end, "position"] = 0
    benchmark["raw_signal"] = side
    benchmark["action"] = "benchmark"
    return benchmark


def summarize_regime_predictions(prediction_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for regime_label, regime_name in REGIME_NAME_MAP.items():
        group = prediction_df.loc[prediction_df["REGIME_LABEL"] == regime_label].copy()
        summary[regime_name] = {
            "rows": int(len(group)),
            "metrics": calc_prediction_metrics(group["future_return"], group["pred_return"]) if not group.empty else {},
        }
    return summary


def build_monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        daily.assign(month=daily["TRADE_DATE"].dt.to_period("M"))
        .groupby("month", sort=True)["net_ret"]
        .apply(lambda x: float((1.0 + x).prod() - 1.0))
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly


def add_daily_exposure_ratios(daily: pd.DataFrame) -> pd.DataFrame:
    exposure = daily.copy()
    total_bars = (
        exposure["long_bars"].astype("float64")
        + exposure["short_bars"].astype("float64")
        + exposure["flat_bars"].astype("float64")
    )
    total_bars = total_bars.replace(0.0, np.nan)
    exposure["long_ratio"] = (exposure["long_bars"] / total_bars).fillna(0.0)
    exposure["short_ratio"] = (exposure["short_bars"] / total_bars).fillna(0.0)
    exposure["flat_ratio"] = (exposure["flat_bars"] / total_bars).fillna(0.0)
    return exposure


def plot_split_switch_and_return(
    ax,
    prediction_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    split_name: str,
) -> None:
    if prediction_df.empty or daily_df.empty:
        ax.set_axis_off()
        return

    regime_daily = (
        prediction_df.groupby("TRADE_DATE", sort=True)
        .agg(REGIME_LABEL=("REGIME_LABEL", "first"))
        .reset_index()
    )
    regime_daily["TRADE_DATE"] = pd.to_datetime(regime_daily["TRADE_DATE"])
    regime_daily["REGIME_NAME"] = regime_daily["REGIME_LABEL"].map(REGIME_NAME_MAP)
    merged = daily_df.merge(regime_daily, on="TRADE_DATE", how="left")

    color_map = {"low_vol": "#7BC96F", "high_vol": "#F28B82"}
    current_name = None
    start_date = None
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []

    for row in regime_daily.itertuples(index=False):
        regime_name = str(row.REGIME_NAME)
        trade_date = pd.to_datetime(row.TRADE_DATE)
        if current_name is None:
            current_name = regime_name
            start_date = trade_date
            continue
        if regime_name != current_name:
            segments.append((start_date, trade_date, current_name))
            current_name = regime_name
            start_date = trade_date
    if current_name is not None and start_date is not None:
        segments.append((start_date, pd.to_datetime(regime_daily["TRADE_DATE"].iloc[-1]) + pd.Timedelta(days=1), current_name))

    for seg_start, seg_end, regime_name in segments:
        ax.axvspan(seg_start, seg_end, color=color_map[regime_name], alpha=0.28, linewidth=0)

    bar_colors = np.where(merged["net_ret"] >= 0, "#1f77b4", "#8c2d04")
    ax.bar(merged["TRADE_DATE"], merged["net_ret"], color=bar_colors, alpha=0.75, width=1.0, label="Daily net return")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"{split_name} Model Switch And Daily Return")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(merged["TRADE_DATE"], merged["nav_net"], color="#111111", linewidth=1.2, label="Net NAV")
    ax2.set_ylabel("Net NAV")
    ax2.grid(False)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color_map["low_vol"], alpha=0.28, label="low_vol model active"),
        Patch(facecolor=color_map["high_vol"], alpha=0.28, label="high_vol model active"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    counts = regime_daily["REGIME_NAME"].value_counts().to_dict()
    ax.text(
        0.01,
        0.02,
        f"low_vol days={int(counts.get('low_vol', 0))}, high_vol days={int(counts.get('high_vol', 0))}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )


def plot_backtest(
    pnl_df: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark_long_daily: pd.DataFrame,
    benchmark_short_daily: pd.DataFrame,
    validation_prediction_df: pd.DataFrame,
    test_prediction_df: pd.DataFrame,
    validation_daily: pd.DataFrame,
    output_path: Path,
) -> None:
    if daily.empty or pnl_df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR.resolve()))
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    exposure = add_daily_exposure_ratios(daily)
    fig, axes = plt.subplots(5, 2, figsize=(20, 27))

    sampled_step = max(len(pnl_df) // 5000, 1)
    sampled_pnl = pnl_df.iloc[::sampled_step].copy()
    rolling_window = max(min(len(pnl_df) // 50, 240), 30)
    pred_roll = pnl_df["pred_return"].rolling(rolling_window, min_periods=1).mean()
    actual_roll = pnl_df["future_return"].rolling(rolling_window, min_periods=1).mean()

    axes[0, 0].plot(daily["TRADE_DATE"], daily["nav_gross"], color="#7f7f7f", linestyle="--", label="Strategy Gross")
    axes[0, 0].plot(daily["TRADE_DATE"], daily["nav_net"], color="#1f77b4", label="Strategy Net")
    axes[0, 0].plot(
        benchmark_long_daily["TRADE_DATE"],
        benchmark_long_daily["nav_net"],
        color="#2ca02c",
        label="Benchmark Long Only (+1 whenever tradable)",
    )
    axes[0, 0].plot(
        benchmark_short_daily["TRADE_DATE"],
        benchmark_short_daily["nav_net"],
        color="#d62728",
        label="Benchmark Short Only (-1 whenever tradable)",
    )
    axes[0, 0].set_title("NAV Curve")
    axes[0, 0].legend()
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].text(
        0.01,
        0.02,
        "Long only: always hold +1 benchmark\nShort only: always hold -1 benchmark",
        transform=axes[0, 0].transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )

    axes[0, 1].fill_between(daily["TRADE_DATE"], daily["net_drawdown"], 0.0, color="#c44e52", alpha=0.35)
    axes[0, 1].plot(daily["TRADE_DATE"], daily["net_drawdown"], color="#8c2d04", linewidth=1.1)
    axes[0, 1].set_title("Net Drawdown")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0, 1].grid(alpha=0.25)

    colors = np.where(daily["net_ret"] >= 0, "#2ca02c", "#d62728")
    axes[1, 0].bar(daily["TRADE_DATE"], daily["net_ret"], color=colors, alpha=0.8, width=1.0)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Daily Net Return")
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].bar(daily["TRADE_DATE"], daily["rebalance_count"], color="#4c78a8", alpha=0.7, label="Rebalance count")
    axes[1, 1].plot(daily["TRADE_DATE"], daily["turnover"], color="#f58518", linewidth=1.3, label="Turnover")
    axes[1, 1].set_title("Daily Turnover And Rebalance Count")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1, 1].grid(alpha=0.25)

    axes[2, 0].step(sampled_pnl["TDATE"], sampled_pnl["position"], where="post", color="#1f77b4", linewidth=1.0)
    axes[2, 0].set_title("Sampled Position Path")
    axes[2, 0].set_ylim(-1.2, 1.2)
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2, 0].grid(alpha=0.25)

    axes[2, 1].stackplot(
        exposure["TRADE_DATE"],
        exposure["short_ratio"],
        exposure["flat_ratio"],
        exposure["long_ratio"],
        labels=["Short ratio", "Flat ratio", "Long ratio"],
        colors=["#d62728", "#bdbdbd", "#2ca02c"],
        alpha=0.85,
    )
    axes[2, 1].set_ylim(0.0, 1.0)
    axes[2, 1].set_title("Daily Holding Ratio")
    axes[2, 1].legend(loc="upper right")
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2, 1].grid(alpha=0.25)

    axes[3, 0].plot(
        sampled_pnl["TDATE"],
        sampled_pnl["future_return"],
        color="#7f7f7f",
        linewidth=0.8,
        alpha=0.75,
        label="Actual future return",
    )
    axes[3, 0].plot(
        sampled_pnl["TDATE"],
        sampled_pnl["pred_return"],
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.85,
        label="Predicted return",
    )
    axes[3, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[3, 0].set_title("Sampled Predicted vs Actual Return")
    axes[3, 0].legend(loc="upper right")
    axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3, 0].grid(alpha=0.25)

    axes[3, 1].plot(pnl_df["TDATE"], actual_roll, color="#7f7f7f", linewidth=1.2, label=f"Actual rolling {rolling_window}")
    axes[3, 1].plot(pnl_df["TDATE"], pred_roll, color="#1f77b4", linewidth=1.2, label=f"Predicted rolling {rolling_window}")
    axes[3, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[3, 1].set_title("Rolling Predicted vs Actual Return")
    axes[3, 1].legend(loc="upper right")
    axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3, 1].grid(alpha=0.25)

    plot_split_switch_and_return(
        ax=axes[4, 0],
        prediction_df=validation_prediction_df,
        daily_df=validation_daily,
        split_name="Validation",
    )
    plot_split_switch_and_return(
        ax=axes[4, 1],
        prediction_df=test_prediction_df,
        daily_df=daily,
        split_name="Test",
    )
    axes[4, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[4, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def run_backtest(config_path: str | None = None, force_rebuild: bool | None = None) -> dict[str, Any]:
    settings = build_backtest_settings(config_path)
    settings["paths"]["backtest_dir"].mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild)
    artifact_map = load_dual_regime_models(config_path)
    val_pred = predict_dual_regime(prepared.val_data, prepared.feature_cols, prepared.target_col, artifact_map)
    test_pred = predict_dual_regime(prepared.test_data, prepared.feature_cols, prepared.target_col, artifact_map)

    rule_map = build_signal_rule_map(val_pred, settings)

    val_position = generate_positions(val_pred, rule_map, settings)
    test_position = generate_positions(test_pred, rule_map, settings)

    val_pnl, val_daily = calc_pnl(
        position_df=val_position,
        commission_rate=settings["commission_rate"],
        slippage_rate=settings["slippage_rate"],
        hold_to_next_bar=settings["hold_to_next_bar"],
    )
    test_pnl, test_daily = calc_pnl(
        position_df=test_position,
        commission_rate=settings["commission_rate"],
        slippage_rate=settings["slippage_rate"],
        hold_to_next_bar=settings["hold_to_next_bar"],
    )

    benchmark_long = build_benchmark_positions(test_pred, side=1, flatten_at_day_end=settings["flatten_at_day_end"])
    benchmark_short = build_benchmark_positions(test_pred, side=-1, flatten_at_day_end=settings["flatten_at_day_end"])
    _, benchmark_long_daily = calc_pnl(
        position_df=benchmark_long,
        commission_rate=settings["commission_rate"],
        slippage_rate=settings["slippage_rate"],
        hold_to_next_bar=settings["hold_to_next_bar"],
    )
    _, benchmark_short_daily = calc_pnl(
        position_df=benchmark_short,
        commission_rate=settings["commission_rate"],
        slippage_rate=settings["slippage_rate"],
        hold_to_next_bar=settings["hold_to_next_bar"],
    )

    plot_backtest(
        pnl_df=test_pnl,
        daily=test_daily,
        benchmark_long_daily=benchmark_long_daily,
        benchmark_short_daily=benchmark_short_daily,
        validation_prediction_df=val_pred,
        test_prediction_df=test_pred,
        validation_daily=val_daily,
        output_path=settings["paths"]["backtest_plot"],
    )

    summary = {
        "dataset": prepared.metadata,
        "signal_rules": {REGIME_NAME_MAP[key]: value for key, value in rule_map.items()},
        "validation_prediction_metrics": calc_prediction_metrics(val_pred["future_return"], val_pred["pred_return"]),
        "validation_prediction_metrics_by_regime": summarize_regime_predictions(val_pred),
        "test_prediction_metrics": calc_prediction_metrics(test_pred["future_return"], test_pred["pred_return"]),
        "test_prediction_metrics_by_regime": summarize_regime_predictions(test_pred),
        "validation_backtest": performance_summary(val_daily, val_pnl, settings["annualization_days"]),
        "test_backtest": performance_summary(test_daily, test_pnl, settings["annualization_days"]),
        "test_monthly_returns": build_monthly_returns(test_daily).to_dict(orient="records"),
    }

    summary_path = settings["paths"]["backtest_dir"] / "backtest_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_native(summary), f, indent=2, ensure_ascii=False)

    if settings["save_prediction_table"]:
        test_position.to_parquet(settings["paths"]["prediction_cache"], index=False)

    return _to_native(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CTA_vol backtest with trained dual-regime LightGBM models.")
    parser.add_argument("--config", default=None, help="Path to CTA_vol config.yaml")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached merged dataset before backtesting.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    summary = run_backtest(config_path=args.config, force_rebuild=args.force_rebuild)
    print(json.dumps(summary["test_backtest"], indent=2, ensure_ascii=False))
