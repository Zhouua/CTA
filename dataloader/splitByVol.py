from pathlib import Path

import numpy as np
import pandas as pd


LOW_VOL_COLOR = "#7BC96F"
HIGH_VOL_COLOR = "#F28B82"
RETURN_LINE_COLOR = "#16324F"
DAILY_VOL_LINE_COLOR = "#8B1E3F"
CUTOFF_LINE_COLOR = "#B22222"
VOL_SPAN_ALPHA = 0.32


def _validate_split_ratio(train_ratio, valid_ratio, test_ratio):
    total = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1.0")


def _prepare_minute_data(data):
    df = data.copy()
    df["TDATE"] = pd.to_datetime(df["TDATE"])
    df = df.sort_values("TDATE").reset_index(drop=True)

    df["TRADE_DATE"] = df["TDATE"].dt.normalize()
    df["MONTH"] = df["TDATE"].dt.to_period("M")
    df["SPLIT_MONTH"] = df["MONTH"]
    df["SPLIT_WEEK"] = df["TDATE"].dt.to_period("W")
    df["SPLIT_DAY"] = df["TRADE_DATE"]
    return df


def _resolve_split_col(granularity):
    granularity = str(granularity).lower()
    if granularity == "month":
        return "SPLIT_MONTH"
    if granularity == "week":
        return "SPLIT_WEEK"
    if granularity == "day":
        return "SPLIT_DAY"
    raise ValueError("split_granularity 必须是 month/week/day 之一")


def _build_time_split_map(data, split_col, train_ratio, valid_ratio, test_ratio):
    _validate_split_ratio(train_ratio, valid_ratio, test_ratio)

    periods = pd.Series(data[split_col].drop_duplicates().sort_values().tolist())
    total_periods = len(periods)
    train_periods = int(total_periods * train_ratio)
    valid_periods = int(total_periods * valid_ratio)

    split_map = pd.DataFrame({split_col: periods})
    split_map["DATA_SPLIT"] = "test"
    split_map.loc[: train_periods - 1, "DATA_SPLIT"] = "train"
    split_map.loc[train_periods : train_periods + valid_periods - 1, "DATA_SPLIT"] = "valid"
    return split_map


def _label_by_cutoff(values, cutoff):
    return np.where(values >= cutoff, 1, -1)


def _add_monthly_background(ax, data):
    span_df = (
        data.dropna(subset=["VOL_LABEL"])
        .groupby("MONTH", as_index=False)
        .agg(
            start_time=("TDATE", "min"),
            end_time=("TDATE", "max"),
            VOL_LABEL=("VOL_LABEL", "first"),
        )
    )

    for row in span_df.itertuples(index=False):
        color = HIGH_VOL_COLOR if row.VOL_LABEL == 1 else LOW_VOL_COLOR
        ax.axvspan(row.start_time, row.end_time, color=color, alpha=VOL_SPAN_ALPHA, linewidth=0)


def _add_split_boundaries(ax, data):
    valid_start = data.loc[data["DATA_SPLIT"] == "valid", "TDATE"]
    test_start = data.loc[data["DATA_SPLIT"] == "test", "TDATE"]

    if not valid_start.empty:
        ax.axvline(valid_start.iloc[0], color="black", linestyle="--", linewidth=1.1)
    if not test_start.empty:
        ax.axvline(test_start.iloc[0], color="black", linestyle="--", linewidth=1.1)


def _build_concatenated_regime_samples(data, vol_label, return_col):
    regime_df = (
        data.loc[data["VOL_LABEL"] == vol_label]
        .sort_values(["MONTH", "TDATE"])
        .reset_index(drop=True)
        .copy()
    )
    if regime_df.empty:
        regime_df["BLOCK_ID"] = pd.Series(dtype="Int64")
        regime_df["concat_index"] = pd.Series(dtype="Int64")
        return regime_df

    month_order = regime_df["MONTH"].drop_duplicates().reset_index(drop=True)
    block_map = pd.DataFrame(
        {
            "MONTH": month_order,
            "BLOCK_ID": np.arange(len(month_order), dtype=int),
        }
    )
    regime_df = regime_df.merge(block_map, on="MONTH", how="left")
    regime_df["BLOCK_ID"] = regime_df["BLOCK_ID"].astype("Int64")
    regime_df["concat_index"] = np.arange(len(regime_df), dtype=int)
    return regime_df


def split_by_vol(
    data,
    vol_threshold=None,
    vol_percentage=None,
    window=20,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    label_train_only=True,
    split_granularity="month",
    rolling_regime_window: int = 0,
):
    if (vol_threshold is None) == (vol_percentage is None):
        raise ValueError("vol_threshold 和 vol_percentage 必须二选一")

    if vol_percentage is not None and not 0 < vol_percentage < 1:
        raise ValueError("vol_percentage 必须在 0 和 1 之间")

    df = _prepare_minute_data(data)
    split_col = _resolve_split_col(split_granularity)
    split_map = _build_time_split_map(df, split_col, train_ratio, valid_ratio, test_ratio)
    df = df.merge(split_map, on=split_col, how="left")

    daily_close = (
        df.groupby("TRADE_DATE", as_index=False)
        .agg(
            CLOSE=("CLOSE", "last"),
            MONTH=("MONTH", "first"),
            DATA_SPLIT=("DATA_SPLIT", "first"),
            day_start=("TDATE", "min"),
            day_end=("TDATE", "max"),
        )
    )
    # 计算日度波动率（仅用于 regime 分类）
    _raw_ret = np.log(daily_close["CLOSE"] / daily_close["CLOSE"].shift(1))
    daily_close["daily_vol_20"] = _raw_ret.rolling(window).std()
    # ret_1d = 前一日全程收益，日内常数，无前视
    daily_close["ret_1d"] = _raw_ret.shift(1)

    train_daily_vol = daily_close.loc[
        daily_close["DATA_SPLIT"] == "train", "daily_vol_20"
    ].dropna()
    if train_daily_vol.empty:
        raise ValueError("train 区间没有可用的 daily_vol_20，无法计算 cutoff")

    if vol_threshold is not None:
        daily_cutoff = vol_threshold
    else:
        daily_cutoff = train_daily_vol.quantile(vol_percentage)

    daily_close["DAILY_VOL_LABEL"] = pd.Series(pd.NA, index=daily_close.index, dtype="Int64")
    daily_label_mask = daily_close["daily_vol_20"].notna()
    if label_train_only:
        daily_label_mask &= daily_close["DATA_SPLIT"] == "train"

    if rolling_regime_window > 0:
        # 滚动 regime：每天相对于过去 rolling_regime_window 天的 vol_percentage 分位数判断高/低波动。
        # 使用与固定阈值相同的分位数（vol_percentage），维持约 (1-vol_percentage) 的高波动占比，
        # 同时自适应波动率制度性变化，无前视。
        _pct = vol_percentage if vol_percentage is not None else 0.65
        rolling_cutoff = (
            daily_close["daily_vol_20"]
            .rolling(rolling_regime_window, min_periods=rolling_regime_window // 3)
            .quantile(_pct)
        )
        valid_mask = daily_label_mask & rolling_cutoff.notna()
        daily_close.loc[valid_mask, "DAILY_VOL_LABEL"] = _label_by_cutoff(
            daily_close.loc[valid_mask, "daily_vol_20"],
            rolling_cutoff[valid_mask],
        )
    else:
        daily_close.loc[daily_label_mask, "DAILY_VOL_LABEL"] = _label_by_cutoff(
            daily_close.loc[daily_label_mask, "daily_vol_20"],
            daily_cutoff,
        )
    # 按照月度划分，取当月日级波动率的平均值进行比较
    monthly_close = (
        daily_close.groupby("MONTH", as_index=False)
        .agg(
            monthly_vol=("daily_vol_20", "mean"),
            DATA_SPLIT=("DATA_SPLIT", "first"),
            month_start=("day_start", "min"),
            month_end=("day_end", "max"),
        )
    )

    train_monthly_vol = monthly_close.loc[
        monthly_close["DATA_SPLIT"] == "train", "monthly_vol"
    ].dropna()
    if train_monthly_vol.empty:
        raise ValueError("train 区间没有可用的 monthly_vol，无法计算 cutoff")

    if vol_threshold is not None:
        monthly_cutoff = vol_threshold
    else:
        monthly_cutoff = train_monthly_vol.quantile(vol_percentage)

    monthly_close["VOL_LABEL"] = pd.Series(pd.NA, index=monthly_close.index, dtype="Int64")
    month_label_mask = monthly_close["monthly_vol"].notna()
    if label_train_only:
        month_label_mask &= monthly_close["DATA_SPLIT"] == "train"
    monthly_close.loc[month_label_mask, "VOL_LABEL"] = _label_by_cutoff(
        monthly_close.loc[month_label_mask, "monthly_vol"],
        monthly_cutoff,
    )

    merged_data = df.merge(
        daily_close[
            [
                "TRADE_DATE",
                "daily_vol_20",
                "DAILY_VOL_LABEL",
                "ret_1d",
            ]
        ],
        on="TRADE_DATE",
        how="left",
    )
    merged_data = merged_data.merge(
        monthly_close[
            [
                "MONTH",
                "monthly_vol",
                "VOL_LABEL",
            ]
        ],
        on="MONTH",
        how="left",
    )

    low_vol = merged_data[merged_data["VOL_LABEL"] == -1].copy()
    high_vol = merged_data[merged_data["VOL_LABEL"] == 1].copy()

    merged_data.attrs["daily_cutoff"] = float(daily_cutoff)
    merged_data.attrs["monthly_cutoff"] = float(monthly_cutoff)
    merged_data.attrs["split_granularity"] = str(split_granularity)
    daily_close.attrs["daily_cutoff"] = float(daily_cutoff)
    monthly_close.attrs["monthly_cutoff"] = float(monthly_cutoff)
    return merged_data, low_vol, high_vol, daily_close, monthly_close


def summarize_daily_vol(daily_close):
    vol = daily_close["daily_vol_20"].dropna()
    if vol.empty:
        raise ValueError("daily_vol_20 全部为空，无法计算统计量")

    stats = {
        "count": int(vol.count()),
        "mean": float(vol.mean()),
        "median": float(vol.median()),
        "std": float(vol.std()),
        "min": float(vol.min()),
        "p10": float(vol.quantile(0.10)),
        "p25": float(vol.quantile(0.25)),
        "p75": float(vol.quantile(0.75)),
        "p90": float(vol.quantile(0.90)),
        "max": float(vol.max()),
        "daily_cutoff": float(daily_close.attrs.get("daily_cutoff", np.nan)),
    }
    return pd.Series(stats, name="daily_vol_20_stats")


def summarize_monthly_vol(monthly_close):
    vol = monthly_close["monthly_vol"].dropna()
    if vol.empty:
        raise ValueError("monthly_vol 全部为空，无法计算统计量")

    stats = {
        "count": int(vol.count()),
        "mean": float(vol.mean()),
        "median": float(vol.median()),
        "std": float(vol.std()),
        "min": float(vol.min()),
        "p10": float(vol.quantile(0.10)),
        "p25": float(vol.quantile(0.25)),
        "p75": float(vol.quantile(0.75)),
        "p90": float(vol.quantile(0.90)),
        "max": float(vol.max()),
        "monthly_cutoff": float(monthly_close.attrs.get("monthly_cutoff", np.nan)),
    }
    return pd.Series(stats, name="monthly_vol_stats")


def plot_target_return_by_vol(
    merged_data,
    daily_close=None,
    monthly_close=None,
    output_path=None,
    return_col: str = "future_return",
    target_horizon: int | None = None,
):
    """画 regime 划分图：bar-level 预测目标收益 + 日波动率 + 高/低波区段背景。

    return_col 默认走 ``future_return``（由 ``_add_targets`` 按 ``data.target_horizon``
    计算的同日 horizon 收益）。需要换其他列名（如策略 B 的极值目标）时显式指定。
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = _prepare_minute_data(merged_data)
    # _prepare_minute_data does data.copy(), so return_col is already in df if it was in merged_data.
    # No separate merge needed — merging again would rename to return_col_x / return_col_y.

    required_columns = {"VOL_LABEL", "DATA_SPLIT", return_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列: {sorted(missing_columns)}")

    if daily_close is None:
        daily_close = (
            df.groupby("TRADE_DATE", as_index=False)
            .agg(
                plot_time=("TDATE", "max"),
                daily_vol_20=("daily_vol_20", "first"),
            )
        )
    else:
        daily_close = daily_close.copy()
        if "plot_time" not in daily_close.columns:
            daily_close["plot_time"] = daily_close["day_end"]

    monthly_cutoff = None
    if monthly_close is not None:
        monthly_cutoff = monthly_close.attrs.get("monthly_cutoff")
    if monthly_cutoff is None:
        monthly_cutoff = merged_data.attrs.get("monthly_cutoff")

    horizon_label = f"h={target_horizon}" if target_horizon is not None else "horizon"
    return_title = f"{return_col} ({horizon_label}) with Monthly Volatility Labels"

    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.6, 1.8],
        height_ratios=[3, 1.5],
        wspace=0.18,
        hspace=0.28,
    )
    ax_return = fig.add_subplot(grid[0, 0])
    ax_vol = fig.add_subplot(grid[1, 0], sharex=ax_return)
    ax_high_concat = fig.add_subplot(grid[0, 1])
    ax_low_concat = fig.add_subplot(grid[1, 1])

    _add_monthly_background(ax_return, df)
    _add_monthly_background(ax_vol, df)

    plot_df = df.dropna(subset=[return_col])
    ax_return.plot(
        plot_df["TDATE"],
        plot_df[return_col],
        color=RETURN_LINE_COLOR,
        linewidth=0.8,
    )
    _add_split_boundaries(ax_return, df)

    vol_plot_df = daily_close.dropna(subset=["daily_vol_20"])
    ax_vol.plot(
        vol_plot_df["plot_time"],
        vol_plot_df["daily_vol_20"],
        color=DAILY_VOL_LINE_COLOR,
        linewidth=1.1,
    )
    if monthly_cutoff is not None and not pd.isna(monthly_cutoff):
        ax_vol.axhline(
            monthly_cutoff,
            color=CUTOFF_LINE_COLOR,
            linestyle="--",
            linewidth=1.3,
        )
    _add_split_boundaries(ax_vol, df)

    ax_return.set_title(return_title)
    ax_return.set_ylabel(return_col)
    ax_return.grid(alpha=0.15)
    ax_vol.set_title("Daily Volatility (daily_vol_20)")
    ax_vol.set_xlabel("TDATE")
    ax_vol.set_ylabel("daily_vol_20")
    ax_vol.grid(alpha=0.15)
    ax_vol.tick_params(axis="x", rotation=30)

    high_concat_df = _build_concatenated_regime_samples(df, vol_label=1, return_col=return_col)
    low_concat_df = _build_concatenated_regime_samples(df, vol_label=-1, return_col=return_col)

    ax_high_concat.plot(
        high_concat_df["concat_index"],
        high_concat_df[return_col],
        color=HIGH_VOL_COLOR,
        linewidth=0.8,
    )
    ax_high_concat.set_title(f"Concatenated High-Vol Samples (n={len(high_concat_df):,})")
    ax_high_concat.set_ylabel(return_col)
    ax_high_concat.set_xlabel("concatenated sample index")
    ax_high_concat.grid(alpha=0.15)

    ax_low_concat.plot(
        low_concat_df["concat_index"],
        low_concat_df[return_col],
        color=LOW_VOL_COLOR,
        linewidth=0.8,
    )
    ax_low_concat.set_title(f"Concatenated Low-Vol Samples (n={len(low_concat_df):,})")
    ax_low_concat.set_ylabel(return_col)
    ax_low_concat.set_xlabel("concatenated sample index")
    ax_low_concat.grid(alpha=0.15)

    legend_handles = [
        Line2D([0], [0], color=RETURN_LINE_COLOR, linewidth=1.2, label=return_col),
        Patch(facecolor=LOW_VOL_COLOR, edgecolor="none", alpha=VOL_SPAN_ALPHA, label="VOL_LABEL = -1"),
        Patch(facecolor=HIGH_VOL_COLOR, edgecolor="none", alpha=VOL_SPAN_ALPHA, label="VOL_LABEL = 1"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, label="split boundary"),
    ]
    ax_return.legend(handles=legend_handles, loc="upper right")

    vol_legend_handles = [
        Line2D([0], [0], color=DAILY_VOL_LINE_COLOR, linewidth=1.2, label="daily_vol_20"),
        Line2D(
            [0],
            [0],
            color=CUTOFF_LINE_COLOR,
            linestyle="--",
            linewidth=1.3,
            label=f"monthly_cutoff = {monthly_cutoff:.6f}" if monthly_cutoff is not None else "monthly_cutoff",
        ),
        Patch(facecolor=LOW_VOL_COLOR, edgecolor="none", alpha=VOL_SPAN_ALPHA, label="VOL_LABEL = -1"),
        Patch(facecolor=HIGH_VOL_COLOR, edgecolor="none", alpha=VOL_SPAN_ALPHA, label="VOL_LABEL = 1"),
    ]
    ax_vol.legend(handles=vol_legend_handles, loc="upper right")
    if monthly_cutoff is not None and not pd.isna(monthly_cutoff):
        ax_vol.set_title(f"Daily Volatility (daily_vol_20, monthly_cutoff={monthly_cutoff:.6f})")

    ax_high_concat.legend(
        handles=[Line2D([0], [0], color=HIGH_VOL_COLOR, linewidth=1.2, label=f"high-vol {return_col}")],
        loc="upper right",
    )
    ax_low_concat.legend(
        handles=[Line2D([0], [0], color=LOW_VOL_COLOR, linewidth=1.2, label=f"low-vol {return_col}")],
        loc="upper right",
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, (ax_return, ax_vol, ax_high_concat, ax_low_concat)
