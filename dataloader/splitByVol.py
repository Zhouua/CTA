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

# 分钟级数据聚合成日级数据
def _prepare_minute_data(data):
    df = data.copy()
    df["TDATE"] = pd.to_datetime(df["TDATE"])
    df = df.sort_values("TDATE").reset_index(drop=True)

    if "5min_return" not in df.columns:
        df["5min_return"] = np.log(df["CLOSE"].shift(-5) / df["CLOSE"])

    df["TRADE_DATE"] = df["TDATE"].dt.normalize()
    df["MONTH"] = df["TDATE"].dt.to_period("M")
    return df


def _build_month_split_map(data, train_ratio, valid_ratio, test_ratio):
    _validate_split_ratio(train_ratio, valid_ratio, test_ratio)

    months = pd.Series(data["MONTH"].drop_duplicates().sort_values().tolist())
    total_months = len(months)
    train_months = int(total_months * train_ratio)
    valid_months = int(total_months * valid_ratio)

    split_map = pd.DataFrame({"MONTH": months})
    split_map["DATA_SPLIT"] = "test"
    split_map.loc[: train_months - 1, "DATA_SPLIT"] = "train"
    split_map.loc[train_months : train_months + valid_months - 1, "DATA_SPLIT"] = "valid"
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


def _build_concatenated_regime_samples(data, vol_label):
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


def export_concatenated_regime_data(merged_data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    high_concat_df = _build_concatenated_regime_samples(merged_data, vol_label=1)
    low_concat_df = _build_concatenated_regime_samples(merged_data, vol_label=-1)

    high_path = output_dir / "high_vol_concatenated.csv"
    low_path = output_dir / "low_vol_concatenated.csv"
    high_concat_df.to_csv(high_path, index=False)
    low_concat_df.to_csv(low_path, index=False)
    return high_concat_df, low_concat_df, high_path, low_path


def split_by_vol(
    data,
    vol_threshold=None,
    vol_percentage=None,
    window=20,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    label_train_only=True,
):
    if (vol_threshold is None) == (vol_percentage is None):
        raise ValueError("vol_threshold 和 vol_percentage 必须二选一")

    if vol_percentage is not None and not 0 < vol_percentage < 1:
        raise ValueError("vol_percentage 必须在 0 和 1 之间")

    df = _prepare_minute_data(data)
    split_map = _build_month_split_map(df, train_ratio, valid_ratio, test_ratio)
    df = df.merge(split_map, on="MONTH", how="left")

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
    # 计算日度波动率
    daily_close["daily_ret"] = np.log(
        daily_close["CLOSE"] / daily_close["CLOSE"].shift(1)
    )
    daily_close["daily_vol_20"] = daily_close["daily_ret"].rolling(window).std()

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
                "daily_ret",
                "daily_vol_20",
                "DAILY_VOL_LABEL",
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


def plot_5min_return_by_vol(merged_data, daily_close=None, monthly_close=None, output_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = _prepare_minute_data(merged_data)

    required_columns = {"VOL_LABEL", "DATA_SPLIT"}
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

    plot_df = df.dropna(subset=["5min_return"])
    ax_return.plot(
        plot_df["TDATE"],
        plot_df["5min_return"],
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

    ax_return.set_title("5min_return with Monthly Volatility Labels")
    ax_return.set_ylabel("5min_return")
    ax_return.grid(alpha=0.15)
    ax_vol.set_title("Daily Volatility (daily_vol_20)")
    ax_vol.set_xlabel("TDATE")
    ax_vol.set_ylabel("daily_vol_20")
    ax_vol.grid(alpha=0.15)
    ax_vol.tick_params(axis="x", rotation=30)

    high_concat_df = _build_concatenated_regime_samples(df, vol_label=1)
    low_concat_df = _build_concatenated_regime_samples(df, vol_label=-1)

    ax_high_concat.plot(
        high_concat_df["concat_index"],
        high_concat_df["5min_return"],
        color=HIGH_VOL_COLOR,
        linewidth=0.8,
    )
    ax_high_concat.set_title(f"Concatenated High-Vol Samples (n={len(high_concat_df):,})")
    ax_high_concat.set_ylabel("5min_return")
    ax_high_concat.set_xlabel("concatenated sample index")
    ax_high_concat.grid(alpha=0.15)

    ax_low_concat.plot(
        low_concat_df["concat_index"],
        low_concat_df["5min_return"],
        color=LOW_VOL_COLOR,
        linewidth=0.8,
    )
    ax_low_concat.set_title(f"Concatenated Low-Vol Samples (n={len(low_concat_df):,})")
    ax_low_concat.set_ylabel("5min_return")
    ax_low_concat.set_xlabel("concatenated sample index")
    ax_low_concat.grid(alpha=0.15)

    legend_handles = [
        Line2D([0], [0], color=RETURN_LINE_COLOR, linewidth=1.2, label="5min_return"),
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

    concat_high_legend = [
        Line2D([0], [0], color=HIGH_VOL_COLOR, linewidth=1.2, label="high-vol 5min_return"),
    ]
    concat_low_legend = [
        Line2D([0], [0], color=LOW_VOL_COLOR, linewidth=1.2, label="low-vol 5min_return"),
    ]
    ax_high_concat.legend(handles=concat_high_legend, loc="upper right")
    ax_low_concat.legend(handles=concat_low_legend, loc="upper right")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, (ax_return, ax_vol, ax_high_concat, ax_low_concat)


def split_and_plot_by_vol(
    data,
    vol_threshold=None,
    vol_percentage=None,
    window=20,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    label_train_only=True,
    output_path=None,
    concat_output_dir=None,
):
    merged_data, low_vol, high_vol, daily_close, monthly_close = split_by_vol(
        data=data,
        vol_threshold=vol_threshold,
        vol_percentage=vol_percentage,
        window=window,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        label_train_only=label_train_only,
    )
    fig, axes = plot_5min_return_by_vol(
        merged_data,
        daily_close=daily_close,
        monthly_close=monthly_close,
        output_path=output_path,
    )
    high_concat_df = _build_concatenated_regime_samples(merged_data, vol_label=1)
    low_concat_df = _build_concatenated_regime_samples(merged_data, vol_label=-1)

    concat_paths = None
    if concat_output_dir is not None:
        _, _, high_path, low_path = export_concatenated_regime_data(
            merged_data,
            output_dir=concat_output_dir,
        )
        concat_paths = (high_path, low_path)

    return (
        merged_data,
        low_vol,
        high_vol,
        daily_close,
        monthly_close,
        high_concat_df,
        low_concat_df,
        fig,
        axes,
        concat_paths,
    )


if __name__ == "__main__":
    data_path = Path("data/RBZL.SHF.csv")
    raw_data = pd.read_csv(
        data_path,
        dtype={"CONTRACT": str, "CONTRACTID": str},
        index_col=0,
    )

    (
        merged_data,
        low_vol,
        high_vol,
        daily_close,
        monthly_close,
        high_concat_df,
        low_concat_df,
        fig,
        axes,
        concat_paths,
    ) = split_and_plot_by_vol(
        raw_data,
        vol_percentage=0.7,
        label_train_only=True,
        output_path="plots/5min_return_by_vol.png",
        concat_output_dir="data/concatenated_regimes",
    )

    print(merged_data[["TDATE", "DATA_SPLIT", "DAILY_VOL_LABEL", "VOL_LABEL"]].head())
    print(summarize_daily_vol(daily_close))
    print(summarize_monthly_vol(monthly_close))
    print(f"monthly_cutoff used for split: {monthly_close.attrs['monthly_cutoff']:.6f}")
    print(f"low_vol rows: {len(low_vol)}")
    print(f"high_vol rows: {len(high_vol)}")
    print(high_concat_df[["TDATE", "MONTH", "BLOCK_ID", "concat_index", "VOL_LABEL"]].head())
    print(low_concat_df[["TDATE", "MONTH", "BLOCK_ID", "concat_index", "VOL_LABEL"]].head())
    if concat_paths is not None:
        print(f"high-vol concatenated csv: {concat_paths[0]}")
        print(f"low-vol concatenated csv: {concat_paths[1]}")
