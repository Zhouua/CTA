from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-8
DEFAULT_WINDOWS = [5, 10, 20, 30, 60]
DEFAULT_LAGS = [0, 1, 2, 3, 4]
DEFAULT_GROUPS = [
    "price_lags",
    "kline",
    "trend",
    "volatility",
    "direction",
    "volume",
    "correlation",
]


@dataclass
class RuntimeFactorResult:
    frame: pd.DataFrame
    factor_cols: list[str]
    group_map: dict[str, list[str]]
    manifest: dict[str, Any]


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def normalize_runtime_factor_config(runtime_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime_cfg = runtime_cfg or {}
    groups = runtime_cfg.get("groups", DEFAULT_GROUPS)
    normalized_groups = [str(group) for group in groups] if groups else list(DEFAULT_GROUPS)

    config = {
        "enabled": bool(runtime_cfg.get("enabled", False)),
        "windows": sorted({int(window) for window in runtime_cfg.get("windows", DEFAULT_WINDOWS)}),
        "lags": sorted({int(lag) for lag in runtime_cfg.get("lags", DEFAULT_LAGS)}),
        "groups": normalized_groups,
        "cache_generated_features": bool(runtime_cfg.get("cache_generated_features", True)),
    }
    config["spec_hash"] = _stable_hash(
        {
            "windows": config["windows"],
            "lags": config["lags"],
            "groups": config["groups"],
        }
    )
    return config


def _rolling_position(series: pd.Series, window: int, fn_name: str) -> pd.Series:
    def _calc(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        if fn_name == "argmax":
            return float((np.argmax(values) + 1) / len(values))
        return float((np.argmin(values) + 1) / len(values))

    return series.rolling(window, min_periods=window).apply(_calc, raw=True)


def _rolling_positive_ratio(series: pd.Series, window: int) -> pd.Series:
    return (series > 0).astype("float32").rolling(window, min_periods=window).mean()


def _rolling_negative_ratio(series: pd.Series, window: int) -> pd.Series:
    return (series < 0).astype("float32").rolling(window, min_periods=window).mean()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype("float64") / denominator.astype("float64").replace(0.0, np.nan)


def _append_columns(output: pd.DataFrame, group_map: dict[str, list[str]], group_name: str, columns: dict[str, pd.Series]) -> pd.DataFrame:
    output = pd.concat(
        [
            output,
            pd.DataFrame({col_name: series.astype("float32") for col_name, series in columns.items()}, index=output.index),
        ],
        axis=1,
    )
    group_map[group_name] = sorted(columns.keys())
    return output


def generate_runtime_factors(raw_df: pd.DataFrame, runtime_cfg: dict[str, Any] | None = None) -> RuntimeFactorResult:
    config = normalize_runtime_factor_config(runtime_cfg)
    if not config["enabled"]:
        return RuntimeFactorResult(
            frame=raw_df.copy(),
            factor_cols=[],
            group_map={},
            manifest={
                "enabled": False,
                "windows": config["windows"],
                "lags": config["lags"],
                "groups": config["groups"],
                "spec_hash": config["spec_hash"],
            },
        )

    output = raw_df.copy()
    close = pd.to_numeric(output["CLOSE"], errors="coerce").astype("float64")
    open_ = pd.to_numeric(output["OPEN"], errors="coerce").astype("float64")
    high = pd.to_numeric(output["HIGH"], errors="coerce").astype("float64")
    low = pd.to_numeric(output["LOW"], errors="coerce").astype("float64")
    volume = (
        pd.to_numeric(output.get("VOLUME", pd.Series(0.0, index=output.index)), errors="coerce")
        .fillna(0.0)
        .astype("float64")
    )
    amount = (
        pd.to_numeric(output.get("AMOUNT", pd.Series(np.nan, index=output.index)), errors="coerce")
        .astype("float64")
    )

    ret1 = close.pct_change()
    log_ret1 = np.log(close / close.shift(1))
    dvol1 = np.log1p(volume).diff()
    vwap = amount / volume.replace(0.0, np.nan)
    vwap = vwap.where(np.isfinite(vwap), close)
    output["RET1"] = ret1.astype("float32")
    output["LOGRET1"] = log_ret1.astype("float32")
    output["DVOL1"] = dvol1.astype("float32")
    output["VWAP"] = vwap.astype("float32")

    windows = config["windows"]
    lags = config["lags"]
    groups = set(config["groups"])
    group_map: dict[str, list[str]] = {}

    if "price_lags" in groups:
        lag_cols: dict[str, pd.Series] = {}
        for lag in lags:
            lag_cols[f"OPEN{lag}"] = _safe_divide(open_.shift(lag), close)
            lag_cols[f"HIGH{lag}"] = _safe_divide(high.shift(lag), close)
            lag_cols[f"LOW{lag}"] = _safe_divide(low.shift(lag), close)
        output = _append_columns(output, group_map, "price_lags", lag_cols)

    if "kline" in groups:
        candle_range = (high - low).replace(0.0, np.nan)
        body_high = pd.concat([open_, close], axis=1).max(axis=1)
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        close_location = (2.0 * close) - high - low
        kline_cols = {
            "KLEN": _safe_divide(high - low, open_),
            "KUP": _safe_divide(high - body_high, open_),
            "KLOW": _safe_divide(body_low - low, open_),
            "KMID": _safe_divide(close - open_, open_),
            "KSFT": _safe_divide(close_location, open_),
            "KUP2": _safe_divide(high - body_high, candle_range),
            "KLOW2": _safe_divide(body_low - low, candle_range),
            "KMID2": _safe_divide(close - open_, candle_range),
            "KSFT2": _safe_divide(close_location, candle_range),
        }
        output = _append_columns(output, group_map, "kline", kline_cols)

    if "trend" in groups:
        trend_cols: dict[str, pd.Series] = {}
        for window in windows:
            trend_cols[f"MA{window}"] = _safe_divide(close.rolling(window, min_periods=window).mean(), close)
            trend_cols[f"ROC{window}"] = _safe_divide(close.shift(window), close)
            trend_cols[f"MAX{window}"] = _safe_divide(high.rolling(window, min_periods=window).max(), close)
            trend_cols[f"MIN{window}"] = _safe_divide(low.rolling(window, min_periods=window).min(), close)
        output = _append_columns(output, group_map, "trend", trend_cols)

    if "volatility" in groups:
        vol_cols: dict[str, pd.Series] = {}
        for window in windows:
            rolling_low = low.rolling(window, min_periods=window).min()
            rolling_high = high.rolling(window, min_periods=window).max()
            vol_cols[f"STD{window}"] = _safe_divide(close.rolling(window, min_periods=window).std(), close)
            vol_cols[f"RSV{window}"] = _safe_divide(close - rolling_low, rolling_high - rolling_low + EPSILON)
            vol_cols[f"QTLU{window}"] = _safe_divide(
                close.rolling(window, min_periods=window).quantile(0.8),
                close,
            )
            vol_cols[f"QTLD{window}"] = _safe_divide(
                close.rolling(window, min_periods=window).quantile(0.2),
                close,
            )
            vol_cols[f"IMAX{window}"] = _rolling_position(high, window, "argmax")
            vol_cols[f"IMIN{window}"] = _rolling_position(low, window, "argmin")
            vol_cols[f"IMXD{window}"] = vol_cols[f"IMAX{window}"] - vol_cols[f"IMIN{window}"]
        output = _append_columns(output, group_map, "volatility", vol_cols)

    if "direction" in groups:
        direction_cols: dict[str, pd.Series] = {}
        positive_ret = ret1.clip(lower=0.0)
        negative_ret = (-ret1.clip(upper=0.0)).abs()
        abs_ret_sum = ret1.abs()
        for window in windows:
            direction_cols[f"CNTP{window}"] = _rolling_positive_ratio(ret1, window)
            direction_cols[f"CNTN{window}"] = _rolling_negative_ratio(ret1, window)
            direction_cols[f"CNTD{window}"] = direction_cols[f"CNTP{window}"] - direction_cols[f"CNTN{window}"]
            abs_ret_roll = abs_ret_sum.rolling(window, min_periods=window).sum()
            direction_cols[f"SUMP{window}"] = _safe_divide(
                positive_ret.rolling(window, min_periods=window).sum(),
                abs_ret_roll,
            )
            direction_cols[f"SUMN{window}"] = _safe_divide(
                negative_ret.rolling(window, min_periods=window).sum(),
                abs_ret_roll,
            )
            direction_cols[f"SUMD{window}"] = direction_cols[f"SUMP{window}"] - direction_cols[f"SUMN{window}"]
        output = _append_columns(output, group_map, "direction", direction_cols)

    if "volume" in groups:
        volume_cols: dict[str, pd.Series] = {}
        vol_diff = volume.diff()
        abs_ret_volume = ret1.abs() * volume
        for lag in lags:
            volume_cols[f"VOLUME{lag}"] = _safe_divide(volume.shift(lag), volume)
        for window in windows:
            volume_cols[f"VMA{window}"] = _safe_divide(volume.rolling(window, min_periods=window).mean(), volume)
            volume_cols[f"VSTD{window}"] = _safe_divide(volume.rolling(window, min_periods=window).std(), volume)
            abs_vol_diff_roll = vol_diff.abs().rolling(window, min_periods=window).sum()
            volume_cols[f"VSUMP{window}"] = _safe_divide(
                vol_diff.clip(lower=0.0).rolling(window, min_periods=window).sum(),
                abs_vol_diff_roll,
            )
            volume_cols[f"VSUMN{window}"] = _safe_divide(
                (-vol_diff.clip(upper=0.0)).abs().rolling(window, min_periods=window).sum(),
                abs_vol_diff_roll,
            )
            volume_cols[f"VSUMD{window}"] = volume_cols[f"VSUMP{window}"] - volume_cols[f"VSUMN{window}"]
            volume_cols[f"WVMA{window}"] = _safe_divide(
                abs_ret_volume.rolling(window, min_periods=window).std(),
                abs_ret_volume.rolling(window, min_periods=window).mean() + EPSILON,
            )
        output = _append_columns(output, group_map, "volume", volume_cols)

    if "correlation" in groups:
        corr_cols: dict[str, pd.Series] = {}
        volume_ratio = _safe_divide(volume, volume.shift(1))
        log_volume_ratio = np.log1p(volume_ratio)
        close_ratio = _safe_divide(close, close.shift(1))
        for window in windows:
            corr_cols[f"CORR{window}"] = close.rolling(window, min_periods=window).corr(np.log1p(volume))
            corr_cols[f"CORD{window}"] = close_ratio.rolling(window, min_periods=window).corr(log_volume_ratio)
        output = _append_columns(output, group_map, "correlation", corr_cols)

    factor_cols = [col for cols in group_map.values() for col in cols]
    manifest = {
        "enabled": True,
        "windows": windows,
        "lags": lags,
        "groups": config["groups"],
        "spec_hash": config["spec_hash"],
        "factor_count": len(factor_cols),
        "factor_groups": {group: len(cols) for group, cols in group_map.items()},
        "factor_cols": factor_cols,
    }
    return RuntimeFactorResult(frame=output, factor_cols=factor_cols, group_map=group_map, manifest=manifest)
