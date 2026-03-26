from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CURRENT_DIR_STR = str(CURRENT_DIR)
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if CURRENT_DIR_STR not in sys.path:
    sys.path.insert(0, CURRENT_DIR_STR)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from config_utils import get_section, load_project_config, resolve_paths
from dataloader.splitByVol import (
    plot_5min_return_by_vol,
    split_by_vol,
    summarize_daily_vol,
    summarize_monthly_vol,
)


REGIME_NAME_MAP = {-1: "low_vol", 1: "high_vol"}


@dataclass
class PreparedData:
    full_data: pd.DataFrame
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    feature_cols: list[str]
    factor_cols: list[str]
    engineered_cols: list[str]
    target_col: str
    metadata: dict[str, Any]


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


def build_data_settings(config_path: str | None = None) -> dict[str, Any]:
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(
        config_dir,
        get_section(config, "paths"),
        ["raw_data", "factor_dir", "merged_cache", "cache_meta", "regime_plot"],
    )
    data_cfg = get_section(config, "data")
    vol_cfg = get_section(config, "vol_split")
    model_cfg = get_section(config, "model")

    return {
        "config_path": str((Path(config_path).expanduser().resolve() if config_path else (PROJECT_ROOT / "config.yaml"))),
        "paths": paths,
        "timestamp_col": str(data_cfg.get("timestamp_col", "TDATE")),
        "trade_date_col": str(data_cfg.get("trade_date_col", "TRADE_DATE")),
        "target_horizon": int(data_cfg.get("target_horizon", 5)),
        "factor_pattern": str(data_cfg.get("factor_pattern", "*.csv")),
        "factor_include": list(data_cfg.get("factor_include", [])),
        "factor_exclude": list(data_cfg.get("factor_exclude", [])),
        "max_factor_missing_ratio": float(data_cfg.get("max_factor_missing_ratio", 0.35)),
        "min_factor_std": float(data_cfg.get("min_factor_std", 1e-8)),
        "fill_method": str(data_cfg.get("fill_method", "forward_fill")),
        "cache_merged_dataset": bool(data_cfg.get("cache_merged_dataset", True)),
        "force_rebuild_cache": bool(data_cfg.get("force_rebuild_cache", False)),
        "use_engineered_features": bool(data_cfg.get("use_engineered_features", True)),
        "engineered_windows": dict(data_cfg.get("engineered_windows", {})),
        "train_ratio": float(data_cfg.get("train_ratio", 0.7)),
        "valid_ratio": float(data_cfg.get("valid_ratio", 0.15)),
        "test_ratio": float(data_cfg.get("test_ratio", 0.15)),
        "vol_window": int(vol_cfg.get("window", 20)),
        "vol_percentage": float(vol_cfg.get("vol_percentage", 0.7)),
        "label_train_only": bool(vol_cfg.get("label_train_only", False)),
        "regime_label_source": str(vol_cfg.get("regime_label_source", "daily")).lower(),
        "min_train_rows_per_regime": int(vol_cfg.get("min_train_rows_per_regime", 10000)),
        "target_col": str(model_cfg.get("target_column", "target_vol_norm")),
        "target_vol_window": int(model_cfg.get("target_vol_window", 20)),
        "target_vol_epsilon": float(model_cfg.get("target_vol_epsilon", 1e-8)),
        "target_vol_floor_quantile": float(model_cfg.get("target_vol_floor_quantile", 0.05)),
    }


class FactorDatasetBuilder:
    def __init__(self, config_path: str | None = None):
        self.settings = build_data_settings(config_path)
        self.paths = self.settings["paths"]
        self.timestamp_col = self.settings["timestamp_col"]
        self.trade_date_col = self.settings["trade_date_col"]

    def _read_raw_data(self) -> pd.DataFrame:
        dtype_map = {
            "CODE": str,
            "CONTRACT": str,
            "CONTRACTID": str,
            "MARKET": str,
            "product": str,
        }
        raw = pd.read_csv(self.paths["raw_data"], dtype=dtype_map, low_memory=False)
        unnamed_cols = [col for col in raw.columns if str(col).startswith("Unnamed:")]
        raw = raw.drop(columns=unnamed_cols, errors="ignore")
        raw[self.timestamp_col] = pd.to_datetime(raw[self.timestamp_col])
        raw = raw.sort_values(self.timestamp_col).drop_duplicates(subset=[self.timestamp_col]).reset_index(drop=True)
        raw[self.trade_date_col] = raw[self.timestamp_col].dt.normalize()

        numeric_candidates = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMOUNT", "POSITION"]
        for col in numeric_candidates:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("float64")
        return raw

    def _discover_factor_files(self) -> list[Path]:
        factor_dir = self.paths["factor_dir"]
        files = sorted(factor_dir.glob(self.settings["factor_pattern"]))
        include = set(self.settings["factor_include"])
        exclude = set(self.settings["factor_exclude"])

        selected: list[Path] = []
        for path in files:
            stem = path.stem
            short_name = stem.split("RBZL_", 1)[-1] if stem.startswith("RBZL_") else stem
            if include and stem not in include and short_name not in include:
                continue
            if stem in exclude or short_name in exclude:
                continue
            selected.append(path)
        return selected

    def _read_factor_series(self, factor_path: Path) -> pd.Series:
        factor_df = pd.read_csv(factor_path, low_memory=False)
        timestamp_col = next((col for col in factor_df.columns if col.lower() == "tdate"), None)
        if timestamp_col is None:
            raise ValueError(f"Factor file {factor_path} does not contain a tdate column.")
        value_cols = [col for col in factor_df.columns if col != timestamp_col]
        if len(value_cols) != 1:
            raise ValueError(f"Factor file {factor_path} must contain exactly one factor column.")

        factor_name = value_cols[0]
        factor_df = factor_df[[timestamp_col, factor_name]].copy()
        factor_df[timestamp_col] = pd.to_datetime(factor_df[timestamp_col])
        factor_df = factor_df.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="last")
        series = pd.to_numeric(factor_df[factor_name], errors="coerce").astype("float32")
        series.index = factor_df[timestamp_col]
        series.name = factor_name
        return series

    def _merge_factors(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        factor_series = [self._read_factor_series(path) for path in self._discover_factor_files()]
        if factor_series:
            factor_frame = pd.concat(factor_series, axis=1, join="outer").sort_index()
            merged = raw_df.set_index(self.timestamp_col).join(factor_frame, how="left").reset_index()
            factor_cols = factor_frame.columns.tolist()
        else:
            merged = raw_df.copy()
            factor_cols = []
        return merged, factor_cols

    def _add_engineered_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if not self.settings["use_engineered_features"]:
            return df, []

        engineered = df.copy()
        engineered = engineered.sort_values(self.timestamp_col).reset_index(drop=True)
        close = engineered["CLOSE"].astype("float64")
        high = engineered["HIGH"].astype("float64")
        low = engineered["LOW"].astype("float64")
        open_ = engineered["OPEN"].astype("float64")
        volume = engineered["VOLUME"].astype("float64") if "VOLUME" in engineered.columns else pd.Series(0.0, index=engineered.index)
        position = engineered["POSITION"].astype("float64") if "POSITION" in engineered.columns else pd.Series(0.0, index=engineered.index)
        amount = engineered["AMOUNT"].astype("float64") if "AMOUNT" in engineered.columns else pd.Series(0.0, index=engineered.index)

        short_w = int(self.settings["engineered_windows"].get("short", 5))
        med_w = int(self.settings["engineered_windows"].get("medium", 20))
        long_w = int(self.settings["engineered_windows"].get("long", 60))
        eps = 1e-8

        ret1 = close.pct_change()
        high_low_range = high - low
        true_range = pd.concat(
            [
                high_low_range,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        engineered["ENG_RET_1"] = ret1
        engineered["ENG_LOG_RET_1"] = np.log(close / close.shift(1))
        engineered["ENG_RANGE_1"] = high / (low + eps) - 1.0
        engineered["ENG_BODY_1"] = close / (open_ + eps) - 1.0
        engineered["ENG_BODY_ABS"] = engineered["ENG_BODY_1"].abs()
        engineered["ENG_CLOSE_TO_RANGE"] = (close - low) / (high_low_range + eps)
        engineered["ENG_INTRABAR_VOL"] = high_low_range / (close + eps)

        for window in sorted({short_w, med_w, long_w}):
            min_periods = max(3, window // 3)
            ma = close.rolling(window, min_periods=min_periods).mean()
            rv = ret1.rolling(window, min_periods=min_periods).std()
            engineered[f"ENG_RET_{window}"] = close.pct_change(window)
            engineered[f"ENG_RV_{window}"] = rv
            engineered[f"ENG_PRICE_TO_MA_{window}"] = close / (ma + eps) - 1.0
            engineered[f"ENG_VOLUME_RATIO_{window}"] = volume / (volume.rolling(window, min_periods=min_periods).mean() + eps)
            engineered[f"ENG_POSITION_RATIO_{window}"] = position / (position.rolling(window, min_periods=min_periods).mean() + eps)
            engineered[f"ENG_AMOUNT_RATIO_{window}"] = amount / (amount.rolling(window, min_periods=min_periods).mean() + eps)

        engineered[f"ENG_ATR_{med_w}"] = true_range.rolling(med_w, min_periods=max(3, med_w // 3)).mean() / (close + eps)
        engineered[f"ENG_ATR_{long_w}"] = true_range.rolling(long_w, min_periods=max(3, long_w // 3)).mean() / (close + eps)
        engineered[f"ENG_VOL_RATIO_{short_w}_{med_w}"] = engineered[f"ENG_RV_{short_w}"] / (engineered[f"ENG_RV_{med_w}"] + eps)
        engineered[f"ENG_RET_DIFF_{short_w}_{med_w}"] = engineered[f"ENG_RET_{short_w}"] - engineered[f"ENG_RET_{med_w}"]
        engineered[f"ENG_PRICE_BREAKOUT_{med_w}"] = close / (
            high.shift(1).rolling(med_w, min_periods=max(3, med_w // 3)).max() + eps
        ) - 1.0
        engineered[f"ENG_PRICE_BREAKDOWN_{med_w}"] = close / (
            low.shift(1).rolling(med_w, min_periods=max(3, med_w // 3)).min() + eps
        ) - 1.0

        minute_of_day = engineered[self.timestamp_col].dt.hour * 60 + engineered[self.timestamp_col].dt.minute
        engineered["ENG_TOD_SIN"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
        engineered["ENG_TOD_COS"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
        engineered["ENG_WEEKDAY"] = engineered[self.timestamp_col].dt.dayofweek.astype("float32")
        engineered["ENG_IS_DAY_SESSION"] = engineered[self.timestamp_col].dt.hour.between(9, 15).astype("float32")

        engineered_cols = [col for col in engineered.columns if col.startswith("ENG_")]
        engineered[engineered_cols] = engineered[engineered_cols].astype("float32")
        return engineered, engineered_cols

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy().sort_values(self.timestamp_col).reset_index(drop=True)
        horizon = int(self.settings["target_horizon"])
        vol_window = int(self.settings["target_vol_window"])
        eps = float(self.settings["target_vol_epsilon"])
        floor_quantile = float(self.settings["target_vol_floor_quantile"])

        same_day_future_close = prepared.groupby(self.trade_date_col, dropna=False)["CLOSE"].shift(-horizon)
        prepared["future_close"] = same_day_future_close
        prepared["future_return"] = prepared["future_close"] / prepared["CLOSE"] - 1.0
        prepared["target"] = prepared["future_return"].astype("float32")

        daily_ret_1 = prepared.groupby(self.trade_date_col, dropna=False)["CLOSE"].pct_change()
        intraday_vol = (
            daily_ret_1.groupby(prepared[self.trade_date_col], dropna=False)
            .rolling(vol_window, min_periods=max(3, vol_window // 3))
            .std()
            .reset_index(level=0, drop=True)
        )
        target_vol_scale = intraday_vol * np.sqrt(float(horizon))
        finite_scale = target_vol_scale.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_scale.empty:
            floor_value = 0.0
        else:
            floor_value = float(np.nanquantile(finite_scale, floor_quantile))
        prepared["target_vol_scale"] = np.maximum(target_vol_scale, floor_value).astype("float32")
        prepared["target_vol_norm"] = (prepared["future_return"] / (prepared["target_vol_scale"] + eps)).astype("float32")

        # splitByVol uses this column name in its plot helper.
        prepared["5min_return"] = np.log(
            prepared.groupby(self.trade_date_col, dropna=False)["CLOSE"].shift(-5) / prepared["CLOSE"]
        ).astype("float32")
        return prepared

    def _write_cache_meta(self, feature_df: pd.DataFrame, factor_cols: list[str], engineered_cols: list[str]) -> None:
        meta = {
            "rows": int(len(feature_df)),
            "columns": int(len(feature_df.columns)),
            "factor_count": int(len(factor_cols)),
            "engineered_count": int(len(engineered_cols)),
            "target_horizon": int(self.settings["target_horizon"]),
            "date_range": {
                "start": feature_df[self.timestamp_col].min(),
                "end": feature_df[self.timestamp_col].max(),
            },
        }
        self.paths["cache_meta"].parent.mkdir(parents=True, exist_ok=True)
        with self.paths["cache_meta"].open("w", encoding="utf-8") as f:
            json.dump(_to_native(meta), f, indent=2, ensure_ascii=False)

    def build_feature_frame(self) -> tuple[pd.DataFrame, list[str], list[str]]:
        raw_df = self._read_raw_data()
        merged_df, factor_cols = self._merge_factors(raw_df)
        merged_df, engineered_cols = self._add_engineered_features(merged_df)
        merged_df = self._add_targets(merged_df)

        self.paths["merged_cache"].parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(self.paths["merged_cache"], index=False)
        self._write_cache_meta(merged_df, factor_cols, engineered_cols)
        return merged_df, factor_cols, engineered_cols

    def load_or_build_feature_frame(self, force_rebuild: bool | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
        use_cache = bool(self.settings["cache_merged_dataset"])
        force_rebuild = self.settings["force_rebuild_cache"] if force_rebuild is None else force_rebuild

        if use_cache and self.paths["merged_cache"].exists() and not force_rebuild:
            cached = pd.read_parquet(self.paths["merged_cache"])
            cached[self.timestamp_col] = pd.to_datetime(cached[self.timestamp_col])
            cached[self.trade_date_col] = pd.to_datetime(cached[self.trade_date_col])
            factor_cols = [col for col in cached.columns if not col.startswith("ENG_") and col not in self._non_factor_columns()]
            engineered_cols = [col for col in cached.columns if col.startswith("ENG_")]
            return cached, factor_cols, engineered_cols
        return self.build_feature_frame()

    def _non_factor_columns(self) -> set[str]:
        return {
            self.timestamp_col,
            self.trade_date_col,
            "CODE",
            "CONTRACT",
            "CONTRACTID",
            "MARKET",
            "product",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOLUME",
            "AMOUNT",
            "POSITION",
            "future_close",
            "future_return",
            "target",
            "target_vol_scale",
            "target_vol_norm",
            "5min_return",
        }

    def prepare(self, force_rebuild: bool | None = None) -> PreparedData:
        feature_df, factor_cols, engineered_cols = self.load_or_build_feature_frame(force_rebuild=force_rebuild)
        merged_data, _, _, daily_close, monthly_close = split_by_vol(
            data=feature_df,
            vol_percentage=self.settings["vol_percentage"],
            window=self.settings["vol_window"],
            train_ratio=self.settings["train_ratio"],
            valid_ratio=self.settings["valid_ratio"],
            test_ratio=self.settings["test_ratio"],
            label_train_only=self.settings["label_train_only"],
        )

        os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
        plot_5min_return_by_vol(
            merged_data=merged_data,
            daily_close=daily_close,
            monthly_close=monthly_close,
            output_path=self.paths["regime_plot"],
        )

        source = self.settings["regime_label_source"]
        if source == "daily":
            regime_col = "DAILY_VOL_LABEL"
            cutoff_key = "daily_cutoff"
        elif source == "monthly":
            regime_col = "VOL_LABEL"
            cutoff_key = "monthly_cutoff"
        else:
            raise ValueError("vol_split.regime_label_source must be 'daily' or 'monthly'.")

        merged_data["REGIME_LABEL"] = pd.to_numeric(merged_data[regime_col], errors="coerce")
        merged_data["REGIME_NAME"] = merged_data["REGIME_LABEL"].map(REGIME_NAME_MAP)
        merged_data["DATA_SPLIT"] = merged_data["DATA_SPLIT"].replace({"valid": "val"})

        extra_feature_cols = []
        if "daily_vol_20" in merged_data.columns:
            extra_feature_cols.append("daily_vol_20")
        if "daily_ret" in merged_data.columns:
            extra_feature_cols.append("daily_ret")

        candidate_cols = list(dict.fromkeys(factor_cols + engineered_cols + extra_feature_cols))
        candidate_cols = [col for col in candidate_cols if col in merged_data.columns]
        merged_data[candidate_cols] = merged_data[candidate_cols].replace([np.inf, -np.inf], np.nan)

        fill_method = self.settings["fill_method"]
        if fill_method == "forward_fill":
            merged_data[candidate_cols] = merged_data[candidate_cols].ffill()
        elif fill_method != "none":
            raise ValueError(f"Unsupported fill method: {fill_method}")

        train_mask = merged_data["DATA_SPLIT"] == "train"
        train_snapshot = merged_data.loc[train_mask, candidate_cols].copy()

        usable_factor_cols = []
        if factor_cols:
            factor_missing = train_snapshot[factor_cols].isna().mean()
            usable_factor_cols = factor_missing[factor_missing <= self.settings["max_factor_missing_ratio"]].index.tolist()

        usable_engineered_cols = [col for col in engineered_cols if col in candidate_cols]
        usable_extra_cols = [col for col in extra_feature_cols if col in candidate_cols]
        filtered_cols = list(dict.fromkeys(usable_factor_cols + usable_engineered_cols + usable_extra_cols))

        if not filtered_cols:
            raise ValueError("No usable feature columns remain after factor filtering.")

        std_series = train_snapshot[filtered_cols].std(skipna=True)
        usable_feature_cols = std_series[std_series > self.settings["min_factor_std"]].index.tolist()
        if not usable_feature_cols:
            raise ValueError("All feature columns were removed by train-set variance filtering.")

        target_col = self.settings["target_col"]
        required_cols = usable_feature_cols + [target_col, "future_return", "REGIME_LABEL", "DATA_SPLIT"]
        prepared = merged_data.dropna(subset=required_cols).copy()
        prepared["REGIME_LABEL"] = prepared["REGIME_LABEL"].astype(int)
        prepared["REGIME_NAME"] = prepared["REGIME_LABEL"].map(REGIME_NAME_MAP)

        train_df = prepared.loc[prepared["DATA_SPLIT"] == "train"].copy()
        val_df = prepared.loc[prepared["DATA_SPLIT"] == "val"].copy()
        test_df = prepared.loc[prepared["DATA_SPLIT"] == "test"].copy()

        regime_counts = (
            train_df.groupby("REGIME_NAME", observed=False)
            .size()
            .reindex(["low_vol", "high_vol"], fill_value=0)
            .to_dict()
        )
        min_train_rows = self.settings["min_train_rows_per_regime"]
        for regime_name, row_count in regime_counts.items():
            if int(row_count) < min_train_rows:
                raise ValueError(
                    f"Train split for regime '{regime_name}' has only {row_count} rows, "
                    f"below vol_split.min_train_rows_per_regime={min_train_rows}."
                )

        metadata = {
            "target_col": target_col,
            "feature_count": int(len(usable_feature_cols)),
            "factor_feature_count": int(len([col for col in usable_feature_cols if col in usable_factor_cols])),
            "engineered_feature_count": int(len([col for col in usable_feature_cols if col.startswith("ENG_")])),
            "date_range": {
                "start": prepared[self.timestamp_col].min(),
                "end": prepared[self.timestamp_col].max(),
            },
            "split_rows": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "train_regime_rows": {key: int(val) for key, val in regime_counts.items()},
            "regime_label_source": source,
            "regime_cutoff": float(daily_close.attrs.get(cutoff_key) if source == "daily" else monthly_close.attrs.get(cutoff_key)),
            "daily_vol_stats": summarize_daily_vol(daily_close).to_dict(),
            "monthly_vol_stats": summarize_monthly_vol(monthly_close).to_dict(),
        }

        return PreparedData(
            full_data=prepared.reset_index(drop=True),
            train_data=train_df.reset_index(drop=True),
            val_data=val_df.reset_index(drop=True),
            test_data=test_df.reset_index(drop=True),
            feature_cols=usable_feature_cols,
            factor_cols=[col for col in usable_feature_cols if col in usable_factor_cols],
            engineered_cols=[col for col in usable_feature_cols if col.startswith("ENG_") or col in extra_feature_cols],
            target_col=target_col,
            metadata=_to_native(metadata),
        )


def prepare_data(config_path: str | None = None, force_rebuild: bool | None = None) -> PreparedData:
    builder = FactorDatasetBuilder(config_path=config_path)
    return builder.prepare(force_rebuild=force_rebuild)
