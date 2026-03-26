from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CURRENT_DIR_STR = str(CURRENT_DIR)
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if CURRENT_DIR_STR not in sys.path:
    sys.path.insert(0, CURRENT_DIR_STR)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from config_utils import get_section, load_project_config, resolve_paths
from dataset import PreparedData, REGIME_NAME_MAP


MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"


@dataclass
class RegimeModelArtifact:
    regime_label: int
    regime_name: str
    booster: lgb.Booster
    scaler: RobustScaler | StandardScaler | None
    params: dict[str, Any]
    training_history: dict[str, Any]
    feature_importance: pd.DataFrame
    metrics: dict[str, Any]


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


def build_model_settings(config_path: str | None = None) -> dict[str, Any]:
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(
        config_dir,
        get_section(config, "paths"),
        ["model_dir", "training_summary", "training_plot", "training_comparison_plot"],
    )
    model_cfg = get_section(config, "model")
    return {
        "config_path": str((Path(config_path).expanduser().resolve() if config_path else (PROJECT_ROOT / "config.yaml"))),
        "paths": paths,
        "target_col": str(model_cfg.get("target_column", "target_vol_norm")),
        "scale_method": str(model_cfg.get("scale_method", "robust")).lower(),
        "num_boost_round": int(model_cfg.get("num_boost_round", 400)),
        "early_stopping_rounds": int(model_cfg.get("early_stopping_rounds", 50)),
        "feature_importance_top_n": int(model_cfg.get("feature_importance_top_n", 20)),
        "common_params": dict(model_cfg.get("common_params", {})),
        "low_vol_overrides": dict(model_cfg.get("low_vol_overrides", {})),
        "high_vol_overrides": dict(model_cfg.get("high_vol_overrides", {})),
    }


def calc_ic(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_series = pd.Series(y_true, dtype="float64")
    pred_series = pd.Series(y_pred, dtype="float64")
    pearson_ic = float(true_series.corr(pred_series, method="pearson"))
    spearman_ic = float(true_series.corr(pred_series, method="spearman"))
    return {
        "pearson_ic": 0.0 if np.isnan(pearson_ic) else pearson_ic,
        "spearman_ic": 0.0 if np.isnan(spearman_ic) else spearman_ic,
    }


def calc_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "directional_accuracy": 0.0,
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "target_mean": 0.0,
            "target_std": 0.0,
            "pearson_ic": 0.0,
            "spearman_ic": 0.0,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
        "directional_accuracy": float(np.mean(direction_true == direction_pred)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
        **calc_ic(y_true, y_pred),
    }


def _build_scaler(method: str) -> RobustScaler | StandardScaler | None:
    if method == "robust":
        return RobustScaler()
    if method == "standard":
        return StandardScaler()
    if method in {"none", "identity"}:
        return None
    raise ValueError(f"Unsupported scale method: {method}")


def _resolve_regime_params(settings: dict[str, Any], regime_name: str) -> dict[str, Any]:
    params = dict(settings["common_params"])
    if regime_name == "low_vol":
        params.update(settings["low_vol_overrides"])
    elif regime_name == "high_vol":
        params.update(settings["high_vol_overrides"])
    else:
        raise ValueError(f"Unknown regime_name: {regime_name}")
    return params


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scaler: RobustScaler | StandardScaler | None,
    fit_scaler: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].to_numpy(dtype="float32")
    y = df[target_col].to_numpy(dtype="float32")
    if scaler is not None:
        x = scaler.fit_transform(x) if fit_scaler else scaler.transform(x)
    return x, y


def _convert_prediction(df: pd.DataFrame, raw_pred: np.ndarray, target_col: str) -> np.ndarray:
    if target_col == "target_vol_norm":
        scale = df["target_vol_scale"].to_numpy(dtype="float64")
        return raw_pred.astype("float64") * scale
    return raw_pred.astype("float64")


def predict_single_regime(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    booster: lgb.Booster,
    scaler: RobustScaler | StandardScaler | None,
) -> pd.DataFrame:
    features = df[feature_cols].to_numpy(dtype="float32")
    if scaler is not None:
        features = scaler.transform(features)
    raw_pred = np.asarray(booster.predict(features), dtype="float64")

    pred_df = df.copy()
    pred_df["pred_target"] = raw_pred
    pred_df["pred_return"] = _convert_prediction(pred_df, raw_pred, target_col)
    pred_df["pred_direction"] = np.sign(pred_df["pred_return"]).astype(np.int8)
    pred_df["true_direction"] = np.sign(pred_df["future_return"]).astype(np.int8)
    pred_df["abs_error"] = (pred_df["pred_return"] - pred_df["future_return"]).abs()
    return pred_df


def train_single_regime_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scale_method: str,
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    regime_label: int,
) -> RegimeModelArtifact:
    regime_name = REGIME_NAME_MAP[int(regime_label)]
    scaler = _build_scaler(scale_method)
    x_train, y_train = _prepare_xy(train_df, feature_cols, target_col, scaler, fit_scaler=True)
    x_val, y_val = _prepare_xy(val_df, feature_cols, target_col, scaler, fit_scaler=False)

    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(x_val, label=y_val, feature_name=feature_cols, reference=train_set)
    evals_result: dict[str, Any] = {}
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            *(
                [lgb.early_stopping(early_stopping_rounds, verbose=False)]
                if early_stopping_rounds > 0
                else []
            ),
            lgb.record_evaluation(evals_result),
        ],
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    val_pred_df = predict_single_regime(val_df, feature_cols, target_col, booster, scaler)
    test_pred_df = (
        predict_single_regime(test_df, feature_cols, target_col, booster, scaler)
        if not test_df.empty
        else test_df.copy()
    )

    metrics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "best_iteration": int(booster.best_iteration or num_boost_round),
        "val_metrics": calc_prediction_metrics(val_pred_df["future_return"], val_pred_df["pred_return"]),
        "test_metrics": (
            calc_prediction_metrics(test_pred_df["future_return"], test_pred_df["pred_return"])
            if not test_pred_df.empty
            else {}
        ),
        "top_features": feature_importance.head(20).to_dict(orient="records"),
    }
    return RegimeModelArtifact(
        regime_label=int(regime_label),
        regime_name=regime_name,
        booster=booster,
        scaler=scaler,
        params=params,
        training_history=evals_result,
        feature_importance=feature_importance,
        metrics=_to_native(metrics),
    )


def _save_regime_artifact(base_dir: Path, artifact: RegimeModelArtifact) -> None:
    regime_dir = base_dir / artifact.regime_name
    regime_dir.mkdir(parents=True, exist_ok=True)
    artifact.booster.save_model(str(regime_dir / "model.txt"))
    with (regime_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(artifact.scaler, f)
    artifact.feature_importance.to_json(regime_dir / "feature_importance.json", orient="records", indent=2)
    with (regime_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            _to_native(
                {
                    "regime_label": artifact.regime_label,
                    "regime_name": artifact.regime_name,
                    "params": artifact.params,
                    "metrics": artifact.metrics,
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )


def plot_training_diagnostics(
    artifact_map: dict[int, RegimeModelArtifact],
    prediction_map: dict[int, pd.DataFrame],
    output_path: Path,
    top_n: int,
) -> None:
    if not artifact_map:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR.resolve()))
    import matplotlib.pyplot as plt

    rows = len(artifact_map)
    fig, axes = plt.subplots(rows, 5, figsize=(28, 5.5 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row_idx, regime_label in enumerate(sorted(artifact_map)):
        artifact = artifact_map[regime_label]
        pred_df = prediction_map[regime_label]
        row_axes = axes[row_idx]

        if pred_df.empty:
            continue

        step = max(len(pred_df) // 4000, 1)
        sampled = pred_df.iloc[::step]
        actual = sampled["future_return"].to_numpy(dtype="float64")
        pred = sampled["pred_return"].to_numpy(dtype="float64")
        lower = min(float(np.nanmin(actual)), float(np.nanmin(pred)))
        upper = max(float(np.nanmax(actual)), float(np.nanmax(pred)))
        timestamps = sampled["TDATE"]
        rolling_window = max(min(len(pred_df) // 40, 240), 30)
        pred_roll = pred_df["pred_return"].rolling(rolling_window, min_periods=1).mean()
        actual_roll = pred_df["future_return"].rolling(rolling_window, min_periods=1).mean()
        cum_pred = pred_df["pred_return"].cumsum()
        cum_actual = pred_df["future_return"].cumsum()

        row_axes[0].scatter(actual, pred, s=8, alpha=0.35, color="#1f77b4")
        row_axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
        row_axes[0].set_title(f"{artifact.regime_name} | Validation: Pred vs Actual")
        row_axes[0].set_xlabel("Actual future return")
        row_axes[0].set_ylabel("Predicted return")
        row_axes[0].grid(alpha=0.25)

        row_axes[1].plot(timestamps, sampled["future_return"], color="#7f7f7f", linewidth=0.8, alpha=0.7, label="Actual")
        row_axes[1].plot(timestamps, sampled["pred_return"], color="#1f77b4", linewidth=0.8, alpha=0.8, label="Predicted")
        row_axes[1].axhline(0.0, color="black", linewidth=0.7)
        row_axes[1].set_title(f"{artifact.regime_name} | Sampled return path")
        row_axes[1].legend(loc="upper right")
        row_axes[1].grid(alpha=0.25)

        row_axes[2].plot(pred_df["TDATE"], actual_roll, color="#7f7f7f", linewidth=1.2, label=f"Actual rolling {rolling_window}")
        row_axes[2].plot(pred_df["TDATE"], pred_roll, color="#1f77b4", linewidth=1.2, label=f"Predicted rolling {rolling_window}")
        row_axes[2].plot(pred_df["TDATE"], cum_actual / max(len(pred_df), 1), color="#2ca02c", linewidth=0.9, alpha=0.8, label="Actual cumulative mean")
        row_axes[2].plot(pred_df["TDATE"], cum_pred / max(len(pred_df), 1), color="#d62728", linewidth=0.9, alpha=0.8, label="Pred cumulative mean")
        row_axes[2].axhline(0.0, color="black", linewidth=0.7)
        row_axes[2].set_title(f"{artifact.regime_name} | Smoothed prediction vs actual")
        row_axes[2].legend(loc="upper right")
        row_axes[2].grid(alpha=0.25)

        history_train = artifact.training_history.get("train", {})
        history_val = artifact.training_history.get("val", {})
        metric_name = next(iter(history_train.keys()), None)
        if metric_name and metric_name in history_val:
            row_axes[3].plot(history_train[metric_name], label="train", color="#2ca02c")
            row_axes[3].plot(history_val[metric_name], label="val", color="#d62728")
            row_axes[3].axvline(artifact.metrics["best_iteration"], color="black", linestyle="--", linewidth=1.0)
            row_axes[3].set_title(f"{artifact.regime_name} | Training history")
            row_axes[3].set_xlabel("Iteration")
            row_axes[3].set_ylabel(metric_name)
            row_axes[3].legend()
            row_axes[3].grid(alpha=0.25)

        top_features = artifact.feature_importance.head(top_n)
        row_axes[4].barh(
            top_features["feature"][::-1],
            top_features["importance_gain"][::-1],
            color="#9467bd",
        )
        row_axes[4].set_title(f"{artifact.regime_name} | Top {top_n} features")
        row_axes[4].set_xlabel("Gain")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_regime_model_comparison(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if validation_df.empty and test_df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR.resolve()))
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    color_map = {"low_vol": "#7BC96F", "high_vol": "#F28B82"}
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=False)

    def _plot_single(ax, df: pd.DataFrame, title: str):
        if df.empty:
            ax.set_axis_off()
            return
        daily = (
            df.groupby("TRADE_DATE", sort=True)
            .agg(
                REGIME_LABEL=("REGIME_LABEL", "first"),
                pred_return=("pred_return", "mean"),
                future_return=("future_return", "mean"),
            )
            .reset_index()
        )
        daily["TRADE_DATE"] = pd.to_datetime(daily["TRADE_DATE"])
        daily["REGIME_NAME"] = daily["REGIME_LABEL"].map(REGIME_NAME_MAP)
        current_name = None
        start_date = None
        segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
        for row in daily.itertuples(index=False):
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
            segments.append((start_date, pd.to_datetime(daily["TRADE_DATE"].iloc[-1]) + pd.Timedelta(days=1), current_name))
        for seg_start, seg_end, regime_name in segments:
            ax.axvspan(seg_start, seg_end, color=color_map[regime_name], alpha=0.28, linewidth=0)

        bar_colors = np.where(daily["future_return"] >= 0, "#4c78a8", "#c44e52")
        ax.bar(daily["TRADE_DATE"], daily["future_return"], color=bar_colors, alpha=0.72, width=1.0, label="Actual return")
        ax.plot(daily["TRADE_DATE"], daily["pred_return"], color="#16324F", linewidth=1.1, label="Predicted return")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        counts = daily["REGIME_NAME"].value_counts().to_dict()
        ic_metrics = calc_prediction_metrics(df["future_return"], df["pred_return"])
        ax.text(
            0.01,
            0.02,
            (
                f"low_vol days={int(counts.get('low_vol', 0))}, "
                f"high_vol days={int(counts.get('high_vol', 0))}\n"
                f"IC={float(ic_metrics.get('pearson_ic', 0.0)):.3f}, "
                f"RankIC={float(ic_metrics.get('spearman_ic', 0.0)):.3f}, "
                f"DA={float(ic_metrics.get('directional_accuracy', 0.0)):.3f}"
            ),
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
        )
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=color_map["low_vol"], alpha=0.28, label="low_vol model active"),
            Patch(facecolor=color_map["high_vol"], alpha=0.28, label="high_vol model active"),
        ]
        ax.legend(handles=handles + ax.lines[:1] + ax.containers[:1], loc="upper right", fontsize=8)

    _plot_single(axes[0], validation_df, "Validation Regime Switch And Return")
    _plot_single(axes[1], test_df, "Test Regime Switch And Return")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def predict_dual_regime(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    artifact_map: dict[int, RegimeModelArtifact],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for regime_label, artifact in artifact_map.items():
        regime_df = df.loc[df["REGIME_LABEL"] == regime_label].copy()
        if regime_df.empty:
            continue
        outputs.append(
            predict_single_regime(
                df=regime_df,
                feature_cols=feature_cols,
                target_col=target_col,
                booster=artifact.booster,
                scaler=artifact.scaler,
            )
        )
    if not outputs:
        return df.iloc[0:0].copy()
    return pd.concat(outputs, axis=0).sort_values("TDATE").reset_index(drop=True)


def train_dual_regime_models(
    prepared: PreparedData,
    config_path: str | None = None,
) -> tuple[dict[int, RegimeModelArtifact], dict[str, Any], dict[str, pd.DataFrame]]:
    settings = build_model_settings(config_path)
    model_dir = settings["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_map: dict[int, RegimeModelArtifact] = {}
    val_prediction_map: dict[int, pd.DataFrame] = {}
    summary = {
        "dataset": prepared.metadata,
        "feature_cols": prepared.feature_cols,
        "regimes": {},
    }

    for regime_label in sorted(REGIME_NAME_MAP):
        regime_name = REGIME_NAME_MAP[regime_label]
        train_df = prepared.train_data.loc[prepared.train_data["REGIME_LABEL"] == regime_label].copy()
        val_df = prepared.val_data.loc[prepared.val_data["REGIME_LABEL"] == regime_label].copy()
        test_df = prepared.test_data.loc[prepared.test_data["REGIME_LABEL"] == regime_label].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Regime '{regime_name}' has an empty train/val split and cannot be trained safely."
            )

        params = _resolve_regime_params(settings, regime_name)
        artifact = train_single_regime_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=prepared.feature_cols,
            target_col=prepared.target_col,
            scale_method=settings["scale_method"],
            params=params,
            num_boost_round=settings["num_boost_round"],
            early_stopping_rounds=settings["early_stopping_rounds"],
            regime_label=regime_label,
        )
        artifact_map[regime_label] = artifact
        val_prediction_map[regime_label] = predict_single_regime(
            df=val_df,
            feature_cols=prepared.feature_cols,
            target_col=prepared.target_col,
            booster=artifact.booster,
            scaler=artifact.scaler,
        )
        summary["regimes"][regime_name] = {
            "params": params,
            "metrics": artifact.metrics,
        }
        _save_regime_artifact(model_dir, artifact)

    combined_val = predict_dual_regime(prepared.val_data, prepared.feature_cols, prepared.target_col, artifact_map)
    combined_test = predict_dual_regime(prepared.test_data, prepared.feature_cols, prepared.target_col, artifact_map)
    summary["combined_validation_metrics"] = calc_prediction_metrics(combined_val["future_return"], combined_val["pred_return"])
    summary["combined_test_metrics"] = calc_prediction_metrics(combined_test["future_return"], combined_test["pred_return"])

    plot_training_diagnostics(
        artifact_map=artifact_map,
        prediction_map=val_prediction_map,
        output_path=settings["paths"]["training_plot"],
        top_n=settings["feature_importance_top_n"],
    )
    plot_regime_model_comparison(
        validation_df=combined_val,
        test_df=combined_test,
        output_path=settings["paths"]["training_comparison_plot"],
    )

    with settings["paths"]["training_summary"].open("w", encoding="utf-8") as f:
        json.dump(_to_native(summary), f, indent=2, ensure_ascii=False)

    return artifact_map, _to_native(summary), {"val": combined_val, "test": combined_test}


def load_dual_regime_models(config_path: str | None = None) -> dict[int, RegimeModelArtifact]:
    settings = build_model_settings(config_path)
    model_dir = settings["paths"]["model_dir"]
    artifact_map: dict[int, RegimeModelArtifact] = {}

    for regime_label, regime_name in REGIME_NAME_MAP.items():
        regime_dir = model_dir / regime_name
        meta_path = regime_dir / "meta.json"
        model_path = regime_dir / "model.txt"
        scaler_path = regime_dir / "scaler.pkl"
        importance_path = regime_dir / "feature_importance.json"
        if not meta_path.exists() or not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Missing model artifact under {regime_dir}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)
        feature_importance = pd.read_json(importance_path)

        artifact_map[regime_label] = RegimeModelArtifact(
            regime_label=regime_label,
            regime_name=regime_name,
            booster=lgb.Booster(model_file=str(model_path)),
            scaler=scaler,
            params=meta.get("params", {}),
            training_history={},
            feature_importance=feature_importance,
            metrics=meta.get("metrics", {}),
        )
    return artifact_map
