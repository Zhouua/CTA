"""Optuna 超参搜索：全品种联合目标 = mean(val Spearman IC)。

设计：
- 全品种共用一组 LightGBM 参数（不做品种级调参）
- 目标函数：每个 trial 在所有品种上训 dual-regime，按 val 行数加权得到 product-level
  val Spearman IC，再对品种取均值 → 单一标量目标，最大化
- 缓存：prepare_data + audit_and_filter 每品种只跑一次；trial 内只做 LGB fit + predict
- 早停：保留 num_boost_round=600 + early_stopping_rounds=50（与 config.yaml 一致）
- 不触碰 test 集
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

from build_product_registry import build_from_config
from config_utils import load_project_config, get_section, resolve_paths
from dataset import prepare_data, REGIME_NAME_MAP
from factor_audit import audit_and_filter
from train_products import (
    build_product_config_override,
    _load_audit_thresholds,
    _apply_factor_selection,
    _normalize_product_id,
)


@dataclass
class ProductCache:
    product_id: str
    feature_cols: list[str]
    target_col: str
    # per regime: scaled X_train, y_train, X_val, y_val, future_return_val
    by_regime: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def _slice_regime(df: pd.DataFrame, regime_label: int, feature_cols: list[str], target_col: str):
    sub = df.loc[df["REGIME_LABEL"] == regime_label]
    if sub.empty:
        return None
    x = sub[feature_cols].astype("float32").values
    y = sub[target_col].astype("float32").values
    fr = sub["future_return"].astype("float32").values if "future_return" in sub.columns else None
    return {"x": x, "y": y, "future_return": fr}


def build_product_cache(product_meta: dict[str, Any], config_path: str) -> ProductCache | None:
    """Run prepare → audit → store scaled (X, y) for each regime in memory."""
    pid = _normalize_product_id(product_meta["product_id"])
    # Use a tmp dir for registry; we never read it back.
    tmp_dir = ROOT / ".optuna_tmp" / pid
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_override = build_product_config_override(product_meta, tmp_dir)
    # 关闭模型持久化
    config_override.setdefault("model", {})["persist_models"] = False
    # 关闭 feature cache 写入：merged_features.parquet 与 runtime factor cache 都不落盘
    config_override.setdefault("data", {})["cache_merged_dataset"] = False
    config_override.setdefault("factors", {}).setdefault("runtime", {})["cache_generated_features"] = False

    prepared = prepare_data(config_path=config_path, force_rebuild=False, config_override=config_override)
    min_abs_ic, min_icir, min_obs, mid_audit_cfg = _load_audit_thresholds(config_path, config_override)
    registry_path = tmp_dir / "factor_registry.json"
    selected_cols, _ = audit_and_filter(
        prepared=prepared,
        output_path=registry_path,
        min_abs_ic=min_abs_ic,
        min_icir=min_icir,
        min_obs_per_window=min_obs,
        mid_audit_cfg=mid_audit_cfg,
    )
    if not selected_cols:
        print(f"  [{pid}] 0 features selected — skipping")
        return None
    prepared = _apply_factor_selection(prepared, selected_cols)

    cache = ProductCache(
        product_id=pid,
        feature_cols=list(prepared.feature_cols),
        target_col=prepared.target_col,
    )
    for regime_label in (-1, 1):
        train = _slice_regime(prepared.train_data, regime_label, cache.feature_cols, cache.target_col)
        val = _slice_regime(prepared.val_data, regime_label, cache.feature_cols, cache.target_col)
        if train is None or val is None:
            continue
        # Pre-fit scaler on train, transform both
        scaler = RobustScaler()
        train_xs = scaler.fit_transform(train["x"]).astype("float32")
        val_xs = scaler.transform(val["x"]).astype("float32")
        cache.by_regime[regime_label] = {
            "x_train": train_xs,
            "y_train": train["y"],
            "x_val": val_xs,
            "y_val": val["y"],
            "fr_val": val["future_return"],
        }
    return cache if cache.by_regime else None


def _fit_one(params: dict, x_train, y_train, x_val, y_val, num_boost_round: int, early_stopping_rounds: int) -> np.ndarray:
    """Fit one LGB and return val predictions."""
    train_set = lgb.Dataset(x_train, label=y_train)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return booster.predict(x_val, num_iteration=booster.best_iteration or num_boost_round)


def _spearman_ic(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10:
        return 0.0
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return 0.0
    rho, _ = spearmanr(pred[mask], target[mask])
    return 0.0 if not np.isfinite(rho) else float(rho)


def evaluate_trial(params_common: dict, caches: list[ProductCache], num_boost_round: int, early_stopping_rounds: int, regime_overrides: dict[int, dict] | None = None, trial: optuna.Trial | None = None) -> float:
    """Run one trial: fit dual-regime per product, return mean(val IC) across products."""
    regime_overrides = regime_overrides or {}
    product_ics: list[float] = []
    for idx, cache in enumerate(caches):
        regime_ics: list[tuple[float, int]] = []
        for regime_label, regime_data in cache.by_regime.items():
            params = dict(params_common)
            params.update(regime_overrides.get(regime_label, {}))
            # 用预测目标 target_vol_norm 算 IC（与 modeling.py 一致）
            try:
                pred = _fit_one(
                    params,
                    regime_data["x_train"], regime_data["y_train"],
                    regime_data["x_val"], regime_data["y_val"],
                    num_boost_round, early_stopping_rounds,
                )
            except Exception as exc:
                print(f"    [{cache.product_id} regime={regime_label}] fit failed: {exc}")
                continue
            ic = _spearman_ic(pred, regime_data["y_val"])
            regime_ics.append((ic, len(regime_data["y_val"])))
        if not regime_ics:
            continue
        total_w = sum(w for _, w in regime_ics)
        prod_ic = sum(ic * w for ic, w in regime_ics) / total_w if total_w > 0 else 0.0
        product_ics.append(prod_ic)
        # Optional Optuna pruning: report intermediate mean so far
        if trial is not None:
            running_mean = float(np.mean(product_ics))
            trial.report(running_mean, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not product_ics:
        return -1.0
    return float(np.mean(product_ics))


# 搜索空间预设：
#   wide：v1 默认（pure-micro 调参），lr 已窄化、子采样允许下探到 0.5
#   narrow_mid：v3 用（mid 启用），ff/bf 不低于 0.7、min_child 不高于 200，
#               让中观因子在更多被采样的特征里有出场机会
SEARCH_PROFILES = {
    "wide": {
        "learning_rate":     ("float_log", 0.005, 0.05),
        "num_leaves":        ("int", 31, 127),
        "max_depth":         ("int", 4, 10),
        "min_child_samples": ("int", 50, 300),
        "feature_fraction":  ("float", 0.5, 1.0),
        "bagging_fraction":  ("float", 0.5, 1.0),
        "reg_alpha":         ("float_log", 1e-8, 1.0),
        "reg_lambda":        ("float_log", 1e-8, 5.0),
    },
    "narrow_mid": {
        "learning_rate":     ("float_log", 0.005, 0.05),
        "num_leaves":        ("int", 31, 127),
        "max_depth":         ("int", 4, 10),
        "min_child_samples": ("int", 50, 200),
        "feature_fraction":  ("float", 0.7, 1.0),
        "bagging_fraction":  ("float", 0.7, 1.0),
        "reg_alpha":         ("float_log", 1e-8, 1.0),
        "reg_lambda":        ("float_log", 1e-8, 5.0),
    },
}


def _suggest(trial, name, spec):
    kind, lo, hi = spec
    if kind == "float":
        return trial.suggest_float(name, lo, hi)
    if kind == "float_log":
        return trial.suggest_float(name, lo, hi, log=True)
    if kind == "int":
        return trial.suggest_int(name, lo, hi)
    raise ValueError(f"unknown spec kind: {kind}")


def make_objective(caches, num_boost_round, early_stopping_rounds, profile: str = "wide"):
    space = SEARCH_PROFILES[profile]

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
            "bagging_freq": 5,
            **{k: _suggest(trial, k, v) for k, v in space.items()},
        }
        # high_vol regime override：min_child_samples 至少 150（与 config.yaml 一致的方向）
        regime_overrides = {1: {"min_child_samples": max(int(params["min_child_samples"]), 150)}}
        return evaluate_trial(
            params, caches, num_boost_round, early_stopping_rounds,
            regime_overrides=regime_overrides, trial=trial,
        )
    return _objective


def load_registry(config_path: str) -> list[dict[str, Any]]:
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(config_dir, get_section(config, "paths"), ["product_registry"])
    registry_path = paths["product_registry"]
    if not registry_path.exists():
        registry = build_from_config(config, config_dir)
    else:
        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    return registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--products", nargs="*", default=None,
                    help="限定品种（默认 registry 中全部 enabled）")
    ap.add_argument("--output", default="results/optuna/tune_v1.json")
    ap.add_argument("--storage", default=None,
                    help="optional Optuna storage URI (e.g. sqlite:///results/optuna/study.db)")
    ap.add_argument("--study-name", default="cta_lgb_tune_v1")
    ap.add_argument("--search-profile", choices=list(SEARCH_PROFILES.keys()), default="wide",
                    help="搜索空间档位：wide（v1，纯微观）/ narrow_mid（v3，中观启用）")
    args = ap.parse_args()

    registry = load_registry(args.config)
    if args.products:
        wanted = {_normalize_product_id(p) for p in args.products}
        registry = [r for r in registry if _normalize_product_id(r["product_id"]) in wanted]
    else:
        registry = [r for r in registry if r.get("enabled", True)]

    print(f"[optuna] preparing caches for {len(registry)} products...")
    caches: list[ProductCache] = []
    t0 = time.time()
    for r in registry:
        pid = _normalize_product_id(r["product_id"])
        try:
            cache = build_product_cache(r, args.config)
        except Exception as exc:
            print(f"  [{pid}] cache failed: {exc}")
            continue
        if cache is None:
            continue
        n_train = sum(len(v["y_train"]) for v in cache.by_regime.values())
        n_val = sum(len(v["y_val"]) for v in cache.by_regime.values())
        n_feat = len(cache.feature_cols)
        print(f"  [{pid}] cached n_feat={n_feat} train={n_train:,} val={n_val:,}")
        caches.append(cache)
    print(f"[optuna] cache built in {time.time()-t0:.1f}s, n_products={len(caches)}")

    if not caches:
        print("[optuna] no products available; aborting")
        return

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(3, len(caches)//6))
    if args.storage:
        Path(args.storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(direction="maximize", storage=args.storage,
                                    study_name=args.study_name, load_if_exists=True, pruner=pruner)
    else:
        study = optuna.create_study(direction="maximize", pruner=pruner)

    obj = make_objective(caches, args.num_boost_round, args.early_stopping_rounds, profile=args.search_profile)
    print(f"[optuna] running {args.n_trials} trials profile={args.search_profile} (num_boost_round={args.num_boost_round}, ES={args.early_stopping_rounds})")
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n[optuna] done. completed={len(completed)} pruned={len(pruned)}")

    if completed:
        best = study.best_trial
        print(f"\n=== BEST ===")
        print(f"  mean(val_spearman_ic) = {best.value:.5f}")
        print(f"  params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump({
                "best_value": best.value,
                "best_params": best.params,
                "n_completed": len(completed),
                "n_pruned": len(pruned),
                "n_products": len(caches),
                "products": [c.product_id for c in caches],
                "trials": [
                    {
                        "number": t.number,
                        "state": t.state.name,
                        "value": t.value,
                        "params": t.params,
                    }
                    for t in study.trials
                ],
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[optuna] saved → {out_path}")


if __name__ == "__main__":
    main()
