"""Per-product factor IC audit + filter, executed at training time.

设计要点（强约束，请勿改动）：
- IC **当场计算**，不读任何预先存在的 registry 文件。每个产品训练时对自己的
  train_data 算 walk-forward 月度 Spearman IC，再按 config 阈值切 train/skip,
  当场写入 `results/runs/<run_id>/<PID>/factor_registry.json`。
- 流程顺序固定：数据处理 → 因子计算（prepare_data 内）→ **因子筛选（本模块）**
  → 模型训练 → 回测。

字段语义：
- mean_ic / std_ic / icir / n_windows : walk-forward 月度 Spearman IC 的统计
- use_in_training : abs(mean_ic) ≥ min_abs_ic AND abs(icir) ≥ min_icir AND n_windows ≥ 3
- importance_gain : 训练后由 modeling 回填，仅作元数据 / 报告图（非过滤依据）

Registry JSON 结构（与旧的全局 data/factor_registry.json 对齐）：
{
  "metadata": {
    "schema_version": 3,
    "product_id": "CU",
    "raw_data": "data/.../CUZL.SHF.csv",
    "target_col": "target_vol_norm",
    "target_horizon": 30,
    "thresholds": {"min_abs_ic": 0.005, "min_icir": 0.3},
    "filter_rule": "abs(mean_ic) >= min_abs_ic AND abs(icir) >= min_icir AND n_windows >= 3",
    "last_audit_at": "2026-04-30T...",
    "n_factors": 306,
    "n_train_factor": 170,
    "n_not_train_factor": 136
  },
  "train_factor":     [ {name, category, source, mean_ic, std_ic, icir, abs_ic, n_windows,
                         importance_gain, use_in_training=true,  last_audit_at, notes}, ... ],
  "not_train_factor": [ {同上 + reason, use_in_training=false}, ... ]
}

因子分类信息（runtime/engineered/mid_weekly 各组列表）已经存在
training_summary.json::feature_manifest 与 backtest_summary.json::feature_manifest
里，registry 不重复保存。
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────── 因子分类（仅作 registry 元数据） ───────────────────────────

def classify_factor(name: str, runtime_cols: set[str], extra_cols: set[str]) -> tuple[str, str]:
    """返回 (category, source)。"""
    if name.startswith("MIDxMICRO_"):
        return ("mid_micro_interaction", "engineered")
    if name.startswith("MID_"):
        return ("mid_weekly", "mid_weekly")
    if name in runtime_cols:
        return ("runtime", "factor_engine")
    if name in extra_cols:
        return ("extra", "extra")
    if not name.startswith("ENG_"):
        return ("unknown", "unknown")
    if name.startswith("ENG_DAILY_") or name in {"ENG_OVERNIGHT_GAP", "ENG_OVERNIGHT_GAP_ABS"}:
        return ("daily", "engineered")
    if (
        name.startswith("ENG_INTRADAY_")
        or name.startswith("ENG_PREV_CLOSE_")
        or name.startswith("ENG_TYPICAL_")
        or name.startswith("ENG_HIST_SAME_MIN_")
    ):
        return ("minute_intraday", "engineered")
    if "_X_" in name or name.endswith("_X_RET20") or name.endswith("_SIGN_AGREE"):
        return ("interaction", "engineered")
    if (
        name.startswith("ENG_SEMIVAR_")
        or name.startswith("ENG_CONSECUTIVE_")
        or name.startswith("ENG_BREAKOUT_")
        or name.startswith("ENG_MULTI_HORIZON_")
    ):
        return ("composite_alpha", "engineered")
    return ("engineered", "engineered")


# ─────────────────────────── walk-forward IC 计算 ───────────────────────────

def compute_walk_forward_ic(
    train_data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    *,
    trade_date_col: str = "TRADE_DATE",
    min_obs_per_window: int = 200,
) -> pd.DataFrame:
    """在已切好的 train_data 上按月计算每因子的 Spearman IC，再聚合 mean / std / ICIR。

    Args:
        train_data: 已经过 split_by_vol、DATA_SPLIT == "train" 的子集。必须包含
                    `trade_date_col`、`target_col` 和 feature_cols 中的列。
        feature_cols: 候选因子列名（缺列容忍，会自动跳过）。

    Returns:
        DataFrame, index = factor name, columns: mean_ic, std_ic, icir, n_windows
    """
    if train_data.empty:
        raise ValueError("train_data 为空，无法计算 walk-forward IC")
    if target_col not in train_data.columns:
        raise KeyError(f"target column {target_col} 不在 train_data 中")
    if trade_date_col not in train_data.columns:
        raise KeyError(f"trade_date column {trade_date_col} 不在 train_data 中")

    df = train_data.dropna(subset=[target_col]).copy()
    df[trade_date_col] = pd.to_datetime(df[trade_date_col])
    df["__month__"] = df[trade_date_col].dt.to_period("M")
    months = sorted(df["__month__"].dropna().unique())

    available = [c for c in feature_cols if c in df.columns]
    print(
        f"[audit] walk-forward windows: {len(months)} months on train split, "
        f"{len(available)}/{len(feature_cols)} factors found in train_data"
    )

    monthly_ics: dict[str, list[float]] = {col: [] for col in feature_cols}
    started_at = time.time()
    progress_every = max(1, len(months) // 10)  # ~10 条进度

    for idx, month in enumerate(months, start=1):
        if idx == 1 or idx % progress_every == 0 or idx == len(months):
            elapsed = time.time() - started_at
            print(
                f"[audit] month {idx}/{len(months)} ({month}) "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        block = df.loc[df["__month__"] == month]
        if len(block) < min_obs_per_window:
            continue
        target_ranks = block[target_col].rank(method="average")
        target_centered = target_ranks - target_ranks.mean()
        target_var = float((target_centered ** 2).sum())
        if target_var <= 0:
            continue

        block_feat = block[available]
        notna_mask = block_feat.notna()
        col_n = notna_mask.sum(axis=0)
        usable_cols = col_n.index[col_n >= min_obs_per_window].tolist()
        if not usable_cols:
            continue

        feat_ranks = block_feat[usable_cols].rank(method="average")
        feat_centered = feat_ranks.sub(feat_ranks.mean(axis=0, skipna=True), axis=1)

        target_aligned = target_centered.values[:, None]
        feat_arr = feat_centered.values
        valid_pairs = ~np.isnan(feat_arr)
        prod = np.where(valid_pairs, feat_arr * target_aligned, np.nan)
        num = np.nansum(prod, axis=0)
        feat_var = np.nansum(np.where(valid_pairs, feat_arr ** 2, np.nan), axis=0)
        target_var_per_col = np.nansum(
            np.where(valid_pairs, target_aligned ** 2, np.nan), axis=0
        )
        denom = np.sqrt(feat_var * target_var_per_col)
        with np.errstate(divide="ignore", invalid="ignore"):
            ic_arr = np.where(denom > 0, num / denom, 0.0)

        for col_name, ic_value in zip(usable_cols, ic_arr):
            if not np.isnan(ic_value):
                monthly_ics[col_name].append(float(ic_value))

    rows = []
    for col in feature_cols:
        ics = np.asarray(monthly_ics.get(col, []), dtype=float)
        n = int(len(ics))
        if n >= 3:
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics, ddof=1)) if n > 1 else 0.0
            icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
        else:
            mean_ic = float(np.mean(ics)) if n > 0 else 0.0
            std_ic = 0.0
            icir = 0.0
        rows.append({
            "name": col,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "n_windows": n,
        })
    return pd.DataFrame(rows).set_index("name")


# ─────────────────────────── 筛选 + 写 registry ───────────────────────────

def audit_and_filter(
    *,
    prepared,
    output_path: Path,
    min_abs_ic: float,
    min_icir: float,
    min_obs_per_window: int = 200,
    trade_date_col: str = "TRADE_DATE",
) -> tuple[list[str], dict[str, Any]]:
    """对 prepared 的候选因子做 IC 审计 + 阈值筛选 + 写 per-product registry。

    Args:
        prepared: PreparedData，含 train_data / feature_cols / feature_manifest 等
        output_path: 写入路径，例如 `results/runs/<run_id>/<PID>/factor_registry.json`
        min_abs_ic / min_icir: 阈值（来自 config.yaml::factors.audit_thresholds）

    Returns:
        (selected_feature_cols, registry_payload)
    """
    feature_cols = list(prepared.feature_cols)
    runtime_cols = set(prepared.runtime_factor_cols)
    extra_cols = set(prepared.feature_manifest.get("extra_feature_cols", []))
    target_col = prepared.target_col
    product_id = prepared.feature_manifest.get("product_id", "?")

    print(
        f"[factor_audit] start product={product_id} "
        f"n_candidates={len(feature_cols)} "
        f"thresholds(min_abs_ic={min_abs_ic}, min_icir={min_icir})"
    )

    ic_df = compute_walk_forward_ic(
        train_data=prepared.train_data,
        target_col=target_col,
        feature_cols=feature_cols,
        trade_date_col=trade_date_col,
        min_obs_per_window=min_obs_per_window,
    )

    today = dt.date.today().isoformat()
    audit_ts = dt.datetime.now().isoformat(timespec="seconds")
    train_records: list[dict[str, Any]] = []
    skip_records: list[dict[str, Any]] = []
    selected: list[str] = []

    for col in feature_cols:
        if col in ic_df.index:
            mean_ic = float(ic_df.at[col, "mean_ic"])
            std_ic = float(ic_df.at[col, "std_ic"])
            icir = float(ic_df.at[col, "icir"])
            n_windows = int(ic_df.at[col, "n_windows"])
        else:
            mean_ic = std_ic = icir = 0.0
            n_windows = 0
        abs_ic = abs(mean_ic)
        abs_icir = abs(icir)
        passes = (abs_ic >= min_abs_ic) and (abs_icir >= min_icir) and (n_windows >= 3)
        category, source = classify_factor(col, runtime_cols, extra_cols)
        record: dict[str, Any] = {
            "name": col,
            "category": category,
            "source": source,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "abs_ic": abs_ic,
            "n_windows": n_windows,
            "importance_gain": 0.0,
            "use_in_training": bool(passes),
            "last_audit_at": today,
            "notes": "",
        }
        if passes:
            train_records.append(record)
            selected.append(col)
        else:
            reasons: list[str] = []
            if n_windows < 3:
                reasons.append("n_windows<3")
            if abs_ic < min_abs_ic:
                reasons.append(f"|mean_ic|<{min_abs_ic}")
            if abs_icir < min_icir:
                reasons.append(f"|icir|<{min_icir}")
            record["reason"] = ";".join(reasons) if reasons else "below_threshold"
            skip_records.append(record)

    def _sort_key(r: dict) -> tuple[float, float]:
        return (-abs(float(r.get("icir", 0.0))), -abs(float(r.get("mean_ic", 0.0))))

    train_records.sort(key=_sort_key)
    skip_records.sort(key=_sort_key)

    def _avg_abs(records: list[dict], field: str) -> float:
        if not records:
            return 0.0
        vals = [abs(float(r.get(field, 0.0))) for r in records]
        return float(np.mean(vals))

    def _median_int(records: list[dict], field: str) -> int:
        if not records:
            return 0
        return int(np.median([int(r.get(field, 0)) for r in records]))

    all_records = train_records + skip_records

    manifest = prepared.feature_manifest
    payload: dict[str, Any] = {
        "metadata": {
            "schema_version": 3,
            "product_id": manifest.get("product_id"),
            "raw_data": manifest.get("raw_data"),
            "target_col": target_col,
            "target_horizon": int(prepared.metadata.get("target_horizon", 0)),
            "thresholds": {"min_abs_ic": min_abs_ic, "min_icir": min_icir},
            "filter_rule": "abs(mean_ic) >= min_abs_ic AND abs(icir) >= min_icir AND n_windows >= 3",
            "last_audit_at": audit_ts,
            "n_factors": len(feature_cols),
            "n_train_factor": len(train_records),
            "n_not_train_factor": len(skip_records),
            # 聚合统计（以 |mean_ic|, |icir| 的均值反映整体因子质量）
            "aggregate_stats": {
                "mean_abs_ic_all": _avg_abs(all_records, "mean_ic"),
                "mean_abs_ic_train": _avg_abs(train_records, "mean_ic"),
                "mean_abs_ic_skip": _avg_abs(skip_records, "mean_ic"),
                "mean_abs_icir_all": _avg_abs(all_records, "icir"),
                "mean_abs_icir_train": _avg_abs(train_records, "icir"),
                "mean_abs_icir_skip": _avg_abs(skip_records, "icir"),
                "median_n_windows": _median_int(all_records, "n_windows"),
                "n_factors_with_enough_windows": sum(
                    1 for r in all_records if int(r.get("n_windows", 0)) >= 3
                ),
            },
        },
        "train_factor": train_records,
        "not_train_factor": skip_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[factor_audit] kept {len(selected)}/{len(feature_cols)} "
        f"(min_abs_ic={min_abs_ic}, min_icir={min_icir}); "
        f"registry → {output_path}"
    )
    return selected, payload


def backfill_importance(registry_path: Path, importance: dict[str, float]) -> None:
    """训练完成后把 LightGBM importance_gain 回填进 per-product registry。

    同时刷新 metadata.aggregate_stats 里 importance 相关的聚合（n_used / mean / sum）。
    importance_gain 仅为元数据（章节 1.6 图依赖），不影响过滤结果。
    """
    if not registry_path.exists():
        return
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    for bucket in ("train_factor", "not_train_factor"):
        for rec in payload.get(bucket, []):
            name = rec.get("name", "")
            if name in importance:
                rec["importance_gain"] = float(importance[name])

    train_records = payload.get("train_factor", [])
    train_imps = [float(r.get("importance_gain", 0.0)) for r in train_records]
    n_used = sum(1 for v in train_imps if v > 0)
    agg = payload.setdefault("metadata", {}).setdefault("aggregate_stats", {})
    agg["n_train_factor_used_by_model"] = n_used
    agg["mean_importance_train"] = float(np.mean(train_imps)) if train_imps else 0.0
    agg["sum_importance_train"] = float(np.sum(train_imps)) if train_imps else 0.0

    registry_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
