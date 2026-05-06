"""Per-product factor IC audit + filter, executed at training time.

设计要点（强约束，请勿改动）：
- IC **当场计算**，不读任何预先存在的 registry 文件。每个产品训练时对自己的
  train_data 算 walk-forward IC，再按 config 阈值切 train/skip,
  当场写入 `results/runs/<run_id>/<PID>/factor_registry.json`。
- 流程顺序固定：数据处理 → 因子计算（prepare_data 内）→ **因子筛选（本模块）**
  → 模型训练 → 回测。

三层审计体系（详见 docs/factor_thoughts.md §7）：
  Tier A  微观因子 (ENG_*, runtime)：1分钟月度窗口 Spearman IC
  Tier B  中观日频因子 (MID_*, freq=日)：日度月度窗口
  Tier C  中观周/月频因子 (MID_*, freq=周/月)：周度季度窗口
  MIDxMICRO：不做独立 IC 审计，靠父因子门 + 硬上限

Registry JSON 结构（schema_version=4）：
{
  "metadata": {
    "schema_version": 4,
    "product_id": "CU",
    "thresholds": {...},
    ...
  },
  "train_factor":     [ {name, category, source, audit_tier,
                         mean_ic, std_ic, icir, abs_ic, n_windows,
                         val_ic, val_ic_passes,
                         importance_gain, use_in_training, ...} ],
  "not_train_factor": [ {同上 + reason} ]
}
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────── 因子分类 ───────────────────────────

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


# ─────────────────────────── 中观频率映射 ───────────────────────────

def _build_mid_freq_map(feature_manifest: dict[str, Any]) -> dict[str, str]:
    """从 feature_manifest["mid_weekly_metadata"] 构建 col → 频率字符串 的映射。

    原始指标列直接映射；派生列（_RET_*/_ZSCORE_* 等）和 MIDxMICRO_* 通过前缀
    匹配找到原始父列，继承其频率。前缀取最长匹配，防止短 ID 误匹配长 ID。
    """
    meta: dict[str, dict[str, str]] = feature_manifest.get("mid_weekly_metadata", {})
    if not meta:
        return {}

    # 按长度降序排列，最长优先匹配（避免短前缀误匹配长名称）
    raw_keys = sorted(meta.keys(), key=len, reverse=True)

    def _lookup_parent_freq(col: str) -> str:
        for key in raw_keys:
            if col == key or col.startswith(key + "_"):
                return meta[key].get("frequency", "")
        return ""

    freq_map: dict[str, str] = {}
    all_cols: list[str] = feature_manifest.get("all_feature_cols", [])

    for col in all_cols:
        if col.startswith("MIDxMICRO_"):
            # 格式：MIDxMICRO_<micro>_X_<mid_col>；取 _X_ 后半部分找频率
            mid_part = col.split("_X_", 1)[1] if "_X_" in col else ""
            freq_map[col] = _lookup_parent_freq(mid_part) if mid_part else ""
        elif col.startswith("MID_"):
            freq_map[col] = _lookup_parent_freq(col)

    return freq_map


def _raw_mid_parent(col: str, raw_keys: list[str]) -> str | None:
    """返回派生 MID_* 列对应的原始父列名（最长前缀匹配）。"""
    for key in raw_keys:
        if col == key or col.startswith(key + "_"):
            return key
    return None


# ─────────────────────────── 数据聚合 ───────────────────────────

def _aggregate_by_period(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    period: str,
    trade_date_col: str = "TRADE_DATE",
    tdate_col: str = "TDATE",
) -> pd.DataFrame:
    """将 1分钟 bar 数据聚合到日度或周度，用于原生频率 IC 计算。

    period="day"：按 TRADE_DATE 分组，目标取均值，特征取 first（ffill 后为常数）。
    period="week"：按自然周分组（从 TDATE 派生），同上；代表日期为周内第一个交易日。

    返回 DataFrame，含 TRADE_DATE（代表日期）、target_col、feature_cols。
    MID_* 列在 prepare() 已做 fillna(0)；AVAILABLE 为 0 的行特征值不可信，
    但聚合时 first() 保留原值，IC 计算本身对稀疏 0 不敏感（与 target 无相关性）。
    """
    avail_features = [c for c in feature_cols if c in data.columns]
    if not avail_features:
        return pd.DataFrame()

    df = data[[tdate_col if period == "week" else trade_date_col,
               trade_date_col, target_col] + avail_features].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    if period == "day":
        group_col = trade_date_col
        agg_dict = {target_col: "mean", **{f: "first" for f in avail_features}}
        result = df.groupby(group_col, sort=True).agg(agg_dict).reset_index()
        result = result.rename(columns={group_col: "TRADE_DATE"})

    else:  # week
        df["__week__"] = pd.to_datetime(df[tdate_col]).dt.to_period("W")
        agg_dict = {
            trade_date_col: "first",   # 周内第一个交易日作代表日期
            target_col: "mean",
            **{f: "first" for f in avail_features},
        }
        result = df.groupby("__week__", sort=True).agg(agg_dict).reset_index(drop=True)
        result = result.rename(columns={trade_date_col: "TRADE_DATE"})

    result["TRADE_DATE"] = pd.to_datetime(result["TRADE_DATE"])
    return result.dropna(subset=[target_col]).reset_index(drop=True)


# ─────────────────────────── walk-forward IC 计算 ───────────────────────────

def compute_walk_forward_ic(
    train_data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    *,
    trade_date_col: str = "TRADE_DATE",
    min_obs_per_window: int = 200,
    window_period: str = "M",   # "M"=月度, "Q"=季度
) -> pd.DataFrame:
    """按时间窗口计算 Spearman IC，返回 mean_ic/std_ic/icir/n_windows。

    window_period="M"：月度切窗（Tier A 微观 / Tier B 日度中观）。
    window_period="Q"：季度切窗（Tier C 周度中观）。
    """
    if train_data.empty:
        raise ValueError("train_data 为空，无法计算 walk-forward IC")

    df = train_data.dropna(subset=[target_col]).copy()
    df[trade_date_col] = pd.to_datetime(df[trade_date_col])
    df["__window__"] = df[trade_date_col].dt.to_period(window_period)
    windows = sorted(df["__window__"].dropna().unique())

    available = [c for c in feature_cols if c in df.columns]
    print(
        f"[audit] walk-forward windows={len(windows)}({window_period}) "
        f"factors={len(available)}/{len(feature_cols)}",
        flush=True,
    )

    monthly_ics: dict[str, list[float]] = {col: [] for col in feature_cols}
    started_at = time.time()
    progress_every = max(1, len(windows) // 10)

    for idx, window in enumerate(windows, start=1):
        if idx == 1 or idx % progress_every == 0 or idx == len(windows):
            print(
                f"[audit] window {idx}/{len(windows)} ({window}) "
                f"elapsed={time.time()-started_at:.1f}s",
                flush=True,
            )
        block = df.loc[df["__window__"] == window]
        if len(block) < min_obs_per_window:
            continue
        target_ranks = block[target_col].rank(method="average")
        target_centered = target_ranks - target_ranks.mean()
        target_var = float((target_centered ** 2).sum())
        if target_var <= 0:
            continue

        block_feat = block[available]
        col_n = block_feat.notna().sum(axis=0)
        usable_cols = col_n.index[col_n >= min_obs_per_window].tolist()
        if not usable_cols:
            continue

        feat_ranks = block_feat[usable_cols].rank(method="average")
        feat_centered = feat_ranks.sub(feat_ranks.mean(axis=0, skipna=True), axis=1)
        target_aligned = target_centered.values[:, None]
        feat_arr = feat_centered.values
        valid_pairs = ~np.isnan(feat_arr)
        num = np.nansum(np.where(valid_pairs, feat_arr * target_aligned, np.nan), axis=0)
        feat_var = np.nansum(np.where(valid_pairs, feat_arr ** 2, np.nan), axis=0)
        tvar_col = np.nansum(np.where(valid_pairs, target_aligned ** 2, np.nan), axis=0)
        denom = np.sqrt(feat_var * tvar_col)
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
        rows.append({"name": col, "mean_ic": mean_ic, "std_ic": std_ic,
                     "icir": icir, "n_windows": n})
    return pd.DataFrame(rows).set_index("name")


def _compute_single_ic(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, float]:
    """在整段数据上计算一次 Spearman IC（无 walk-forward）。用于 val 一致性检验。

    样本量不足 10 时返回 NaN，调用方跳过该因子的一致性检验。
    """
    df = data.dropna(subset=[target_col]).copy()
    available = [c for c in feature_cols if c in df.columns]
    if len(df) < 10 or not available:
        return {col: float("nan") for col in feature_cols}

    target_ranks = df[target_col].rank(method="average")
    target_centered = (target_ranks - target_ranks.mean()).values[:, None]
    target_var = float((target_centered ** 2).sum())
    if target_var <= 0:
        return {col: float("nan") for col in feature_cols}

    feat_ranks = df[available].rank(method="average")
    feat_centered = feat_ranks.sub(feat_ranks.mean(axis=0, skipna=True), axis=1).values
    valid = ~np.isnan(feat_centered)
    num = np.nansum(np.where(valid, feat_centered * target_centered, np.nan), axis=0)
    fvar = np.nansum(np.where(valid, feat_centered ** 2, np.nan), axis=0)
    tvar = np.nansum(np.where(valid, target_centered ** 2, np.nan), axis=0)
    denom = np.sqrt(fvar * tvar)
    with np.errstate(divide="ignore", invalid="ignore"):
        ics = np.where(denom > 0, num / denom, float("nan"))

    result = {col: float("nan") for col in feature_cols}
    for col_name, ic_val in zip(available, ics):
        result[col_name] = float(ic_val)
    return result


# ─────────────────────────── 中观分层审计 ───────────────────────────

def _audit_mid_tier(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    target_col: str,
    mid_cols: list[str],
    tier_cfg: dict[str, Any],
    val_cfg: dict[str, Any],
    period: str,
    tdate_col: str = "TDATE",
    trade_date_col: str = "TRADE_DATE",
) -> pd.DataFrame:
    """对一组 MID_* 列做原生频率 IC 审计 + val 一致性检验。

    返回 DataFrame，index=col_name，列包含 mean_ic/std_ic/icir/n_windows/val_ic/val_ic_passes。
    period: "day"（Tier B）或 "week"（Tier C）。
    """
    min_abs_ic = float(tier_cfg.get("min_abs_ic", 0.015))
    min_icir = float(tier_cfg.get("min_icir", 0.40))
    min_obs = int(tier_cfg.get("min_obs_per_window", 15))
    window_period = "M" if period == "day" else "Q"
    retention = float(val_cfg.get(
        "daily_min_retention" if period == "day" else "weekly_min_retention", 0.35
    ))

    # 聚合到原生频率
    train_agg = _aggregate_by_period(
        train_data, target_col, mid_cols, period, trade_date_col, tdate_col
    )
    if train_agg.empty:
        return pd.DataFrame()

    agg_features = [c for c in mid_cols if c in train_agg.columns]
    ic_df = compute_walk_forward_ic(
        train_agg, target_col, agg_features,
        trade_date_col="TRADE_DATE",
        min_obs_per_window=min_obs,
        window_period=window_period,
    )

    # val 一致性：全段 IC（不做 walk-forward），样本量不足时跳过
    val_ics: dict[str, float] = {}
    if val_cfg.get("required", True) and not val_data.empty:
        val_agg = _aggregate_by_period(
            val_data, target_col, mid_cols, period, trade_date_col, tdate_col
        )
        if not val_agg.empty:
            val_ics = _compute_single_ic(val_agg, target_col, agg_features)

    rows = []
    for col in mid_cols:
        if col not in ic_df.index:
            rows.append({
                "name": col, "mean_ic": 0.0, "std_ic": 0.0,
                "icir": 0.0, "n_windows": 0,
                "val_ic": float("nan"), "val_ic_passes": False,
                "passes_tier": False,
            })
            continue

        mean_ic = float(ic_df.at[col, "mean_ic"])
        std_ic = float(ic_df.at[col, "std_ic"])
        icir = float(ic_df.at[col, "icir"])
        n_windows = int(ic_df.at[col, "n_windows"])
        val_ic = val_ics.get(col, float("nan"))

        # Tier 通过条件
        passes_threshold = (
            abs(mean_ic) >= min_abs_ic
            and abs(icir) >= min_icir
            and n_windows >= 3
        )
        # val 一致性：NaN 时跳过检验（不惩罚）
        if val_cfg.get("required", True) and not np.isnan(val_ic):
            val_same_sign = (mean_ic * val_ic) > 0
            val_retained = abs(val_ic) >= retention * abs(mean_ic)
            val_passes = val_same_sign and val_retained
        else:
            val_passes = True  # 样本不足时不强制

        rows.append({
            "name": col,
            "mean_ic": mean_ic, "std_ic": std_ic,
            "icir": icir, "n_windows": n_windows,
            "val_ic": val_ic if not np.isnan(val_ic) else None,
            "val_ic_passes": bool(val_passes),
            "passes_tier": bool(passes_threshold and val_passes),
        })

    return pd.DataFrame(rows).set_index("name")


def _select_midxmicro(
    midxmicro_cols: list[str],
    selected_mid_set: set[str],
    selected_micro_set: set[str],
    max_n: int,
    ic_df_1min: pd.DataFrame,
) -> list[str]:
    """MIDxMICRO 准入：父因子双门 + 按 1分钟 |ICIR| 排序取 top-max_n。

    MIDxMICRO 不做独立 IC 审计。准入条件：
      1. 中观父因子通过 Tier B/C（在 selected_mid_set 中）
      2. 微观父因子通过 Tier A（在 selected_micro_set 中）
    两个父因子都通过后，按该交互特征在 1分钟层面的 |ICIR| 排序，取 top-max_n。
    1分钟 IC 这里仅作排序依据（不作准入门），选最有区分度的交互。
    """
    candidates = []
    for col in midxmicro_cols:
        if "_X_" not in col:
            continue
        # 格式：MIDxMICRO_<micro>_X_<mid_col>
        parts = col.split("_X_", 1)
        micro_part = parts[0].replace("MIDxMICRO_", "", 1)   # e.g. ENG_RET_60
        mid_part = parts[1]                                   # e.g. MID_M_S004_PCT_RANK_52
        if micro_part not in selected_micro_set:
            continue
        # mid_part 可能是派生列；只要其在 selected_mid_set 中即可
        if mid_part not in selected_mid_set:
            continue
        icir_val = abs(float(ic_df_1min.at[col, "icir"])) if col in ic_df_1min.index else 0.0
        candidates.append((col, icir_val))

    candidates.sort(key=lambda x: -x[1])
    return [col for col, _ in candidates[:max_n]]


# ─────────────────────────── 筛选 + 写 registry ───────────────────────────

def audit_and_filter(
    *,
    prepared,
    output_path: Path,
    min_abs_ic: float,
    min_icir: float,
    min_obs_per_window: int = 200,
    mid_audit_cfg: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """三层因子审计 + 筛选 + 写 per-product registry。

    Tier A  微观（ENG_*, runtime）：1分钟月度 walk-forward IC，沿用 min_abs_ic/min_icir。
    Tier B  日频中观（MID_*, freq=日）：日度月度 walk-forward IC + val 一致性。
    Tier C  周/月频中观（MID_*, freq=周/月）：周度季度 walk-forward IC + val 一致性 + 预算。
    MIDxMICRO：不做 IC 审计，靠父因子门 + max_n 硬上限。

    mid_audit_cfg 为 None 时退化为旧行为（所有因子走 Tier A），保持向后兼容。
    """
    feature_cols = list(prepared.feature_cols)
    runtime_cols = set(prepared.runtime_factor_cols)
    extra_cols = set(prepared.feature_manifest.get("extra_feature_cols", []))
    target_col = prepared.target_col
    product_id = prepared.feature_manifest.get("product_id", "?")
    tdate_col = "TDATE"
    trade_date_col = "TRADE_DATE"

    print(
        f"[factor_audit] product={product_id} n_candidates={len(feature_cols)} "
        f"mid_audit={'on' if mid_audit_cfg else 'off(legacy)'}",
        flush=True,
    )

    # ── 分层路由 ──────────────────────────────────────────────────────────────
    if mid_audit_cfg:
        freq_map = _build_mid_freq_map(prepared.feature_manifest)
        daily_freqs = {"日", "日度", "daily"}
        weekly_freqs = {"周", "月", "周度", "月度", "weekly", "monthly"}

        micro_cols, mid_daily_cols, mid_weekly_cols, midxmicro_cols = [], [], [], []
        for col in feature_cols:
            if col.startswith("MIDxMICRO_"):
                midxmicro_cols.append(col)
            elif col.startswith("MID_"):
                freq = freq_map.get(col, "")
                if freq in daily_freqs:
                    mid_daily_cols.append(col)
                else:
                    # 未知频率归入周度，更保守
                    mid_weekly_cols.append(col)
            else:
                micro_cols.append(col)
    else:
        # 旧模式：全部走 Tier A
        micro_cols = feature_cols
        mid_daily_cols, mid_weekly_cols, midxmicro_cols = [], [], []

    # ── Tier A：微观（含旧模式下的全部因子） ────────────────────────────────
    print(f"[factor_audit] Tier A: {len(micro_cols)} micro cols", flush=True)
    ic_df_micro = compute_walk_forward_ic(
        train_data=prepared.train_data,
        target_col=target_col,
        feature_cols=micro_cols,
        trade_date_col=trade_date_col,
        min_obs_per_window=min_obs_per_window,
        window_period="M",
    )

    # ── Tier B：日频中观 ─────────────────────────────────────────────────────
    tier_b_df = pd.DataFrame()
    if mid_daily_cols and mid_audit_cfg:
        print(f"[factor_audit] Tier B: {len(mid_daily_cols)} daily-mid cols", flush=True)
        tier_b_df = _audit_mid_tier(
            prepared.train_data, prepared.val_data,
            target_col, mid_daily_cols,
            mid_audit_cfg.get("daily", {}),
            mid_audit_cfg.get("val_consistency", {}),
            period="day",
            tdate_col=tdate_col, trade_date_col=trade_date_col,
        )

    # ── Tier C：周/月频中观 ──────────────────────────────────────────────────
    tier_c_df = pd.DataFrame()
    if mid_weekly_cols and mid_audit_cfg:
        print(f"[factor_audit] Tier C: {len(mid_weekly_cols)} weekly-mid cols", flush=True)
        tier_c_df = _audit_mid_tier(
            prepared.train_data, prepared.val_data,
            target_col, mid_weekly_cols,
            mid_audit_cfg.get("weekly", {}),
            mid_audit_cfg.get("val_consistency", {}),
            period="week",
            tdate_col=tdate_col, trade_date_col=trade_date_col,
        )
        # 周度硬预算：通过阈值的原始指标最多保留 max_raw_features 个
        max_raw = int(mid_audit_cfg.get("weekly", {}).get("max_raw_features", 6))
        _apply_weekly_raw_budget(tier_c_df, prepared.feature_manifest, max_raw)

    # ── MIDxMICRO：父因子门 + 硬上限 ────────────────────────────────────────
    selected_mid = set()
    if not tier_b_df.empty:
        selected_mid.update(
            tier_b_df.index[tier_b_df["passes_tier"]].tolist()
        )
    if not tier_c_df.empty:
        selected_mid.update(
            tier_c_df.index[tier_c_df["passes_tier"]].tolist()
        )
    # Tier A IC 用于 MIDxMICRO 排序（微观父因子必须通过 Tier A）
    selected_micro = set(
        ic_df_micro.index[
            (ic_df_micro["mean_ic"].abs() >= min_abs_ic)
            & (ic_df_micro["icir"].abs() >= min_icir)
            & (ic_df_micro["n_windows"] >= 3)
        ].tolist()
    )
    max_mx = int(mid_audit_cfg.get("midxmicro_max_per_product", 15)) if mid_audit_cfg else 100
    selected_midxmicro = _select_midxmicro(
        midxmicro_cols, selected_mid, selected_micro, max_mx, ic_df_micro
    )

    # ── 汇总选择结果 ─────────────────────────────────────────────────────────
    today = dt.date.today().isoformat()
    audit_ts = dt.datetime.now().isoformat(timespec="seconds")
    train_records: list[dict[str, Any]] = []
    skip_records: list[dict[str, Any]] = []
    selected: list[str] = []

    def _make_record(
        col: str,
        mean_ic: float,
        std_ic: float,
        icir: float,
        n_windows: int,
        passes: bool,
        audit_tier: str,
        val_ic: float | None = None,
        val_ic_passes: bool = True,
        reason: str = "",
    ) -> dict[str, Any]:
        category, source = classify_factor(col, runtime_cols, extra_cols)
        rec: dict[str, Any] = {
            "name": col,
            "category": category,
            "source": source,
            "audit_tier": audit_tier,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "abs_ic": abs(mean_ic),
            "n_windows": n_windows,
            "val_ic": val_ic,
            "val_ic_passes": val_ic_passes,
            "importance_gain": 0.0,
            "use_in_training": bool(passes),
            "last_audit_at": today,
            "notes": "",
        }
        if not passes:
            rec["reason"] = reason
        return rec

    # Tier A 微观
    for col in micro_cols:
        if col in ic_df_micro.index:
            mean_ic = float(ic_df_micro.at[col, "mean_ic"])
            std_ic = float(ic_df_micro.at[col, "std_ic"])
            icir = float(ic_df_micro.at[col, "icir"])
            n_windows = int(ic_df_micro.at[col, "n_windows"])
        else:
            mean_ic = std_ic = icir = 0.0
            n_windows = 0
        passes = (
            abs(mean_ic) >= min_abs_ic
            and abs(icir) >= min_icir
            and n_windows >= 3
        )
        reasons = []
        if n_windows < 3:
            reasons.append("n_windows<3")
        if abs(mean_ic) < min_abs_ic:
            reasons.append(f"|mean_ic|<{min_abs_ic}")
        if abs(icir) < min_icir:
            reasons.append(f"|icir|<{min_icir}")
        rec = _make_record(col, mean_ic, std_ic, icir, n_windows, passes,
                           "A", reason=";".join(reasons))
        if passes:
            train_records.append(rec)
            selected.append(col)
        else:
            skip_records.append(rec)

    # Tier B/C 中观
    for mid_col, tier_df, tier_label in [
        (mid_daily_cols, tier_b_df, "B"),
        (mid_weekly_cols, tier_c_df, "C"),
    ]:
        for col in mid_col:
            if not tier_df.empty and col in tier_df.index:
                row = tier_df.loc[col]
                mean_ic = float(row.get("mean_ic", 0.0))
                std_ic = float(row.get("std_ic", 0.0))
                icir = float(row.get("icir", 0.0))
                n_windows = int(row.get("n_windows", 0))
                val_ic_raw = row.get("val_ic")
                val_ic = float(val_ic_raw) if val_ic_raw is not None else None
                val_passes = bool(row.get("val_ic_passes", False))
                passes = bool(row.get("passes_tier", False))
            else:
                mean_ic = std_ic = icir = 0.0
                n_windows = 0
                val_ic = None
                val_passes = False
                passes = False
            reasons = _mid_skip_reasons(
                col, mean_ic, icir, n_windows, val_ic, val_passes, mid_audit_cfg, tier_label
            )
            rec = _make_record(col, mean_ic, std_ic, icir, n_windows, passes,
                               tier_label, val_ic, val_passes,
                               reason=";".join(reasons))
            if passes:
                train_records.append(rec)
                selected.append(col)
            else:
                skip_records.append(rec)

    # MIDxMICRO
    selected_mx_set = set(selected_midxmicro)
    for col in midxmicro_cols:
        passes = col in selected_mx_set
        # IC 数据来自 Tier A（排序用途）
        if col in ic_df_micro.index:
            mean_ic = float(ic_df_micro.at[col, "mean_ic"])
            std_ic = float(ic_df_micro.at[col, "std_ic"])
            icir = float(ic_df_micro.at[col, "icir"])
            n_windows = int(ic_df_micro.at[col, "n_windows"])
        else:
            mean_ic = std_ic = icir = 0.0
            n_windows = 0
        reason = "" if passes else "parent_gate_or_budget"
        rec = _make_record(col, mean_ic, std_ic, icir, n_windows, passes,
                           "MIDxMICRO", reason=reason)
        if passes:
            train_records.append(rec)
            selected.append(col)
        else:
            skip_records.append(rec)

    def _sort_key(r: dict) -> tuple[float, float]:
        return (-abs(float(r.get("icir", 0.0))), -abs(float(r.get("mean_ic", 0.0))))

    train_records.sort(key=_sort_key)
    skip_records.sort(key=_sort_key)

    def _avg_abs(records: list[dict], field: str) -> float:
        vals = [abs(float(r.get(field, 0.0))) for r in records]
        return float(np.mean(vals)) if vals else 0.0

    all_records = train_records + skip_records
    payload: dict[str, Any] = {
        "metadata": {
            "schema_version": 4,
            "product_id": prepared.feature_manifest.get("product_id"),
            "raw_data": prepared.feature_manifest.get("raw_data"),
            "target_col": target_col,
            "target_horizon": int(prepared.metadata.get("target_horizon", 0)),
            "thresholds": {
                "micro": {"min_abs_ic": min_abs_ic, "min_icir": min_icir},
                "mid_audit": mid_audit_cfg or {},
            },
            "filter_rule": (
                "Tier A: abs(mean_ic)>=min_abs_ic AND abs(icir)>=min_icir AND n_windows>=3; "
                "Tier B/C: native-freq IC + val consistency + weekly budget; "
                "MIDxMICRO: parent gate + hard cap"
            ),
            "last_audit_at": audit_ts,
            "n_factors": len(feature_cols),
            "n_train_factor": len(train_records),
            "n_not_train_factor": len(skip_records),
            "aggregate_stats": {
                "mean_abs_ic_all": _avg_abs(all_records, "mean_ic"),
                "mean_abs_ic_train": _avg_abs(train_records, "mean_ic"),
                "mean_abs_ic_skip": _avg_abs(skip_records, "mean_ic"),
                "mean_abs_icir_all": _avg_abs(all_records, "icir"),
                "mean_abs_icir_train": _avg_abs(train_records, "icir"),
                "mean_abs_icir_skip": _avg_abs(skip_records, "icir"),
                "median_n_windows": int(np.median(
                    [int(r.get("n_windows", 0)) for r in all_records]
                )) if all_records else 0,
                "n_factors_with_enough_windows": sum(
                    1 for r in all_records if int(r.get("n_windows", 0)) >= 3
                ),
            },
        },
        "train_factor": train_records,
        "not_train_factor": skip_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[factor_audit] kept {len(selected)}/{len(feature_cols)} "
        f"(A={sum(1 for r in train_records if r['audit_tier']=='A')} "
        f"B={sum(1 for r in train_records if r['audit_tier']=='B')} "
        f"C={sum(1 for r in train_records if r['audit_tier']=='C')} "
        f"MX={sum(1 for r in train_records if r['audit_tier']=='MIDxMICRO')}); "
        f"registry → {output_path}",
        flush=True,
    )
    return selected, payload


# ─────────────────────────── 辅助函数 ───────────────────────────

def _apply_weekly_raw_budget(
    tier_c_df: pd.DataFrame,
    feature_manifest: dict[str, Any],
    max_raw: int,
) -> None:
    """对 Tier C 通过的特征按原始指标预算排序截断（in-place 修改 passes_tier）。

    先找出通过阈值的原始指标，按 |ICIR| 取 top max_raw，其余原始指标（及其派生列）
    的 passes_tier 设为 False。
    """
    if tier_c_df.empty or max_raw <= 0:
        return

    meta: dict[str, dict] = feature_manifest.get("mid_weekly_metadata", {})
    raw_keys = sorted(meta.keys(), key=len, reverse=True)

    # 哪些原始指标的派生列有通过的
    passing_cols = tier_c_df.index[tier_c_df["passes_tier"]].tolist()
    raw_icir: dict[str, float] = {}
    for col in passing_cols:
        parent = _raw_mid_parent(col, raw_keys)
        if parent and parent not in raw_icir:
            raw_icir[parent] = abs(float(tier_c_df.at[col, "icir"]))
        elif parent:
            raw_icir[parent] = max(raw_icir[parent], abs(float(tier_c_df.at[col, "icir"])))

    if len(raw_icir) <= max_raw:
        return  # 不超预算，无需截断

    # 取 top-max_raw 原始指标
    top_raws = set(
        sorted(raw_icir, key=lambda k: -raw_icir[k])[:max_raw]
    )
    # 不在 top_raws 中的原始指标的所有派生列 → passes_tier = False
    for col in passing_cols:
        parent = _raw_mid_parent(col, raw_keys)
        if parent and parent not in top_raws:
            tier_c_df.at[col, "passes_tier"] = False
            tier_c_df.at[col, "val_ic_passes"] = False


def _mid_skip_reasons(
    col: str,
    mean_ic: float,
    icir: float,
    n_windows: int,
    val_ic: float | None,
    val_passes: bool,
    mid_audit_cfg: dict | None,
    tier: str,
) -> list[str]:
    reasons = []
    if mid_audit_cfg is None:
        return reasons
    tier_key = "daily" if tier == "B" else "weekly"
    tier_cfg = mid_audit_cfg.get(tier_key, {})
    min_abs_ic = float(tier_cfg.get("min_abs_ic", 0.015))
    min_icir = float(tier_cfg.get("min_icir", 0.40))
    if n_windows < 3:
        reasons.append("n_windows<3")
    if abs(mean_ic) < min_abs_ic:
        reasons.append(f"|mean_ic|<{min_abs_ic}(Tier{tier})")
    if abs(icir) < min_icir:
        reasons.append(f"|icir|<{min_icir}(Tier{tier})")
    if not val_passes:
        reasons.append("val_consistency_fail")
    return reasons


# ─────────────────────────── importance 回填 ───────────────────────────

def backfill_importance(registry_path: Path, importance: dict[str, float]) -> None:
    """训练完成后把 LightGBM importance_gain 回填进 per-product registry。

    importance_gain 仅作元数据/报告图，不影响过滤结果。
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
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
