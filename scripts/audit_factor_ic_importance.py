"""因子 IC × ICIR 审计脚本（驱动 data/factor_registry.json）。

用途：
  1. 在 train split 上按月切窗（walk-forward）计算每个因子与 target_vol_norm 的
     Spearman IC，得到 N 个月度 IC，再求 mean_ic / std_ic / icir = mean_ic / std_ic。
  2. 同时收集 LightGBM 训练得到的 importance gain（low_vol + high_vol 求和），
     仅作为元数据保存以供报告章节图使用，不参与训练时因子过滤。
  3. 读取 / 初始化 data/factor_registry.json：每个因子记录
       name, category, source, use_in_training,
       mean_ic, std_ic, icir, n_windows,            ← 训练时过滤依据
       abs_ic (= |mean_ic|, 兼容字段),
       importance_gain (仅供图表展示),
       last_audit_at, notes
  4. 训练用因子判定（AND 关系）：
       |mean_ic| ≥ min_abs_ic   AND   |icir| ≥ min_icir   → train_factor
       否则 → not_train_factor
  5. IC 仅基于 train split 计算，避免 val 信息泄漏到"保留哪些因子"决策中。

调用：
  python scripts/audit_factor_ic_importance.py                 # 写回 registry JSON
  python scripts/audit_factor_ic_importance.py --dry-run       # 仅查看，不修改 JSON
  python scripts/audit_factor_ic_importance.py --min-icir 0.5  # 命令行覆盖阈值
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加 repo root 到 path 以便复用 splitByVol 等
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_audit_defaults_from_config() -> dict:
    """从 config.yaml::factors.audit_thresholds 读取默认阈值。
    若 config 缺失则退回硬编码默认（0.005 / 0.3）。
    """
    config_path = PROJECT_ROOT / "config.yaml"
    defaults = {"min_abs_ic": 0.005, "min_icir": 0.3, "min_obs_per_window": 200}
    if not config_path.exists():
        return defaults
    try:
        import yaml  # PyYAML 依赖已存在
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        thresholds = (cfg.get("factors", {}) or {}).get("audit_thresholds", {}) or {}
        if "min_abs_ic" in thresholds:
            defaults["min_abs_ic"] = float(thresholds["min_abs_ic"])
        if "min_icir" in thresholds:
            defaults["min_icir"] = float(thresholds["min_icir"])
        if "min_obs_per_window" in thresholds:
            defaults["min_obs_per_window"] = int(thresholds["min_obs_per_window"])
    except Exception as exc:
        print(f"[audit] warning: 读 config.yaml 阈值失败，使用硬编码默认值：{exc}")
    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factor walk-forward IC + ICIR audit → registry JSON updater."
    )
    parser.add_argument(
        "--cache",
        default=str(PROJECT_ROOT / "results/cache/merged_features.parquet"),
        help="特征缓存 parquet 路径",
    )
    parser.add_argument(
        "--training-summary",
        default=str(PROJECT_ROOT / "results/models/training_summary.json"),
        help="训练摘要 JSON 路径（用于读取 importance_gain，仅作元数据）",
    )
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "data/factor_registry.json"),
        help="因子注册表 JSON 路径（被读取并写回）",
    )
    audit_defaults = _load_audit_defaults_from_config()
    parser.add_argument(
        "--min-abs-ic",
        type=float,
        default=audit_defaults["min_abs_ic"],
        help=f"|mean_ic| 下限（默认从 config.yaml 读取，当前 {audit_defaults['min_abs_ic']}）",
    )
    parser.add_argument(
        "--min-icir",
        type=float,
        default=audit_defaults["min_icir"],
        help=f"|ICIR| 下限（默认从 config.yaml 读取，当前 {audit_defaults['min_icir']}）",
    )
    parser.add_argument(
        "--min-obs-per-window",
        type=int,
        default=audit_defaults["min_obs_per_window"],
        help="单个月度窗口的最小有效样本数，低于则该月跳过该因子",
    )
    parser.add_argument("--target-col", default="target_vol_norm")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅查看建议，不写回 JSON。",
    )
    return parser.parse_args()


# ─────────────────────────── 因子分类（用于 registry 元数据） ───────────────────────────


def _classify_factor(name: str, runtime_cols: set[str], extra_cols: set[str]) -> tuple[str, str]:
    """返回 (category, source)。
    category: minute_intraday / daily / cross_section / interaction / mid_weekly /
              composite_alpha / runtime / extra / engineered / unknown
    source:   engineered / factor_engine / extra / mid_weekly
    """
    if name.startswith("MIDxMICRO_"):
        # A1.2: mid × micro 显式交互（旧前缀 XINT_ 已废弃）
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
    cache_path: Path,
    target_col: str,
    feature_cols: list[str],
    *,
    min_obs_per_window: int = 200,
) -> pd.DataFrame:
    """在 train split 内按月切窗计算每因子的月度 IC，再求 mean / std / ICIR。

    实现要点：
      - 仅使用 DATA_SPLIT == "train" 的行，避免 val 信息泄漏到选因子决策。
      - 按 TRADE_DATE 月份分组，每月内独立计算 Spearman IC。
      - 每月样本不足 min_obs_per_window 的因子在该月跳过（IC 不参与统计）。
      - 整个 train 期至少有 N≥3 个有效月才参与 ICIR 计算；不满足 N 个月的因子
        ICIR=NaN，会被自动判为 not_train_factor。

    Returns:
        DataFrame，index = factor name，columns:
          - mean_ic: 月度 IC 均值
          - std_ic:  月度 IC 标准差
          - icir:    mean_ic / std_ic（std 为 0 / NaN 时为 0）
          - n_windows: 有效月数
    """
    from dataloader.splitByVol import split_by_vol

    df = pd.read_parquet(cache_path)
    df["TDATE"] = pd.to_datetime(df["TDATE"])
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])

    merged, _, _, _, _ = split_by_vol(
        data=df,
        vol_percentage=0.70,
        window=20,
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15,
        label_train_only=False,
        split_granularity="month",
    )
    if target_col not in merged.columns:
        raise KeyError(f"target {target_col} 不在 cache 列中")
    train = merged.loc[merged["DATA_SPLIT"] == "train"].copy()
    train = train.dropna(subset=[target_col])
    if train.empty:
        raise ValueError("train split 为空，无法计算 walk-forward IC")

    train["__month__"] = train["TRADE_DATE"].dt.to_period("M")
    months = sorted(train["__month__"].dropna().unique())

    # 预先把候选因子限定为 cache 中实际存在的列
    available = [c for c in feature_cols if c in train.columns]
    print(f"[audit] walk-forward windows: {len(months)} months on train split, "
          f"{len(available)}/{len(feature_cols)} factors found in cache")

    # 用 (factor, month) 矩阵积累 IC
    monthly_ics: dict[str, list[float]] = {col: [] for col in feature_cols}

    for month in months:
        block = train.loc[train["__month__"] == month]
        if len(block) < min_obs_per_window:
            continue
        target_ranks = block[target_col].rank(method="average")
        target_centered = target_ranks - target_ranks.mean()
        target_var = float((target_centered ** 2).sum())
        if target_var <= 0:
            continue

        # 一次性 rank 所有 available 因子（向量化，比逐列 .corr 快）
        block_feat = block[available]
        # mask: 每列各自的有效样本
        notna_mask = block_feat.notna()
        # 只保留每列样本数 >= min_obs_per_window 的因子
        col_n = notna_mask.sum(axis=0)
        usable_cols = col_n.index[col_n >= min_obs_per_window].tolist()
        if not usable_cols:
            continue

        feat_ranks = block_feat[usable_cols].rank(method="average")
        # 对每列各自做中心化（按该列的有效均值）：用 rank 下 NaN 仍是 NaN，
        # 直接 sub mean(skipna) 即可。
        feat_centered = feat_ranks.sub(feat_ranks.mean(axis=0, skipna=True), axis=1)

        # 协方差分子：sum(t_c * f_c)，注意要在 t_c 上对齐 NaN
        target_aligned = target_centered.values[:, None]  # (n, 1)
        feat_arr = feat_centered.values  # (n, k)
        valid_pairs = ~np.isnan(feat_arr)
        # 每列的协方差
        prod = np.where(valid_pairs, feat_arr * target_aligned, np.nan)
        num = np.nansum(prod, axis=0)
        # 每列各自的方差，target 部分按列各自的有效掩码切
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


# ─────────────────────────── importance 收集（仅供元数据 / 图表） ───────────────────────────


def collect_importance(training_summary_path: Path) -> tuple[pd.Series, list[str], set[str], set[str]]:
    """合并 low_vol + high_vol 两个 regime 的 importance_gain。

    优先读取 results/models/{regime}/feature_importance.json（含全部特征 gain），
    退回 training_summary.json 的 top_features（仅 top 20）。
    importance_gain 仅作为元数据写入 registry，不参与训练时因子过滤。
    """
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    feature_cols = list(summary.get("feature_cols", []))
    runtime_cols = set(summary.get("runtime_factor_cols", []))
    feature_manifest = summary.get("feature_manifest", {})
    extra_cols = set(feature_manifest.get("extra_feature_cols", []))

    importance: dict[str, float] = {col: 0.0 for col in feature_cols}
    model_dir = training_summary_path.parent
    for regime_name in ["low_vol", "high_vol"]:
        fi_path = model_dir / regime_name / "feature_importance.json"
        if fi_path.exists():
            try:
                fi_df = pd.read_json(fi_path)
                if "feature" in fi_df.columns and "importance_gain" in fi_df.columns:
                    for _, row in fi_df.iterrows():
                        name = str(row["feature"])
                        importance[name] = importance.get(name, 0.0) + float(row["importance_gain"])
                    continue
            except Exception:
                pass
        regime = summary.get("regimes", {}).get(regime_name, {})
        top = regime.get("metrics", {}).get("top_features", [])
        for item in top:
            name = item["feature"]
            importance[name] = importance.get(name, 0.0) + float(item.get("importance_gain", 0.0))
    return pd.Series(importance, name="importance_gain"), feature_cols, runtime_cols, extra_cols


# ─────────────────────────── registry 更新 ───────────────────────────


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {
            "metadata": {
                "schema_version": 3,
                "target_col": "target_vol_norm",
                "thresholds": {},
                "last_audit_at": None,
            },
            "train_factor": [],
            "not_train_factor": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    # 向后兼容：旧 schema (factors[] + use_in_training)
    if "factors" in data and "train_factor" not in data:
        train_list = [f for f in data["factors"] if f.get("use_in_training", True)]
        skip_list = [f for f in data["factors"] if not f.get("use_in_training", True)]
        data["train_factor"] = train_list
        data["not_train_factor"] = skip_list
        data.pop("factors", None)
    data.setdefault("train_factor", [])
    data.setdefault("not_train_factor", [])
    return data


def update_registry(
    registry: dict,
    feature_cols: list[str],
    ic_df: pd.DataFrame,
    importance: pd.Series,
    runtime_cols: set[str],
    extra_cols: set[str],
    min_abs_ic: float,
    min_icir: float,
    target_col: str,
) -> dict:
    """合并审计结果到 registry 中。判定规则：
        |mean_ic| ≥ min_abs_ic  AND  |icir| ≥ min_icir  → train_factor
    importance_gain 仅作元数据保存，不参与判定。
    """
    today = dt.date.today().isoformat()
    existing_by_name: dict[str, dict] = {}
    for bucket in ("train_factor", "not_train_factor"):
        for f in registry.get(bucket, []):
            existing_by_name[f["name"]] = f

    new_records: list[dict] = []
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

        imp_val = float(importance.get(col, 0.0))
        existing = existing_by_name.get(col, {})
        category, source = _classify_factor(col, runtime_cols, extra_cols)
        record = {
            "name": col,
            "category": existing.get("category", category),
            "source": existing.get("source", source),
            "use_in_training": bool(passes),
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "n_windows": n_windows,
            # abs_ic 兼容字段：等于 |mean_ic|，保留是为了下游脚本不需要重写
            "abs_ic": abs_ic,
            # importance_gain：仅作元数据 / 图表展示，不参与 keep / skip 判定
            "importance_gain": imp_val,
            "last_audit_at": today,
            "notes": existing.get("notes", ""),
        }
        new_records.append(record)

    # 已在 registry 但本次未出现的因子：保留记录、标记 stale，归入 not_train
    seen_set = set(feature_cols)
    for name, rec in existing_by_name.items():
        if name in seen_set:
            continue
        rec = dict(rec)
        rec["stale"] = True
        rec.setdefault("category", "unknown")
        rec.setdefault("source", "unknown")
        rec["use_in_training"] = False
        new_records.append(rec)

    def _sort_key(r: dict) -> tuple[float, float, float]:
        # 训练用 / 不训练 内部都按 |ICIR| 降序、再按 |mean_ic| 降序、最后按 importance 降序
        return (
            -abs(float(r.get("icir", 0.0))),
            -abs(float(r.get("mean_ic", r.get("abs_ic", 0.0)))),
            -float(r.get("importance_gain", 0.0)),
        )

    train_records = sorted(
        [r for r in new_records if r.get("use_in_training")],
        key=_sort_key,
    )
    skip_records = sorted(
        [r for r in new_records if not r.get("use_in_training")],
        key=_sort_key,
    )

    registry["metadata"] = {
        "schema_version": 3,
        "target_col": target_col,
        "thresholds": {
            "min_abs_ic": min_abs_ic,
            "min_icir": min_icir,
        },
        "filter_rule": "abs(mean_ic) >= min_abs_ic AND abs(icir) >= min_icir",
        "last_audit_at": today,
        "n_factors": len(new_records),
        "n_train_factor": len(train_records),
        "n_not_train_factor": len(skip_records),
    }
    registry["train_factor"] = train_records
    registry["not_train_factor"] = skip_records
    registry.pop("factors", None)
    return registry


# ─────────────────────────── 主流程 ───────────────────────────


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache)
    summary_path = Path(args.training_summary)
    registry_path = Path(args.registry)

    # 如果默认 cache 不存在（单品种模式 cache 在 per-product 路径下），
    # 从 training_summary.json::dataset.cache_path 自动取回，避免用户手动指定 --cache。
    if not cache_path.exists() and summary_path.exists():
        try:
            ts = json.loads(summary_path.read_text(encoding="utf-8"))
            ts_cache = (ts.get("dataset") or {}).get("cache_path")
            if ts_cache and Path(ts_cache).exists():
                cache_path = Path(ts_cache)
                print(f"[audit] auto-detected cache from training_summary: {cache_path}")
        except Exception:
            pass

    if not cache_path.exists():
        print(f"[error] cache not found: {cache_path}", file=sys.stderr)
        return 1
    if not summary_path.exists():
        print(f"[error] training summary not found: {summary_path}", file=sys.stderr)
        return 1

    importance, feature_cols, runtime_cols, extra_cols = collect_importance(summary_path)
    if not feature_cols:
        print("[error] training_summary 没有 feature_cols", file=sys.stderr)
        return 1

    ic_df = compute_walk_forward_ic(
        cache_path,
        args.target_col,
        feature_cols,
        min_obs_per_window=args.min_obs_per_window,
    )
    print(f"[audit] mean |mean_ic| = {ic_df['mean_ic'].abs().mean():.4f}, "
          f"mean |icir| = {ic_df['icir'].abs().mean():.3f}, "
          f"median n_windows = {int(ic_df['n_windows'].median())}")

    # 合并到 registry
    registry = load_registry(registry_path)
    registry = update_registry(
        registry,
        feature_cols=feature_cols,
        ic_df=ic_df,
        importance=importance,
        runtime_cols=runtime_cols,
        extra_cols=extra_cols,
        min_abs_ic=args.min_abs_ic,
        min_icir=args.min_icir,
        target_col=args.target_col,
    )

    n_train = registry["metadata"]["n_train_factor"]
    n_skip = registry["metadata"]["n_not_train_factor"]
    n_total = registry["metadata"]["n_factors"]
    print(f"[audit] total={n_total} train_factor={n_train} not_train_factor={n_skip}")
    print(f"[audit] thresholds: min_abs_ic={args.min_abs_ic}  min_icir={args.min_icir}  "
          f"(rule: abs(mean_ic) ≥ min_abs_ic AND abs(icir) ≥ min_icir)")

    if args.dry_run:
        print("[audit] --dry-run: 不写回 JSON")
    else:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[audit] registry → {registry_path}")

    # 简要输出 not_train_factor 中的 ENG_*/ret_1d（runtime 量大不展示）
    skip_eng = [
        r for r in registry["not_train_factor"]
        if (r["name"].startswith("ENG_") or r["name"] == "ret_1d")
        and not r.get("stale", False)
    ]
    if skip_eng:
        print(f"\n[audit] not_train_factor 中的 ENG / ret_1d 清单（共 {len(skip_eng)} 个）：")
        for r in skip_eng[:30]:
            print(
                f"  - {r['name']:42s}  mean_ic={r.get('mean_ic',0):+.4f}  "
                f"icir={r.get('icir',0):+.3f}  n={r.get('n_windows',0)}  "
                f"imp={r.get('importance_gain',0):>9.1f}"
            )
        if len(skip_eng) > 30:
            print(f"  ... ({len(skip_eng) - 30} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
