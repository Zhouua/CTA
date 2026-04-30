"""
check_VolNorm.py — 比较是否进行 vol 归一化的实验。

  A. vol_norm    — target = target_vol_norm，预测时反归一化（当前默认）
  B. raw_return  — target = future_return，预测直接使用 raw_pred，无需反归一化

两个变体共享同一次 prepare_data + factor_audit（因子集相同，仅训练目标不同）。
结果写入 results/runs/check_volnorm/comparison.json 和 comparison.md。

用法：
  python scripts/check_VolNorm.py
  python scripts/check_VolNorm.py --products RB CU AU M
  python scripts/check_VolNorm.py --rerun-all          # 覆盖已有结果
  python scripts/check_VolNorm.py --force-rebuild       # 重建 feature cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace as dc_replace
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
for _p in (str(PIPELINE_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PRODUCTS = [
    "RB", "CU", "AU", "M", "AG", "AL", "ZN", "SN", "HC", "JD",
    "RU", "BU", "FU", "J", "JM", "L", "PP", "V", "Y", "P",
]
RUN_NAME = "check_volnorm"


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _extract_metrics(bs: dict) -> dict:
    net = (bs.get("test_backtest") or {}).get("net") or {}
    return {
        "annual_return": net.get("annual_return"),
        "sharpe": net.get("sharpe"),
        "max_drawdown": net.get("max_drawdown"),
        "trade_count": (bs.get("test_backtest") or {}).get("trade_count"),
        "spearman_ic": (bs.get("test_prediction_metrics") or {}).get("spearman_ic"),
    }


def _pct(v):
    return f"{v*100:+.2f}%" if isinstance(v, float) else "—"

def _f2(v):
    return f"{v:+.2f}" if isinstance(v, float) else "—"

def _f4(v):
    return f"{v:+.4f}" if isinstance(v, float) else "—"

def _i(v):
    return str(int(v)) if isinstance(v, (int, float)) and v == v else "—"


# ─────────────────────────────────────────────
# core: train dual-regime with given target_col
# ─────────────────────────────────────────────

def _train_and_backtest(prepared, target_col: str, config_path: str, config_override: dict) -> dict:
    """Train dual-regime LightGBM with `target_col` and run backtest.

    Supports two modes:
      target_vol_norm — train on vol-normalised target; de-normalise pred at inference
      future_return   — train directly on raw return; pred_return = raw_pred (no de-norm)
    """
    from modeling import train_dual_regime_models
    from backtest import build_backtest_settings, execute_backtest

    # swap target_col in prepared without touching any other field
    prepared_for_training = dc_replace(prepared, target_col=target_col)

    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared_for_training,
        config_path=config_path,
        config_override=config_override,
    )

    settings = build_backtest_settings(config_path=config_path, config_override=config_override)
    arts = execute_backtest(prepared=prepared_for_training, artifact_map=artifact_map, settings=settings)

    return _extract_metrics(arts.summary)


# ─────────────────────────────────────────────
# per-product runner
# ─────────────────────────────────────────────

def _run_for_product(
    product_meta: dict,
    config_path: str,
    run_dir: Path,
    force_rebuild: bool,
) -> dict:
    """Run vol_norm and raw_return variants for one product."""
    from dataset import prepare_data
    from factor_audit import audit_and_filter
    from train_products import (
        build_product_config_override,
        _load_audit_thresholds,
        _apply_factor_selection,
    )

    product_id = str(product_meta["product_id"]).upper()
    product_dir = run_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    config_override = build_product_config_override(product_meta, product_dir, persist_to_shared_dir=False)
    config_override.setdefault("data", {})["cache_merged_dataset"] = False
    config_override.setdefault("factors", {}).setdefault("runtime", {})["cache_generated_features"] = False

    # ① prepare_data — shared; target_col defaults to target_vol_norm from config
    print(f"  [{product_id}] prepare_data ...", flush=True)
    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild, config_override=config_override)

    # ② factor_audit — shared; IC computed against prepared.target_col (= target_vol_norm)
    #    Using the same feature set for both variants controls for confounders.
    print(f"  [{product_id}] factor_audit ...", flush=True)
    min_abs_ic, min_icir, min_obs = _load_audit_thresholds(config_path, config_override)
    selected_cols, _ = audit_and_filter(
        prepared=prepared,
        output_path=product_dir / "factor_registry.json",
        min_abs_ic=min_abs_ic, min_icir=min_icir, min_obs_per_window=min_obs,
    )
    prepared = _apply_factor_selection(prepared, selected_cols)
    print(f"  [{product_id}] n_selected_features={len(prepared.feature_cols)}", flush=True)

    # ③a vol_norm (default)
    print(f"  [{product_id}] training vol_norm ...", flush=True)
    t0 = time.time()
    vol_norm_metrics = _train_and_backtest(prepared, "target_vol_norm", config_path, config_override)
    print(
        f"  [{product_id}] vol_norm done ({time.time()-t0:.0f}s)  "
        f"net={_pct(vol_norm_metrics['annual_return'])}  sharpe={_f2(vol_norm_metrics['sharpe'])}",
        flush=True,
    )

    # ③b raw_return
    print(f"  [{product_id}] training raw_return ...", flush=True)
    t0 = time.time()
    raw_return_metrics = _train_and_backtest(prepared, "future_return", config_path, config_override)
    print(
        f"  [{product_id}] raw_return done ({time.time()-t0:.0f}s)  "
        f"net={_pct(raw_return_metrics['annual_return'])}  sharpe={_f2(raw_return_metrics['sharpe'])}",
        flush=True,
    )

    return {"vol_norm": vol_norm_metrics, "raw_return": raw_return_metrics}


# ─────────────────────────────────────────────
# report
# ─────────────────────────────────────────────

def _write_comparison(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# check_volnorm — vol_norm vs raw_return\n",
        "共享 prepare_data + factor_audit；仅训练目标不同。\n",
        "| product | vol_norm net | raw_return net | vol_norm sharpe | raw_return sharpe "
        "| vol_norm IC | raw_return IC | vol_norm trades | raw_return trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _row(pid, v, r):
        return (
            f"| **{pid}** "
            f"| {_pct(v.get('annual_return'))} | {_pct(r.get('annual_return'))} "
            f"| {_f2(v.get('sharpe'))} | {_f2(r.get('sharpe'))} "
            f"| {_f4(v.get('spearman_ic'))} | {_f4(r.get('spearman_ic'))} "
            f"| {_i(v.get('trade_count'))} | {_i(r.get('trade_count'))} |"
        )

    ok_rows = [r for r in rows if "vol_norm" in r and "raw_return" in r]
    for r in ok_rows:
        lines.append(_row(r["product_id"], r["vol_norm"], r["raw_return"]))

    if ok_rows:
        import statistics as st
        def _avg(key, variant):
            vals = [r[variant].get(key) for r in ok_rows
                    if isinstance(r[variant].get(key), float)]
            return st.mean(vals) if vals else None

        lines.append(
            _row(
                "**avg**",
                {k: _avg(k, "vol_norm") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
                {k: _avg(k, "raw_return") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
            )
        )

    for r in rows:
        if "error" in r:
            lines.append(f"\n> **{r['product_id']} failed**: {r['error']}")

    out_path = run_dir / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[check_volnorm] comparison.md → {out_path}", flush=True)


# ─────────────────────────────────────────────
# registry helpers
# ─────────────────────────────────────────────

def _load_registry(config_path: str | None) -> dict[str, dict]:
    from config_utils import get_section, load_project_config, resolve_paths
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(config_dir, get_section(config, "paths"), ["product_registry"])
    payload = json.loads(paths["product_registry"].read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("products", [])
    return {str(r["product_id"]).upper(): r for r in records}


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vol_norm vs raw_return target.")
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS,
                        help="Product IDs to test (default: RB CU AU M)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild feature cache before running")
    parser.add_argument("--rerun-all", action="store_true",
                        help="Ignore existing results and re-run all products")
    args = parser.parse_args()

    config_path = args.config or str(PROJECT_ROOT / "config.yaml")
    run_dir = PROJECT_ROOT / "results" / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(config_path)
    products = [p.upper() for p in args.products]

    rows: list[dict] = []
    result_cache = run_dir / "comparison.json"

    existing: dict[str, dict] = {}
    if not args.rerun_all and result_cache.exists():
        try:
            for r in json.loads(result_cache.read_text(encoding="utf-8")):
                if "vol_norm" in r and "raw_return" in r:
                    existing[r["product_id"]] = r
        except Exception:
            pass

    for pid in products:
        if pid in existing:
            print(f"[{pid}] skip (already done; use --rerun-all to re-run)", flush=True)
            rows.append(existing[pid])
            continue

        if pid not in registry:
            print(f"[{pid}] skip: not in product_registry", flush=True)
            rows.append({"product_id": pid, "error": "not in product_registry"})
            continue

        print(f"\n[check_volnorm] === {pid} ===", flush=True)
        try:
            result = _run_for_product(
                product_meta=registry[pid],
                config_path=config_path,
                run_dir=run_dir,
                force_rebuild=args.force_rebuild,
            )
            rows.append({"product_id": pid, **result})
        except Exception as exc:
            print(f"[{pid}] ERROR: {exc}", flush=True)
            traceback.print_exc()
            rows.append({"product_id": pid, "error": f"{type(exc).__name__}: {exc}"})

    _write_comparison(run_dir, rows)

    print("\n[check_volnorm] summary:", flush=True)
    for r in rows:
        if "vol_norm" in r:
            v, rr = r["vol_norm"], r["raw_return"]
            print(
                f"  {r['product_id']:4s}  vol_norm net={_pct(v['annual_return'])} sharpe={_f2(v['sharpe'])}"
                f"  |  raw_return net={_pct(rr['annual_return'])} sharpe={_f2(rr['sharpe'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
