"""
check_EntryQuantile.py — 对比不同 entry_quantile 对回测表现的影响。

第一性原理：backtest 只在 |pred| > entry_quantile 分位数时开仓。
提高 entry_quantile（如 0.88→0.92）意味着只在预测置信度最高的 top 8% 时开仓，
减少换手、降低成本，但信号数量减少。核心问题：这个 precision vs recall 的权衡在哪里最优。

两个变体共享 prepare_data + factor_audit + 相同训练（只有回测时 threshold 不同）。
结果写入 results/runs/check_entry_quantile/comparison.json 和 comparison.md。

用法:
  python scripts/check_EntryQuantile.py
  python scripts/check_EntryQuantile.py --products RB CU AU
  python scripts/check_EntryQuantile.py --candidate-quantile 0.92
  python scripts/check_EntryQuantile.py --rerun-all
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
for _p in (str(PIPELINE_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PRODUCTS = [
    "RB", "CU", "AU", "M", "AG", "AL", "ZN", "SN", "JD",
    "RU", "BU", "FU", "J", "JM", "Y",
]
RUN_NAME = "check_entry_quantile"


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
# core: train once, backtest twice with different entry_quantile
# ─────────────────────────────────────────────

def _backtest_only(prepared, artifact_map, entry_quantile: float, config_path: str, config_override: dict) -> dict:
    from modeling import REGIME_NAME_MAP
    from backtest import build_backtest_settings, execute_backtest

    override = copy.deepcopy(config_override)
    override.setdefault("signal", {})["entry_quantile"] = entry_quantile

    val_ics = {
        REGIME_NAME_MAP[label]: round(float(art.val_spearman_ic), 4)
        for label, art in artifact_map.items()
    }

    settings = build_backtest_settings(config_path=config_path, config_override=override)
    min_ic = settings.get("min_regime_val_ic", -0.01)

    arts = execute_backtest(prepared=prepared, artifact_map=artifact_map, settings=settings)
    metrics = _extract_metrics(arts.summary)
    metrics["val_ics"] = val_ics

    degenerate = [name for name, ic in val_ics.items() if ic < min_ic]
    if degenerate:
        print(
            f"    [quality gate] {', '.join(degenerate)} val_IC < {min_ic} → position=0",
            flush=True,
        )
    return metrics


# ─────────────────────────────────────────────
# per-product runner
# ─────────────────────────────────────────────

def _run_for_product(
    product_meta: dict,
    config_path: str,
    run_dir: Path,
    force_rebuild: bool,
    baseline_quantile: float,
    candidate_quantile: float,
) -> dict:
    from dataset import prepare_data
    from factor_audit import audit_and_filter
    from modeling import train_dual_regime_models
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

    print(f"  [{product_id}] prepare_data ...", flush=True)
    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild, config_override=config_override)

    print(f"  [{product_id}] factor_audit ...", flush=True)
    min_abs_ic, min_icir, min_obs = _load_audit_thresholds(config_path, config_override)
    selected_cols, _ = audit_and_filter(
        prepared=prepared,
        output_path=product_dir / "factor_registry.json",
        min_abs_ic=min_abs_ic, min_icir=min_icir, min_obs_per_window=min_obs,
    )
    prepared = _apply_factor_selection(prepared, selected_cols)
    print(f"  [{product_id}] n_features={len(prepared.feature_cols)}", flush=True)

    # 只训练一次，两个 entry_quantile 共享同一套模型
    print(f"  [{product_id}] training (shared model) ...", flush=True)
    t0 = time.time()
    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared,
        config_path=config_path,
        config_override=config_override,
    )
    print(f"  [{product_id}] training done ({time.time()-t0:.0f}s)", flush=True)

    print(f"  [{product_id}] backtest entry_q={baseline_quantile} ...", flush=True)
    baseline = _backtest_only(prepared, artifact_map, baseline_quantile, config_path, config_override)
    print(
        f"  [{product_id}] baseline (q={baseline_quantile}) "
        f"net={_pct(baseline['annual_return'])}  sharpe={_f2(baseline['sharpe'])}  "
        f"trades={_i(baseline['trade_count'])}",
        flush=True,
    )

    print(f"  [{product_id}] backtest entry_q={candidate_quantile} ...", flush=True)
    candidate = _backtest_only(prepared, artifact_map, candidate_quantile, config_path, config_override)
    print(
        f"  [{product_id}] candidate (q={candidate_quantile}) "
        f"net={_pct(candidate['annual_return'])}  sharpe={_f2(candidate['sharpe'])}  "
        f"trades={_i(candidate['trade_count'])}",
        flush=True,
    )

    return {"baseline": baseline, "candidate": candidate}


# ─────────────────────────────────────────────
# report
# ─────────────────────────────────────────────

def _write_comparison(
    run_dir: Path, rows: list[dict],
    baseline_quantile: float, candidate_quantile: float,
) -> None:
    (run_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# check_entry_quantile — baseline (q={baseline_quantile}) vs candidate (q={candidate_quantile})\n",
        "共享 prepare_data + factor_audit + 模型训练；仅回测时 entry_quantile 不同。\n",
        "> 第一性原理：提高 entry_quantile = 只在预测置信度最高时开仓，减少换手降成本，代价是信号数量减少。\n",
        "> ⚠️ val_IC 列为 low_vol/high_vol regime 的验证集 Spearman IC；任意 regime < -0.01 → quality gate 触发，trades=0。\n",
        f"| product | base net | cand net | base sharpe | cand sharpe "
        f"| base IC | cand IC | base trades | cand trades "
        f"| val_IC (lv/hv) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def _val_ic_str(m: dict) -> str:
        ics = m.get("val_ics") or {}
        lv = ics.get("low_vol")
        hv = ics.get("high_vol")
        lv_s = f"{lv:+.4f}" if isinstance(lv, float) else "—"
        hv_s = f"{hv:+.4f}" if isinstance(hv, float) else "—"
        return f"{lv_s} / {hv_s}"

    def _row(pid, b, c):
        return (
            f"| **{pid}** "
            f"| {_pct(b.get('annual_return'))} | {_pct(c.get('annual_return'))} "
            f"| {_f2(b.get('sharpe'))} | {_f2(c.get('sharpe'))} "
            f"| {_f4(b.get('spearman_ic'))} | {_f4(c.get('spearman_ic'))} "
            f"| {_i(b.get('trade_count'))} | {_i(c.get('trade_count'))} "
            f"| {_val_ic_str(b)} |"
        )

    ok_rows = [r for r in rows if "baseline" in r and "candidate" in r]
    for r in ok_rows:
        lines.append(_row(r["product_id"], r["baseline"], r["candidate"]))

    if ok_rows:
        import statistics as st
        def _avg(key, variant):
            vals = [r[variant].get(key) for r in ok_rows
                    if isinstance(r[variant].get(key), float)]
            return st.mean(vals) if vals else None
        lines.append(_row(
            "**avg**",
            {k: _avg(k, "baseline") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
            {k: _avg(k, "candidate") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
        ))

    for r in rows:
        if "error" in r:
            lines.append(f"\n> **{r['product_id']} failed**: {r['error']}")

    out_path = run_dir / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[check_entry_quantile] comparison.md → {out_path}", flush=True)


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
    parser = argparse.ArgumentParser(
        description="Compare two entry_quantile thresholds on the same trained model."
    )
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument("--baseline-quantile", type=float, default=0.88,
                        help="baseline entry_quantile（默认 0.88 = top 12%%）")
    parser.add_argument("--candidate-quantile", type=float, default=0.92,
                        help="candidate entry_quantile（默认 0.92 = top 8%%）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--rerun-all", action="store_true")
    args = parser.parse_args()

    config_path = args.config or str(PROJECT_ROOT / "config.yaml")
    run_dir = PROJECT_ROOT / "results" / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = _load_registry(config_path)
    products = list(dict.fromkeys(p.upper() for p in args.products))

    rows: list[dict] = []
    result_cache = run_dir / "comparison.json"
    existing: dict[str, dict] = {}
    if not args.rerun_all and result_cache.exists():
        try:
            for r in json.loads(result_cache.read_text(encoding="utf-8")):
                if "baseline" in r and "candidate" in r:
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

        print(f"\n[check_entry_quantile] === {pid} ===", flush=True)
        try:
            result = _run_for_product(
                product_meta=registry[pid],
                config_path=config_path,
                run_dir=run_dir,
                force_rebuild=args.force_rebuild,
                baseline_quantile=args.baseline_quantile,
                candidate_quantile=args.candidate_quantile,
            )
            rows.append({"product_id": pid, **result})
        except Exception as exc:
            print(f"[{pid}] ERROR: {exc}", flush=True)
            traceback.print_exc()
            rows.append({"product_id": pid, "error": f"{type(exc).__name__}: {exc}"})

        _write_comparison(run_dir, rows, args.baseline_quantile, args.candidate_quantile)

    _write_comparison(run_dir, rows, args.baseline_quantile, args.candidate_quantile)
    print("\n[check_entry_quantile] summary:", flush=True)
    for r in rows:
        if "baseline" in r:
            b, c = r["baseline"], r["candidate"]
            delta = (c["sharpe"] - b["sharpe"]) if isinstance(c.get("sharpe"), float) and isinstance(b.get("sharpe"), float) else float("nan")
            print(
                f"  {r['product_id']:4s}  base sharpe={_f2(b['sharpe'])} trades={_i(b['trade_count'])}"
                f"  |  cand sharpe={_f2(c['sharpe'])} trades={_i(c['trade_count'])}"
                f"  |  ΔSharpe={delta:+.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
