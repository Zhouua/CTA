"""
check_StrategyA.py — 对比 sample_weight_alpha=0（均匀权重）vs alpha=1（按|target|加权）。

backtest 只有 top 12% 极端预测 bar 才开仓，P&L 由尾部方向准确率决定。
L2 均等对待所有 bar，中性 bar 梯度主导训练。alpha=1 让极端 bar 获得更多梯度。
val set 不传 weight，early stopping 保持无偏。

两个变体共享 prepare_data + factor_audit（因子集相同，仅训练 weight 不同）。
结果写入 results/runs/check_strategy_a/comparison.json 和 comparison.md。

用法:
  python scripts/check_StrategyA.py
  python scripts/check_StrategyA.py --products RB CU AU
  python scripts/check_StrategyA.py --alpha 2          # 对比 alpha=2 vs baseline
  python scripts/check_StrategyA.py --rerun-all
"""
from __future__ import annotations

import argparse
import copy
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
    "RB", "CU", "AU", "M", "AG", "AL", "ZN", "SN", "JD",
    "RU", "BU", "FU", "J", "JM", "Y",
]
RUN_NAME = "check_strategy_a"


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
# core: train + backtest with given alpha
# ─────────────────────────────────────────────

def _train_and_backtest(prepared, alpha: float, config_path: str, config_override: dict) -> dict:
    from modeling import train_dual_regime_models, REGIME_NAME_MAP
    from backtest import build_backtest_settings, execute_backtest

    # 深拷贝避免多个 alpha 调用间互相污染
    override = copy.deepcopy(config_override)
    override.setdefault("model", {})["sample_weight_alpha"] = alpha

    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared,
        config_path=config_path,
        config_override=override,
    )

    # 把每个 regime 的 val_IC 带出来，用于诊断 quality gate 是否触发
    val_ics = {
        REGIME_NAME_MAP[label]: round(float(art.val_spearman_ic), 4)
        for label, art in artifact_map.items()
    }

    settings = build_backtest_settings(config_path=config_path, config_override=override)
    min_ic = settings.get("min_regime_val_ic", -0.01)

    arts = execute_backtest(prepared=prepared, artifact_map=artifact_map, settings=settings)
    metrics = _extract_metrics(arts.summary)
    metrics["val_ics"] = val_ics

    # 打印触发 quality gate 的 regime，方便即时诊断
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
    candidate_alpha: float,
) -> dict:
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

    # baseline: alpha=0（均匀权重，等价于当前默认）
    print(f"  [{product_id}] training baseline (alpha=0) ...", flush=True)
    t0 = time.time()
    baseline = _train_and_backtest(prepared, 0.0, config_path, config_override)
    print(
        f"  [{product_id}] baseline done ({time.time()-t0:.0f}s)  "
        f"net={_pct(baseline['annual_return'])}  sharpe={_f2(baseline['sharpe'])}  "
        f"val_ic={baseline['val_ics']}",
        flush=True,
    )

    # candidate: alpha=candidate_alpha（按|target|^alpha 加权）
    print(f"  [{product_id}] training strategy_a (alpha={candidate_alpha}) ...", flush=True)
    t0 = time.time()
    strategy_a = _train_and_backtest(prepared, candidate_alpha, config_path, config_override)
    print(
        f"  [{product_id}] strategy_a done ({time.time()-t0:.0f}s)  "
        f"net={_pct(strategy_a['annual_return'])}  sharpe={_f2(strategy_a['sharpe'])}  "
        f"val_ic={strategy_a['val_ics']}",
        flush=True,
    )

    return {"baseline": baseline, "strategy_a": strategy_a}


# ─────────────────────────────────────────────
# report
# ─────────────────────────────────────────────

def _write_comparison(run_dir: Path, rows: list[dict], candidate_alpha: float) -> None:
    (run_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# check_strategy_a — baseline (alpha=0) vs strategy_a (alpha={candidate_alpha})\n",
        "共享 prepare_data + factor_audit；仅训练 sample weight 不同。\n",
        "> ⚠️ val_IC 列为 low_vol/high_vol regime 的验证集 Spearman IC；任意 regime < -0.01 → quality gate 触发，trades=0。\n",
        "| product | baseline net | strategy_a net | baseline sharpe | strategy_a sharpe "
        "| baseline IC | strategy_a IC | baseline trades | strategy_a trades "
        "| baseline val_IC (lv/hv) | strategy_a val_IC (lv/hv) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
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
            f"| {_val_ic_str(b)} | {_val_ic_str(c)} |"
        )

    ok_rows = [r for r in rows if "baseline" in r and "strategy_a" in r]
    for r in ok_rows:
        lines.append(_row(r["product_id"], r["baseline"], r["strategy_a"]))

    if ok_rows:
        import statistics as st
        def _avg(key, variant):
            vals = [r[variant].get(key) for r in ok_rows
                    if isinstance(r[variant].get(key), float)]
            return st.mean(vals) if vals else None
        lines.append(_row(
            "**avg**",
            {k: _avg(k, "baseline") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
            {k: _avg(k, "strategy_a") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
        ))

    for r in rows:
        if "error" in r:
            lines.append(f"\n> **{r['product_id']} failed**: {r['error']}")

    out_path = run_dir / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[check_strategy_a] comparison.md → {out_path}", flush=True)


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
    parser = argparse.ArgumentParser(description="Compare baseline (alpha=0) vs strategy_a (alpha>0).")
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument("--alpha", type=float, default=1.0, help="candidate sample_weight_alpha (default 1.0)")
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
                if "baseline" in r and "strategy_a" in r:
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

        print(f"\n[check_strategy_a] === {pid} ===", flush=True)
        try:
            result = _run_for_product(
                product_meta=registry[pid],
                config_path=config_path,
                run_dir=run_dir,
                force_rebuild=args.force_rebuild,
                candidate_alpha=args.alpha,
            )
            rows.append({"product_id": pid, **result})
        except Exception as exc:
            print(f"[{pid}] ERROR: {exc}", flush=True)
            traceback.print_exc()
            rows.append({"product_id": pid, "error": f"{type(exc).__name__}: {exc}"})

        _write_comparison(run_dir, rows, args.alpha)

    _write_comparison(run_dir, rows, args.alpha)
    print("\n[check_strategy_a] summary:", flush=True)
    for r in rows:
        if "baseline" in r:
            b, c = r["baseline"], r["strategy_a"]
            delta = (c["sharpe"] - b["sharpe"]) if isinstance(c.get("sharpe"), float) and isinstance(b.get("sharpe"), float) else float("nan")
            print(
                f"  {r['product_id']:4s}  baseline net={_pct(b['annual_return'])} sharpe={_f2(b['sharpe'])}"
                f"  |  strategy_a net={_pct(c['annual_return'])} sharpe={_f2(c['sharpe'])}"
                f"  |  ΔSharpe={delta:+.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
