"""
check_Regime.py — 比较是否分 regime 的实验。

  A. dual_regime  — 高低波各训练一个 LightGBM（当前默认流程）
  B. single_model — 忽略 regime，所有 train 行统一训练一个 LightGBM

两个变体共享同一次 prepare_data + factor_audit，只在训练阶段区分。
结果写入 results/runs/check_regime/comparison.json 和 comparison.md。

用法：
  python scripts/check_Regime.py
  python scripts/check_Regime.py --products RB CU AU M
  python scripts/check_Regime.py --rerun-all          # 覆盖已有结果
  python scripts/check_Regime.py --force-rebuild       # 重建 feature cache
"""
from __future__ import annotations

import argparse
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
    "RB", "CU", "AU", "M", "AG", "AL", "ZN", "SN", "HC", "JD",
    "RU", "BU", "FU", "J", "JM", "L", "PP", "V", "Y", "P",
]
RUN_NAME = "check_regime"


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
# core: single-model backtest
# ─────────────────────────────────────────────

def _backtest_single_model(prepared, config_path: str, config_override: dict) -> dict:
    """Train ONE LightGBM on all train rows (no regime split) and backtest."""
    import numpy as np
    from config_utils import get_section, load_project_config
    from dataset import REGIME_NAME_MAP
    from modeling import train_single_regime_model, predict_single_regime, calc_prediction_metrics
    from backtest import (
        build_backtest_settings,
        build_signal_rule_map,
        generate_positions,
        calc_pnl,
        performance_summary,
    )

    config, _ = load_project_config(config_path, config_override=config_override)
    model_cfg = get_section(config, "model")
    params = dict(model_cfg.get("common_params", {}))
    num_boost_round = int(model_cfg.get("num_boost_round", 600))
    early_stopping_rounds = int(model_cfg.get("early_stopping_rounds", 50))
    scale_method = str(model_cfg.get("scale_method", "robust")).lower()

    # single unified model on all train rows (regime_label=-1 is just a naming token)
    artifact = train_single_regime_model(
        train_df=prepared.train_data,
        val_df=prepared.val_data,
        test_df=prepared.test_data,
        feature_cols=prepared.feature_cols,
        target_col=prepared.target_col,
        scale_method=scale_method,
        params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        regime_label=-1,
    )

    val_pred = predict_single_regime(
        prepared.val_data, prepared.feature_cols,
        prepared.target_col, artifact.booster, artifact.scaler,
    )
    test_pred = predict_single_regime(
        prepared.test_data, prepared.feature_cols,
        prepared.target_col, artifact.booster, artifact.scaler,
    )

    settings = build_backtest_settings(config_path=config_path, config_override=config_override)
    # compute per-regime signal thresholds from single-model val predictions
    rule_map = build_signal_rule_map(val_pred, settings)
    fallback = next(iter(rule_map.values())) if rule_map else {}
    for label, name in REGIME_NAME_MAP.items():
        if label not in rule_map and fallback:
            rule_map[label] = {**fallback, "regime_name": name}

    test_pos = generate_positions(test_pred, rule_map, settings)
    test_pnl, test_daily = calc_pnl(
        test_pos, settings["commission_rate"], settings["slippage_rate"],
        settings["hold_to_next_bar"],
    )
    test_perf = performance_summary(test_daily, test_pnl, settings["annualization_days"])
    test_ic = calc_prediction_metrics(
        test_pred["future_return"].values, test_pred["pred_return"].values,
    )
    net = test_perf.get("net") or {}
    return {
        "annual_return": net.get("annual_return"),
        "sharpe": net.get("sharpe"),
        "max_drawdown": net.get("max_drawdown"),
        "trade_count": test_perf.get("trade_count"),
        "spearman_ic": test_ic.get("spearman_ic"),
    }


# ─────────────────────────────────────────────
# per-product runner
# ─────────────────────────────────────────────

def _run_for_product(
    product_meta: dict,
    config_path: str,
    run_dir: Path,
    force_rebuild: bool,
) -> dict:
    """Run dual_regime and single_model for one product. Returns {dual, single} metrics."""
    from dataset import prepare_data
    from factor_audit import audit_and_filter
    from modeling import train_dual_regime_models
    from backtest import build_backtest_settings, execute_backtest
    from train_products import (
        build_product_config_override,
        _load_audit_thresholds,
        _apply_factor_selection,
    )

    product_id = str(product_meta["product_id"]).upper()
    product_dir = run_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    config_override = build_product_config_override(product_meta, product_dir, persist_to_shared_dir=False)
    # disable cache writes to keep results directory clean
    config_override.setdefault("data", {})["cache_merged_dataset"] = False
    config_override.setdefault("factors", {}).setdefault("runtime", {})["cache_generated_features"] = False

    # ① prepare_data (shared)
    print(f"  [{product_id}] prepare_data ...", flush=True)
    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild, config_override=config_override)

    # ② factor audit (shared)
    print(f"  [{product_id}] factor_audit ...", flush=True)
    min_abs_ic, min_icir, min_obs = _load_audit_thresholds(config_path, config_override)
    selected_cols, _ = audit_and_filter(
        prepared=prepared,
        output_path=product_dir / "factor_registry.json",
        min_abs_ic=min_abs_ic, min_icir=min_icir, min_obs_per_window=min_obs,
    )
    prepared = _apply_factor_selection(prepared, selected_cols)
    print(f"  [{product_id}] n_selected_features={len(prepared.feature_cols)}", flush=True)

    # ③a dual_regime
    print(f"  [{product_id}] training dual_regime ...", flush=True)
    t0 = time.time()
    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared, config_path=config_path, config_override=config_override,
    )
    settings = build_backtest_settings(config_path=config_path, config_override=config_override)
    arts = execute_backtest(prepared=prepared, artifact_map=artifact_map, settings=settings)
    dual_metrics = _extract_metrics(arts.summary)
    print(f"  [{product_id}] dual_regime done ({time.time()-t0:.0f}s)  "
          f"net={_pct(dual_metrics['annual_return'])}  sharpe={_f2(dual_metrics['sharpe'])}", flush=True)

    # ③b single_model
    print(f"  [{product_id}] training single_model ...", flush=True)
    t0 = time.time()
    single_metrics = _backtest_single_model(prepared, config_path, config_override)
    print(f"  [{product_id}] single_model done ({time.time()-t0:.0f}s)  "
          f"net={_pct(single_metrics['annual_return'])}  sharpe={_f2(single_metrics['sharpe'])}", flush=True)

    return {"dual_regime": dual_metrics, "single_model": single_metrics}


# ─────────────────────────────────────────────
# report
# ─────────────────────────────────────────────

def _write_comparison(run_dir: Path, rows: list[dict]) -> None:
    """Write comparison.json and comparison.md."""
    (run_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# check_regime — dual_regime vs single_model\n",
        "共享 prepare_data + factor_audit；仅训练策略不同。\n",
        "| product | dual net | single net | dual sharpe | single sharpe "
        "| dual IC | single IC | dual trades | single trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _row(pid, d, s):
        return (
            f"| **{pid}** "
            f"| {_pct(d.get('annual_return'))} | {_pct(s.get('annual_return'))} "
            f"| {_f2(d.get('sharpe'))} | {_f2(s.get('sharpe'))} "
            f"| {_f4(d.get('spearman_ic'))} | {_f4(s.get('spearman_ic'))} "
            f"| {_i(d.get('trade_count'))} | {_i(s.get('trade_count'))} |"
        )

    ok_rows = [r for r in rows if "dual_regime" in r and "single_model" in r]
    for r in ok_rows:
        lines.append(_row(r["product_id"], r["dual_regime"], r["single_model"]))

    if ok_rows:
        import statistics as st
        def _avg(key, variant):
            vals = [r[variant].get(key) for r in ok_rows
                    if isinstance((r[variant].get(key)), float)]
            return st.mean(vals) if vals else None

        lines.append(
            _row(
                "**avg**",
                {k: _avg(k, "dual_regime") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
                {k: _avg(k, "single_model") for k in ["annual_return", "sharpe", "spearman_ic", "trade_count"]},
            )
        )

    for r in rows:
        if "error" in r:
            lines.append(f"\n> **{r['product_id']} failed**: {r['error']}")

    out_path = run_dir / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[check_regime] comparison.md → {out_path}", flush=True)


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
    parser = argparse.ArgumentParser(description="Compare dual_regime vs single_model.")
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

    # resume: load existing results
    existing: dict[str, dict] = {}
    if not args.rerun_all and result_cache.exists():
        try:
            for r in json.loads(result_cache.read_text(encoding="utf-8")):
                if "dual_regime" in r and "single_model" in r:
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

        print(f"\n[check_regime] === {pid} ===", flush=True)
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

    print("\n[check_regime] summary:", flush=True)
    for r in rows:
        if "dual_regime" in r:
            d, s = r["dual_regime"], r["single_model"]
            print(
                f"  {r['product_id']:4s}  dual net={_pct(d['annual_return'])} sharpe={_f2(d['sharpe'])}"
                f"  |  single net={_pct(s['annual_return'])} sharpe={_f2(s['sharpe'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
