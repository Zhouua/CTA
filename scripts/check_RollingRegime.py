"""
check_RollingRegime.py — 比较固定 vol cutoff（rolling_regime_window=0）
vs 滚动中位数（rolling_regime_window=240）对 regime 分布和回测结果的影响。

动机：RB 测试期 99% 被标记为 low_vol（训练期只有 38%），
导致 quality gate 误触发、0 成交。滚动窗口让每天的 regime
标签相对近期 N 天中位数计算，自适应制度性波动率变化。

用法：
  python scripts/check_RollingRegime.py
  python scripts/check_RollingRegime.py --products RB CU AU M AG
  python scripts/check_RollingRegime.py --force-rebuild
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

DEFAULT_PRODUCTS = ["RB", "CU", "AU", "M", "AG", "RU"]
RUN_NAME = "check_rolling_regime"
ROLLING_WINDOW = 240


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _pct(v):
    return f"{v*100:+.2f}%" if isinstance(v, float) else "—"

def _f2(v):
    return f"{v:+.2f}" if isinstance(v, float) else "—"

def _f4(v):
    return f"{v:+.4f}" if isinstance(v, float) else "—"

def _i(v):
    return str(int(v)) if isinstance(v, (int, float)) and v == v else "—"

def _pp(v):
    return f"{v*100:.1f}%" if isinstance(v, float) else "—"


def _regime_balance(prepared) -> dict:
    """Fraction of high_vol bars in each split."""
    from dataset import REGIME_NAME_MAP
    result = {}
    full = prepared.full_data
    for split in ("train", "val", "test"):
        sub = full[full["DATA_SPLIT"] == split]
        if sub.empty:
            result[split] = None
            continue
        hv = (sub["REGIME_LABEL"] == 1).sum()
        total = len(sub)
        result[split] = hv / total if total > 0 else None
    return result


def _extract_metrics(arts) -> dict:
    bs = arts.summary
    net = (bs.get("test_backtest") or {}).get("net") or {}
    val_by_regime = bs.get("validation_prediction_metrics_by_regime") or {}
    lv_m = (val_by_regime.get("low_vol") or {}).get("metrics") or {}
    hv_m = (val_by_regime.get("high_vol") or {}).get("metrics") or {}
    return {
        "annual_return": net.get("annual_return"),
        "sharpe": net.get("sharpe"),
        "max_drawdown": net.get("max_drawdown"),
        "trade_count": (bs.get("test_backtest") or {}).get("trade_count"),
        "spearman_ic": (bs.get("test_prediction_metrics") or {}).get("spearman_ic"),
        "val_ic_lv": lv_m.get("spearman_ic"),
        "val_ic_hv": hv_m.get("spearman_ic"),
    }


# ─────────────────────────────────────────────
# core runner
# ─────────────────────────────────────────────

def _run_variant(
    product_meta: dict,
    config_path: str,
    product_dir: Path,
    force_rebuild: bool,
    rolling_window: int,
) -> dict:
    from dataset import prepare_data
    from factor_audit import audit_and_filter
    from modeling import train_dual_regime_models
    from backtest import build_backtest_settings, execute_backtest
    from train_products import (
        build_product_config_override,
        _load_audit_thresholds,
        _apply_factor_selection,
    )

    config_override = build_product_config_override(product_meta, product_dir, persist_to_shared_dir=False)
    config_override.setdefault("data", {})["cache_merged_dataset"] = False
    config_override.setdefault("factors", {}).setdefault("runtime", {})["cache_generated_features"] = False
    config_override.setdefault("vol_split", {})["rolling_regime_window"] = rolling_window

    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild, config_override=config_override)
    balance = _regime_balance(prepared)

    min_abs_ic, min_icir, min_obs = _load_audit_thresholds(config_path, config_override)
    selected_cols, _ = audit_and_filter(
        prepared=prepared,
        output_path=product_dir / "factor_registry.json",
        min_abs_ic=min_abs_ic, min_icir=min_icir, min_obs_per_window=min_obs,
    )
    prepared = _apply_factor_selection(prepared, selected_cols)

    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared, config_path=config_path, config_override=config_override,
    )
    settings = build_backtest_settings(config_path=config_path, config_override=config_override)
    arts = execute_backtest(prepared=prepared, artifact_map=artifact_map, settings=settings)

    m = _extract_metrics(arts)
    m["balance"] = balance
    return m


def _run_for_product(product_meta: dict, config_path: str, run_dir: Path, force_rebuild: bool) -> dict:
    pid = str(product_meta["product_id"]).upper()
    prod_dir_fixed = run_dir / pid / "fixed"
    prod_dir_roll = run_dir / pid / "rolling"
    prod_dir_fixed.mkdir(parents=True, exist_ok=True)
    prod_dir_roll.mkdir(parents=True, exist_ok=True)

    print(f"  [{pid}] variant=fixed (rolling_window=0) ...", flush=True)
    t0 = time.time()
    fixed = _run_variant(product_meta, config_path, prod_dir_fixed, force_rebuild, rolling_window=0)
    print(
        f"  [{pid}] fixed  net={_pct(fixed['annual_return'])} "
        f"trades={_i(fixed['trade_count'])} "
        f"testHV={_pp(fixed['balance'].get('test'))} "
        f"valIC_lv={_f4(fixed.get('val_ic_lv'))} "
        f"valIC_hv={_f4(fixed.get('val_ic_hv'))} "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    print(f"  [{pid}] variant=rolling (rolling_window={ROLLING_WINDOW}) ...", flush=True)
    t0 = time.time()
    rolling = _run_variant(product_meta, config_path, prod_dir_roll, force_rebuild, rolling_window=ROLLING_WINDOW)
    print(
        f"  [{pid}] rolling net={_pct(rolling['annual_return'])} "
        f"trades={_i(rolling['trade_count'])} "
        f"testHV={_pp(rolling['balance'].get('test'))} "
        f"valIC_lv={_f4(rolling.get('val_ic_lv'))} "
        f"valIC_hv={_f4(rolling.get('val_ic_hv'))} "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    return {"fixed": fixed, "rolling": rolling}


# ─────────────────────────────────────────────
# report
# ─────────────────────────────────────────────

def _write_report(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# check_rolling_regime — fixed cutoff vs rolling-240 median\n",
        f"rolling_regime_window=0（固定训练期 cutoff）vs {ROLLING_WINDOW}（滚动中位数）。\n",
        "| product | fixed net | roll net | fixed trades | roll trades "
        "| fixed IC | roll IC "
        "| fixed valIC_lv | fixed valIC_hv "
        "| roll valIC_lv | roll valIC_hv "
        "| fixed testHV% | roll testHV% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _row(pid, f, r):
        return (
            f"| **{pid}** "
            f"| {_pct(f.get('annual_return'))} | {_pct(r.get('annual_return'))} "
            f"| {_i(f.get('trade_count'))} | {_i(r.get('trade_count'))} "
            f"| {_f4(f.get('spearman_ic'))} | {_f4(r.get('spearman_ic'))} "
            f"| {_f4(f.get('val_ic_lv'))} | {_f4(f.get('val_ic_hv'))} "
            f"| {_f4(r.get('val_ic_lv'))} | {_f4(r.get('val_ic_hv'))} "
            f"| {_pp((f.get('balance') or {}).get('test'))} "
            f"| {_pp((r.get('balance') or {}).get('test'))} |"
        )

    ok = [r for r in rows if "fixed" in r and "rolling" in r]
    for r in ok:
        lines.append(_row(r["product_id"], r["fixed"], r["rolling"]))

    if ok:
        import statistics as st
        def _avg(key, variant):
            vals = [r[variant].get(key) for r in ok if isinstance(r[variant].get(key), float)]
            return st.mean(vals) if vals else None
        def _avg_bal(variant):
            vals = [
                (r[variant].get("balance") or {}).get("test")
                for r in ok
                if isinstance((r[variant].get("balance") or {}).get("test"), float)
            ]
            return st.mean(vals) if vals else None

        f_avg = {k: _avg(k, "fixed") for k in ["annual_return", "trade_count", "spearman_ic", "val_ic_lv", "val_ic_hv"]}
        r_avg = {k: _avg(k, "rolling") for k in ["annual_return", "trade_count", "spearman_ic", "val_ic_lv", "val_ic_hv"]}
        f_avg["balance"] = {"test": _avg_bal("fixed")}
        r_avg["balance"] = {"test": _avg_bal("rolling")}
        lines.append(_row("**avg**", f_avg, r_avg))

    for r in rows:
        if "error" in r:
            lines.append(f"\n> **{r['product_id']} failed**: {r['error']}")

    out_path = run_dir / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[check_rolling_regime] report → {out_path}", flush=True)


# ─────────────────────────────────────────────
# registry
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
    parser = argparse.ArgumentParser(description="Compare fixed vs rolling regime window.")
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
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
                if "fixed" in r and "rolling" in r:
                    existing[r["product_id"]] = r
        except Exception:
            pass

    for pid in products:
        if pid in existing:
            print(f"[{pid}] skip (cached; --rerun-all to redo)", flush=True)
            rows.append(existing[pid])
            continue
        if pid not in registry:
            print(f"[{pid}] not in registry", flush=True)
            rows.append({"product_id": pid, "error": "not in product_registry"})
            continue

        print(f"\n[check_rolling_regime] === {pid} ===", flush=True)
        try:
            result = _run_for_product(registry[pid], config_path, run_dir, args.force_rebuild)
            rows.append({"product_id": pid, **result})
        except Exception as exc:
            traceback.print_exc()
            rows.append({"product_id": pid, "error": f"{type(exc).__name__}: {exc}"})

        _write_report(run_dir, rows)

    _write_report(run_dir, rows)

    print("\n[check_rolling_regime] summary:")
    for r in rows:
        if "fixed" in r:
            f, rol = r["fixed"], r["rolling"]
            fb = (f.get("balance") or {}).get("test")
            rb = (rol.get("balance") or {}).get("test")
            print(
                f"  {r['product_id']:4s}  fixed={_pct(f['annual_return'])} trades={_i(f['trade_count'])} "
                f"testHV={_pp(fb)}  |  "
                f"rolling={_pct(rol['annual_return'])} trades={_i(rol['trade_count'])} "
                f"testHV={_pp(rb)}"
            )


if __name__ == "__main__":
    main()
