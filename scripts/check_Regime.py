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

def _backtest_single_model(
    prepared, config_path: str, config_override: dict
) -> tuple[dict, "pd.DataFrame"]:
    """Train ONE LightGBM on all train rows (no regime split) and backtest.

    Returns (metrics_dict, test_daily_df).
    """
    import numpy as np
    import pandas as pd
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
    metrics = {
        "annual_return": net.get("annual_return"),
        "sharpe": net.get("sharpe"),
        "max_drawdown": net.get("max_drawdown"),
        "trade_count": test_perf.get("trade_count"),
        "spearman_ic": test_ic.get("spearman_ic"),
    }
    return metrics, test_daily


# ─────────────────────────────────────────────
# comparison chart
# ─────────────────────────────────────────────

def _plot_nav_comparison(
    dual_daily: "pd.DataFrame",
    single_daily: "pd.DataFrame",
    dual_metrics: dict,
    single_metrics: dict,
    output_path: Path,
    product_id: str,
) -> None:
    """Plot dual_regime vs single_model cumulative net NAV on a single chart."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    })

    fig, (ax_nav, ax_dd) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    def _prep(daily):
        d = daily.copy()
        d["TRADE_DATE"] = pd.to_datetime(d["TRADE_DATE"])
        return d.sort_values("TRADE_DATE")

    dual = _prep(dual_daily)
    single = _prep(single_daily)

    def _label(name, m):
        net = _pct(m.get("annual_return"))
        sh = _f2(m.get("sharpe"))
        tr = _i(m.get("trade_count"))
        return f"{name}  net={net}  Sharpe={sh}  trades={tr}"

    ax_nav.plot(
        dual["TRADE_DATE"], dual["nav_net"],
        color="#1f77b4", linewidth=1.6,
        label=_label("双域模型", dual_metrics),
    )
    ax_nav.plot(
        single["TRADE_DATE"], single["nav_net"],
        color="#d62728", linewidth=1.4, linestyle="--",
        label=_label("单域基线", single_metrics),
    )
    ax_nav.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax_nav.set_ylabel("累计净值")
    ax_nav.set_title(f"双域模型 vs 单域基线——净值对比（{product_id}，测试集）")
    ax_nav.legend(fontsize=9)
    ax_nav.grid(alpha=0.25)

    ax_dd.fill_between(
        dual["TRADE_DATE"], dual["net_drawdown"], 0,
        color="#1f77b4", alpha=0.35, label="双域回撤",
    )
    ax_dd.fill_between(
        single["TRADE_DATE"], single["net_drawdown"], 0,
        color="#d62728", alpha=0.25, label="单域回撤",
    )
    ax_dd.set_ylabel("回撤")
    ax_dd.legend(fontsize=8)
    ax_dd.grid(alpha=0.25)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] regime_model_comparison → {output_path}", flush=True)


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
    dual_daily = arts.test_daily
    print(f"  [{product_id}] dual_regime done ({time.time()-t0:.0f}s)  "
          f"net={_pct(dual_metrics['annual_return'])}  sharpe={_f2(dual_metrics['sharpe'])}", flush=True)

    # ③b single_model
    print(f"  [{product_id}] training single_model ...", flush=True)
    t0 = time.time()
    single_metrics, single_daily = _backtest_single_model(prepared, config_path, config_override)
    print(f"  [{product_id}] single_model done ({time.time()-t0:.0f}s)  "
          f"net={_pct(single_metrics['annual_return'])}  sharpe={_f2(single_metrics['sharpe'])}", flush=True)

    # ④ comparison chart
    try:
        _plot_nav_comparison(
            dual_daily=dual_daily,
            single_daily=single_daily,
            dual_metrics=dual_metrics,
            single_metrics=single_metrics,
            output_path=product_dir / "regime_model_comparison.png",
            product_id=product_id,
        )
    except Exception as exc:
        print(f"  [{product_id}] chart failed: {exc}", flush=True)

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
    from train_products import annotate_products_for_batch_skip, load_batch_training_settings
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(config_dir, get_section(config, "paths"), ["product_registry"])
    payload = json.loads(paths["product_registry"].read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("products", [])
    annotated = annotate_products_for_batch_skip(
        records, **load_batch_training_settings(config_path=config_path)
    )
    return {str(r["product_id"]).upper(): r for r in annotated}


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dual_regime vs single_model.")
    parser.add_argument("--products", nargs="+", default=None,
                        help="Product IDs to test (default: built-in subset; use --all for full registry)")
    parser.add_argument("--all", action="store_true",
                        help="Run on every enabled product in product_registry.json")
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
    if args.all:
        products = sorted(pid for pid, rec in registry.items() if rec.get("enabled", True))
    else:
        products = [p.upper() for p in (args.products or DEFAULT_PRODUCTS)]

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

        skip_status = registry[pid].get("_batch_skip_status")
        if skip_status:
            skip_error = registry[pid].get("_batch_skip_error", "")
            print(f"[{pid}] skip ({skip_status}): {skip_error}", flush=True)
            rows.append({"product_id": pid, "error": f"{skip_status}: {skip_error}"})
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
