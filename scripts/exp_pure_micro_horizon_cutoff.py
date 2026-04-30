"""Pure-micro factor experiment: sweep (product, target_horizon, vol_cutoff).

For each (product, horizon, cutoff) combo, train dual-regime models with
mid-weekly factors disabled and write outputs to:
    results/runs/exp_pure_micro_h{H}_c{C}/{PRODUCT}/

After all combos finish (or --report-only), aggregate every produced
backtest_summary.json into a concise markdown table at
docs/exp_pure_micro_horizon_vs_cutoff.md.

Scope:
    products: AU, AG (high-vol heavy)
              JD, M  (balanced)
              CU, SN (low-vol heavy)
    horizons: 1, 5, 10, 30, 60, 120  (bars; 1 bar = 5min)
    cutoffs:  0.60, 0.65, 0.70, 0.75, 0.80  (vol_split.vol_percentage)

Key overrides applied per combo:
    data.use_mid_weekly             -> False
    data.target_horizon             -> H
    data.cache_merged_dataset       -> False  (no cache write per user pref)
    factors.runtime.cache_generated_features -> False
    factors.registry.enabled        -> False  (registry was audited at h=60,
                                              disable for fair cross-horizon
                                              comparison)
    vol_split.vol_percentage        -> C
    signal.min_hold_bars            -> max(2, H // 2)  (scale hold to horizon)

Resume / fill-in support:
    By default, any (product, horizon, cutoff) cell whose
    results/runs/exp_pure_micro_h{H}_c{C}/{PRODUCT}/backtest_summary.json
    already exists is SKIPPED. So you can re-invoke with the same args to fill
    in only the missing cells (e.g. crashed product, or a combo you killed
    mid-run). Pass --rerun-all to disable and retrain every cell from scratch.

Examples:
    # full default sweep (skip any cells already done)
    python scripts/exp_pure_micro_horizon_cutoff.py
    # fill in only missing AU/CU at h=120
    python scripts/exp_pure_micro_horizon_cutoff.py --horizons 120 --products AU CU
    # rebuild a single combo from scratch
    python scripts/exp_pure_micro_horizon_cutoff.py \\
        --horizons 60 --cutoffs 0.70 --rerun-all
    # only regenerate the markdown report
    python scripts/exp_pure_micro_horizon_cutoff.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
for p in (str(PIPELINE_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────────
# scope
# ──────────────────────────────────────────────────────────────────
PRODUCTS = ["AU", "AG", "JD", "M", "CU", "SN"]
PRODUCT_CATEGORY = {
    "AU": "high-vol heavy",
    "AG": "high-vol heavy",
    "JD": "balanced",
    "M":  "balanced",
    "CU": "low-vol heavy",
    "SN": "low-vol heavy",
}
HORIZONS = [1, 5, 10, 30, 60, 120]
CUTOFFS = [0.60, 0.65, 0.70, 0.75, 0.80]

RUN_PREFIX = "exp_pure_micro"
RUN_RE = re.compile(rf"^{RUN_PREFIX}_h(\d+)_c(\d+)$")
DEFAULT_OUT = PROJECT_ROOT / "docs" / "exp_pure_micro_horizon_vs_cutoff.md"

CATEGORY_ORDER = ["high-vol heavy", "balanced", "low-vol heavy"]


def _scaled_min_hold(horizon: int) -> int:
    return max(2, horizon // 2)


# ──────────────────────────────────────────────────────────────────
# training: per-combo executor + driver
# ──────────────────────────────────────────────────────────────────
def _make_executor(horizon: int, cutoff: float):
    """Return a closure compatible with execute_training_session's executor signature
    that injects (horizon, cutoff, no-mid, no-cache, no-registry) overrides on top
    of the per-product override.
    """

    def _executor(product_meta, config_path, run_dir, force_rebuild, persist_to_shared_dir):
        from backtest import build_backtest_settings, execute_backtest, write_backtest_outputs
        from dataset import prepare_data
        from modeling import train_dual_regime_models
        from train_products import (
            _normalize_product_id,
            build_product_config_override,
            save_product_artifacts,
            summarize_product_run,
        )

        product_id = _normalize_product_id(product_meta["product_id"])
        product_dir = Path(run_dir) / product_id
        base_override = build_product_config_override(
            product_meta, product_dir, persist_to_shared_dir=persist_to_shared_dir
        )

        base_override.setdefault("product", {})["mid_weekly_files"] = []
        base_override["data"] = {
            "use_mid_weekly": False,
            "target_horizon": int(horizon),
            "cache_merged_dataset": False,
        }
        base_override.setdefault("factors", {}).setdefault("runtime", {})[
            "cache_generated_features"
        ] = False
        base_override["factors"].setdefault("registry", {})["enabled"] = False
        base_override["vol_split"] = {"vol_percentage": float(cutoff)}
        base_override["signal"] = {"min_hold_bars": _scaled_min_hold(int(horizon))}

        prepared = prepare_data(
            config_path=config_path,
            force_rebuild=force_rebuild,
            config_override=base_override,
        )
        artifact_map, training_summary, _ = train_dual_regime_models(
            prepared=prepared, config_path=config_path, config_override=base_override
        )
        backtest_settings = build_backtest_settings(
            config_path=config_path, config_override=base_override
        )
        backtest_artifacts = execute_backtest(
            prepared=prepared, artifact_map=artifact_map, settings=backtest_settings
        )
        write_backtest_outputs(backtest_artifacts, backtest_settings)
        save_product_artifacts(
            product_dir=product_dir,
            product_meta=product_meta,
            prepared=prepared,
            backtest_artifacts=backtest_artifacts,
        )
        return summarize_product_run(
            product_meta=product_meta,
            training_summary=training_summary,
            backtest_summary=backtest_artifacts.summary,
            product_dir=product_dir,
        )

    return _executor


def _existing_product_ids(run_dir: Path) -> set[str]:
    """Return PIDs whose backtest_summary.json already exists under run_dir/<PID>/."""
    if not run_dir.exists():
        return set()
    done: set[str] = set()
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "backtest_summary.json").exists():
            done.add(child.name.upper())
    return done


def run_one_combo(
    *,
    horizon: int,
    cutoff: float,
    products: list[str],
    config_path: str | None,
    skip_existing: bool = True,
) -> tuple[Path, list[dict]]:
    from train_products import (
        annotate_products_for_batch_skip,
        execute_training_session,
        load_batch_training_settings,
        load_product_registry,
        resolve_run_root,
        select_products,
    )

    cutoff_int = int(round(cutoff * 100))
    run_id = f"{RUN_PREFIX}_h{horizon}_c{cutoff_int}"
    run_dir = resolve_run_root(config_path=config_path) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pending_products = list(products)
    if skip_existing:
        done = _existing_product_ids(run_dir)
        pending_products = [p for p in products if p.upper() not in done]
        if done:
            print(
                f"[skip-existing] combo h={horizon} c={cutoff:.2f}: "
                f"{len(done)}/{len(products)} already done ({sorted(done)}); "
                f"running {len(pending_products)} pending",
                flush=True,
            )
        if not pending_products:
            print(
                f"=== combo h={horizon} c={cutoff:.2f} skipped (all products done) ===",
                flush=True,
            )
            return run_dir, []

    registry = load_product_registry(config_path=config_path)
    selected = select_products(registry=registry, requested_products=pending_products, run_all=False)
    selected = annotate_products_for_batch_skip(
        selected, **load_batch_training_settings(config_path=config_path)
    )

    print(f"\n=== combo h={horizon} c={cutoff:.2f} run_id={run_id} ===", flush=True)
    t0 = time.time()
    results, _, manifest = execute_training_session(
        product_records=selected,
        config_path=config_path,
        run_dir=run_dir,
        run_id=run_id,
        requested_products=pending_products,
        force_rebuild=False,
        executor=_make_executor(horizon, cutoff),
    )
    dt = time.time() - t0
    print(
        f"=== combo h={horizon} c={cutoff:.2f} done in {dt/60:.1f}min "
        f"success={manifest['success_count']} failed={manifest['failure_count']} ===",
        flush=True,
    )
    return run_dir, results


# ──────────────────────────────────────────────────────────────────
# report: aggregate run outputs into a markdown table
# ──────────────────────────────────────────────────────────────────
def _pct(x, digits=2):
    if x is None:
        return "—"
    try:
        return f"{float(x)*100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _f(x, digits=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):+.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _i(x):
    if x is None:
        return "—"
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def collect_rows(run_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not run_root.exists():
        return rows
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        horizon = int(m.group(1))
        cutoff = int(m.group(2))
        for product_dir in sorted(run_dir.iterdir()):
            if not product_dir.is_dir():
                continue
            bs_path = product_dir / "backtest_summary.json"
            if not bs_path.exists():
                continue
            try:
                s = json.loads(bs_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tb = s.get("test_backtest", {}) or {}
            vb = s.get("validation_backtest", {}) or {}
            tp = s.get("test_prediction_metrics", {}) or {}
            vp = s.get("validation_prediction_metrics", {}) or {}
            by_test = s.get("test_prediction_metrics_by_regime", {}) or {}
            by_val = s.get("validation_prediction_metrics_by_regime", {}) or {}

            tn = tb.get("net", {}) or {}
            tg = tb.get("gross", {}) or {}
            vn = vb.get("net", {}) or {}

            test_low = (by_test.get("low_vol", {}) or {}).get("rows", 0)
            test_high = (by_test.get("high_vol", {}) or {}).get("rows", 0)
            val_low = (by_val.get("low_vol", {}) or {}).get("rows", 0)
            val_high = (by_val.get("high_vol", {}) or {}).get("rows", 0)

            rows.append(
                {
                    "product": product_dir.name,
                    "category": PRODUCT_CATEGORY.get(product_dir.name, "?"),
                    "horizon": horizon,
                    "cutoff": cutoff,
                    "val_low": val_low,
                    "val_high": val_high,
                    "test_low": test_low,
                    "test_high": test_high,
                    "val_ic": vp.get("spearman_ic"),
                    "test_ic": tp.get("spearman_ic"),
                    "net_annual": tn.get("annual_return"),
                    "gross_annual": tg.get("annual_return"),
                    "sharpe": tn.get("sharpe"),
                    "mdd": tn.get("max_drawdown"),
                    "trades": tb.get("trade_count"),
                    "win_rate": tb.get("trade_win_rate"),
                    "val_net": vn.get("annual_return"),
                }
            )
    return rows


def write_md(rows: list[dict], out_path: Path) -> None:
    rows.sort(
        key=lambda r: (
            CATEGORY_ORDER.index(r["category"]) if r["category"] in CATEGORY_ORDER else 99,
            PRODUCTS.index(r["product"]) if r["product"] in PRODUCTS else 99,
            r["horizon"],
            r["cutoff"],
        )
    )

    lines: list[str] = []
    lines.append("# Pure-micro factor: horizon × vol_cutoff sweep\n")
    lines.append(
        "实验设置：禁用中观因子（`data.use_mid_weekly=false`）+ 关闭 factor_registry 过滤"
        "（让所有 horizon 公平比较），仅使用微观（runtime）+ engineered 因子，训练 "
        "dual-regime LightGBM。Test 段为 backtest 测试集，净收益已扣除 commission(1bp) "
        "+ slippage(1bp)。"
    )
    lines.append("")
    lines.append("品种分类（基于 `vol_percentage=0.70` cutoff 下 val/test 的 high-vol 比例）：")
    lines.append("- **high-vol heavy**: AU (val=51% / test=61%), AG (val=42% / test=48%)")
    lines.append("- **balanced**: JD (val=31% / test=34%), M (val=21% / test=36%)")
    lines.append("- **low-vol heavy**: CU (val=16% / test=11%), SN (val=10% / test=8.5%)")
    lines.append("")
    lines.append(
        "横轴 horizon (bars，1 bar = 5min)：1 / 5 / 10 / 30 / 60 / 120  "
        "| vol_cutoff (vol_split.vol_percentage)：60 / 65 / 70 / 75 / 80"
    )
    lines.append("")
    lines.append("## 完整结果表")
    lines.append("")
    lines.append(
        "| 品种 | 类别 | horizon | cutoff | test high% | val IC | test IC | "
        "net ann | gross ann | sharpe | mdd | trades | win% |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        n_test = (r["test_low"] or 0) + (r["test_high"] or 0)
        thp = f"{(r['test_high'] or 0) / n_test * 100:.1f}%" if n_test else "—"
        wr = r.get("win_rate")
        wr_str = f"{wr*100:.1f}%" if isinstance(wr, (int, float)) else "—"
        lines.append(
            f"| **{r['product']}** | {r['category']} | "
            f"{r['horizon']} | {r['cutoff']} | {thp} | "
            f"{_f(r['val_ic'], 4)} | {_f(r['test_ic'], 4)} | "
            f"{_pct(r['net_annual'])} | {_pct(r['gross_annual'])} | "
            f"{_f(r['sharpe'])} | {_pct(r['mdd'])} | {_i(r['trades'])} | {wr_str} |"
        )
    lines.append("")

    # 每品种最优 (horizon, cutoff)
    lines.append("## 每品种最优组合（按 net_annual 排序）")
    lines.append("")
    lines.append("| 品种 | 类别 | best horizon | best cutoff | net ann | sharpe | trades |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    by_product: dict[str, list[dict]] = {}
    for r in rows:
        by_product.setdefault(r["product"], []).append(r)
    for pid in PRODUCTS:
        prod_rows = by_product.get(pid, [])
        if not prod_rows:
            continue
        valid = [r for r in prod_rows if isinstance(r.get("net_annual"), (int, float))]
        if not valid:
            continue
        best = max(valid, key=lambda r: r["net_annual"])
        lines.append(
            f"| **{pid}** | {best['category']} | "
            f"{best['horizon']} | {best['cutoff']} | "
            f"{_pct(best['net_annual'])} | {_f(best['sharpe'])} | "
            f"{_i(best['trades'])} |"
        )
    lines.append("")

    # (horizon, cutoff) × 全品种均值
    lines.append("## (horizon, cutoff) × 全品种均值")
    lines.append("")
    lines.append("| horizon | cutoff | mean net ann | mean sharpe | mean test IC | n |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    grid: dict[tuple, list[dict]] = {}
    for r in rows:
        grid.setdefault((r["horizon"], r["cutoff"]), []).append(r)
    for key in sorted(grid.keys()):
        bucket = grid[key]
        nets = [r["net_annual"] for r in bucket if isinstance(r.get("net_annual"), (int, float))]
        sharpes = [r["sharpe"] for r in bucket if isinstance(r.get("sharpe"), (int, float))]
        ics = [r["test_ic"] for r in bucket if isinstance(r.get("test_ic"), (int, float))]
        n_net = sum(nets) / len(nets) if nets else None
        n_sh = sum(sharpes) / len(sharpes) if sharpes else None
        n_ic = sum(ics) / len(ics) if ics else None
        lines.append(
            f"| {key[0]} | {key[1]} | {_pct(n_net)} | {_f(n_sh)} | {_f(n_ic, 4)} | {len(bucket)} |"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_report(*, config_path: str | None, out_path: Path) -> int:
    from train_products import resolve_run_root

    run_root = resolve_run_root(config_path=config_path)
    rows = collect_rows(run_root)
    if not rows:
        print(f"[report] WARN: no exp_pure_micro_* runs with backtest_summary.json under {run_root}")
        return 0
    write_md(rows, out_path)
    print(f"[report] wrote {len(rows)} rows -> {out_path}", flush=True)
    return len(rows)


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--products", nargs="+", default=PRODUCTS)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument(
        "--cutoffs", nargs="+", type=float, default=CUTOFFS,
        help="vol_percentage values; e.g. 0.60 0.70 0.80",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="markdown report output path")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip training; only aggregate existing run outputs into the markdown report",
    )
    parser.add_argument(
        "--rerun-all", action="store_true",
        help="Default behavior is to skip (product, horizon, cutoff) cells whose "
             "backtest_summary.json already exists. Pass --rerun-all to disable that "
             "and re-train every cell from scratch.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.report_only:
        build_report(config_path=args.config, out_path=out_path)
        return

    products = [p.strip().upper() for p in args.products]
    horizons = [int(h) for h in args.horizons]
    cutoffs = [float(c) for c in args.cutoffs]

    overall_t0 = time.time()
    matrix_log: list[dict] = []
    skip_existing = not args.rerun_all
    for h in horizons:
        for c in cutoffs:
            try:
                run_dir, results = run_one_combo(
                    horizon=h, cutoff=c, products=products, config_path=args.config,
                    skip_existing=skip_existing,
                )
                matrix_log.append(
                    {
                        "horizon": h,
                        "cutoff": c,
                        "run_dir": str(run_dir),
                        "n_success": sum(
                            1 for r in results if str(r.get("status", "")).lower() == "success"
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"!!! combo h={h} c={c} crashed: {type(exc).__name__}: {exc}", flush=True)
                matrix_log.append(
                    {"horizon": h, "cutoff": c, "error": f"{type(exc).__name__}: {exc}"}
                )

    total_min = (time.time() - overall_t0) / 60
    print(f"\n=== ALL TRAINING DONE in {total_min:.1f}min ===", flush=True)

    from train_products import resolve_run_root

    log_path = (
        resolve_run_root(config_path=args.config) / "exp_pure_micro_matrix_log.json"
    )
    log_path.write_text(json.dumps(matrix_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"matrix_log -> {log_path}", flush=True)

    build_report(config_path=args.config, out_path=out_path)


if __name__ == "__main__":
    main()
