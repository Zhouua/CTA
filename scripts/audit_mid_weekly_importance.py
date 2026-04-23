"""Audit MID_*_AVAILABLE dummy-variable effectiveness after a batch run.

Scope: T3.3 of docs/mid_weekly_integration_plan.md.

Two modes:
    1) Report (default): read LightGBM feature-importance artifacts produced
       by a run (either full ``feature_importance.json`` files per regime,
       or the top-N ``top_features`` lists in each product's
       ``training_summary.json``). Emits a markdown/JSON report with:
         - ranking of each MID_*_AVAILABLE column in each regime,
         - gain ratio (sum AVAILABLE gain / sum MID_* gain),
         - classification per the plan's thresholds (>=5% → keep, <5% →
           candidate for removal pending ablation).
    2) Ablation (``--ablation --product <PID>``): re-runs single-product
       training with ``mid_weekly.available_dummy=false``, compares the
       regime validation Sharpe (or l2) against the baseline, and folds
       the Δ into the same report. Produces
       ``docs/mid_weekly_dummy_decision.md`` for the user to sign off
       (plan § 9 Q2).

The decision itself is never written automatically — the script only
*proposes* keep/remove. See § 9 Q2 in the plan.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MID_AVAILABLE_SUFFIX = "_AVAILABLE"
MID_PREFIX = "MID_"
GAIN_SHARE_THRESHOLD = 0.05  # ≥ 5% → keep; < 5% → candidate removal
ABLATION_DELTA_THRESHOLD = 0.1  # |Δ val_sharpe| ≥ 0.1 → keep


@dataclass
class RegimeImportance:
    regime: str
    product_id: str
    source: str  # "feature_importance.json" or "training_summary.top_features"
    total_rows: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    def gain_col(self) -> str:
        if self.rows and "importance_gain" in self.rows[0]:
            return "importance_gain"
        if self.rows and "gain" in self.rows[0]:
            return "gain"
        return ""

    def total_gain(self) -> float:
        col = self.gain_col()
        if not col:
            return 0.0
        return float(sum(r.get(col, 0.0) or 0.0 for r in self.rows))


def _load_regime_importance(product_dir: Path, regime: str) -> RegimeImportance | None:
    """Try full feature_importance.json; fall back to training_summary top_features."""
    pid = product_dir.name
    fi_path = product_dir / "models" / regime / "feature_importance.json"
    if fi_path.exists():
        rows = json.loads(fi_path.read_text(encoding="utf-8"))
        return RegimeImportance(
            regime=regime, product_id=pid, source=str(fi_path), total_rows=len(rows), rows=rows
        )
    ts_path = product_dir / "training_summary.json"
    if ts_path.exists():
        ts = json.loads(ts_path.read_text(encoding="utf-8"))
        regimes = ts.get("regimes", {})
        top = regimes.get(regime, {}).get("top_features") or []
        if top:
            return RegimeImportance(
                regime=regime,
                product_id=pid,
                source=f"{ts_path} → regimes.{regime}.top_features",
                total_rows=len(top),
                rows=top,
            )
    return None


def _analyze_regime(regime_imp: RegimeImportance) -> dict[str, Any]:
    gain_col = regime_imp.gain_col()
    if not gain_col:
        return {
            "regime": regime_imp.regime,
            "product_id": regime_imp.product_id,
            "source": regime_imp.source,
            "error": "no gain column in feature-importance payload",
        }
    rows = sorted(regime_imp.rows, key=lambda r: r.get(gain_col, 0.0) or 0.0, reverse=True)
    feature_key = "feature" if rows and "feature" in rows[0] else "name"
    total = sum((r.get(gain_col) or 0.0) for r in rows) or 1.0
    mid_rows = [r for r in rows if str(r.get(feature_key, "")).startswith(MID_PREFIX)]
    avail_rows = [r for r in mid_rows if str(r.get(feature_key, "")).endswith(MID_AVAILABLE_SUFFIX)]
    level_rows = [r for r in mid_rows if not str(r.get(feature_key, "")).endswith(MID_AVAILABLE_SUFFIX)]
    rank_map = {r.get(feature_key): idx + 1 for idx, r in enumerate(rows)}
    avail_detail = [
        {
            "feature": r.get(feature_key),
            "rank": rank_map.get(r.get(feature_key)),
            "rank_pct_top": rank_map.get(r.get(feature_key), 0) / max(len(rows), 1),
            "gain": float(r.get(gain_col) or 0.0),
            "gain_share_of_total": float(r.get(gain_col) or 0.0) / total,
        }
        for r in avail_rows
    ]
    avail_gain_sum = sum(d["gain"] for d in avail_detail)
    mid_gain_sum = sum(float(r.get(gain_col) or 0.0) for r in mid_rows) or 1.0
    level_gain_sum = sum(float(r.get(gain_col) or 0.0) for r in level_rows)
    available_over_mid = avail_gain_sum / mid_gain_sum if mid_gain_sum else 0.0
    return {
        "regime": regime_imp.regime,
        "product_id": regime_imp.product_id,
        "source": regime_imp.source,
        "total_rows": len(rows),
        "mid_feature_count": len(mid_rows),
        "available_feature_count": len(avail_rows),
        "level_feature_count": len(level_rows),
        "total_gain": total,
        "mid_total_gain": mid_gain_sum,
        "available_total_gain": avail_gain_sum,
        "level_total_gain": level_gain_sum,
        "available_gain_ratio_vs_mid": available_over_mid,
        "available_passes_threshold": bool(available_over_mid >= GAIN_SHARE_THRESHOLD),
        "available_detail": avail_detail,
    }


def _scan_run(run_dir: Path) -> dict[str, Any]:
    product_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")])
    # filter to plausible product dirs: those containing training_summary.json
    product_dirs = [p for p in product_dirs if (p / "training_summary.json").exists()]
    results: list[dict[str, Any]] = []
    for pdir in product_dirs:
        for regime in ("low_vol", "high_vol"):
            regime_imp = _load_regime_importance(pdir, regime)
            if regime_imp is None:
                continue
            results.append(_analyze_regime(regime_imp))
    return {"run_dir": str(run_dir), "regime_reports": results}


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Mid-Weekly Dummy-Variable Audit")
    lines.append("")
    lines.append(f"Run: `{report['run_dir']}`")
    lines.append("")
    lines.append("## 裁决规则（plan § 9 Q2）")
    lines.append("")
    lines.append(f"- 若 `sum(MID_*_AVAILABLE gain) / sum(MID_* gain) >= {GAIN_SHARE_THRESHOLD:.0%}` **或** 消融 |Δval_sharpe| >= {ABLATION_DELTA_THRESHOLD:.1f} → **保留** `available_dummy`。")
    lines.append("- 否则 → **关闭** `available_dummy`。")
    lines.append("")
    lines.append("## 每 (product, regime) 的 AVAILABLE 贡献")
    lines.append("")
    lines.append("| Product | Regime | #MID总 | #AVAILABLE | #LEVEL | AVAILABLE gain / MID gain | 通过 5% 阈值 | 数据来源 |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    by_pid: dict[str, list[dict[str, Any]]] = {}
    for r in report.get("regime_reports", []):
        by_pid.setdefault(r.get("product_id", "?"), []).append(r)
    for pid in sorted(by_pid):
        for r in by_pid[pid]:
            if "error" in r:
                lines.append(f"| {pid} | {r['regime']} | - | - | - | ERROR | - | {r['source']} |")
                continue
            pass_mark = "✓" if r["available_passes_threshold"] else "✗"
            src = r["source"]
            src_short = Path(src).name if "feature_importance" in src else "top_features (≤N)"
            lines.append(
                f"| {pid} | {r['regime']} | {r['mid_feature_count']} | {r['available_feature_count']} | {r['level_feature_count']} | {r['available_gain_ratio_vs_mid']:.3f} | {pass_mark} | {src_short} |"
            )
    lines.append("")
    lines.append("## 汇总建议")
    lines.append("")
    passes = sum(1 for r in report.get("regime_reports", []) if r.get("available_passes_threshold"))
    total = sum(1 for r in report.get("regime_reports", []) if "error" not in r)
    lines.append(f"- (product, regime) 对中 `AVAILABLE` 通过 5% 阈值：{passes} / {total}")
    if total:
        pct = passes / total
        lines.append(f"- 通过比例：{pct:.1%}")
        if pct >= 0.33:
            lines.append(f"- **建议：保留 `available_dummy`**（≥ 1/3 的 (pid, regime) 通过阈值）")
        else:
            lines.append(f"- **建议：观察消融结果，倾向关闭 `available_dummy`**（<1/3 通过阈值）")
    ablation = report.get("ablation")
    if ablation:
        lines.append("")
        lines.append("## 消融结果（T3.3\\*）")
        lines.append("")
        lines.append(f"- product: `{ablation['product_id']}`")
        lines.append(f"- baseline val sharpe: {ablation.get('baseline_val_sharpe', 'n/a')}")
        lines.append(f"- ablation val sharpe (available_dummy=false): {ablation.get('ablation_val_sharpe', 'n/a')}")
        dl = ablation.get("delta_val_sharpe")
        if dl is not None:
            lines.append(f"- Δval_sharpe: {dl:+.4f}")
            if abs(dl) >= ABLATION_DELTA_THRESHOLD:
                lines.append(f"- |Δ| >= {ABLATION_DELTA_THRESHOLD:.1f} → 建议保留")
            else:
                lines.append(f"- |Δ| < {ABLATION_DELTA_THRESHOLD:.1f} → 建议与前面的 gain-share 结论一致取舍")
    lines.append("")
    lines.append("> 最终由用户填入 `docs/mid_weekly_integration_plan.md § 9 Q2`。")
    lines.append("")
    return "\n".join(lines)


def _run_ablation(run_dir: Path, product_id: str, baseline_summary: dict[str, Any]) -> dict[str, Any]:
    """Re-train a single product with available_dummy=false and return the Δ."""
    import yaml  # local: only needed in ablation path

    override_cfg = {
        "product": {
            "product_id": product_id,
            "raw_data_file": None,  # fallback to registry lookup in train_products
        },
        "mid_weekly": {"available_dummy": False},
        "model": {"persist_models": False},
    }
    ablation_dir = run_dir / "_ablation" / product_id
    ablation_dir.mkdir(parents=True, exist_ok=True)
    override_path = ablation_dir / "config_override.yaml"
    override_path.write_text(yaml.safe_dump(override_cfg, allow_unicode=True), encoding="utf-8")

    # Prefer using train_products.py with a single-product filter so it
    # inherits the registry wiring. That script doesn't currently accept a
    # mid_weekly override path — document the gap.
    baseline_val_sharpe = _extract_val_sharpe(baseline_summary)

    cmd = [
        sys.executable,
        "pipeline/train_products.py",
        "--product", product_id,
    ]
    print(f"[ablation] would run: {' '.join(cmd)}  (config override: {override_path})")
    print("[ablation] NOTE: train_products.py does not yet accept a mid_weekly override inline.")
    print("[ablation] To actually run ablation, add the override-pass-through or run train.py directly with mid_weekly.available_dummy=false.")
    return {
        "product_id": product_id,
        "baseline_val_sharpe": baseline_val_sharpe,
        "ablation_val_sharpe": None,
        "delta_val_sharpe": None,
        "status": "skeleton_only",
        "override_path": str(override_path),
    }


def _extract_val_sharpe(training_summary: dict[str, Any]) -> float | None:
    metrics = training_summary.get("combined_validation_metrics") or {}
    # Common sharpe-ish key names
    for k in ("sharpe", "net_sharpe", "gross_sharpe"):
        if k in metrics:
            return float(metrics[k])
    # Try per-regime
    regimes = training_summary.get("regimes", {})
    sharpe_values: list[float] = []
    for regime_payload in regimes.values():
        m = regime_payload.get("metrics") or {}
        for k in ("sharpe", "val_sharpe", "sharpe_val"):
            if k in m:
                sharpe_values.append(float(m[k]))
                break
    return sum(sharpe_values) / len(sharpe_values) if sharpe_values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory like results/runs/<run_id>")
    parser.add_argument("--output-md", default="docs/mid_weekly_dummy_decision.md")
    parser.add_argument("--output-json", default=None, help="Optional JSON dump of the audit payload")
    parser.add_argument("--ablation", action="store_true", help="Also run single-product ablation (requires --product)")
    parser.add_argument("--product", default=None, help="Product id to ablate on (required with --ablation)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run-dir {run_dir} does not exist", file=sys.stderr)
        return 2

    report = _scan_run(run_dir)

    if args.ablation:
        if not args.product:
            print("--ablation requires --product", file=sys.stderr)
            return 2
        baseline_summary_path = run_dir / args.product / "training_summary.json"
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8")) if baseline_summary_path.exists() else {}
        report["ablation"] = _run_ablation(run_dir, args.product, baseline_summary)

    md = _markdown(report)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
