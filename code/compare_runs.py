"""Compare baseline vs candidate batch runs — T4.3 of mid_weekly_integration_plan.md.

Outputs:
    <output>.csv  — per-product row (baseline / candidate / delta metrics)
    <output>.md   — narrative sections: "全面提升" / "持平" / "退步" / "结构变化"
    <output>.png  — 3-panel figure: scatter, Δ-bar, top-10 NAV overlay

Verdict rules printed in the md tail:
    - Median ΔSharpe > 0             → mid_weekly helps on balance.
    - > 1/3 of products regress      → block; do T3.3 ablation before conclusion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STAT_KEYS = {
    "sharpe": "net.sharpe",
    "annual_return": "net.annual_return",
    "max_drawdown": "net.max_drawdown",
    "trade_count": "trade_count",
    "trade_win_rate": "trade_win_rate",
    "daily_win_rate_net": "net.daily_win_rate",
}


def _load_summary(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "run_summary.csv")
    return df[df["status"] == "success"].copy()


def _backtest_payload(run_dir: Path, product_id: str, retries: int = 3) -> dict[str, Any]:
    """Read backtest_summary.json with retries — APFS near-full filesystem
    intermittently returns empty reads that bubble up as JSONDecodeError."""
    path = run_dir / product_id / "backtest_summary.json"
    if not path.exists():
        return {}
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                raise ValueError("empty file on read")
            return json.loads(raw)
        except Exception as e:
            last_err = e
            import time

            time.sleep(0.25 * attempt)
    warnings.warn(f"Failed to read {path} after {retries} attempts: {last_err}", RuntimeWarning, stacklevel=2)
    return {}


def _pick(payload: dict[str, Any], dotted: str) -> float | None:
    cur: Any = payload
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _gather(run_dir: Path, pids: list[str], which: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pid in pids:
        payload = _backtest_payload(run_dir, pid)
        block = payload.get("test_backtest", {})
        row = {"product_id": pid, "run": which}
        for col, dotted in STAT_KEYS.items():
            row[col] = _pick(block, dotted)
        rows.append(row)
    return pd.DataFrame(rows)


def _merge(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["product_id"] + list(STAT_KEYS.keys())
    b = baseline_df[cols].rename(columns={k: f"b_{k}" for k in STAT_KEYS})
    c = candidate_df[cols].rename(columns={k: f"c_{k}" for k in STAT_KEYS})
    merged = b.merge(c, on="product_id", how="inner")
    merged["delta_sharpe"] = merged["c_sharpe"] - merged["b_sharpe"]
    merged["delta_annual_return"] = merged["c_annual_return"] - merged["b_annual_return"]
    merged["delta_max_drawdown"] = merged["c_max_drawdown"] - merged["b_max_drawdown"]
    merged["delta_trade_count"] = merged["c_trade_count"] - merged["b_trade_count"]
    merged["delta_trade_win_rate"] = merged["c_trade_win_rate"] - merged["b_trade_win_rate"]
    merged["delta_daily_win_rate_net"] = merged["c_daily_win_rate_net"] - merged["b_daily_win_rate_net"]
    merged["trade_count_pct_change"] = np.where(
        merged["b_trade_count"] > 0,
        (merged["c_trade_count"] - merged["b_trade_count"]) / merged["b_trade_count"],
        np.nan,
    )
    merged = merged.sort_values("delta_sharpe", ascending=False).reset_index(drop=True)
    return merged


def _nav_overlay(baseline_dir: Path, candidate_dir: Path, product_ids: list[str], out_ax: plt.Axes) -> None:
    colors = plt.cm.tab10.colors
    for idx, pid in enumerate(product_ids[:10]):
        try:
            bn = pd.read_csv(baseline_dir / pid / "nav_curve.csv")
            cn = pd.read_csv(candidate_dir / pid / "nav_curve.csv")
        except Exception:
            continue
        bn["TRADE_DATE"] = pd.to_datetime(bn["TRADE_DATE"])
        cn["TRADE_DATE"] = pd.to_datetime(cn["TRADE_DATE"])
        col = colors[idx % len(colors)]
        out_ax.plot(bn["TRADE_DATE"], bn["nav_net"], linestyle="--", alpha=0.7, color=col, label=f"{pid} (baseline)")
        out_ax.plot(cn["TRADE_DATE"], cn["nav_net"], linestyle="-", alpha=0.95, color=col, label=f"{pid} (candidate)")
    out_ax.set_title("(c) Top-10 baseline products — test NAV (solid = candidate, dashed = baseline)")
    out_ax.set_xlabel("date")
    out_ax.set_ylabel("NAV (net)")
    out_ax.legend(loc="upper left", ncol=2, fontsize=6)
    out_ax.grid(True, alpha=0.3)


def _render_png(merged: pd.DataFrame, baseline_dir: Path, candidate_dir: Path, out_png: Path) -> None:
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_nav = fig.add_subplot(gs[2, 0])

    # (a) Scatter
    ax_scatter.scatter(merged["b_sharpe"], merged["c_sharpe"], s=50, alpha=0.75)
    lo = float(min(merged["b_sharpe"].min(), merged["c_sharpe"].min()))
    hi = float(max(merged["b_sharpe"].max(), merged["c_sharpe"].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 0.5
    ax_scatter.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="grey", alpha=0.7, label="y=x")
    for _, r in merged.iterrows():
        ax_scatter.annotate(r["product_id"], (r["b_sharpe"], r["c_sharpe"]), fontsize=7, alpha=0.8)
    ax_scatter.set_xlabel("baseline test Sharpe")
    ax_scatter.set_ylabel("candidate test Sharpe")
    ax_scatter.set_title("(a) Baseline vs candidate Sharpe")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()

    # (b) Bar
    bar_df = merged.sort_values("delta_sharpe")
    colors = ["#1b7f3a" if d > 0 else "#c0322b" for d in bar_df["delta_sharpe"]]
    ax_bar.barh(bar_df["product_id"], bar_df["delta_sharpe"], color=colors, alpha=0.85)
    ax_bar.axvline(0, color="black", linewidth=0.6)
    ax_bar.set_title(f"(b) ΔSharpe (candidate − baseline), n={len(bar_df)}")
    ax_bar.set_xlabel("ΔSharpe")
    ax_bar.grid(True, axis="x", alpha=0.3)

    # (c) NAV overlay for top-10 baseline products
    top10_pids = merged.sort_values("b_sharpe", ascending=False).head(10)["product_id"].tolist()
    _nav_overlay(baseline_dir, candidate_dir, top10_pids, ax_nav)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _section(title: str, rows: pd.DataFrame, fmt_fn) -> list[str]:
    lines = [f"### {title} ({len(rows)})", ""]
    if rows.empty:
        lines.append("_无_\n")
        return lines
    lines.append("| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for _, r in rows.iterrows():
        lines.append(fmt_fn(r))
    lines.append("")
    return lines


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['product_id']} | {r['delta_sharpe']:+.3f} | "
        f"{r['b_sharpe']:.3f} → {r['c_sharpe']:.3f} | "
        f"{r['delta_annual_return']:+.3f} | {r['delta_max_drawdown']:+.3f} | "
        f"{int(r['delta_trade_count']):+d} | {r['delta_trade_win_rate']:+.3f} |"
    )


def _render_md(merged: pd.DataFrame, baseline_id: str, candidate_id: str, out_md: Path) -> None:
    eps = 0.3
    improved = merged[merged["delta_sharpe"] >= eps]
    regressed = merged[merged["delta_sharpe"] <= -eps]
    flat_mask = (merged["delta_sharpe"].abs() < eps)
    flat = merged[flat_mask]
    # structural change: |pct-change of trade_count| >= 0.25
    struct = merged[merged["trade_count_pct_change"].abs() >= 0.25].copy()

    verdict_bits: list[str] = []
    med_delta = float(np.nanmedian(merged["delta_sharpe"]))
    frac_regress = float((merged["delta_sharpe"] < 0).mean())
    verdict_bits.append(f"- 中位数 ΔSharpe = **{med_delta:+.3f}** （> 0 时整体加分）")
    verdict_bits.append(f"- 退步 (ΔSharpe<0) 占比 = **{frac_regress:.1%}** （> 33.3% 时必须先做 T3.3 ablation 才能定论）")
    verdict = "mid_weekly **整体加分**" if med_delta > 0 else "mid_weekly **整体未加分**"
    if frac_regress > 1 / 3:
        verdict += "；且退步占比超过 1/3 → 强制进入 T3.3 ablation 复核"
    else:
        verdict += "；退步占比 ≤ 1/3 → 可按 gain-share 结论决定 AVAILABLE 去留"

    lines: list[str] = [
        f"# Mid-Weekly A/B Report — {baseline_id} vs {candidate_id}",
        "",
        f"- Baseline run: `results/runs/{baseline_id}`（`mid_weekly_feature_count = 0`）",
        f"- Candidate run: `results/runs/{candidate_id}`（`mid_weekly_feature_count > 0`）",
        f"- 比较口径：各品种 test 集 net 指标（`backtest_summary.json → test_backtest`）",
        "",
        "## 摘要",
        "",
        *verdict_bits,
        f"- 比较品种数 = {len(merged)}",
        f"- 结论：{verdict}",
        "",
    ]
    lines += _section("全面提升的品种（ΔSharpe ≥ 0.30）", improved, _fmt_row)
    lines += _section("持平品种（|ΔSharpe| < 0.30）", flat, _fmt_row)
    lines += _section("退步品种（ΔSharpe ≤ -0.30）", regressed, _fmt_row)
    # Structural change
    lines.append(f"### 结构变化（|ΔTradeCount / baseline TradeCount| ≥ 25%）（{len(struct)}）")
    lines.append("")
    if struct.empty:
        lines.append("_无_")
        lines.append("")
    else:
        lines.append("| Product | baseline trades | candidate trades | % change | ΔSharpe |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in struct.sort_values("trade_count_pct_change", key=lambda s: s.abs(), ascending=False).iterrows():
            lines.append(
                f"| {r['product_id']} | {int(r['b_trade_count'])} | {int(r['c_trade_count'])} | "
                f"{r['trade_count_pct_change']:+.1%} | {r['delta_sharpe']:+.3f} |"
            )
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline run_id (under results/runs/)")
    parser.add_argument("--candidate", required=True, help="Candidate run_id")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output", default="results/comparison/midweekly_vs_baseline", help="Stem for output files (.csv/.md/.png appended)")
    args = parser.parse_args(argv)

    baseline_dir = Path(args.runs_root) / args.baseline
    candidate_dir = Path(args.runs_root) / args.candidate
    if not baseline_dir.exists():
        print(f"ERROR: missing {baseline_dir}", file=sys.stderr)
        return 2
    if not candidate_dir.exists():
        print(f"ERROR: missing {candidate_dir}", file=sys.stderr)
        return 2

    base_success = _load_summary(baseline_dir)
    cand_success = _load_summary(candidate_dir)
    intersect_pids = sorted(set(base_success["product_id"]) & set(cand_success["product_id"]))
    if not intersect_pids:
        print("ERROR: no intersecting success products", file=sys.stderr)
        return 2

    print(f"Baseline success: {len(base_success)}; Candidate success: {len(cand_success)}; Intersect: {len(intersect_pids)}")

    baseline_metrics = _gather(baseline_dir, intersect_pids, "baseline")
    candidate_metrics = _gather(candidate_dir, intersect_pids, "candidate")
    merged = _merge(baseline_metrics, candidate_metrics)

    out_stem = Path(args.output)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_cols = [
        "product_id",
        "b_sharpe", "c_sharpe", "delta_sharpe",
        "b_annual_return", "c_annual_return", "delta_annual_return",
        "b_max_drawdown", "c_max_drawdown", "delta_max_drawdown",
        "b_trade_count", "c_trade_count", "delta_trade_count", "trade_count_pct_change",
        "b_trade_win_rate", "c_trade_win_rate", "delta_trade_win_rate",
        "b_daily_win_rate_net", "c_daily_win_rate_net", "delta_daily_win_rate_net",
    ]
    csv_path = out_stem.with_suffix(".csv")
    merged[csv_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Wrote {csv_path}")

    # Markdown
    md_path = out_stem.with_suffix(".md")
    _render_md(merged, args.baseline, args.candidate, md_path)
    print(f"Wrote {md_path}")

    # Plot
    png_path = out_stem.with_suffix(".png")
    _render_png(merged, baseline_dir, candidate_dir, png_path)
    print(f"Wrote {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
