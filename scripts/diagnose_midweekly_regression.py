"""Root-cause diagnosis for the 9 mid_weekly regression products.

Plan § 14. Single script, zero side-artifacts except:
  - docs/midweekly_regression_diagnosis.md  (appended per task)
  - results/comparison/_audit/T14_{n}_check.txt  (by caller)

Usage:
  python scripts/diagnose_midweekly_regression.py --task 1
  python scripts/diagnose_midweekly_regression.py --task 2
  python scripts/diagnose_midweekly_regression.py --task 3
  python scripts/diagnose_midweekly_regression.py --task 4
  python scripts/diagnose_midweekly_regression.py --task 5

First-principles question: why do these 9 specifically regress when
mid_weekly is on? Each --task answers a narrow hypothesis; the report
markdown is the single source of truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BASELINE_RUN = Path("results/runs/20260416_170652")
CANDIDATE_RUN = Path("results/runs/20260422_154414")
AB_CSV = Path("results/comparison/midweekly_vs_baseline.csv")
REPORT = Path("docs/midweekly_regression_diagnosis.md")
REGISTRY = Path("data/product_registry.json")
MID_CLEANED_DIR = Path("data/mid_weekly/_cleaned")
RAW_DATA_DIR = Path("data/分产品1min主连")

REGRESSED: tuple[str, ...] = ("FB", "FU", "Y", "B", "SN", "RU", "BB", "JD", "M")
IMPROVED_DELTA_THRESHOLD = 0.3

MID_PREFIX = "MID_"
MID_AVAILABLE_SUFFIX = "_AVAILABLE"
_DERIVED_TRANSFORMS: tuple[str, ...] = ("RET", "ZSCORE", "PCT_RANK")


# ---------- shared helpers ----------

def _classify(name: str) -> str:
    if not name.startswith(MID_PREFIX):
        return "other"
    if name.endswith(MID_AVAILABLE_SUFFIX):
        return "available"
    tokens = name.split("_")
    if len(tokens) >= 2 and tokens[-1].isdigit():
        tail_two = "_".join(tokens[-3:-1]) if len(tokens) >= 3 else ""
        tail_one = tokens[-2]
        if tail_two == "PCT_RANK" or tail_one in _DERIVED_TRANSFORMS:
            return "derived"
    return "level"


def _load_top_features(run_dir: Path, pid: str, regime: str) -> list[dict[str, Any]]:
    path = run_dir / pid / "training_summary.json"
    if not path.exists():
        return []
    ts = json.loads(path.read_text(encoding="utf-8"))
    return (
        ts.get("regimes", {}).get(regime, {}).get("metrics", {}).get("top_features") or []
    )


def _load_regime_metrics(run_dir: Path, pid: str, regime: str) -> dict[str, Any]:
    path = run_dir / pid / "training_summary.json"
    if not path.exists():
        return {}
    ts = json.loads(path.read_text(encoding="utf-8"))
    return ts.get("regimes", {}).get(regime, {}).get("metrics", {}) or {}


def _append_section(section_md: str) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        header = (
            "# Mid-Weekly Regression Root-Cause Diagnosis\n\n"
            "生成者：`scripts/diagnose_midweekly_regression.py`。每个 `--task` "
            "追加一节；最终 § 14.5 为人手写结论。\n\n"
            f"- baseline run: `{BASELINE_RUN}`\n"
            f"- candidate run: `{CANDIDATE_RUN}`\n"
            f"- 9 回归品种: `{', '.join(REGRESSED)}`\n\n"
        )
        REPORT.write_text(header, encoding="utf-8")
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(section_md)
        if not section_md.endswith("\n"):
            f.write("\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- T14.1: top-20 composition delta ----------

def task1() -> int:
    md = [
        f"\n## 14.1 top-20 组成对比（baseline vs candidate）",
        "",
        f"_generated at {_now_utc()}_",
        "",
        "- baseline = 无 mid_weekly；candidate = mid_weekly 全开。",
        "- 对每个 (pid, regime) 统计候选 top-20 里 MID_* 的桶占比，"
        "以及 baseline top-20 里哪些列在候选中落榜（按 baseline gain 降序取前 5）。",
        "",
        "| PID | Regime | ΔSharpe | MID level | MID derived | AVAILABLE | other | 落榜 baseline top-5 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    # precompute ΔSharpe map
    dsharpe: dict[str, float] = {}
    if AB_CSV.exists():
        ab = pd.read_csv(AB_CSV)
        dsharpe = dict(zip(ab["product_id"], ab["delta_sharpe"]))

    missing: list[str] = []
    for pid in REGRESSED:
        for regime in ("low_vol", "high_vol"):
            base = _load_top_features(BASELINE_RUN, pid, regime)
            cand = _load_top_features(CANDIDATE_RUN, pid, regime)
            if not base:
                missing.append(f"{pid}/{regime} (baseline)")
            if not cand:
                missing.append(f"{pid}/{regime} (candidate)")
            if not base or not cand:
                md.append(f"| {pid} | {regime} | {dsharpe.get(pid, float('nan')):.2f} | ? | ? | ? | ? | DATA MISSING |")
                continue
            cand_feats = [r["feature"] for r in cand]
            buckets = {"level": 0, "derived": 0, "available": 0, "other": 0}
            for f in cand_feats:
                buckets[_classify(f)] += 1
            base_by_gain = sorted(base, key=lambda r: r["importance_gain"], reverse=True)
            dropped = [r["feature"] for r in base_by_gain if r["feature"] not in cand_feats][:5]
            md.append(
                f"| {pid} | {regime} | {dsharpe.get(pid, float('nan')):+.2f} "
                f"| {buckets['level']} | {buckets['derived']} | {buckets['available']} | {buckets['other']} "
                f"| {', '.join(f'`{f}`' for f in dropped)} |"
            )
    md.append("")
    # aggregate interpretation
    md.append("**观察**：")
    md.append("- `MID derived` 占据多少席 → 派生列是否主导；")
    md.append("- `AVAILABLE` 栏如果全 0（top-20 口径下），说明哑变量不是 top 特征；")
    md.append("- 落榜列里常见 `ENG_TOD_*` / `MAX*` / `MA*` 则意味着时间/趋势骨干被挤掉——这是直接的信号劣化因果链。")
    md.append("")
    if missing:
        md.append(f"> 数据缺失：{len(missing)} 条。样本：`{', '.join(missing[:5])}`")
        md.append("")

    _append_section("\n".join(md))
    print(f"T14.1 written to {REPORT}")
    if missing:
        print(f"WARN: {len(missing)} missing entries", file=sys.stderr)
        return 3
    return 0


# ---------- T14.2: val vs test divergence ----------

def task2() -> int:
    md = [
        f"\n## 14.2 val vs test 指标分化",
        "",
        f"_generated at {_now_utc()}_",
        "",
        "- 对 9 回归品种 × 2 regime，比较 baseline 与 candidate 在 val / test 上的",
        "  `pearson_ic` / `spearman_ic` / `directional_accuracy` / `rmse` 差值。",
        "- tag 规则：",
        "  - `OVERFIT`：Δval_ic ≥ -0.01 而 Δtest_ic ≤ -0.01 **且** |Δtest_ic| ≥ 2·|Δval_ic|。",
        "  - `SIGNAL_NOISE`：Δval_ic ≤ -0.005 **且** Δtest_ic ≤ -0.005（两头同降）。",
        "  - `UNCHANGED`：否则。",
        "",
        "| PID | Regime | Δval_pic | Δtest_pic | Δval_da | Δtest_da | Δval_rmse | Δtest_rmse | tag |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    tag_counts = {"OVERFIT": 0, "SIGNAL_NOISE": 0, "UNCHANGED": 0}
    missing_rate_num = 0
    total = 0
    for pid in REGRESSED:
        for regime in ("low_vol", "high_vol"):
            total += 1
            bm = _load_regime_metrics(BASELINE_RUN, pid, regime)
            cm = _load_regime_metrics(CANDIDATE_RUN, pid, regime)
            bv = (bm.get("val_metrics") or {})
            bt = (bm.get("test_metrics") or {})
            cv = (cm.get("val_metrics") or {})
            ct = (cm.get("test_metrics") or {})

            def d(a: dict, b: dict, k: str) -> float:
                if k not in a or k not in b:
                    return float("nan")
                return float(b[k]) - float(a[k])

            dv_pic = d(bv, cv, "pearson_ic")
            dt_pic = d(bt, ct, "pearson_ic")
            dv_da = d(bv, cv, "directional_accuracy")
            dt_da = d(bt, ct, "directional_accuracy")
            dv_rmse = d(bv, cv, "rmse")
            dt_rmse = d(bt, ct, "rmse")

            import math
            if any(math.isnan(x) for x in (dv_pic, dt_pic)):
                tag = "N/A"
                missing_rate_num += 1
            elif dv_pic >= -0.01 and dt_pic <= -0.01 and abs(dt_pic) >= 2 * max(abs(dv_pic), 1e-9):
                tag = "OVERFIT"
            elif dv_pic <= -0.005 and dt_pic <= -0.005:
                tag = "SIGNAL_NOISE"
            else:
                tag = "UNCHANGED"
            if tag in tag_counts:
                tag_counts[tag] += 1
            md.append(
                f"| {pid} | {regime} | {dv_pic:+.4f} | {dt_pic:+.4f} | {dv_da:+.4f} | {dt_da:+.4f} "
                f"| {dv_rmse:+.5f} | {dt_rmse:+.5f} | **{tag}** |"
            )
    md.append("")
    md.append(f"**计数**：OVERFIT={tag_counts['OVERFIT']} · SIGNAL_NOISE={tag_counts['SIGNAL_NOISE']} · UNCHANGED={tag_counts['UNCHANGED']} · total={total}")
    md.append("")
    _append_section("\n".join(md))
    print(f"T14.2 written to {REPORT}")
    if missing_rate_num > total * 0.5:
        print(f"WARN: >50% entries missing val/test metrics", file=sys.stderr)
        return 3
    return 0


# ---------- T14.3: MID input coverage audit ----------

@dataclass
class MidColAudit:
    column: str
    indicator_name: str
    frequency: str
    first_value_date: str | None
    last_value_date: str | None
    non_null_ratio: float
    step_dummy_suspect: bool


def _parse_mid_xlsx(path: Path) -> tuple[list[MidColAudit], str | None]:
    """Return (per-column audit rows, err). xlsx rows are stored reverse-chronological
    (row 0 = newest). We sort ascending before analysis so 'first_value_date' means
    the chronologically earliest non-null timestamp."""
    try:
        header = pd.read_excel(path, header=None, nrows=4)
    except Exception as exc:
        return [], f"open failed: {exc}"
    raw = pd.read_excel(path, header=None, skiprows=4)
    if raw.empty:
        return [], "empty data"
    ts = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    mask = ts.notna()
    raw = raw[mask].reset_index(drop=True)
    ts = ts[mask].reset_index(drop=True)
    # Sort ascending by timestamp (xlsx is reverse-chronological)
    order = ts.sort_values(kind="stable").index
    raw = raw.loc[order].reset_index(drop=True)
    ts = ts.loc[order].reset_index(drop=True)
    out: list[MidColAudit] = []
    for col_idx in range(1, raw.shape[1]):
        series = pd.to_numeric(raw.iloc[:, col_idx], errors="coerce")
        name = str(header.iloc[1, col_idx])
        freq = str(header.iloc[2, col_idx])
        nonnull = series.notna()
        if not nonnull.any():
            out.append(MidColAudit(
                column=str(header.iloc[3, col_idx]),
                indicator_name=name, frequency=freq,
                first_value_date=None, last_value_date=None,
                non_null_ratio=0.0, step_dummy_suspect=False,
            ))
            continue
        first_pos = int(nonnull.idxmax())
        last_pos = int(len(nonnull) - 1 - nonnull.iloc[::-1].reset_index(drop=True).idxmax())
        first_dt = ts.iloc[first_pos].strftime("%Y-%m-%d")
        last_dt = ts.iloc[last_pos].strftime("%Y-%m-%d")
        nn_ratio = float(nonnull.sum() / len(series))
        # Step-dummy = before first valid date all null AND after first valid date
        # non-null rate >= 0.9 (i.e., the indicator genuinely "starts" mid-timeline).
        # first_pos > 0 guarantees "before" is non-empty and all-null by construction.
        post_nn = float(nonnull.iloc[first_pos:].mean())
        step_suspect = (first_pos > 0) and (post_nn >= 0.9)
        out.append(MidColAudit(
            column=str(header.iloc[3, col_idx]),
            indicator_name=name, frequency=freq,
            first_value_date=first_dt, last_value_date=last_dt,
            non_null_ratio=nn_ratio, step_dummy_suspect=step_suspect,
        ))
    return out, None


def task3() -> int:
    md = [
        f"\n## 14.3 MID 输入审计（9 回归品种）",
        "",
        f"_generated at {_now_utc()}_",
        "",
        "- 对每个品种，读 `data/mid_weekly/_cleaned/<PID>.xlsx`，按列统计："
        "起始日期 / 非空比 / 相对期货起点的覆盖延迟 / 阶梯 dummy 嫌疑。",
        "- **阶梯 dummy 嫌疑**：该列在 first_value_date 之后非空比 ≥ 0.9，之前全空——"
        "这样 `MID_*_AVAILABLE` 近似 `I{t ≥ first_value_date}`，会给 LightGBM 提供一个伪'regime 切换'信号。",
        "",
        "| PID | futures_start | 指标数 | 覆盖延迟>1y 数 | 阶梯 dummy 数 | 非空比<0.3 数 | 备注 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg_by_pid = {r["product_id"]: r for r in reg}
    missing_any = False
    detail_sections: list[str] = []

    for pid in REGRESSED:
        entry = reg_by_pid.get(pid, {})
        raw_start_str = entry.get("data_start", "")
        try:
            raw_start = pd.to_datetime(raw_start_str)
        except Exception:
            raw_start = None
        mid_path = MID_CLEANED_DIR / f"{pid}.xlsx"
        if not mid_path.exists():
            md.append(f"| {pid} | {raw_start_str} | - | - | - | - | MID FILE MISSING |")
            missing_any = True
            continue
        rows, err = _parse_mid_xlsx(mid_path)
        if err:
            md.append(f"| {pid} | {raw_start_str} | - | - | - | - | {err} |")
            missing_any = True
            continue
        n = len(rows)
        delayed = 0
        step_cnt = 0
        sparse_cnt = 0
        for r in rows:
            if r.non_null_ratio < 0.3:
                sparse_cnt += 1
            if r.step_dummy_suspect:
                step_cnt += 1
            if raw_start is not None and r.first_value_date:
                fv = pd.to_datetime(r.first_value_date)
                if (fv - raw_start).days > 365:
                    delayed += 1
        md.append(
            f"| {pid} | {raw_start.strftime('%Y-%m-%d') if raw_start is not None else '?'} "
            f"| {n} | {delayed} | {step_cnt} | {sparse_cnt} | |"
        )
        # detail per product
        detail_sections.append(f"\n### {pid} 列级明细\n")
        detail_sections.append(
            "| indicator_id | 指标名称 | 频率 | first | last | 非空比 | 阶梯嫌疑 |\n"
            "|---|---|---|---|---|---:|---|"
        )
        for r in rows:
            name_short = r.indicator_name[:40]
            susp = "⚠ YES" if r.step_dummy_suspect else ""
            detail_sections.append(
                f"| `{r.column}` | {name_short} | {r.frequency} "
                f"| {r.first_value_date or '?'} | {r.last_value_date or '?'} "
                f"| {r.non_null_ratio:.2f} | {susp} |"
            )
    md.append("")
    md.extend(detail_sections)
    md.append("")
    _append_section("\n".join(md))
    print(f"T14.3 written to {REPORT}")
    return 3 if missing_any else 0


# ---------- T14.4: MID class share (regressed vs improved) ----------

def task4() -> int:
    if not AB_CSV.exists():
        print(f"ERROR: {AB_CSV} missing", file=sys.stderr)
        return 3
    ab = pd.read_csv(AB_CSV)
    improved = sorted(ab.query("delta_sharpe > @IMPROVED_DELTA_THRESHOLD")["product_id"].tolist())
    regressed = list(REGRESSED)

    def tally(pids: Iterable[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pid in pids:
            for regime in ("low_vol", "high_vol"):
                feats = _load_top_features(CANDIDATE_RUN, pid, regime)
                if not feats:
                    continue
                buckets = {"level": 0, "derived": 0, "available": 0, "other": 0}
                for r in feats:
                    buckets[_classify(r["feature"])] += 1
                rows.append({"pid": pid, "regime": regime, **buckets})
        return rows

    reg_rows = tally(regressed)
    imp_rows = tally(improved)
    reg_df = pd.DataFrame(reg_rows)
    imp_df = pd.DataFrame(imp_rows)

    def stats(df: pd.DataFrame, col: str) -> str:
        if df.empty:
            return "-"
        return f"{df[col].median():.1f} (p25={df[col].quantile(0.25):.1f}, p75={df[col].quantile(0.75):.1f})"

    md = [
        f"\n## 14.4 MID 类占比对比（回归组 vs 改善组）",
        "",
        f"_generated at {_now_utc()}_",
        "",
        f"- 回归组 (N={len(regressed)}) = `{', '.join(regressed)}`",
        f"- 改善组 (N={len(improved)}, ΔSharpe>{IMPROVED_DELTA_THRESHOLD}) = `{', '.join(improved)}`",
        f"- 样本数 (pid, regime)：回归组 {len(reg_df)}，改善组 {len(imp_df)}",
        "",
        "| Group | level 中位数 | derived 中位数 | AVAILABLE 中位数 | other 中位数 |",
        "|---|---|---|---|---|",
        f"| 回归组 | {stats(reg_df, 'level')} | {stats(reg_df, 'derived')} | {stats(reg_df, 'available')} | {stats(reg_df, 'other')} |",
        f"| 改善组 | {stats(imp_df, 'level')} | {stats(imp_df, 'derived')} | {stats(imp_df, 'available')} | {stats(imp_df, 'other')} |",
        "",
    ]

    # simple judgment
    if not reg_df.empty and not imp_df.empty:
        gap_derived = reg_df["derived"].median() - imp_df["derived"].median()
        gap_other = imp_df["other"].median() - reg_df["other"].median()
        md.append("**解读**：")
        if gap_derived >= 2:
            md.append(f"- 回归组 top-20 中 `MID derived` 中位数比改善组多 {gap_derived:.1f} 列 → 派生列挤占率显著更高，指向 **R-D（derived 错配）**。")
        elif gap_derived <= -2:
            md.append(f"- 改善组反而派生列更多 {-gap_derived:.1f} → 派生列并非问题所在。")
        else:
            md.append(f"- 两组 `MID derived` 中位数差 {gap_derived:+.1f}，不显著。")
        if gap_other >= 2:
            md.append(f"- 改善组 `other`（非 MID 骨干列）比回归组多 {gap_other:.1f} → 时间/趋势骨干在回归组被挤出，指向信号劣化。")
        md.append("")

    _append_section("\n".join(md))
    print(f"T14.4 written to {REPORT}")
    return 0


# ---------- T14.5: synthesis stub ----------

def task5() -> int:
    md = [
        f"\n## 14.5 根因结论 + 路径决策（stub）",
        "",
        f"_generated at {_now_utc()}_",
        "",
        "> 本节由会话手写补齐。本 stub 仅提醒字段完整性。",
        "",
        "- **ROUTE=**（R-A / R-D / R-P 三选一）",
        "- **根因一句话**：",
        "- **证据链**：",
        "  - 来自 § 14.1 的支持点：",
        "  - 来自 § 14.2 的支持点：",
        "  - 来自 § 14.3 的支持点：",
        "  - 来自 § 14.4 的支持点：",
        "- **下一步动作**：（T3.3\\*-b / T3.3\\*-e / § 9 Q5）",
        "",
    ]
    _append_section("\n".join(md))
    print(f"T14.5 stub written to {REPORT}")
    return 0


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["1", "2", "3", "4", "5"])
    args = parser.parse_args(argv)
    return {"1": task1, "2": task2, "3": task3, "4": task4, "5": task5}[args.task]()


if __name__ == "__main__":
    raise SystemExit(main())
