"""Audit mid_weekly xlsx files — produce docs/mid_weekly_audit.md + data/mid_weekly/_cleaned/*.xlsx.

Scope: T2 of docs/mid_weekly_integration_plan.md.

Header convention (strict, verified by scanning all 25 files):
    row 0  -> 单位 (unit)            (col 0 value is 单位 or similar label)
    row 1  -> 指标名称 (indicator)
    row 2  -> 频率 (frequency)
    row 3  -> 指标ID  (indicator id)
    row 4+ -> data rows, col 0 = date, col 1..N = numeric values

Definitions:
    hard duplicate: two data columns share identical (指标名称, 频率, 指标ID).
        -> dropped in the cleaned xlsx (first occurrence kept).
    soft duplicate: two *different* data columns have
        corr(overlap) > 0.99 AND nan-mask overlap ratio > 0.9.
        -> listed in the audit for user decision; NOT auto-dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

HEADER_ROWS = 4  # 单位 / 指标名称 / 频率 / 指标ID
META_UNIT = 0
META_NAME = 1
META_FREQ = 2
META_ID = 3


@dataclass
class ColumnMeta:
    col_index: int  # 1-based data column (col 0 is date)
    unit: str
    name: str
    freq: str
    ind_id: str
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    non_null_ratio: float
    non_null_count: int
    total_count: int


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _read_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None)
    if df.shape[0] < HEADER_ROWS + 1:
        raise ValueError(f"{path}: too few rows ({df.shape[0]}) for header convention")
    return df


def _extract_metas(df: pd.DataFrame) -> list[ColumnMeta]:
    data = df.iloc[HEADER_ROWS:, :].copy()
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    data = data.dropna(subset=[data.columns[0]])
    total = len(data)
    metas: list[ColumnMeta] = []
    for col in range(1, df.shape[1]):
        unit = _s(df.iat[META_UNIT, col])
        name = _s(df.iat[META_NAME, col])
        freq = _s(df.iat[META_FREQ, col])
        ind_id = _s(df.iat[META_ID, col])
        values = pd.to_numeric(data.iloc[:, col], errors="coerce")
        non_null = values.notna()
        non_null_count = int(non_null.sum())
        if non_null_count:
            dates = data.iloc[:, 0][non_null]
            start_date = pd.Timestamp(dates.min())
            end_date = pd.Timestamp(dates.max())
        else:
            start_date = end_date = None
        metas.append(
            ColumnMeta(
                col_index=col,
                unit=unit,
                name=name,
                freq=freq,
                ind_id=ind_id,
                start_date=start_date,
                end_date=end_date,
                non_null_ratio=(non_null_count / total) if total else 0.0,
                non_null_count=non_null_count,
                total_count=total,
            )
        )
    return metas


def _hard_duplicates(metas: list[ColumnMeta]) -> tuple[set[int], list[list[ColumnMeta]]]:
    """Return set of col_indices to drop + the list of duplicate groups."""
    groups: dict[tuple[str, str, str], list[ColumnMeta]] = defaultdict(list)
    for m in metas:
        key = (m.name, m.freq, m.ind_id)
        groups[key].append(m)
    drop: set[int] = set()
    dup_groups: list[list[ColumnMeta]] = []
    for key, members in groups.items():
        if len(members) <= 1:
            continue
        dup_groups.append(members)
        # keep the first — drop the rest
        for m in members[1:]:
            drop.add(m.col_index)
    return drop, dup_groups


def _soft_duplicates(
    df: pd.DataFrame,
    metas: list[ColumnMeta],
    kept_cols: list[int],
    corr_threshold: float = 0.99,
    nan_overlap_threshold: float = 0.9,
) -> list[tuple[ColumnMeta, ColumnMeta, float, float]]:
    """Pairwise test on columns that survive hard-dup removal."""
    data = df.iloc[HEADER_ROWS:, :].copy()
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    data = data.dropna(subset=[data.columns[0]])
    num = pd.DataFrame({
        c: pd.to_numeric(data.iloc[:, c], errors="coerce").values for c in kept_cols
    })
    meta_by_col = {m.col_index: m for m in metas}
    results: list[tuple[ColumnMeta, ColumnMeta, float, float]] = []
    cols = list(num.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = num[cols[i]], num[cols[j]]
            mask_a, mask_b = a.notna(), b.notna()
            union = mask_a | mask_b
            inter = mask_a & mask_b
            if union.sum() == 0 or inter.sum() < 20:
                continue
            nan_overlap = inter.sum() / union.sum()
            if nan_overlap < nan_overlap_threshold:
                continue
            corr = a[inter].corr(b[inter])
            if pd.isna(corr):
                continue
            if abs(corr) >= corr_threshold:
                results.append((meta_by_col[cols[i]], meta_by_col[cols[j]], float(corr), float(nan_overlap)))
    return results


def _write_cleaned_xlsx(src: Path, dst: Path, drop_cols: set[int]) -> int:
    """Copy src xlsx to dst, removing the specified 1-based data columns.

    Uses openpyxl on a filesystem copy so formatting/metadata is preserved.
    Returns number of columns kept (data cols only).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if not drop_cols:
        # count data columns via df shape
        df = pd.read_excel(src, sheet_name=0, header=None, nrows=1)
        return df.shape[1] - 1
    wb = load_workbook(dst)
    ws = wb.active
    # openpyxl columns are 1-based; our col_index is also 1-based w.r.t. pandas
    # but pandas col 0 == openpyxl col 1, so our data col i -> openpyxl col i+1.
    drop_openpyxl = sorted({c + 1 for c in drop_cols}, reverse=True)
    for col in drop_openpyxl:
        ws.delete_cols(col, 1)
    wb.save(dst)
    wb.close()
    df_after = pd.read_excel(dst, sheet_name=0, header=None, nrows=1)
    return df_after.shape[1] - 1


def _short_hash(s: str, n: int = 6) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def _render_markdown(
    product_id: str,
    src_path: Path,
    metas: list[ColumnMeta],
    drop_cols: set[int],
    hard_groups: list[list[ColumnMeta]],
    soft_pairs: list[tuple[ColumnMeta, ColumnMeta, float, float]],
) -> str:
    lines: list[str] = []
    lines.append(f"### {product_id}")
    lines.append("")
    lines.append(f"- 源文件：`{src_path.as_posix()}`")
    lines.append(f"- 原始数据列数：{len(metas)}")
    lines.append(f"- 硬重复被移除的列数：{len(drop_cols)}")
    lines.append(f"- 清洗后保留列数：{len(metas) - len(drop_cols)}")
    lines.append(f"- 软重复对数（待裁决）：{len(soft_pairs)}")
    lines.append("")
    lines.append("#### 指标一览")
    lines.append("")
    lines.append("| Col | 指标名称 | 频率 | 指标ID | 单位 | 起止 | 非空率 | 硬重复处置 |")
    lines.append("|---:|---|---|---|---|---|---:|---|")
    for m in metas:
        start = m.start_date.strftime("%Y-%m-%d") if m.start_date is not None else "-"
        end = m.end_date.strftime("%Y-%m-%d") if m.end_date is not None else "-"
        rng = f"{start} → {end}"
        action = "drop" if m.col_index in drop_cols else "keep"
        lines.append(
            f"| {m.col_index} | {m.name} | {m.freq} | {m.ind_id} | {m.unit} | {rng} | {m.non_null_ratio:.3f} | {action} |"
        )
    lines.append("")
    if hard_groups:
        lines.append("#### 硬重复（脚本已移除，仅保留每组首列）")
        lines.append("")
        lines.append("| 指标名称 | 频率 | 指标ID | 重复列 |")
        lines.append("|---|---|---|---|")
        for grp in hard_groups:
            head = grp[0]
            cols_str = ", ".join(str(m.col_index) for m in grp)
            lines.append(f"| {head.name} | {head.freq} | {head.ind_id} | {cols_str} |")
        lines.append("")
    if soft_pairs:
        lines.append("#### 软重复（需要用户裁决，保留还是删除）")
        lines.append("")
        lines.append("| Col A | 指标 A | Col B | 指标 B | |corr| | NaN 重叠率 |")
        lines.append("|---:|---|---:|---|---:|---:|")
        for a, b, corr, overlap in soft_pairs:
            lines.append(
                f"| {a.col_index} | {a.name} ({a.ind_id}) | {b.col_index} | {b.name} ({b.ind_id}) | {abs(corr):.3f} | {overlap:.3f} |"
            )
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def _render_header() -> str:
    return """# Mid-Weekly Audit

> T2 产出。脚本：`scripts/audit_mid_weekly_inputs.py`。输入：`data/mid_weekly/*.xlsx`。
> 清洗结果（仅移除硬重复）：`data/mid_weekly/_cleaned/*.xlsx`。原文件保留不动。

## 原则

- 周频指标在分钟数据上必然有大段空，训练管道用 `merge_asof(backward) + ffill` 处理（见 `pipeline/dataset.py::_merge_mid_weekly_features`）。
- 指标自身周内断更：先 `ffill`，**禁止 `bfill`**（会未来泄漏）。
- 起始时间晚于行情：交给 § 6 策略 B（哑变量 + ffill 上限）处理，audit 阶段不做干预。

## 重复定义

- **硬重复**：(指标名称, 频率, 指标ID) 三元组完全相同——脚本直接删除多余列，仅保留首列。
- **软重复**：两列数值相关系数 |corr| ≥ 0.99 **且** NaN 掩码重叠率 ≥ 0.9——列入文档，**等用户裁决**。

## 待用户裁决汇总

所有软重复见各品种 "软重复" 表。如果判定为"选保留其一"，请在 `docs/mid_weekly_integration_plan.md § 9` 新增 Q1 子项记录选择。

"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mid-weekly-dir", default="data/mid_weekly")
    parser.add_argument("--output", default="docs/mid_weekly_audit.md")
    parser.add_argument("--cleaned-subdir", default="_cleaned")
    args = parser.parse_args(argv)

    src_dir = Path(args.mid_weekly_dir)
    out_md = Path(args.output)
    cleaned_dir = src_dir / args.cleaned_subdir

    xlsx_files = sorted(p for p in src_dir.glob("*.xlsx") if not p.name.startswith("~"))
    if not xlsx_files:
        print(f"No xlsx found under {src_dir}", file=sys.stderr)
        return 1

    doc_parts: list[str] = [_render_header()]
    per_product_summary: list[dict[str, Any]] = []

    for src in xlsx_files:
        product_id = src.stem
        try:
            df = _read_xlsx(src)
        except Exception as e:  # noqa: BLE001
            doc_parts.append(f"### {product_id}\n\n- **ERROR** reading `{src}`: {e}\n\n")
            per_product_summary.append({"pid": product_id, "error": str(e)})
            continue
        metas = _extract_metas(df)
        drop_cols, hard_groups = _hard_duplicates(metas)
        kept_cols = [m.col_index for m in metas if m.col_index not in drop_cols]
        soft_pairs = _soft_duplicates(df, metas, kept_cols)
        dst = cleaned_dir / src.name
        kept_count = _write_cleaned_xlsx(src, dst, drop_cols)
        doc_parts.append(_render_markdown(product_id, src, metas, drop_cols, hard_groups, soft_pairs))
        per_product_summary.append({
            "pid": product_id,
            "raw_cols": len(metas),
            "kept_cols": kept_count,
            "hard_groups": len(hard_groups),
            "soft_pairs": len(soft_pairs),
        })

    # prepend a summary table
    header_table = [
        "## 每品种汇总",
        "",
        "| Product | 原列数 | 保留列数 | 硬重复组数 | 软重复对数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_product_summary:
        if "error" in row:
            header_table.append(f"| {row['pid']} | ERROR | - | - | - |")
        else:
            header_table.append(
                f"| {row['pid']} | {row['raw_cols']} | {row['kept_cols']} | {row['hard_groups']} | {row['soft_pairs']} |"
            )
    header_table.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(doc_parts[0] + "\n".join(header_table) + "\n\n" + "\n".join(doc_parts[1:]), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Cleaned xlsx under {cleaned_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
