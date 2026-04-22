"""Apply user decisions on soft-duplicate columns to data/mid_weekly/_cleaned/*.xlsx.

The `audit_mid_weekly.py` step only removes HARD duplicates. Soft duplicates
(|corr|>=0.99 AND nan-overlap>=0.9) are listed in docs/mid_weekly_audit.md and
docs/mid_weekly_integration_plan.md § 9 Q1 for the user to resolve.

This script is idempotent: it matches columns by (indicator_id, indicator_name)
rather than by positional index, so re-running it after audit re-run still
drops the correct columns.

Decisions recorded below MUST stay in sync with § 9 Q1 of the plan.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

HEADER_ROWS = 4
META_NAME = 1
META_ID = 3


@dataclass(frozen=True)
class Decision:
    product_id: str
    drop_indicator_id: str
    drop_indicator_name: str
    keep_indicator_id: str
    keep_indicator_name: str
    rationale: str


# Decisions made 2026-04-22 (see plan § 9 Q1).
DECISIONS: list[Decision] = [
    Decision(
        product_id="AU",
        drop_indicator_id="S000025737",
        drop_indicator_name="库存:黄金",
        keep_indicator_id="S000025741",
        keep_indicator_name="期货库存:黄金",
        rationale="期货库存 maps directly to SHFE delivery inventory",
    ),
    Decision(
        product_id="AU",
        drop_indicator_id="S008523183",
        drop_indicator_name="COMEX:黄金:期货(新版):总持仓:持仓数量",
        keep_indicator_id="S002825231",
        keep_indicator_name="COMEX:黄金:总持仓",
        rationale="Keep original-naming COMEX total OI series for historical consistency",
    ),
    Decision(
        product_id="JM",
        drop_indicator_id="M003575153",
        drop_indicator_name="生产资料价格:焦煤(1/3焦煤)",
        keep_indicator_id="S005948590",
        keep_indicator_name="市场价:焦煤(主焦煤):当旬值",
        rationale="主焦煤 corresponds to the deliverable grade",
    ),
    Decision(
        product_id="M",
        drop_indicator_id="S004242412",
        drop_indicator_name="现货价:豆粕",
        keep_indicator_id="S002856699",
        keep_indicator_name="现货价:豆粕:地区均价",
        rationale="地区均价 is more robust than single-location quote",
    ),
    Decision(
        product_id="RB",
        drop_indicator_id="S005476450",
        drop_indicator_name="期货库存:螺纹钢:小计",
        keep_indicator_id="S002853768",
        keep_indicator_name="注册仓单:螺纹钢",
        rationale="注册仓单 maps to immediately deliverable inventory",
    ),
    Decision(
        product_id="RU",
        drop_indicator_id="S002842061",
        drop_indicator_name="仓单数量:天然橡胶",
        keep_indicator_id="S004410360",
        keep_indicator_name="仓单数量:天然橡胶:总计",
        rationale="Explicit 总计 suffix is more precise",
    ),
    Decision(
        product_id="SN",
        drop_indicator_id="S005580375",
        drop_indicator_name="库存:锡:总计",
        keep_indicator_id="S005580468",
        keep_indicator_name="期货库存:锡:总计",
        rationale="Parallel to AU #1 — keep 期货库存 for delivery focus",
    ),
]


def _find_column_to_drop(df: pd.DataFrame, decision: Decision) -> int | None:
    """Return the 1-based column index (pandas) matching the drop target."""
    for col in range(1, df.shape[1]):
        name = str(df.iat[META_NAME, col]).strip()
        ind_id = str(df.iat[META_ID, col]).strip()
        if ind_id == decision.drop_indicator_id and name == decision.drop_indicator_name:
            return col
    # also tolerate match by id alone in case of name-whitespace drift
    for col in range(1, df.shape[1]):
        if str(df.iat[META_ID, col]).strip() == decision.drop_indicator_id:
            return col
    return None


def _keep_column_exists(df: pd.DataFrame, decision: Decision) -> bool:
    for col in range(1, df.shape[1]):
        if str(df.iat[META_ID, col]).strip() == decision.keep_indicator_id:
            return True
    return False


def apply(cleaned_dir: Path) -> int:
    actions = 0
    skipped_already_applied = 0
    errors = 0
    by_pid: dict[str, list[Decision]] = {}
    for d in DECISIONS:
        by_pid.setdefault(d.product_id, []).append(d)

    for pid, decisions in by_pid.items():
        path = cleaned_dir / f"{pid}.xlsx"
        if not path.exists():
            print(f"[ERR] missing file: {path}")
            errors += 1
            continue
        df = pd.read_excel(path, sheet_name=0, header=None, nrows=HEADER_ROWS)
        # plan drops in one pass so we don't have to reload after each deletion;
        # compute the openpyxl column numbers (=pandas col + 1) in descending order.
        drop_openpyxl_cols: list[tuple[int, Decision]] = []
        for dec in decisions:
            if not _keep_column_exists(df, dec):
                print(f"[ERR] {pid}: keep target {dec.keep_indicator_id} missing — abort this product")
                errors += 1
                drop_openpyxl_cols.clear()
                break
            col = _find_column_to_drop(df, dec)
            if col is None:
                print(f"[skip] {pid}: drop target {dec.drop_indicator_id} already absent")
                skipped_already_applied += 1
                continue
            drop_openpyxl_cols.append((col + 1, dec))
        if not drop_openpyxl_cols:
            continue
        drop_openpyxl_cols.sort(key=lambda t: t[0], reverse=True)
        wb = load_workbook(path)
        ws = wb.active
        for col_idx, dec in drop_openpyxl_cols:
            ws.delete_cols(col_idx, 1)
            print(f"[ok] {pid}: dropped col {col_idx - 1} ({dec.drop_indicator_id} {dec.drop_indicator_name})")
            actions += 1
        wb.save(path)
        wb.close()

    print(f"\nSummary: applied={actions}, already-applied={skipped_already_applied}, errors={errors}")
    return 0 if errors == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-dir", default="data/mid_weekly/_cleaned")
    args = parser.parse_args(argv)
    return apply(Path(args.cleaned_dir))


if __name__ == "__main__":
    raise SystemExit(main())
