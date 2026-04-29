from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "macro_monthly_features.csv"
DEFAULT_CORE_OUTPUT = BASE_DIR / "macro_monthly_features_core.csv"
DEFAULT_STATE_OUTPUT = BASE_DIR / "macro_state_monthly.csv"
DEFAULT_PBOC_WORKBOOK = BASE_DIR / "央行月度数据.xlsx"

# FACTOR_GROUPS = {
#     # 这里按“全量提取后的原始列”分组，不是去冗余后的 core 版本。
#     "库存周期": [
#         "ths_规模以上工业企业_产成品存货_期末同比",
#         "ths_规模以上工业企业_产成品存货_期末值",
#     ],
#     "地产需求": [
#         "nbs_房地产投资_累计值_亿元",
#         "nbs_房地产投资_累计增长_pct",
#         "nbs_房地产新开工施工面积累计值_万平方米",
#         "nbs_房地产新开工施工面积累计增长_pct",
#         "nbs_新建商品房销售面积累计值_万平方米",
#         "nbs_新建商品房销售面积累计增长_pct",
#     ],
#     "信用": [
#         "afre_社会融资规模存量_stock",
#         "afre_社会融资规模存量_growth_pct",
#         "money_货币和准货币_M2",
#         "ths_金融机构_人民币贷款余额",
#         "ths_金融机构_新增人民币贷款_中长期贷款_当月值",
#     ],
#     "宏观": [
#         "ths_规模以上工业增加值_当月同比",
#         "nbs_固定资产投资额累计增长_pct",
#         "ths_进出口差额_人民币计价_当月值",
#         "ths_CPI_当月同比",
#         "ths_PPI_当月同比",
#     ],
#     "PMI": [
#         "nbs_制造业采购经理指数_pct",
#         "nbs_新订单指数_pct",
#         "nbs_原材料库存指数_pct",
#         "nbs_产成品库存指数_pct",
#         "nbs_主要原材料购进价格指数_pct",
#         "nbs_出厂价格指数_pct",
#     ],
# }

CORE_FEATURE_COLUMNS = [
    "ths_规模以上工业企业_产成品存货_期末同比",
    "nbs_房地产投资_累计增长_pct",
    "nbs_房地产新开工施工面积累计增长_pct",
    "nbs_新建商品房销售面积累计增长_pct",
    "afre_社会融资规模存量_growth_pct",
    "money_货币和准货币_M2",
    "ths_金融机构_人民币贷款余额",
    "ths_金融机构_新增人民币贷款_中长期贷款_当月值",
    "ths_规模以上工业增加值_当月同比",
    "nbs_固定资产投资额累计增长_pct",
    "ths_进出口差额_人民币计价_当月值",
    "ths_CPI_当月同比",
    "ths_PPI_当月同比",
    "nbs_制造业采购经理指数_pct",
    "nbs_新订单指数_pct",
    "nbs_原材料库存指数_pct",
    "nbs_产成品库存指数_pct",
    "nbs_主要原材料购进价格指数_pct",
    "nbs_出厂价格指数_pct",
]


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_feature_name(value: object, prefix: str) -> str:
    text = _normalize_text(value)
    text = text.replace("%", "pct")
    text = text.replace("（", "_").replace("）", "")
    text = text.replace("(", "_").replace(")", "")
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace(":", "_")
    text = text.replace("：", "_")
    text = text.replace("、", "_")
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return f"{prefix}_{text}"


def _parse_month_token(value: object) -> pd.Period:
    text = _normalize_text(value)
    if not text:
        raise ValueError("Empty month token.")

    if match := re.search(r"(\d{4})年(\d{1,2})月", text):
        year, month = int(match.group(1)), int(match.group(2))
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    if match := re.search(r"^(\d{4})\.(\d{1,2})$", text):
        year, month = int(match.group(1)), int(match.group(2))
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    if match := re.search(r"^(\d{4})-(\d{1,2})$", text):
        year, month = int(match.group(1)), int(match.group(2))
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    raise ValueError(f"Unsupported month token: {value!r}")


def _first_non_empty_label(values: list[object]) -> str:
    for value in values:
        text = _normalize_text(value)
        if text:
            return text
    return ""


def _make_month_start_index(period_index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return period_index.to_timestamp(how="start")


def _coerce_numeric(values: pd.Series | list[object]) -> pd.Series:
    series = pd.Series(values, dtype="object")
    normalized = series.map(_normalize_text).replace({"": pd.NA, "--": pd.NA})
    normalized = normalized.str.replace(",", "", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


def _build_asof_series(series: pd.Series) -> pd.Series:
    # Monthly macro data should only affect trading after publication, so we use lag-1 then forward-fill.
    return series.shift(1).ffill()


def extract_nbs_monthly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    periods = [_parse_month_token(value) for value in raw.iloc[2, 1:]]
    rows: list[pd.Series] = []

    for row_idx in range(3, len(raw)):
        label = _normalize_text(raw.iloc[row_idx, 0])
        if not label:
            continue
        if label.startswith("注：") or label.startswith("数据来源") or re.match(r"^\d+\.", label):
            continue

        values = _coerce_numeric(raw.iloc[row_idx, 1:])
        if int(values.notna().sum()) == 0:
            continue

        series = pd.Series(
            values.to_numpy(dtype="float64"),
            index=pd.PeriodIndex(periods, freq="M"),
            name=_clean_feature_name(label, prefix="nbs"),
        )
        rows.append(series)

    if not rows:
        raise ValueError(f"No usable monthly rows found in {path}.")

    return pd.concat(rows, axis=1).sort_index()


def _extract_afre_monthly_raw(raw: pd.DataFrame) -> pd.DataFrame:
    metric_types = [_normalize_text(value) for value in raw.iloc[5, 1:]]
    first_period = _parse_month_token(raw.iloc[4, 1])
    month_count = len(metric_types) // 2
    periods = pd.period_range(start=first_period, periods=month_count, freq="M")
    expanded_periods = [period for period in periods for _ in range(2)]

    frame = pd.DataFrame(index=periods)
    for row_idx in range(7, len(raw)):
        label = _first_non_empty_label(raw.iloc[row_idx, :3].tolist())
        if not label:
            continue

        values = _coerce_numeric(raw.iloc[row_idx, 1:])
        if int(values.notna().sum()) == 0:
            continue

        base_name = _clean_feature_name(label, prefix="afre")
        for period, metric_type, value in zip(expanded_periods, metric_types, values.tolist(), strict=False):
            if pd.isna(value):
                continue
            if "存量" in metric_type:
                column = f"{base_name}_stock"
            elif "增速" in metric_type:
                column = f"{base_name}_growth_pct"
            else:
                column = f"{base_name}_{_clean_feature_name(metric_type, prefix='metric')}"
            frame.loc[period, column] = float(value)

    return frame.sort_index()


def extract_afre_monthly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    return _extract_afre_monthly_raw(raw)


def _extract_money_supply_monthly_raw(raw: pd.DataFrame) -> pd.DataFrame:
    first_period = _parse_month_token(raw.iloc[5, 3])
    month_count = len(raw.columns[3:])
    periods = pd.period_range(start=first_period, periods=month_count, freq="M")
    rows: list[pd.Series] = []

    for row_idx in range(6, len(raw)):
        label = _first_non_empty_label(raw.iloc[row_idx, :3].tolist())
        if not label:
            continue

        values = _coerce_numeric(raw.iloc[row_idx, 3:])
        if int(values.notna().sum()) == 0:
            continue

        series = pd.Series(
            values.to_numpy(dtype="float64"),
            index=periods,
            name=_clean_feature_name(label, prefix="money"),
        )
        rows.append(series)

    if not rows:
        raise ValueError(f"No usable money-supply rows found in {path}.")

    return pd.concat(rows, axis=1).sort_index()


def extract_money_supply_monthly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    return _extract_money_supply_monthly_raw(raw)


def extract_pboc_monthly(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    social_financing_sheet = next(
        (sheet for sheet in workbook.sheet_names if "社会融资规模" in sheet),
        workbook.sheet_names[0],
    )
    money_supply_sheet = next(
        (sheet for sheet in workbook.sheet_names if "货币供应量" in sheet),
        workbook.sheet_names[-1],
    )

    social_financing_raw = workbook.parse(social_financing_sheet, header=None)
    money_supply_raw = workbook.parse(money_supply_sheet, header=None)
    return pd.concat(
        [
            _extract_afre_monthly_raw(social_financing_raw),
            _extract_money_supply_monthly_raw(money_supply_raw),
        ],
        axis=1,
    ).sort_index()


def extract_ths_supplement_monthly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    if len(raw) < 5:
        raise ValueError(f"Unexpected THS supplement layout in {path}.")

    metric_names = raw.iloc[1].tolist()
    dates = pd.to_datetime(raw.iloc[4:, 0], errors="coerce")
    periods = dates.dt.to_period("M")
    valid_periods = periods.dropna()
    if valid_periods.empty:
        raise ValueError(f"No usable dates found in {path}.")

    frame = pd.DataFrame(index=pd.PeriodIndex(sorted(valid_periods.unique()), freq="M"))
    for col_idx in range(1, raw.shape[1]):
        metric_name = _normalize_text(metric_names[col_idx])
        if not metric_name:
            continue

        values = _coerce_numeric(raw.iloc[4:, col_idx])
        series = pd.DataFrame({"period": periods, "value": values}).dropna()
        if series.empty:
            continue

        monthly_series = series.groupby("period", sort=True)["value"].last()
        monthly_series.name = _clean_feature_name(metric_name, prefix="ths")
        frame = frame.join(monthly_series, how="left")

    return frame.sort_index()


def build_macro_monthly_frame(base_dir: Path | None = None) -> pd.DataFrame:
    base_dir = BASE_DIR if base_dir is None else Path(base_dir)
    pboc_workbook = base_dir / DEFAULT_PBOC_WORKBOOK.name
    if not pboc_workbook.exists():
        raise FileNotFoundError(f"Missing PBOC workbook: {pboc_workbook}")

    frames = [
        extract_nbs_monthly(base_dir / "国家统计局月度数据.xlsx"),
        extract_pboc_monthly(pboc_workbook),
    ]
    ths_path = base_dir / "同花顺补充数据.xlsx"
    if ths_path.exists():
        frames.append(extract_ths_supplement_monthly(ths_path))

    combined = pd.concat(frames, axis=1).sort_index()
    combined.index = _make_month_start_index(combined.index)
    combined.index.name = "tdate"
    combined = combined.reset_index()
    return combined


def build_macro_core_frame(monthly_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in CORE_FEATURE_COLUMNS if col not in monthly_df.columns]
    if missing:
        raise ValueError(f"Missing core factor columns: {missing}")

    core = monthly_df[["tdate", *CORE_FEATURE_COLUMNS]].copy()
    non_date_cols = [col for col in core.columns if col != "tdate"]
    core = core.loc[core[non_date_cols].notna().any(axis=1)].reset_index(drop=True)
    return core


def build_macro_state_frame(core_df: pd.DataFrame) -> pd.DataFrame:
    state = core_df.copy()
    state["tdate"] = pd.to_datetime(state["tdate"])
    state = state.sort_values("tdate").reset_index(drop=True)

    raw = state.set_index("tdate")
    derived = pd.DataFrame(index=raw.index)

    derived["inventory_yoy"] = raw["ths_规模以上工业企业_产成品存货_期末同比"]
    derived["real_estate_investment_growth"] = raw["nbs_房地产投资_累计增长_pct"]
    derived["real_estate_starts_growth"] = raw["nbs_房地产新开工施工面积累计增长_pct"]
    derived["real_estate_sales_growth"] = raw["nbs_新建商品房销售面积累计增长_pct"]
    derived["social_financing_growth"] = raw["afre_社会融资规模存量_growth_pct"]
    derived["m2_level"] = raw["money_货币和准货币_M2"]
    derived["m2_yoy"] = raw["money_货币和准货币_M2"].pct_change(12, fill_method=None) * 100.0
    derived["loan_balance"] = raw["ths_金融机构_人民币贷款余额"]
    derived["loan_balance_yoy"] = raw["ths_金融机构_人民币贷款余额"].pct_change(12, fill_method=None) * 100.0
    derived["long_term_new_loan"] = raw["ths_金融机构_新增人民币贷款_中长期贷款_当月值"]
    derived["industrial_output_yoy"] = raw["ths_规模以上工业增加值_当月同比"]
    derived["fixed_asset_investment_growth"] = raw["nbs_固定资产投资额累计增长_pct"]
    derived["trade_balance"] = raw["ths_进出口差额_人民币计价_当月值"]
    derived["trade_balance_yoy_diff"] = raw["ths_进出口差额_人民币计价_当月值"].diff(12)
    derived["cpi_yoy"] = raw["ths_CPI_当月同比"]
    derived["ppi_yoy"] = raw["ths_PPI_当月同比"]
    derived["pmi_manufacturing"] = raw["nbs_制造业采购经理指数_pct"]
    derived["pmi_new_orders"] = raw["nbs_新订单指数_pct"]
    derived["pmi_raw_material_inventory"] = raw["nbs_原材料库存指数_pct"]
    derived["pmi_finished_goods_inventory"] = raw["nbs_产成品库存指数_pct"]
    derived["pmi_input_price"] = raw["nbs_主要原材料购进价格指数_pct"]
    derived["pmi_output_price"] = raw["nbs_出厂价格指数_pct"]

    asof = derived.apply(_build_asof_series)
    asof = asof.add_prefix("asof_").reset_index()
    factor_cols = [col for col in asof.columns if col != "tdate"]
    asof["available_factor_count"] = asof[factor_cols].notna().sum(axis=1)
    asof["available_factor_ratio"] = asof["available_factor_count"] / len(factor_cols)
    asof["tdate"] = asof["tdate"].dt.strftime("%Y-%m-%d")
    return asof


def export_macro_monthly_csv(output_path: Path | None = None) -> Path:
    output_path = DEFAULT_OUTPUT if output_path is None else Path(output_path)
    frame = build_macro_monthly_frame(BASE_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_macro_core_csv(output_path: Path | None = None) -> Path:
    output_path = DEFAULT_CORE_OUTPUT if output_path is None else Path(output_path)
    monthly_df = build_macro_monthly_frame(BASE_DIR)
    core_df = build_macro_core_frame(monthly_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    core_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_macro_state_csv(output_path: Path | None = None) -> Path:
    output_path = DEFAULT_STATE_OUTPUT if output_path is None else Path(output_path)
    monthly_df = build_macro_monthly_frame(BASE_DIR)
    core_df = build_macro_core_frame(monthly_df)
    state_df = build_macro_state_frame(core_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract monthly macro features from raw Excel files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    output_path = export_macro_monthly_csv(args.output)
    core_output_path = export_macro_core_csv(DEFAULT_CORE_OUTPUT)
    state_output_path = export_macro_state_csv(DEFAULT_STATE_OUTPUT)

    frame = pd.read_csv(output_path)
    core_frame = pd.read_csv(core_output_path)
    state_frame = pd.read_csv(state_output_path)
    print(f"Saved monthly macro features to: {output_path}")
    print(f"Rows: {len(frame)}, Columns: {len(frame.columns)}")
    print(f"Date range: {frame['tdate'].min()} -> {frame['tdate'].max()}")
    print(f"Saved macro core factors to: {core_output_path} ({len(core_frame)} rows, {len(core_frame.columns)} cols)")
    print(f"Saved macro as-of state table to: {state_output_path} ({len(state_frame)} rows, {len(state_frame.columns)} cols)")


if __name__ == "__main__":
    main()
