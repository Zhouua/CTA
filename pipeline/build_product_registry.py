from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CURRENT_DIR_STR = str(CURRENT_DIR)
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if CURRENT_DIR_STR not in sys.path:
    sys.path.insert(0, CURRENT_DIR_STR)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from config_utils import get_section, load_project_config, resolve_optional_paths


MANUAL_FIELDS = ["category", "mid_weekly_files", "enabled"]


def _canonical_priority(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit(".", 1)[-1].upper() if "." in path.stem else ""
    order = {"CZCE": 0, "CZC": 1}
    return order.get(suffix, 2), path.name


def _infer_product_id(filename: str, product_value: str) -> str:
    product = (product_value or "").strip().upper()
    if product:
        return product

    stem = Path(filename).stem
    code_base = stem.split(".", 1)[0].upper()
    if code_base.endswith("ZL"):
        code_base = code_base[:-2]
    return code_base


def _scan_single_csv(path: Path) -> dict[str, Any] | None:
    # 先读 header 一行确定有哪些列，再用 list-based usecols（C-engine 走 zero-copy
    # 列裁剪，比 callable usecols 快 10×）。dtype 用最窄的可读类型，避免 dtype=str
    # 让 C-engine 退化到对象数组。
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols_in_file = [c for c in ("TDATE", "VOLUME", "CODE", "product") if c in head.columns]
    if "TDATE" not in cols_in_file:
        return None

    df = pd.read_csv(path, usecols=cols_in_file, low_memory=False)
    if df.empty:
        return None

    row_count = int(len(df))
    if "VOLUME" in df.columns and row_count > 0:
        vol = pd.to_numeric(df["VOLUME"], errors="coerce").fillna(0.0)
        zero_volume_ratio = float((vol == 0.0).sum() / row_count)
    else:
        zero_volume_ratio = None

    first_product = str(df["product"].iloc[0]) if "product" in df.columns else ""
    last_code = str(df["CODE"].iloc[-1]) if "CODE" in df.columns else ""
    first_code = str(df["CODE"].iloc[0]) if "CODE" in df.columns else ""

    product_id = _infer_product_id(path.name, first_product)
    instrument_code = (last_code or first_code or path.stem).strip().upper()
    exchange = instrument_code.rsplit(".", 1)[-1] if "." in instrument_code else path.stem.rsplit(".", 1)[-1].upper()
    return {
        "product_id": product_id,
        "instrument_code": instrument_code,
        "exchange": exchange,
        "raw_data_file": path.name,
        "data_start": str(df["TDATE"].iloc[0]),
        "data_end": str(df["TDATE"].iloc[-1]),
        "row_count": row_count,
        "zero_volume_ratio": zero_volume_ratio,
        "aliases": [path.name],
    }


def _load_existing_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("products", [])
    else:
        records = payload
    existing: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        product_id = str(item.get("product_id", "")).upper()
        if product_id:
            existing[product_id] = item
    return existing


def build_product_registry(
    product_data_dir: Path,
    output_json: Path,
) -> list[dict[str, Any]]:
    existing = _load_existing_registry(output_json)
    grouped: dict[str, list[dict[str, Any]]] = {}

    skipped_files: list[dict[str, str]] = []
    for csv_path in sorted(product_data_dir.glob("*.csv")):
        try:
            scanned = _scan_single_csv(csv_path)
        except Exception as exc:
            skipped_files.append({"file": csv_path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if scanned is None:
            continue
        grouped.setdefault(scanned["product_id"], []).append(scanned)

    registry: list[dict[str, Any]] = []
    for product_id, candidates in sorted(grouped.items()):
        ordered = sorted(candidates, key=lambda item: _canonical_priority(product_data_dir / item["raw_data_file"]))
        chosen = dict(ordered[0])
        aliases = sorted({alias for item in ordered for alias in item.get("aliases", [])})
        chosen["aliases"] = aliases
        chosen["category"] = "unknown"
        chosen["mid_weekly_files"] = []
        chosen["enabled"] = True

        previous = existing.get(product_id, {})
        for field in MANUAL_FIELDS:
            if field in previous:
                chosen[field] = previous[field]
        chosen["aliases"] = sorted(set(chosen["aliases"]) | set(previous.get("aliases", [])))
        registry.append(chosen)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    if skipped_files:
        print(json.dumps({"skipped_files": skipped_files}, ensure_ascii=False), file=sys.stderr)

    return registry


def build_from_config(config_path: str | None = None, config_override: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config, config_dir = load_project_config(config_path, config_override=config_override)
    paths_cfg = get_section(config, "paths")
    resolved = resolve_optional_paths(
        config_dir,
        paths_cfg,
        ["product_data_dir", "product_registry"],
    )
    if "product_data_dir" not in resolved or "product_registry" not in resolved:
        raise KeyError("config.paths must define product_data_dir and product_registry for registry building.")

    output_json = resolved["product_registry"]
    return build_product_registry(
        product_data_dir=resolved["product_data_dir"],
        output_json=output_json,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or refresh product registry from per-product minute files.")
    parser.add_argument("--config", default=None, help="Path to CTA_vol config.yaml")
    parser.add_argument("--refresh", action="store_true", help="Refresh the registry by scanning product minute files.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if not args.refresh:
        raise SystemExit("Pass --refresh to rebuild product_registry.json from the product data directory.")
    records = build_from_config(config_path=args.config)
    print(json.dumps({"product_count": len(records)}, indent=2, ensure_ascii=False))
