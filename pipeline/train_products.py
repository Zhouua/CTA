from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
CURRENT_DIR_STR = str(CURRENT_DIR)
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if CURRENT_DIR_STR not in sys.path:
    sys.path.insert(0, CURRENT_DIR_STR)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from config_utils import get_section, load_project_config, resolve_paths


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return value
    return value


def _normalize_product_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _is_success_status(status: Any) -> bool:
    return str(status or "").strip().lower() == "success"


def _is_skipped_status(status: Any) -> bool:
    return str(status or "").strip().lower().startswith("skipped")


def _is_insufficient_coverage_status(status: Any) -> bool:
    return str(status or "").strip().lower() == "skipped_insufficient_coverage"


def _ordered_rows(rows: list[dict[str, Any]], selected_products: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    if not selected_products:
        return rows

    normalized_selection = [_normalize_product_id(item) for item in selected_products]
    row_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        product_id = _normalize_product_id(row.get("product_id"))
        if product_id:
            row_map[product_id] = row

    ordered = [row_map[product_id] for product_id in normalized_selection if product_id in row_map]
    ordered_ids = {_normalize_product_id(row.get("product_id")) for row in ordered}
    ordered.extend(row for row in rows if _normalize_product_id(row.get("product_id")) not in ordered_ids)
    return ordered


def _read_json_payload(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    payload = path.read_text(encoding="utf-8").strip()
    if not payload:
        return default
    return json.loads(payload)


def _metric_or_na(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _default_logger(message: str) -> None:
    print(message, flush=True)


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def load_batch_training_settings(config_path: str | None = None) -> dict[str, Any]:
    config, _ = load_project_config(config_path)
    batch_cfg = get_section(config, "batch_training", {})
    return {
        "enforce_registry_coverage": bool(batch_cfg.get("enforce_registry_coverage", True)),
        "required_data_start": _coerce_timestamp(batch_cfg.get("required_data_start", "2021-01-01")),
        "required_data_end": _coerce_timestamp(batch_cfg.get("required_data_end", "2026-01-01")),
    }


def _format_required_timestamp(value: pd.Timestamp | None) -> str:
    if value is None:
        return "n/a"
    return value.strftime("%Y-%m-%d")


def build_coverage_skip_message(
    product_meta: dict[str, Any],
    *,
    required_data_start: pd.Timestamp | None,
    required_data_end: pd.Timestamp | None,
) -> str:
    required_data_start = _coerce_timestamp(required_data_start)
    required_data_end = _coerce_timestamp(required_data_end)
    data_start = _coerce_timestamp(product_meta.get("data_start"))
    data_end = _coerce_timestamp(product_meta.get("data_end"))
    if data_start is None or data_end is None:
        return "Available data range is unknown because registry data_start/data_end is missing."

    reasons: list[str] = []
    available_range = f"available_range={data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}"
    if required_data_start is not None and data_start > required_data_start:
        reasons.append(f"data_start={data_start.strftime('%Y-%m-%d')} > required_start={required_data_start.strftime('%Y-%m-%d')}")
    if required_data_end is not None and data_end < required_data_end:
        reasons.append(f"data_end={data_end.strftime('%Y-%m-%d')} < required_end={required_data_end.strftime('%Y-%m-%d')}")
    if not reasons:
        return ""
    return "Insufficient registry date coverage: " + available_range + "; " + "; ".join(reasons)


def annotate_products_for_batch_skip(
    product_records: list[dict[str, Any]],
    *,
    enforce_registry_coverage: bool,
    required_data_start: pd.Timestamp | None,
    required_data_end: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    required_data_start = _coerce_timestamp(required_data_start)
    required_data_end = _coerce_timestamp(required_data_end)
    annotated: list[dict[str, Any]] = []
    for product_meta in product_records:
        updated_meta = dict(product_meta)
        if enforce_registry_coverage:
            skip_message = build_coverage_skip_message(
                updated_meta,
                required_data_start=required_data_start,
                required_data_end=required_data_end,
            )
            if skip_message:
                updated_meta["_batch_skip_status"] = "skipped_insufficient_coverage"
                updated_meta["_batch_skip_error"] = skip_message
                updated_meta["_required_data_start"] = _format_required_timestamp(required_data_start)
                updated_meta["_required_data_end"] = _format_required_timestamp(required_data_end)
        annotated.append(updated_meta)
    return annotated


def load_product_registry(config_path: str | None = None) -> list[dict[str, Any]]:
    from build_product_registry import build_from_config

    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(config_dir, get_section(config, "paths"), ["product_registry"])
    registry_path = paths["product_registry"]
    if not registry_path.exists():
        return build_from_config(config_path=config_path)

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("products", []))
    if not isinstance(payload, list):
        raise TypeError(f"Unsupported product registry payload in {registry_path}")
    return payload


def resolve_run_root(config_path: str | None = None) -> Path:
    config, config_dir = load_project_config(config_path)
    paths = resolve_paths(config_dir, get_section(config, "paths"), ["run_root"])
    return paths["run_root"]


def resolve_existing_run_dir(run_ref: str, config_path: str | None = None) -> Path:
    candidate = Path(run_ref).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise NotADirectoryError(f"Resume target is not a directory: {candidate}")
        return candidate.resolve()

    run_dir = resolve_run_root(config_path=config_path) / run_ref
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot find previous run directory: {run_ref}")
    return run_dir.resolve()


def build_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def build_product_config_override(
    product_meta: dict[str, Any],
    product_dir: Path,
    *,
    persist_to_shared_dir: bool = False,
) -> dict[str, Any]:
    """构造 per-product 的 config override。

    Args:
        product_meta: 品种元数据（来自 product_registry.json）
        product_dir: 该品种本次训练的输出目录
        persist_to_shared_dir: 是否同时把模型 / training_summary 写入共享路径
            (results/models/, results/backtest/)。默认 False（适用于批量训练，
            避免 25 个品种互相覆盖）。当只跑单品种时建议设为 True，让后续
            pipeline/backtest.py 仍能从 results/models/ 加载模型重新回测。
    """
    product_id = _normalize_product_id(product_meta["product_id"])
    paths_override = {
        "regime_plot": str(product_dir / "vol_regime_split.png"),
        "training_plot": str(product_dir / "training_diagnostics.png"),
        "training_comparison_plot": str(product_dir / "regime_model_comparison.png"),
        "backtest_plot": str(product_dir / "backtest_curve.png"),
        "prediction_cache": str(product_dir / "test_predictions.parquet"),
    }
    if persist_to_shared_dir:
        # 单品种模式：训练产物同时写入 product_dir（保留批量产物结构）和 results/models/、
        # results/backtest/ 等共享路径（让 pipeline/backtest.py 能直接 reuse）。
        # 注：model_dir / backtest_dir / training_summary 不在 paths_override 里就
        # 走 config.yaml 默认值（即 results/models/、results/backtest/）。
        pass
    else:
        # 批量模式：所有产物隔离到 product_dir，避免 25 个品种互相覆盖。
        paths_override["model_dir"] = str(product_dir / "models")
        paths_override["training_summary"] = str(product_dir / "training_summary.json")
        paths_override["backtest_dir"] = str(product_dir)
    return {
        "product": {
            "product_id": product_id,
            "instrument_code": product_meta.get("instrument_code"),
            "exchange": product_meta.get("exchange"),
            "category": product_meta.get("category", "unknown"),
            "raw_data_file": product_meta.get("raw_data_file"),
            "mid_weekly_files": list(product_meta.get("mid_weekly_files", [])),
            "enabled": bool(product_meta.get("enabled", True)),
            "aliases": list(product_meta.get("aliases", [])),
        },
        "paths": paths_override,
        "factors": {
            "runtime": {
                "enabled": True,
            }
        },
        "model": {
            # 单品种 → True（写入 results/models/，可被 pipeline/backtest.py 复用）
            # 批量    → False（避免 25 品种共用 results/models/ 互相覆盖）
            "persist_models": bool(persist_to_shared_dir),
        },
    }


def summarize_product_run(
    product_meta: dict[str, Any],
    training_summary: dict[str, Any],
    backtest_summary: dict[str, Any],
    product_dir: Path,
    status: str = "success",
    error: str = "",
) -> dict[str, Any]:
    dataset = backtest_summary.get("dataset", {})
    test_metrics = backtest_summary.get("test_backtest", {})
    net_metrics = test_metrics.get("net", {})
    split_rows = dataset.get("split_rows", {})
    return {
        "product_id": product_meta.get("product_id"),
        "instrument_code": product_meta.get("instrument_code"),
        "exchange": product_meta.get("exchange"),
        "category": product_meta.get("category", "unknown"),
        "status": status,
        "error": error,
        "data_start": dataset.get("date_range", {}).get("start"),
        "data_end": dataset.get("date_range", {}).get("end"),
        "feature_count": dataset.get("feature_count"),
        "runtime_factor_feature_count": dataset.get("runtime_factor_feature_count"),
        "mid_weekly_feature_count": dataset.get("mid_weekly_feature_count"),
        "train_rows": split_rows.get("train"),
        "val_rows": split_rows.get("val"),
        "test_rows": split_rows.get("test"),
        "total_return": net_metrics.get("total_return"),
        "annual_return": net_metrics.get("annual_return"),
        "sharpe": net_metrics.get("sharpe"),
        "max_drawdown": net_metrics.get("max_drawdown"),
        "trade_count": test_metrics.get("trade_count"),
        "product_dir": str(product_dir),
        "training_summary_path": str(product_dir / "training_summary.json"),
        "backtest_summary_path": str(product_dir / "backtest_summary.json"),
    }


def _read_full_metrics(product_dir: Path) -> dict[str, Any]:
    """从 product_dir 读完整 backtest_summary.json，返回写 md 需要的所有字段。
    任何字段读不到都退回 None，不抛错。"""
    bs_path = product_dir / "backtest_summary.json"
    if not bs_path.exists():
        # 单品种模式时 backtest_summary 在 results/backtest/ 下，product_dir 没这文件
        bs_path = product_dir.parent.parent.parent / "backtest" / "backtest_summary.json"
        if not bs_path.exists():
            return {}
    try:
        s = json.loads(bs_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    tb = s.get("test_backtest", {}) or {}
    vb = s.get("validation_backtest", {}) or {}
    tp = s.get("test_prediction_metrics", {}) or {}
    vp = s.get("validation_prediction_metrics", {}) or {}
    by_reg = s.get("test_prediction_metrics_by_regime", {}) or {}
    return {
        "test_net":     (tb.get("net", {}) or {}).get("annual_return"),
        "test_gross":   (tb.get("gross", {}) or {}).get("annual_return"),
        "test_sharpe":  (tb.get("net", {}) or {}).get("sharpe"),
        "test_mdd":     (tb.get("net", {}) or {}).get("max_drawdown"),
        "test_trades":  tb.get("trade_count"),
        "test_winrate": tb.get("trade_win_rate"),
        "test_ic":      tp.get("spearman_ic"),
        "val_net":      (vb.get("net", {}) or {}).get("annual_return"),
        "val_sharpe":   (vb.get("net", {}) or {}).get("sharpe"),
        "val_ic":       vp.get("spearman_ic"),
        "low_vol_rows":  (by_reg.get("low_vol", {}) or {}).get("rows"),
        "high_vol_rows": (by_reg.get("high_vol", {}) or {}).get("rows"),
    }


def write_run_summary_md(
    run_dir: Path,
    run_id: str,
    results: list[dict[str, Any]],
    sort_by: str = "test_net",
) -> Path:
    """写入 results/runs/<run_id>/run_summary.md。
    按 sort_by （test_net 或 test_sharpe）降序列出所有 success 品种的关键指标。
    skipped / failed / disabled 单独列出。
    """
    from datetime import datetime

    rows: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []  # skipped / failed / disabled
    for r in results:
        status = str(r.get("status", "")).lower()
        if status != "success":
            other.append(r)
            continue
        pid = _normalize_product_id(r.get("product_id"))
        product_dir = Path(r.get("product_dir") or (run_dir / pid))
        m = _read_full_metrics(product_dir)
        rows.append({
            "product_id": pid,
            "category": r.get("category", ""),
            "feature_count": r.get("feature_count"),
            "mid_weekly_feature_count": r.get("mid_weekly_feature_count"),
            **m,
        })

    # 排序：按指定 metric 降序，None 排到最后
    def _key(rec):
        v = rec.get(sort_by)
        return (-(v if isinstance(v, (int, float)) else float("-inf")),)

    rows.sort(key=_key)

    def _pct(x, digits=2):
        if x is None or (isinstance(x, float) and not (x == x)):  # NaN
            return "—"
        return f"{x*100:+.{digits}f}%"

    def _f(x, digits=2):
        if x is None or (isinstance(x, float) and not (x == x)):
            return "—"
        return f"{x:+.{digits}f}"

    def _i(x):
        if x is None:
            return "—"
        try:
            return f"{int(x):,}"
        except Exception:
            return str(x)

    lines: list[str] = []
    lines.append(f"# Batch run summary: `{run_id}`\n")
    lines.append(f"_generated at {datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_run dir: `{run_dir}`_  ")
    lines.append(f"_sort by: **{sort_by} desc**_\n")

    if rows:
        lines.append(f"## Test 集回测结果（{len(rows)} 品种成功）\n")
        lines.append("| # | product | TEST net | TEST gross | sharpe | trades | win% | IC | VAL net | VAL sharpe | VAL IC | low/high vol rows | features (mid) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(rows, 1):
            low_n = r.get("low_vol_rows")
            high_n = r.get("high_vol_rows")
            regime_str = f"{_i(low_n)} / {_i(high_n)}"
            mid_n = r.get("mid_weekly_feature_count")
            feat_str = f"{_i(r.get('feature_count'))} ({_i(mid_n)} mid)" if mid_n else f"{_i(r.get('feature_count'))}"
            wr = r.get("test_winrate")
            wr_str = f"{wr*100:.1f}%" if isinstance(wr, (int, float)) else "—"
            lines.append(
                f"| {i} | **{r['product_id']}** | "
                f"{_pct(r.get('test_net'))} | {_pct(r.get('test_gross'))} | "
                f"{_f(r.get('test_sharpe'))} | {_i(r.get('test_trades'))} | {wr_str} | "
                f"{_f(r.get('test_ic'), 4)} | "
                f"{_pct(r.get('val_net'))} | {_f(r.get('val_sharpe'))} | "
                f"{_f(r.get('val_ic'), 4)} | {regime_str} | {feat_str} |"
            )
        lines.append("")
        # 简要统计
        net_vals = [r.get("test_net") for r in rows if isinstance(r.get("test_net"), (int, float))]
        sharpe_vals = [r.get("test_sharpe") for r in rows if isinstance(r.get("test_sharpe"), (int, float))]
        if net_vals and sharpe_vals:
            import statistics as st
            lines.append("### 摘要")
            lines.append(f"- TEST net annual：median **{st.median(net_vals)*100:+.2f}%** / mean {st.mean(net_vals)*100:+.2f}% / "
                         f"max {max(net_vals)*100:+.2f}% / min {min(net_vals)*100:+.2f}%")
            lines.append(f"- TEST sharpe：median **{st.median(sharpe_vals):+.2f}** / mean {st.mean(sharpe_vals):+.2f} / "
                         f"max {max(sharpe_vals):+.2f} / min {min(sharpe_vals):+.2f}")
            lines.append(f"- 正收益品种：{sum(1 for v in net_vals if v > 0)} / {len(net_vals)}")
            lines.append("")

    if other:
        lines.append(f"## 其他状态（{len(other)} 品种）\n")
        lines.append("| product | status | error |")
        lines.append("|---|---|---|")
        for r in other:
            err = str(r.get("error", "") or "")[:160]
            lines.append(f"| {_normalize_product_id(r.get('product_id'))} | "
                         f"{r.get('status', '')} | {err} |")
        lines.append("")

    out_path = run_dir / "run_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_product_artifacts(
    product_dir: Path,
    product_meta: dict[str, Any],
    prepared,
    backtest_artifacts,
) -> None:
    from backtest import add_daily_exposure_ratios

    product_dir.mkdir(parents=True, exist_ok=True)

    nav_curve = backtest_artifacts.test_daily[["TRADE_DATE", "nav_gross", "nav_net", "net_drawdown"]].copy()
    nav_curve.to_csv(product_dir / "nav_curve.csv", index=False, encoding="utf-8")

    daily_returns = add_daily_exposure_ratios(backtest_artifacts.test_daily.copy())
    daily_returns.to_csv(product_dir / "daily_returns.csv", index=False, encoding="utf-8")

    with (product_dir / "feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(prepared.feature_manifest, f, indent=2, ensure_ascii=False)


def _build_basic_result(
    product_meta: dict[str, Any],
    run_dir: Path,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    product_id = _normalize_product_id(product_meta.get("product_id"))
    return {
        "product_id": product_id,
        "instrument_code": product_meta.get("instrument_code"),
        "exchange": product_meta.get("exchange"),
        "category": product_meta.get("category", "unknown"),
        "status": status,
        "error": error,
        "data_start": product_meta.get("data_start"),
        "data_end": product_meta.get("data_end"),
        "product_dir": str(run_dir / product_id),
    }


def build_disabled_result(product_meta: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return _build_basic_result(product_meta=product_meta, run_dir=run_dir, status="skipped_disabled")


def build_failure_result(product_meta: dict[str, Any], run_dir: Path, error_message: str) -> dict[str, Any]:
    return _build_basic_result(product_meta=product_meta, run_dir=run_dir, status="failed", error=error_message)


def build_prechecked_skip_result(product_meta: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    result = _build_basic_result(
        product_meta=product_meta,
        run_dir=run_dir,
        status=str(product_meta.get("_batch_skip_status", "skipped")),
        error=str(product_meta.get("_batch_skip_error", "")),
    )
    result["available_data_start"] = product_meta.get("data_start")
    result["available_data_end"] = product_meta.get("data_end")
    if "_required_data_start" in product_meta:
        result["required_data_start"] = product_meta.get("_required_data_start")
    if "_required_data_end" in product_meta:
        result["required_data_end"] = product_meta.get("_required_data_end")
    return result


def build_failure_entry(product_id: str, error_message: str, trace_text: str) -> dict[str, Any]:
    return {
        "product_id": _normalize_product_id(product_id),
        "error": error_message,
        "traceback": trace_text,
    }


def run_single_product_training(
    product_meta: dict[str, Any],
    config_path: str | None,
    run_dir: Path,
    force_rebuild: bool = False,
    persist_to_shared_dir: bool = False,
) -> dict[str, Any]:
    from backtest import build_backtest_settings, execute_backtest, write_backtest_outputs
    from dataset import prepare_data
    from modeling import train_dual_regime_models

    product_id = _normalize_product_id(product_meta["product_id"])
    product_dir = run_dir / product_id
    config_override = build_product_config_override(
        product_meta, product_dir, persist_to_shared_dir=persist_to_shared_dir
    )

    prepared = prepare_data(config_path=config_path, force_rebuild=force_rebuild, config_override=config_override)
    artifact_map, training_summary, _ = train_dual_regime_models(
        prepared=prepared,
        config_path=config_path,
        config_override=config_override,
    )

    backtest_settings = build_backtest_settings(config_path=config_path, config_override=config_override)
    backtest_artifacts = execute_backtest(prepared=prepared, artifact_map=artifact_map, settings=backtest_settings)
    write_backtest_outputs(backtest_artifacts, backtest_settings)
    save_product_artifacts(product_dir=product_dir, product_meta=product_meta, prepared=prepared, backtest_artifacts=backtest_artifacts)

    # 写诊断图到 product_dir：覆盖 report.md 章节 1.2 + 1.3 + 1.6 引用的所有图
    # （vol_regime_split / backtest_curve 由 dataset.py / backtest.py 已经自动产出，
    # 这里补足 dist_* / factor_* / factor_top20_model_importance.png 共 7 张）。
    try:
        from diagnostics import generate_diagnostic_charts
        generate_diagnostic_charts(
            prepared=prepared,
            artifact_map=artifact_map,
            output_dir=product_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] WARNING: 诊断图生成失败 product={product_id}: {type(exc).__name__}: {exc}")

    return summarize_product_run(
        product_meta=product_meta,
        training_summary=training_summary,
        backtest_summary=backtest_artifacts.summary,
        product_dir=product_dir,
    )


def train_selected_products(
    product_records: list[dict[str, Any]],
    config_path: str | None,
    run_dir: Path,
    force_rebuild: bool = False,
    executor: Callable[..., dict[str, Any]] = run_single_product_training,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # 单品种模式：自动 persist 模型到 results/models/，方便 pipeline/backtest.py 复用。
    # 批量（多品种）：保持隔离，每个品种产物独占 product_dir。
    enabled_records = [m for m in product_records if not m.get("_batch_skip_status") and bool(m.get("enabled", True))]
    persist_to_shared_dir = len(enabled_records) == 1

    for product_meta in product_records:
        product_id = _normalize_product_id(product_meta.get("product_id"))
        if product_meta.get("_batch_skip_status"):
            results.append(build_prechecked_skip_result(product_meta=product_meta, run_dir=run_dir))
            continue
        if not bool(product_meta.get("enabled", True)):
            results.append(build_disabled_result(product_meta=product_meta, run_dir=run_dir))
            continue

        try:
            results.append(executor(product_meta, config_path, run_dir, force_rebuild, persist_to_shared_dir))
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            failures.append(build_failure_entry(product_id=product_id, error_message=error_message, trace_text=traceback.format_exc()))
            results.append(build_failure_result(product_meta=product_meta, run_dir=run_dir, error_message=error_message))

    return results, failures


def load_existing_run_outputs(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    results_payload = _read_json_payload(run_dir / "run_summary.json", [])
    failures_payload = _read_json_payload(run_dir / "failed_products.json", [])
    manifest_payload = _read_json_payload(run_dir / "manifest.json", {})

    results = results_payload if isinstance(results_payload, list) else []
    failures = failures_payload if isinstance(failures_payload, list) else []
    manifest = manifest_payload if isinstance(manifest_payload, dict) else {}
    return results, failures, manifest


def write_run_outputs(
    run_dir: Path,
    run_id: str,
    config_path: str | None,
    requested_products: list[str],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    selected_products: list[str] | None = None,
    started_at: str | None = None,
    status: str = "completed",
    resume_from: str | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    normalized_selected = [_normalize_product_id(item) for item in (selected_products or [row.get("product_id") for row in results])]
    ordered_results = _ordered_rows([dict(row) for row in results], normalized_selected)
    ordered_failures = _ordered_rows([dict(row) for row in failures], normalized_selected)

    # 保留：JSON 形式的结果汇总（resume 路径会读 run_summary.json）
    (run_dir / "run_summary.json").write_text(json.dumps(_to_native(ordered_results), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "failed_products.json").write_text(json.dumps(_to_native(ordered_failures), indent=2, ensure_ascii=False), encoding="utf-8")
    # 移除：run_summary.csv / insufficient_coverage_products.{json,csv} / manifest.json
    # 用户偏好：批量运行只保留 JSON + run_summary.md（人类可读）

    insufficient_coverage_rows = [row for row in ordered_results if _is_insufficient_coverage_status(row.get("status"))]
    completed_products = [_normalize_product_id(row.get("product_id")) for row in ordered_results if _normalize_product_id(row.get("product_id"))]
    completed_set = set(completed_products)
    pending_products = [product_id for product_id in normalized_selected if product_id not in completed_set]

    manifest = {
        "run_id": run_id,
        "created_at": started_at or datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": status,
        "config_path": str(Path(config_path).expanduser().resolve()) if config_path else str((PROJECT_ROOT / "config.yaml").resolve()),
        "requested_products": requested_products,
        "selected_products": normalized_selected,
        "completed_products": completed_products,
        "pending_products": pending_products,
        "total_selected": int(len(normalized_selected)),
        "processed_count": int(len(completed_products)),
        "pending_count": int(len(pending_products)),
        "success_count": int(sum(_is_success_status(row.get("status")) for row in ordered_results)),
        "failure_count": int(sum(str(row.get("status", "")).strip().lower() == "failed" for row in ordered_results)),
        "skipped_count": int(sum(_is_skipped_status(row.get("status")) for row in ordered_results)),
        "insufficient_coverage_count": int(len(insufficient_coverage_rows)),
        "insufficient_coverage_products": [
            _normalize_product_id(row.get("product_id")) for row in insufficient_coverage_rows
        ],
        "failed_products": [_normalize_product_id(row.get("product_id")) for row in ordered_failures],
        "run_dir": str(run_dir),
        "run_summary_path": str(run_dir / "run_summary.json"),
        "failed_products_path": str(run_dir / "failed_products.json"),
        "resume_from": resume_from,
    }
    # manifest dict 不再写盘（用户偏好），仅作为函数返回值供调用方使用。
    return manifest


def select_products(registry: list[dict[str, Any]], requested_products: list[str], run_all: bool) -> list[dict[str, Any]]:
    if run_all:
        return registry
    wanted = {_normalize_product_id(product) for product in requested_products}
    selected = [item for item in registry if _normalize_product_id(item.get("product_id")) in wanted]
    missing = sorted(wanted - {_normalize_product_id(item.get("product_id")) for item in selected})
    if missing:
        raise KeyError(f"Missing products in registry: {', '.join(missing)}")
    return selected


def resolve_selection_from_args(
    *,
    explicit_products: list[str],
    run_all: bool,
    existing_manifest: dict[str, Any],
) -> tuple[list[str], list[str], bool]:
    normalized_products = [_normalize_product_id(item) for item in explicit_products]
    if run_all or normalized_products:
        requested_products = ["__all__"] if run_all else normalized_products
        return requested_products, normalized_products, run_all

    prior_requested = list(existing_manifest.get("requested_products", []))
    prior_selected = [_normalize_product_id(item) for item in existing_manifest.get("selected_products", []) if _normalize_product_id(item)]
    if prior_selected:
        requested_products = prior_requested or prior_selected
        return requested_products, prior_selected, False
    if prior_requested == ["__all__"]:
        return prior_requested, [], True
    if prior_requested:
        normalized_requested = [_normalize_product_id(item) for item in prior_requested]
        return prior_requested, normalized_requested, False
    raise SystemExit("Pass --all or at least one --product, or provide --resume-run with an existing manifest.json.")


def split_resume_products(
    product_records: list[dict[str, Any]],
    existing_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success_map = {
        _normalize_product_id(row.get("product_id")): dict(row)
        for row in existing_results
        if _is_success_status(row.get("status"))
    }
    retained_results: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    for product_meta in product_records:
        product_id = _normalize_product_id(product_meta.get("product_id"))
        previous_success = success_map.get(product_id)
        if previous_success is not None and not product_meta.get("_batch_skip_status"):
            retained_results.append(previous_success)
        else:
            pending_records.append(product_meta)
    return retained_results, pending_records


def _log_top_results(results: list[dict[str, Any]], logger: Callable[[str], None]) -> None:
    successful_rows = [row for row in results if _is_success_status(row.get("status"))]
    sortable_rows = [row for row in successful_rows if row.get("sharpe") is not None]
    sortable_rows = [row for row in sortable_rows if _metric_or_na(row.get("sharpe")) != "n/a"]
    if not sortable_rows:
        return

    top_rows = sorted(sortable_rows, key=lambda row: float(row["sharpe"]), reverse=True)[:5]
    for row in top_rows:
        logger(
            "[batch] top_result "
            f"product={_normalize_product_id(row.get('product_id'))} "
            f"sharpe={_metric_or_na(row.get('sharpe'), digits=3)} "
            f"total_return={_metric_or_na(row.get('total_return'), digits=4)}"
        )


def execute_training_session(
    product_records: list[dict[str, Any]],
    config_path: str | None,
    run_dir: Path,
    run_id: str,
    requested_products: list[str],
    *,
    force_rebuild: bool = False,
    existing_results: list[dict[str, Any]] | None = None,
    resume_from: str | None = None,
    executor: Callable[..., dict[str, Any]] = run_single_product_training,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    logger = logger or _default_logger
    selected_products = [_normalize_product_id(item.get("product_id")) for item in product_records]
    started_at = datetime.now().isoformat()
    # 单品种模式：把模型 / training_summary / backtest_summary 写入共享路径
    # （results/models/、results/backtest/），让 pipeline/backtest.py 等下游脚本
    # 可以直接读取，避免和批量 (--all) 模式互相覆盖的问题。
    enabled_for_run = [
        m for m in product_records
        if not m.get("_batch_skip_status") and bool(m.get("enabled", True))
    ]
    persist_to_shared_dir = len(enabled_for_run) == 1

    results_map: dict[str, dict[str, Any]] = {
        _normalize_product_id(row.get("product_id")): dict(row)
        for row in (existing_results or [])
        if _normalize_product_id(row.get("product_id"))
    }
    failures_map: dict[str, dict[str, Any]] = {}

    initial_results = _ordered_rows(list(results_map.values()), selected_products)
    manifest = write_run_outputs(
        run_dir=run_dir,
        run_id=run_id,
        config_path=config_path,
        requested_products=requested_products,
        results=initial_results,
        failures=[],
        selected_products=selected_products,
        started_at=started_at,
        status="running",
        resume_from=resume_from,
    )

    logger(
        "[batch] start "
        f"run_id={run_id} total_selected={len(selected_products)} "
        f"pending={manifest['pending_count']} run_dir={run_dir}"
    )
    if resume_from:
        logger(f"[batch] resume_from={resume_from}")

    processed_count = len(initial_results)
    try:
        for product_meta in product_records:
            product_id = _normalize_product_id(product_meta.get("product_id"))
            if product_id in results_map:
                continue

            current_step = processed_count + 1
            logger(f"[batch] [{current_step}/{len(selected_products)}] start product={product_id}")

            if product_meta.get("_batch_skip_status"):
                result = build_prechecked_skip_result(product_meta=product_meta, run_dir=run_dir)
                results_map[product_id] = result
                logger(
                    f"[batch] [{current_step}/{len(selected_products)}] "
                    f"{result['status']} product={product_id} reason={result['error']}"
                )
            elif not bool(product_meta.get("enabled", True)):
                result = build_disabled_result(product_meta=product_meta, run_dir=run_dir)
                results_map[product_id] = result
                logger(f"[batch] [{current_step}/{len(selected_products)}] skipped_disabled product={product_id}")
            else:
                try:
                    result = executor(product_meta, config_path, run_dir, force_rebuild, persist_to_shared_dir)
                    results_map[product_id] = result
                    failures_map.pop(product_id, None)
                    logger(
                        "[batch] "
                        f"[{current_step}/{len(selected_products)}] success product={product_id} "
                        f"sharpe={_metric_or_na(result.get('sharpe'), digits=3)} "
                        f"total_return={_metric_or_na(result.get('total_return'), digits=4)}"
                    )
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                    failures_map[product_id] = build_failure_entry(
                        product_id=product_id,
                        error_message=error_message,
                        trace_text=traceback.format_exc(),
                    )
                    results_map[product_id] = build_failure_result(
                        product_meta=product_meta,
                        run_dir=run_dir,
                        error_message=error_message,
                    )
                    logger(f"[batch] [{current_step}/{len(selected_products)}] failed product={product_id} error={error_message}")

            processed_count += 1
            manifest = write_run_outputs(
                run_dir=run_dir,
                run_id=run_id,
                config_path=config_path,
                requested_products=requested_products,
                results=list(results_map.values()),
                failures=list(failures_map.values()),
                selected_products=selected_products,
                started_at=started_at,
                status="running",
                resume_from=resume_from,
            )
    except KeyboardInterrupt:
        manifest = write_run_outputs(
            run_dir=run_dir,
            run_id=run_id,
            config_path=config_path,
            requested_products=requested_products,
            results=list(results_map.values()),
            failures=list(failures_map.values()),
            selected_products=selected_products,
            started_at=started_at,
            status="interrupted",
            resume_from=resume_from,
        )
        logger(f"[batch] interrupted run_id={run_id} pending={manifest['pending_count']} run_dir={run_dir}")
        raise

    final_results = _ordered_rows(list(results_map.values()), selected_products)
    final_failures = _ordered_rows(list(failures_map.values()), selected_products)
    manifest = write_run_outputs(
        run_dir=run_dir,
        run_id=run_id,
        config_path=config_path,
        requested_products=requested_products,
        results=final_results,
        failures=final_failures,
        selected_products=selected_products,
        started_at=started_at,
        status="completed",
        resume_from=resume_from,
    )

    logger(
        "[batch] completed "
        f"run_id={run_id} success={manifest['success_count']} "
        f"failed={manifest['failure_count']} skipped={manifest['skipped_count']} "
        f"pending={manifest['pending_count']}"
    )
    if manifest["failed_products"]:
        logger(f"[batch] failed_products={', '.join(manifest['failed_products'])}")
    _log_top_results(final_results, logger=logger)
    logger(f"[batch] run_dir={run_dir}")

    # 写人类可读的 markdown 总结：按 TEST net annual_return 降序列出全部成功品种
    try:
        summary_md = write_run_summary_md(run_dir=run_dir, run_id=run_id, results=final_results, sort_by="test_net")
        logger(f"[batch] summary_md={summary_md}")
    except Exception as exc:  # noqa: BLE001
        logger(f"[batch] WARNING: write_run_summary_md failed: {type(exc).__name__}: {exc}")

    return final_results, final_failures, manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CTA_vol models for multiple products with runtime-generated factors.")
    parser.add_argument("--config", default=None, help="Path to CTA_vol config.yaml")
    parser.add_argument("--all", action="store_true", help="Train all products in product_registry.json")
    parser.add_argument("--product", action="append", default=[], help="Train one specific product_id; can be repeated")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached merged dataset for selected products")
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Resume an existing batch run by run_id or run directory; only non-success products will be retrained.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="自定义本次 run 的目录名（默认为 build_run_id() 生成的时间戳 YYYYMMDD_HHMMSS）。"
             "结果会写到 results/runs/<run_name>/ 下；与 --resume-run 互斥。",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    existing_manifest: dict[str, Any] = {}
    existing_results: list[dict[str, Any]] = []
    resume_run_dir: Path | None = None
    resume_from: str | None = None
    if args.resume_run:
        resume_run_dir = resolve_existing_run_dir(args.resume_run, config_path=args.config)
        existing_results, _, existing_manifest = load_existing_run_outputs(resume_run_dir)
        resume_from = str(resume_run_dir)

    requested_products, selection_ids, run_all = resolve_selection_from_args(
        explicit_products=args.product,
        run_all=args.all,
        existing_manifest=existing_manifest,
    )

    registry = load_product_registry(config_path=args.config)
    selected = select_products(registry=registry, requested_products=selection_ids, run_all=run_all)
    selected = annotate_products_for_batch_skip(selected, **load_batch_training_settings(config_path=args.config))

    retained_results: list[dict[str, Any]] = []
    # run_id 命名优先级：--resume-run > --run-name > 自动时间戳
    if args.run_name and args.resume_run:
        raise SystemExit("--run-name 与 --resume-run 互斥；resume 会复用原 run 目录名。")
    run_id = (args.run_name or "").strip() or build_run_id()
    run_dir = resolve_run_root(config_path=args.config) / run_id
    if resume_run_dir is not None:
        retained_results, _ = split_resume_products(selected, existing_results)
        run_dir = resume_run_dir
        run_id = run_dir.name

    _, _, manifest = execute_training_session(
        product_records=selected,
        config_path=args.config,
        run_dir=run_dir,
        run_id=run_id,
        requested_products=requested_products,
        force_rebuild=args.force_rebuild,
        existing_results=retained_results,
        resume_from=resume_from,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
