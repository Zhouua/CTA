"""
Retrain the dual-regime (micro) model for RB and save per-bar test predictions.

Outputs: results/runs/micro_result/RB/test_predictions.csv
Columns: TDATE, TRADE_DATE, REGIME_LABEL, future_return, pred_return

Used by compare_initial_vs_micro.py for the precision/recall threshold analysis.
Run this once; subsequent comparison runs load the cached CSV directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplconfig").resolve()))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

import pandas as pd

from pipeline.dataset import prepare_data
from pipeline.modeling import train_dual_regime_models, predict_dual_regime

OUT_PATH = PROJECT_ROOT / "results" / "runs" / "micro_result" / "RB" / "test_predictions.csv"


def main() -> None:
    config_path = str(PROJECT_ROOT / "config.yaml")
    config_override = {
        "data": {"use_mid_weekly": False},
        "product": {
            "product_id": "RB",
            "instrument_code": "RBZL.SHF",
            "raw_data_file": "RBZL.SHF.csv",
            "mid_weekly_files": [],
        },
        "paths": {
            "run_root": str(PROJECT_ROOT / "results" / "runs"),
        },
    }

    print("Preparing data ...")
    prepared = prepare_data(config_path=config_path, config_override=config_override)
    print(f"  train={len(prepared.train_data)}, val={len(prepared.val_data)}, test={len(prepared.test_data)}")

    print("Training dual-regime micro model ...")
    artifact_map, _, _ = train_dual_regime_models(
        prepared=prepared,
        config_path=config_path,
        config_override=config_override,
    )
    print(f"  Trained {len(artifact_map)} regime models")

    print("Generating test predictions ...")
    test_pred = predict_dual_regime(
        df=prepared.test_data,
        feature_cols=prepared.feature_cols,
        target_col=prepared.target_col,
        artifact_map=artifact_map,
    )

    cols = ["TDATE", "TRADE_DATE", "REGIME_LABEL", "future_return", "pred_return"]
    out = test_pred[[c for c in cols if c in test_pred.columns]].copy()
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUT_PATH}")
    print(f"  pred_return: mean={out['pred_return'].mean():.3e}  std={out['pred_return'].std():.3e}")


if __name__ == "__main__":
    main()
