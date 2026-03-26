from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".mplconfig").resolve()))

from dataset import prepare_data
from modeling import train_dual_regime_models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train low-vol/high-vol LightGBM models for CTA_vol.")
    parser.add_argument("--config", default=None, help="Path to CTA_vol config.yaml")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached merged dataset before training.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepared = prepare_data(config_path=args.config, force_rebuild=args.force_rebuild)
    _, summary, _ = train_dual_regime_models(prepared=prepared, config_path=args.config)
    print(json.dumps(summary["combined_test_metrics"], indent=2, ensure_ascii=False))
