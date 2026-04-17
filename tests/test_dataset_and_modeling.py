from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from code.dataset import FactorDatasetBuilder, PreparedData
from code.modeling import train_dual_regime_models


class DatasetAndModelingTest(unittest.TestCase):
    def test_read_raw_data_retries_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_path = root / "RBZL.SHF.csv"
            merged_cache = root / "cache.parquet"
            cache_meta = root / "cache_meta.json"
            regime_plot = root / "regime_plot.png"

            builder = FactorDatasetBuilder(
                config_override={
                    "paths": {
                        "raw_data": str(raw_path),
                        "merged_cache": str(merged_cache),
                        "cache_meta": str(cache_meta),
                        "regime_plot": str(regime_plot),
                    },
                }
            )

            sample_df = pd.DataFrame(
                {
                    "TDATE": ["2024-01-01 09:00:00"],
                    "OPEN": [1.0],
                    "HIGH": [1.2],
                    "LOW": [0.9],
                    "CLOSE": [1.1],
                }
            )

            with patch("code.dataset.pd.read_csv", side_effect=[TimeoutError(60, "Operation timed out"), sample_df]) as mocked_read_csv:
                raw = builder._read_raw_data()

            self.assertEqual(mocked_read_csv.call_count, 2)
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw["CLOSE"].iloc[0], 1.1)

    def test_mid_weekly_merge_uses_asof_forward_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_path = root / "RBZL.SHF.csv"
            mid_dir = root / "mid_weekly"
            mid_dir.mkdir()
            merged_cache = root / "cache.parquet"
            cache_meta = root / "cache_meta.json"
            regime_plot = root / "regime_plot.png"

            raw_df = pd.DataFrame(
                {
                    "TDATE": pd.to_datetime(
                        [
                            "2024-01-01 09:00:00",
                            "2024-01-02 09:00:00",
                            "2024-01-03 09:00:00",
                        ]
                    ),
                    "OPEN": [1, 2, 3],
                    "HIGH": [2, 3, 4],
                    "LOW": [1, 2, 3],
                    "CLOSE": [1.5, 2.5, 3.5],
                    "VOLUME": [10, 20, 30],
                    "AMOUNT": [15, 50, 105],
                }
            )
            raw_df.to_csv(raw_path, index=False)

            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-03"],
                    "inventory": [100, 200],
                }
            ).to_csv(mid_dir / "inventory.csv", index=False)

            builder = FactorDatasetBuilder(
                config_override={
                    "paths": {
                        "raw_data": str(raw_path),
                        "merged_cache": str(merged_cache),
                        "cache_meta": str(cache_meta),
                        "regime_plot": str(regime_plot),
                        "mid_weekly_dir": str(mid_dir),
                    },
                    "data": {
                        "use_mid_weekly": True,
                    },
                    "product": {
                        "product_id": "RB",
                        "mid_weekly_files": ["inventory.csv"],
                    },
                }
            )
            raw = builder._read_raw_data()
            merged, mid_cols = builder._merge_mid_weekly_features(raw)

            self.assertEqual(mid_cols, ["MID_inventory"])
            self.assertEqual(merged["MID_inventory"].tolist(), [100.0, 100.0, 200.0])

    def test_train_dual_regime_models_skips_model_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "models"
            summary_path = root / "training_summary.json"
            training_plot = root / "training_plot.png"
            comparison_plot = root / "comparison_plot.png"

            def _make_split(start: str, regime_label: int, n: int) -> pd.DataFrame:
                idx = pd.date_range(start, periods=n, freq="min")
                base = np.linspace(-1.0, 1.0, n) + regime_label * 0.2
                future_return = base * 0.01
                return pd.DataFrame(
                    {
                        "TDATE": idx,
                        "TRADE_DATE": idx.normalize(),
                        "REGIME_LABEL": regime_label,
                        "future_return": future_return,
                        "target_vol_scale": np.ones(n),
                        "target_vol_norm": future_return,
                        "f1": base,
                    }
                )

            train_df = pd.concat([_make_split("2024-01-01 09:00:00", -1, 24), _make_split("2024-01-02 09:00:00", 1, 24)], ignore_index=True)
            val_df = pd.concat([_make_split("2024-01-03 09:00:00", -1, 12), _make_split("2024-01-04 09:00:00", 1, 12)], ignore_index=True)
            test_df = pd.concat([_make_split("2024-01-05 09:00:00", -1, 12), _make_split("2024-01-06 09:00:00", 1, 12)], ignore_index=True)
            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

            prepared = PreparedData(
                full_data=full_df,
                train_data=train_df,
                val_data=val_df,
                test_data=test_df,
                feature_cols=["f1"],
                factor_cols=["f1"],
                engineered_cols=[],
                runtime_factor_cols=["f1"],
                mid_weekly_cols=[],
                target_col="target_vol_norm",
                metadata={"feature_count": 1},
                feature_manifest={"all_feature_cols": ["f1"]},
            )

            _, summary, _ = train_dual_regime_models(
                prepared=prepared,
                config_override={
                    "paths": {
                        "model_dir": str(model_dir),
                        "training_summary": str(summary_path),
                        "training_plot": str(training_plot),
                        "training_comparison_plot": str(comparison_plot),
                    },
                    "model": {
                        "persist_models": False,
                        "num_boost_round": 10,
                        "early_stopping_rounds": 0,
                        "feature_importance_top_n": 5,
                        "common_params": {
                            "objective": "regression",
                            "metric": "l2",
                            "boosting_type": "gbdt",
                            "learning_rate": 0.1,
                            "num_leaves": 7,
                            "max_depth": 3,
                            "min_child_samples": 2,
                            "verbosity": -1,
                            "seed": 42,
                        },
                        "low_vol_overrides": {},
                        "high_vol_overrides": {},
                    },
                },
            )

            self.assertFalse(model_dir.exists())
            self.assertTrue(summary_path.exists())
            self.assertFalse((model_dir / "low_vol" / "model.txt").exists())
            self.assertIn("combined_test_metrics", summary)

    def test_train_dual_regime_models_reports_empty_regime_split_counts(self) -> None:
        def _make_split(start: str, regime_label: int, n: int) -> pd.DataFrame:
            idx = pd.date_range(start, periods=n, freq="min")
            base = np.linspace(-1.0, 1.0, n) + regime_label * 0.2
            future_return = base * 0.01
            return pd.DataFrame(
                {
                    "TDATE": idx,
                    "TRADE_DATE": idx.normalize(),
                    "REGIME_LABEL": regime_label,
                    "future_return": future_return,
                    "target_vol_scale": np.ones(n),
                    "target_vol_norm": future_return,
                    "f1": base,
                }
            )

        train_df = pd.concat([_make_split("2024-01-01 09:00:00", -1, 24), _make_split("2024-01-02 09:00:00", 1, 24)], ignore_index=True)
        val_df = _make_split("2024-01-03 09:00:00", -1, 12)
        test_df = pd.concat([_make_split("2024-01-04 09:00:00", -1, 12), _make_split("2024-01-05 09:00:00", 1, 8)], ignore_index=True)
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        prepared = PreparedData(
            full_data=full_df,
            train_data=train_df,
            val_data=val_df,
            test_data=test_df,
            feature_cols=["f1"],
            factor_cols=["f1"],
            engineered_cols=[],
            runtime_factor_cols=["f1"],
            mid_weekly_cols=[],
            target_col="target_vol_norm",
            metadata={"feature_count": 1},
            feature_manifest={"all_feature_cols": ["f1"]},
        )

        with self.assertRaises(ValueError) as ctx:
            train_dual_regime_models(
                prepared=prepared,
                config_override={
                    "model": {
                        "persist_models": False,
                    },
                },
            )

        message = str(ctx.exception)
        self.assertIn("Regime 'high_vol' has empty split(s): val", message)
        self.assertIn("rows_by_split=train:24, val:0, test:8", message)

    def test_load_or_build_feature_frame_falls_back_when_cache_read_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_path = root / "RBZL.SHF.csv"
            merged_cache = root / "cache.parquet"
            cache_meta = root / "cache_meta.json"
            regime_plot = root / "regime_plot.png"

            raw_path.write_text("TDATE,OPEN,HIGH,LOW,CLOSE\n2024-01-01 09:00:00,1,2,1,1.5\n", encoding="utf-8")
            merged_cache.write_bytes(b"placeholder")

            builder = FactorDatasetBuilder(
                config_override={
                    "paths": {
                        "raw_data": str(raw_path),
                        "merged_cache": str(merged_cache),
                        "cache_meta": str(cache_meta),
                        "regime_plot": str(regime_plot),
                    },
                }
            )

            fallback_result = (
                pd.DataFrame({"TDATE": pd.to_datetime(["2024-01-01 09:00:00"]), "TRADE_DATE": pd.to_datetime(["2024-01-01"])}),
                ["f1"],
                ["ENG_X"],
                ["f1"],
                [],
                {"factor_count": 1},
            )

            with patch("code.dataset.pd.read_parquet", side_effect=TimeoutError(60, "Operation timed out")):
                with patch.object(builder, "build_feature_frame", return_value=fallback_result) as mocked_build_feature_frame:
                    result = builder.load_or_build_feature_frame(force_rebuild=False)

            mocked_build_feature_frame.assert_called_once()
            self.assertIs(result, fallback_result)


if __name__ == "__main__":
    unittest.main()
