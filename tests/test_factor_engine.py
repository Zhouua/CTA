from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.factor_engine import generate_runtime_factors


class RuntimeFactorEngineTest(unittest.TestCase):
    def _make_raw_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "TDATE": pd.date_range("2024-01-01 09:00:00", periods=8, freq="min"),
                "OPEN": [10, 11, 12, 13, 14, 15, 16, 17],
                "HIGH": [11, 12, 13, 14, 15, 16, 17, 18],
                "LOW": [9, 10, 11, 12, 13, 14, 15, 16],
                "CLOSE": [10, 12, 11, 15, 14, 18, 17, 19],
                "VOLUME": [100, 110, 90, 130, 120, 140, 150, 160],
                "AMOUNT": [1000, 1320, 990, 1950, 1680, 2520, 2550, 3040],
            }
        )
        df["TRADE_DATE"] = df["TDATE"].dt.normalize()
        return df

    def test_runtime_factor_warmup_and_no_future_leak(self) -> None:
        raw = self._make_raw_frame()
        result = generate_runtime_factors(
            raw,
            {
                "enabled": True,
                "windows": [3],
                "lags": [0, 1],
                "groups": ["price_lags", "trend", "direction"],
            },
        )
        frame = result.frame

        self.assertAlmostEqual(frame.loc[0, "OPEN0"], raw.loc[0, "OPEN"] / raw.loc[0, "CLOSE"])
        self.assertAlmostEqual(frame.loc[1, "OPEN1"], raw.loc[0, "OPEN"] / raw.loc[1, "CLOSE"])
        self.assertTrue(pd.isna(frame.loc[1, "MA3"]))
        self.assertTrue(pd.notna(frame.loc[2, "MA3"]))

        altered = raw.copy()
        altered.loc[7, "CLOSE"] = 9999.0
        altered_result = generate_runtime_factors(
            altered,
            {
                "enabled": True,
                "windows": [3],
                "lags": [0, 1],
                "groups": ["price_lags", "trend", "direction"],
            },
        )
        pd.testing.assert_series_equal(frame.loc[:5, "MA3"], altered_result.frame.loc[:5, "MA3"], check_names=False)
        pd.testing.assert_series_equal(frame.loc[:5, "CNTD3"], altered_result.frame.loc[:5, "CNTD3"], check_names=False)

    def test_runtime_factor_formula_subset(self) -> None:
        raw = self._make_raw_frame()
        result = generate_runtime_factors(
            raw,
            {
                "enabled": True,
                "windows": [3],
                "lags": [0, 1],
                "groups": ["kline", "trend", "volatility", "volume", "correlation"],
            },
        )
        frame = result.frame

        idx = 3
        self.assertAlmostEqual(frame.loc[idx, "KMID"], (raw.loc[idx, "CLOSE"] - raw.loc[idx, "OPEN"]) / raw.loc[idx, "OPEN"])
        self.assertAlmostEqual(
            frame.loc[idx, "KSFT2"],
            ((2 * raw.loc[idx, "CLOSE"]) - raw.loc[idx, "HIGH"] - raw.loc[idx, "LOW"]) / (raw.loc[idx, "HIGH"] - raw.loc[idx, "LOW"]),
        )
        self.assertAlmostEqual(frame.loc[idx, "ROC3"], raw.loc[idx - 3, "CLOSE"] / raw.loc[idx, "CLOSE"])
        self.assertAlmostEqual(frame.loc[idx, "VOLUME0"], 1.0)
        self.assertAlmostEqual(
            frame.loc[idx, "VMA3"],
            raw.loc[idx - 2 : idx, "VOLUME"].mean() / raw.loc[idx, "VOLUME"],
        )

        abs_ret_volume = raw["CLOSE"].pct_change().abs() * raw["VOLUME"]
        expected_wvma = abs_ret_volume.iloc[idx - 2 : idx + 1].std(ddof=1) / abs_ret_volume.iloc[idx - 2 : idx + 1].mean()
        self.assertAlmostEqual(frame.loc[idx, "WVMA3"], expected_wvma)

        close_ratio = raw["CLOSE"] / raw["CLOSE"].shift(1)
        log_volume_ratio = pd.Series(np.log1p(raw["VOLUME"] / raw["VOLUME"].shift(1)))
        expected_corr = raw.loc[idx - 2 : idx, "CLOSE"].corr(pd.Series(np.log1p(raw.loc[idx - 2 : idx, "VOLUME"])))
        expected_cord = close_ratio.iloc[idx - 2 : idx + 1].corr(log_volume_ratio.iloc[idx - 2 : idx + 1])
        self.assertAlmostEqual(frame.loc[idx, "CORR3"], expected_corr)
        self.assertAlmostEqual(frame.loc[idx, "CORD3"], expected_cord)

    def test_rb_reference_alignment_subset(self) -> None:
        raw_path = "data/RBZL.SHF.csv"
        reference_dir = "data/factors"
        try:
            raw = pd.read_csv(raw_path, nrows=400)
        except FileNotFoundError:
            self.skipTest("RB reference data not available")
            return

        raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed:")]
        raw["TDATE"] = pd.to_datetime(raw["TDATE"])
        raw["TRADE_DATE"] = raw["TDATE"].dt.normalize()
        result = generate_runtime_factors(raw, {"enabled": True})

        tolerances = {
            "MA5": 1e-6,
            "ROC10": 1e-6,
            "STD10": 1e-6,
            "MAX20": 1e-6,
            "MIN20": 1e-6,
            "RSV10": 1e-6,
            "CNTP10": 1e-6,
            "CNTN10": 1e-6,
            "CNTD10": 1e-6,
            "SUMP10": 1e-3,
            "SUMN10": 1e-3,
            "SUMD10": 2e-3,
            "KMID": 1e-6,
            "KMID2": 1e-6,
            "KSFT": 1e-6,
            "KSFT2": 1e-6,
            "CORR10": 1e-6,
            "CORD10": 1e-6,
            "VMA10": 1e-6,
            "WVMA10": 1e-6,
            "IMAX10": 1e-6,
            "IMIN10": 1e-6,
            "IMXD10": 1e-6,
        }

        for col, tolerance in tolerances.items():
            ref = pd.read_csv(f"{reference_dir}/RBZL_{col}.csv", nrows=400)
            ref["tdate"] = pd.to_datetime(ref["tdate"])
            merged = result.frame[["TDATE", col]].merge(
                ref.rename(columns={"tdate": "TDATE"}),
                on="TDATE",
                how="inner",
                suffixes=("_new", "_ref"),
            )
            valid = merged.dropna()
            self.assertFalse(valid.empty, msg=f"{col} has no overlapping valid rows")
            max_diff = float((valid[f"{col}_new"] - valid[f"{col}_ref"]).abs().max())
            self.assertLessEqual(max_diff, tolerance, msg=f"{col} max diff {max_diff} exceeds tolerance {tolerance}")


if __name__ == "__main__":
    unittest.main()
