"""
策略A（样本加权）和策略B（Signed Extreme Target + target_hit 止盈）的单元/集成测试。
运行方式（从仓库根目录）: python -m unittest tests.test_strategy_ab
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.backtest import build_backtest_settings, generate_positions
from pipeline.dataset import REGIME_NAME_MAP, FactorDatasetBuilder, PreparedData
from pipeline.modeling import (
    build_model_settings,
    train_dual_regime_models,
    train_single_regime_model,
)


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------

def _make_minimal_model_override(
    tmp_dir: str,
    alpha: float = 0.0,
    target_col: str = "target_vol_norm",
) -> dict:
    root = Path(tmp_dir)
    return {
        "paths": {
            "model_dir": str(root / "models"),
            "training_summary": str(root / "training_summary.json"),
            "training_plot": str(root / "training_plot.png"),
            "training_comparison_plot": str(root / "comparison_plot.png"),
        },
        "model": {
            "target_column": target_col,
            "persist_models": False,
            "num_boost_round": 5,
            "early_stopping_rounds": 0,
            "feature_importance_top_n": 1,
            "sample_weight_alpha": alpha,
            "common_params": {
                "objective": "regression",
                "metric": "l2",
                "boosting_type": "gbdt",
                "learning_rate": 0.1,
                "num_leaves": 4,
                "min_child_samples": 5,
                "verbosity": -1,
                "seed": 42,
            },
            "low_vol_overrides": {},
            "high_vol_overrides": {},
        },
    }


def _make_regime_split(start: str, regime_label: int, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="min")
    base = np.linspace(-1.0, 1.0, n) + regime_label * 0.1
    target = base * 0.01
    return pd.DataFrame({
        "TDATE": idx,
        "TRADE_DATE": idx.normalize(),
        "REGIME_LABEL": regime_label,
        "future_return": target,
        "target_vol_scale": np.full(n, 0.001),
        "target_vol_norm": target,
        "target_extreme_norm": target * 1.5,  # extreme 比终值幅度更大
        "f1": base,
    })


def _make_prepared(alpha: float = 0.0) -> tuple[PreparedData, dict]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        n = 200
        train_df = pd.concat([
            _make_regime_split("2024-01-01", -1, n),
            _make_regime_split("2024-01-02", 1, n),
        ], ignore_index=True)
        val_df = pd.concat([
            _make_regime_split("2024-01-03", -1, n // 4),
            _make_regime_split("2024-01-04", 1, n // 4),
        ], ignore_index=True)
        test_df = pd.concat([
            _make_regime_split("2024-01-05", -1, n // 4),
            _make_regime_split("2024-01-06", 1, n // 4),
        ], ignore_index=True)
        prepared = PreparedData(
            full_data=pd.concat([train_df, val_df, test_df], ignore_index=True),
            train_data=train_df,
            val_data=val_df,
            test_data=test_df,
            feature_cols=["f1"],
            factor_cols=["f1"],
            engineered_cols=[],
            runtime_factor_cols=["f1"],
            mid_weekly_cols=[],
            target_col="target_vol_norm",
            metadata={},
            feature_manifest={},
        )
        return prepared, _make_minimal_model_override(tmp_dir, alpha=alpha)


# ---------------------------------------------------------------------------
# 策略A: 样本加权
# ---------------------------------------------------------------------------

class TestStrategyAWeightFormula(unittest.TestCase):
    """验证加权公式本身的数学性质，不依赖 LightGBM 训练。"""

    def _compute_weights(self, y: np.ndarray, alpha: float) -> np.ndarray | None:
        # 直接复制 train_single_regime_model 中的逻辑
        if alpha <= 0:
            return None
        w = np.abs(y) ** alpha
        return w / (w.mean() + 1e-10)

    def test_alpha_zero_returns_none(self) -> None:
        y = np.array([-2.0, -1.0, 0.1, 1.0, 2.0], dtype="float32")
        self.assertIsNone(self._compute_weights(y, alpha=0.0))

    def test_alpha_one_mean_is_one(self) -> None:
        y = np.array([-2.0, -1.0, 0.1, 1.0, 2.0], dtype="float32")
        w = self._compute_weights(y, alpha=1.0)
        self.assertIsNotNone(w)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)

    def test_alpha_one_larger_absolute_target_gets_more_weight(self) -> None:
        y = np.array([0.5, 1.0, 2.0], dtype="float32")
        w = self._compute_weights(y, alpha=1.0)
        # |2.0| > |1.0| > |0.5| → 权重单调递增
        self.assertGreater(w[2], w[1])
        self.assertGreater(w[1], w[0])

    def test_alpha_two_amplifies_tail_more_than_alpha_one(self) -> None:
        # alpha=2 时尾部权重相对于中间权重更大
        y = np.array([0.1, 1.0, 5.0], dtype="float32")
        w1 = self._compute_weights(y, alpha=1.0)
        w2 = self._compute_weights(y, alpha=2.0)
        ratio1 = w1[2] / w1[0]   # alpha=1 时 tail/head 倍数
        ratio2 = w2[2] / w2[0]   # alpha=2 时 tail/head 倍数
        self.assertGreater(ratio2, ratio1)

    def test_weights_are_non_negative(self) -> None:
        y = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype="float32")
        for alpha in [0.5, 1.0, 2.0]:
            w = self._compute_weights(y, alpha=alpha)
            self.assertTrue((w >= 0).all(), f"alpha={alpha} 产生了负权重")


class TestStrategyALightGBMWeightPassing(unittest.TestCase):
    """验证 train_single_regime_model 在 alpha>0 时把 weight 传给 lgb.Dataset，alpha=0 时传 None。"""

    def _build_xy(self, n: int = 100) -> tuple:
        x = np.random.randn(n, 2).astype("float32")
        y = np.random.randn(n).astype("float32")
        df = pd.DataFrame({
            "TDATE": pd.date_range("2024-01-01", periods=n, freq="min"),
            "TRADE_DATE": pd.to_datetime("2024-01-01"),
            "future_return": y,
            "target_vol_scale": np.ones(n),
            "f1": x[:, 0],
            "f2": x[:, 1],
        })
        return df, ["f1", "f2"]

    def _minimal_params(self) -> dict:
        return {
            "objective": "regression", "metric": "l2",
            "boosting_type": "gbdt", "learning_rate": 0.1,
            "num_leaves": 4, "min_child_samples": 2,
            "verbosity": -1, "seed": 42,
        }

    def _call_and_capture_weight(self, alpha: float) -> object:
        """调用 train_single_regime_model，返回传入 lgb.Dataset 的 weight 参数。"""
        train_df, feat_cols = self._build_xy(100)
        val_df, _ = self._build_xy(30)
        test_df, _ = self._build_xy(20)

        captured: dict = {}
        original_dataset = lgb.Dataset

        def _patched_dataset(data, label=None, weight=None, **kwargs):
            # 只拦截第一次调用（train set），val set 不传 weight
            if "weight" not in captured:
                captured["weight"] = weight
            return original_dataset(data, label=label, weight=weight, **kwargs)

        with patch("pipeline.modeling.lgb.Dataset", side_effect=_patched_dataset):
            train_single_regime_model(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feat_cols,
                target_col="future_return",  # raw target，无需 vol_scale
                scale_method="none",
                params=self._minimal_params(),
                num_boost_round=3,
                early_stopping_rounds=0,
                regime_label=-1,
                sample_weight_alpha=alpha,
            )
        return captured.get("weight")

    def test_alpha_zero_passes_none_weight(self) -> None:
        w = self._call_and_capture_weight(alpha=0.0)
        self.assertIsNone(w)

    def test_alpha_one_passes_array_weight(self) -> None:
        w = self._call_and_capture_weight(alpha=1.0)
        self.assertIsNotNone(w)
        self.assertIsInstance(w, np.ndarray)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=4)


class TestStrategyAConfigRead(unittest.TestCase):
    """build_model_settings 能正确读取 sample_weight_alpha。"""

    def test_reads_sample_weight_alpha_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = build_model_settings(
                config_override={
                    "paths": {
                        "model_dir": str(root / "models"),
                        "training_summary": str(root / "summary.json"),
                        "training_plot": str(root / "train.png"),
                        "training_comparison_plot": str(root / "cmp.png"),
                    },
                    "model": {"sample_weight_alpha": 1.5},
                }
            )
        self.assertAlmostEqual(settings["sample_weight_alpha"], 1.5, places=6)

    def test_default_sample_weight_alpha_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = build_model_settings(
                config_override={
                    "paths": {
                        "model_dir": str(root / "models"),
                        "training_summary": str(root / "summary.json"),
                        "training_plot": str(root / "train.png"),
                        "training_comparison_plot": str(root / "cmp.png"),
                    },
                }
            )
        self.assertAlmostEqual(settings["sample_weight_alpha"], 0.0, places=6)


class TestStrategyAIntegration(unittest.TestCase):
    """train_dual_regime_models 在 alpha=1 时完成训练并产生合法 artifact。"""

    def test_training_with_alpha_one_produces_valid_artifact(self) -> None:
        prepared, override = _make_prepared(alpha=1.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            override["paths"] = {
                "model_dir": str(Path(tmp_dir) / "models"),
                "training_summary": str(Path(tmp_dir) / "summary.json"),
                "training_plot": str(Path(tmp_dir) / "train.png"),
                "training_comparison_plot": str(Path(tmp_dir) / "cmp.png"),
            }
            artifact_map, summary, _ = train_dual_regime_models(
                prepared=prepared, config_override=override
            )
        # 两个 regime 都训练完成
        self.assertIn(-1, artifact_map)
        self.assertIn(1, artifact_map)
        # val_metrics 存在
        for artifact in artifact_map.values():
            self.assertIn("val_metrics", artifact.metrics)


# ---------------------------------------------------------------------------
# 策略B: Signed Extreme Target
# ---------------------------------------------------------------------------

class TestStrategyBTargetExtremeNorm(unittest.TestCase):
    """验证 _add_targets 生成 target_extreme_norm 的正确性。"""

    def _make_builder(self, tmp_dir: str) -> FactorDatasetBuilder:
        root = Path(tmp_dir)
        raw = root / "FAKE.csv"
        return FactorDatasetBuilder(
            config_override={
                "paths": {
                    "raw_data": str(raw),
                    "merged_cache": str(root / "cache.parquet"),
                    "cache_meta": str(root / "cache_meta.json"),
                    "regime_plot": str(root / "regime.png"),
                },
                "data": {
                    "target_horizon": 3,
                    "timestamp_col": "TDATE",
                    "trade_date_col": "TRADE_DATE",
                },
                "model": {
                    "target_vol_window": 3,
                    "target_vol_epsilon": 1e-8,
                    "target_vol_floor_quantile": 0.05,
                },
            }
        )

    def _make_single_day_df(self, closes: list[float], date: str = "2024-01-01") -> pd.DataFrame:
        n = len(closes)
        ts = pd.date_range(f"{date} 09:00:00", periods=n, freq="min")
        return pd.DataFrame({
            "TDATE": ts,
            "TRADE_DATE": pd.to_datetime(date),
            "CLOSE": closes,
        })

    def test_target_extreme_norm_column_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(tmp_dir)
            df = self._make_single_day_df([100.0] * 10)
            result = builder._add_targets(df)
        self.assertIn("target_extreme_norm", result.columns)

    def test_extreme_magnitude_geq_vol_norm_for_valid_bars(self) -> None:
        # |signed_extreme| >= |future_return|，因为 extreme 是窗口内最大偏移
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(tmp_dir)
            # 单调上涨序列：extreme == future_return（最大偏移就是终值）
            closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
            df = self._make_single_day_df(closes)
            result = builder._add_targets(df)
        valid = result.dropna(subset=["target_extreme_norm", "target_vol_norm"])
        for _, row in valid.iterrows():
            self.assertGreaterEqual(
                abs(row["target_extreme_norm"]),
                abs(row["target_vol_norm"]) - 1e-6,
                msg=f"extreme ({row['target_extreme_norm']:.4f}) < vol_norm ({row['target_vol_norm']:.4f})",
            )

    def test_spike_and_revert_captured_by_extreme_not_vol_norm(self) -> None:
        # "先涨再回归"：vol_norm ≈ 0，extreme 捕捉到涨幅。
        # rolling vol 需要 min_periods=3 个非 NaN pct_change，所以 bar0/1/2 的 vol_scale 为 NaN。
        # 用 bars 0-4 作为波动暖机（交替±1%），bar5 作为被测 bar，bars 6-8 构造 spike-revert。
        #   bar5 future_window (horizon=3) = [close6, close7, close8] = [106, 106, 100]
        #   future_return[5] = close8/close5 - 1 = 100/100 - 1 = 0    → vol_norm ≈ 0
        #   future_max[5] = 106/100 - 1 = 6%                          → extreme > 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(tmp_dir)
            closes = [100.0, 101.0, 100.0, 101.0, 100.0,  # 暖机
                      100.0, 106.0, 106.0, 100.0,          # spike-revert：bar5 = 测试目标
                      100.0, 100.0, 100.0, 100.0, 100.0, 100.0]  # 尾部
            df = self._make_single_day_df(closes)
            result = builder._add_targets(df)
        bar5 = result.iloc[5]
        # future_return 通过 shift(-horizon=-3) 同日取值：close[8]=100 / close[5]=100 - 1 = 0
        self.assertAlmostEqual(float(bar5["future_return"]), 0.0, places=4)
        # target_extreme_norm 通过 max(shift[-1,-2,-3]) 取得：max(106,106,100)/100-1=6% > 0
        self.assertFalse(pd.isna(bar5["target_extreme_norm"]), "bar5 的 vol_scale 应非 NaN（已过暖机期）")
        self.assertGreater(float(bar5["target_extreme_norm"]), 0.0)

    def test_same_day_constraint_last_bar_has_nan_extreme(self) -> None:
        # future_closes 使用 groupby(trade_date).shift(-k) 且 max(skipna=True)。
        # 只要同日内还有 ≥1 个未来 close，max/min 就不是 NaN——partial window 合法。
        # 只有最后一根 bar（0 个未来 close）的 extreme 才一定是 NaN。
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(tmp_dir)
            closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
            df = self._make_single_day_df(closes)
            result = builder._add_targets(df)
        # 最后一根 bar：shift(-1/-2/-3) 均为 NaN → max=NaN
        self.assertTrue(
            pd.isna(result.iloc[-1]["target_extreme_norm"]),
            "最后一根 bar 无同日未来 close，extreme 必须为 NaN",
        )
        # 倒数第 2 根 bar：shift(-1) 非 NaN（还有 close[-1]） → max 非 NaN
        self.assertFalse(
            pd.isna(result.iloc[-2]["target_extreme_norm"]),
            "倒数第 2 根 bar 有 1 个同日未来 close，extreme 应为非 NaN（partial window）",
        )

    def test_no_cross_day_bleed(self) -> None:
        # Day2 的高价格不应影响 Day1 末尾 bar 的 extreme
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(tmp_dir)
            day1 = self._make_single_day_df([100.0] * 6, date="2024-01-01")
            day2 = self._make_single_day_df([999.0] * 6, date="2024-01-02")
            df = pd.concat([day1, day2], ignore_index=True)
            result = builder._add_targets(df)
        # Day1 最后一根 bar 的 future close 应全部为 NaN（不跨日）
        day1_last = result[result["TRADE_DATE"] == pd.to_datetime("2024-01-01")].iloc[-1]
        self.assertTrue(
            pd.isna(day1_last["target_extreme_norm"]),
            "Day1 最后一根 bar 的 extreme 不应使用 Day2 的数据",
        )


# ---------------------------------------------------------------------------
# 策略B: target_hit 退出逻辑
# ---------------------------------------------------------------------------

def _make_pred_df(
    closes: list[float],
    preds: list[float],
    trade_date: str = "2024-01-01",
    regime: int = -1,
) -> pd.DataFrame:
    n = len(closes)
    ts = pd.date_range(f"{trade_date} 09:00:00", periods=n, freq="min")
    return pd.DataFrame({
        "TDATE": ts,
        "TRADE_DATE": pd.to_datetime(trade_date),
        "REGIME_LABEL": regime,
        "pred_return": preds,
        "CLOSE": closes,
    })


def _make_rule_map(
    regime: int = -1,
    entry_thresh: float = 0.03,
    exit_thresh: float = 0.01,
    min_hold_bars: int = 1,
    confirmation_bars: int = 2,
) -> dict:
    return {
        regime: {
            "regime_name": REGIME_NAME_MAP[regime],
            "entry_threshold": entry_thresh,
            "exit_threshold": exit_thresh,
            "cost_filter_threshold": 0.0,
            "effective_entry_threshold": entry_thresh,
            "confirmation_bars": confirmation_bars,
            "min_hold_bars": min_hold_bars,
            "cooldown_bars": 0,
            "allow_direct_flip": False,
            "flip_to_flat_first": True,
        }
    }


def _target_hit_settings(tp_fraction: float = 1.0, flatten_at_day_end: bool = False) -> dict:
    return {
        "signal_col": "pred_return",
        "exit_mode": "target_hit",
        "tp_fraction": tp_fraction,
        "flatten_at_day_end": flatten_at_day_end,
    }


def _quantile_settings(flatten_at_day_end: bool = False) -> dict:
    return {
        "signal_col": "pred_return",
        "exit_mode": "quantile",
        "tp_fraction": 1.0,
        "flatten_at_day_end": flatten_at_day_end,
    }


class TestStrategyBTargetHitExit(unittest.TestCase):
    """验证 exit_mode=target_hit 在 generate_positions 中的行为。"""

    def test_position_exits_when_pnl_reaches_target(self) -> None:
        # 预测极值=5%，价格从100涨到105时 pnl=5% >= target=5% → 平仓
        # bar0: pred=0.05 → confirm_count=1
        # bar1: pred=0.05 → confirm_count=2 → OPEN LONG @ CLOSE=100, entry_pred=0.05
        # bar2: CLOSE=102, pnl=2% < 5% → hold
        # bar3: CLOSE=105, pnl=5% >= 5% → EXIT
        closes = [100.0, 100.0, 102.0, 105.0, 106.0]
        preds  = [0.05,  0.05,  0.05,  0.05,  0.05]
        df = _make_pred_df(closes, preds)
        rule_map = _make_rule_map(entry_thresh=0.03, min_hold_bars=1, confirmation_bars=2)
        result = generate_positions(df, rule_map, _target_hit_settings(tp_fraction=1.0))

        positions = result["position"].tolist()
        actions = result["action"].tolist()
        self.assertEqual(positions[0], 0)             # 等待确认
        self.assertEqual(positions[1], 1)             # 开多
        self.assertEqual(positions[2], 1)             # 持仓（pnl 未达目标）
        self.assertEqual(positions[3], 0)             # 止盈平仓
        self.assertIn("exit_to_flat", actions[3])

    def test_position_stays_open_when_pnl_below_target(self) -> None:
        # 预测极值=10%，价格最多涨5%，始终不触达目标 → 全程持仓
        closes = [100.0, 100.0, 102.0, 104.0, 105.0]
        preds  = [0.10,  0.10,  0.10,  0.10,  0.10]
        df = _make_pred_df(closes, preds)
        rule_map = _make_rule_map(entry_thresh=0.03, min_hold_bars=1, confirmation_bars=2)
        result = generate_positions(df, rule_map, _target_hit_settings(tp_fraction=1.0))

        # bar1 开仓后，bar2/3/4 都应保持持仓
        self.assertEqual(result["position"].iloc[1], 1)
        self.assertEqual(result["position"].iloc[2], 1)
        self.assertEqual(result["position"].iloc[3], 1)
        self.assertEqual(result["position"].iloc[4], 1)

    def test_short_position_exits_when_price_drops_to_target(self) -> None:
        # 预测 -5%，做空，价格从100跌到95时 pnl=5% >= target=5% → 平仓
        closes = [100.0, 100.0, 98.0, 95.0, 94.0]
        preds  = [-0.05, -0.05, -0.05, -0.05, -0.05]
        df = _make_pred_df(closes, preds)
        rule_map = _make_rule_map(entry_thresh=0.03, min_hold_bars=1, confirmation_bars=2)
        result = generate_positions(df, rule_map, _target_hit_settings(tp_fraction=1.0))

        self.assertEqual(result["position"].iloc[1], -1)  # 开空
        self.assertEqual(result["position"].iloc[2], -1)  # 持仓
        self.assertEqual(result["position"].iloc[3], 0)   # 止盈平仓

    def test_tp_fraction_below_one_triggers_early_exit(self) -> None:
        # tp_fraction=0.5: 只需到达预测极值的50%即可平仓
        # 预测=10%，tp=5%，价格涨5%时就应平仓
        closes = [100.0, 100.0, 102.0, 105.0, 108.0]
        preds  = [0.10,  0.10,  0.10,  0.10,  0.10]
        df = _make_pred_df(closes, preds)
        rule_map = _make_rule_map(entry_thresh=0.03, min_hold_bars=1, confirmation_bars=2)
        result = generate_positions(df, rule_map, _target_hit_settings(tp_fraction=0.5))

        # bar3: CLOSE=105, pnl=5% >= 0.5*10%=5% → EXIT
        self.assertEqual(result["position"].iloc[1], 1)
        self.assertEqual(result["position"].iloc[3], 0)

    def test_day_end_flat_overrides_target_hit(self) -> None:
        # flatten_at_day_end=True 时，即使 pnl 未达目标，日末仍强平
        closes = [100.0, 100.0, 102.0, 104.0]
        preds  = [0.20,  0.20,  0.20,  0.20]   # 目标20%，价格根本达不到
        df = _make_pred_df(closes, preds, trade_date="2024-01-01")
        rule_map = _make_rule_map(entry_thresh=0.03, min_hold_bars=1, confirmation_bars=2)
        settings = _target_hit_settings(tp_fraction=1.0, flatten_at_day_end=True)
        result = generate_positions(df, rule_map, settings)

        # 最后一根 bar 是日末 → 应被强平
        self.assertEqual(result["position"].iloc[-1], 0)
        self.assertIn("day_end_flat", result["action"].tolist())

    def test_entry_price_resets_after_exit(self) -> None:
        # 平仓后，下一次开仓的 entry_price 应该是新的 CLOSE，而不是上次开仓价
        # bar0/1: 开多 @ 100，bar2/3: 止盈平 @ 105
        # bar4/5: 再次确认，开多 @ 110，bar6: 达到新目标
        closes = [100.0, 100.0, 105.0, 105.0, 110.0, 110.0, 115.5]
        preds  = [0.05,  0.05,  0.05,  0.05,  0.05,  0.05, 0.05]
        df = _make_pred_df(closes, preds)
        rule_map = _make_rule_map(entry_thresh=0.03, min_hold_bars=1, confirmation_bars=2)
        result = generate_positions(df, rule_map, _target_hit_settings(tp_fraction=1.0))

        # bar1: 开多@100; bar2: pnl=5%>=5% → 平; bar5: 再开多@110; bar6: pnl≈5% → 平
        self.assertEqual(result["position"].iloc[1], 1)   # 第一次开多
        self.assertEqual(result["position"].iloc[2], 0)   # 止盈平
        self.assertEqual(result["position"].iloc[5], 1)   # 第二次开多
        self.assertEqual(result["position"].iloc[6], 0)   # 再次止盈平


class TestStrategyBQuantileModeUnchanged(unittest.TestCase):
    """回归测试：exit_mode=quantile 时行为与策略B引入前完全一致。"""

    def test_quantile_exit_uses_signal_threshold_not_price(self) -> None:
        # quantile 模式：pred_value <= exit_thresh 时平仓，与 CLOSE 无关
        # bar0/1: 开多；bar2: 信号跌破阈值 → 平仓；即使 CLOSE 还在上涨
        closes = [100.0, 100.0, 200.0, 300.0]   # 价格一直涨（target_hit 不会平）
        preds  = [0.05,  0.05,  0.005, 0.005]   # pred 跌破 exit_thresh=0.01
        df = _make_pred_df(closes, preds)
        rule_map = _make_rule_map(entry_thresh=0.03, exit_thresh=0.01, min_hold_bars=1, confirmation_bars=2)
        result = generate_positions(df, rule_map, _quantile_settings())

        self.assertEqual(result["position"].iloc[1], 1)   # 开多
        self.assertEqual(result["position"].iloc[2], 0)   # 信号衰减平仓，非价格触达


class TestStrategyBBacktestSettingsRead(unittest.TestCase):
    """build_backtest_settings 能正确读取 exit_mode 和 tp_fraction。"""

    def test_reads_exit_mode_and_tp_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = build_backtest_settings(
                config_override={
                    "paths": {
                        "backtest_dir": str(root / "backtest"),
                        "backtest_plot": str(root / "backtest.png"),
                        "prediction_cache": str(root / "pred.parquet"),
                    },
                    "signal": {"exit_mode": "target_hit", "tp_fraction": 0.8},
                }
            )
        self.assertEqual(settings["exit_mode"], "target_hit")
        self.assertAlmostEqual(settings["tp_fraction"], 0.8, places=6)

    def test_defaults_to_quantile_and_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = build_backtest_settings(
                config_override={
                    "paths": {
                        "backtest_dir": str(root / "backtest"),
                        "backtest_plot": str(root / "backtest.png"),
                        "prediction_cache": str(root / "pred.parquet"),
                    },
                }
            )
        self.assertEqual(settings["exit_mode"], "quantile")
        self.assertAlmostEqual(settings["tp_fraction"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
