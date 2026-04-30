"""
pipeline/cal_factors.py — 工程化特征 (ENG_*) 的集中式动态计算

本模块从 dataset.py 中抽出，所有"按时序滚动计算"的 ENG_* 因子集中维护。
runtime 量价因子 (RET1/STD20/CORR60/...) 仍在 pipeline/factor_engine.py，
两者在 dataset.py::FactorDatasetBuilder 中分别被调用。

----------------------------------------------------------------------
**严格因果性约束（防前视）**
----------------------------------------------------------------------
每根 bar t 的特征值，只允许使用 timestamp ≤ t 的已实现数据。具体保证：

1. 输入 df 必须按 timestamp_col 升序排列（函数入口处会再排一次以保险）。
2. 所有 rolling / pct_change / shift 默认使用过去 + 当前 bar 的数据。
   - 同 bar 的 OPEN/HIGH/LOW/CLOSE/VOLUME/AMOUNT/POSITION 视为"该 bar 收盘时
     已实现"，下一根 bar 才可执行交易（次根撮合，与 backtest 一致）。
3. 日级 context（ENG_DAILY_*_LAG1）一律使用 daily_log_ret.shift(1) 以后再 rolling，
   即"截止到昨日"的统计；广播到当日 bars 时，每个 bar 拿到的都是 ≤ d-1 的信息。
4. 跨日同分钟历史先验（ENG_TYPICAL_*）一律 groupby(minute).shift(1) 排除"今日同
   分钟自身"，再在 minute 组内 rolling N 个历史样本。
5. ENG_INTRADAY_BAR_PROGRESS 改用「昨日 bar 数」作分母，避免
   groupby(today).transform("count") 隐含的"当日总 bar 数"前视（2026-04 修订）。
6. cummax/cummin/expanding/groupby(_tdate).cumcount 等"沿时间累积"的算子是因果的
   （只看 ≤ t 的数据），可放心使用。

----------------------------------------------------------------------
**因子组别（按调用顺序，所有列名以 ENG_ 前缀）**
----------------------------------------------------------------------
A. 单 bar 价格 / K 线形态                 ENG_RET_1, ENG_RANGE_1, ENG_BODY_1, ...
B. 多窗口动量 / 波动 / 价位偏离           ENG_RET_w, ENG_RV_w, ENG_PRICE_TO_MA_w, ...
C. 量 / 持仓 / 成交额比                   ENG_VOLUME_RATIO_w, ENG_POSITION_RATIO_w, ...
D. 突破 / 区间位置 / 距日内极值距离       ENG_PRICE_BREAKOUT_*, ENG_DAY_RANGE_POS, ENG_DIST_*
E. 时间编码                               ENG_TOD_SIN/COS, ENG_WEEKDAY, ENG_IS_DAY_SESSION
F. 隔夜 + 日内运行收益                    ENG_OVERNIGHT_GAP, ENG_INTRADAY_RET, ENG_PREV_CLOSE_LOG_RET
G. 长窗口动量 / 波动率制度                ENG_RET_120/240/480, ENG_VOL_TREND, ENG_VOL_ZSCORE, ...
H. 隔夜 / VWAP / Z-score / 量价加权       ENG_VWAP_DEV_w, ENG_CLOSE_ZSCORE_w, ENG_TICK_DIR_VW_w, ...
I. 日内 expanding 累积 / 进度             ENG_INTRADAY_POS_RATIO_CUM, ENG_INTRADAY_BAR_PROGRESS, ...
J. 跨日同分钟历史先验                     ENG_TYPICAL_INTRADAY_DEV_30, ENG_TYPICAL_RET1_Z_60, ...
K. 半方差比 + 连胜计数                    ENG_SEMIVAR_LOG_RATIO_*, ENG_CONSECUTIVE_DIR_COUNT
L. 持仓量 (OI) 因子                       ENG_OI_CHANGE_w, ENG_OI_PRICE_CONFIRM_w, ENG_OI_DIVERGE_w
M. 合成 / 交互 / 效率比 / 突破触发        ENG_SHARPE_w, ENG_MOM_ACCEL_w, ENG_BREAKOUT_60, ...
N. 日级 lag 上下文（截止昨日）            ENG_DAILY_RET_*_LAG1, ENG_DAILY_VOL_*_LAG1, ...
O. 因子融合（同类簇内取均值/几何均值）    ENG_FUSED_*

----------------------------------------------------------------------
**入口函数**：
    add_engineered_features(df, *, timestamp_col, trade_date_col,
                             engineered_windows, use_engineered_features=True,
                             use_synthetic_factors=True)
    → (df_with_features, eng_cols)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-8

# 当 use_synthetic_factors=False 时只保留的"原始 35 项基础 ENG_*"
# （消融实验用；保持与历史 baseline 完全一致的列名集合）
_ORIGINAL_ENG_COLS: set[str] = {
    "ENG_RET_1", "ENG_LOG_RET_1", "ENG_RANGE_1", "ENG_BODY_1", "ENG_BODY_ABS",
    "ENG_CLOSE_TO_RANGE", "ENG_INTRABAR_VOL",
    "ENG_RET_5", "ENG_RET_20", "ENG_RET_60",
    "ENG_RV_5", "ENG_RV_20", "ENG_RV_60",
    "ENG_PRICE_TO_MA_5", "ENG_PRICE_TO_MA_20", "ENG_PRICE_TO_MA_60",
    "ENG_VOLUME_RATIO_5", "ENG_VOLUME_RATIO_20", "ENG_VOLUME_RATIO_60",
    "ENG_POSITION_RATIO_5", "ENG_POSITION_RATIO_20", "ENG_POSITION_RATIO_60",
    "ENG_AMOUNT_RATIO_5", "ENG_AMOUNT_RATIO_20", "ENG_AMOUNT_RATIO_60",
    "ENG_ATR_20", "ENG_ATR_60", "ENG_VOL_RATIO_5_20", "ENG_RET_DIFF_5_20",
    "ENG_PRICE_BREAKOUT_20", "ENG_PRICE_BREAKDOWN_20",
    "ENG_TOD_SIN", "ENG_TOD_COS", "ENG_WEEKDAY", "ENG_IS_DAY_SESSION",
}


# ════════════════════════════════════════════════════════════════════
# 内部小工具
# ════════════════════════════════════════════════════════════════════

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """缺列回退为 0：VOLUME/AMOUNT/POSITION 在某些品种上可能缺失。"""
    if col in df.columns:
        return df[col].astype("float64")
    return pd.Series(0.0, index=df.index, dtype="float64")


def _min_periods(window: int) -> int:
    """rolling 默认 min_periods：max(3, window // 3)。
    既能在数据起步阶段尽早产生有效值，又避免极端短窗造成噪声。"""
    return max(3, window // 3)


# ════════════════════════════════════════════════════════════════════
# A. 单 bar 价格 / K 线形态  —— 依赖：当根 OHLC（bar 收盘时已实现）
# ════════════════════════════════════════════════════════════════════

def _add_single_bar_features(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, open_, high, low = ctx["close"], ctx["open_"], ctx["high"], ctx["low"]
    high_low_range = ctx["high_low_range"]

    # 当根 1 - bar 简单收益与对数收益
    out["ENG_RET_1"] = ctx["ret1"]
    out["ENG_LOG_RET_1"] = np.log(close / close.shift(1))
    # 当根高低区间幅度（相对 low），用于刻画 bar 内振幅
    out["ENG_RANGE_1"] = high / (low + EPS) - 1.0
    # 实体大小（close 对 open 的比率减 1）与其绝对值
    out["ENG_BODY_1"] = close / (open_ + EPS) - 1.0
    out["ENG_BODY_ABS"] = out["ENG_BODY_1"].abs()
    # 当根 close 在 [low, high] 中的相对位置（多空压力）
    out["ENG_CLOSE_TO_RANGE"] = (close - low) / (high_low_range + EPS)
    # 振幅占价格比（bar 内日内波动率代理）
    out["ENG_INTRABAR_VOL"] = high_low_range / (close + EPS)


# ════════════════════════════════════════════════════════════════════
# B. 多窗口动量 / 波动 / 均线偏离  —— rolling 在过去+当前 bar 上
# ════════════════════════════════════════════════════════════════════

def _add_window_features(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close = ctx["close"]
    ret1 = ctx["ret1"]
    high, low = ctx["high"], ctx["low"]
    volume, position, amount = ctx["volume"], ctx["position"], ctx["amount"]
    short_w, med_w, long_w = ctx["short_w"], ctx["med_w"], ctx["long_w"]

    for w in sorted({short_w, med_w, long_w}):
        mp = _min_periods(w)
        ma = close.rolling(w, min_periods=mp).mean()
        rv = ret1.rolling(w, min_periods=mp).std()
        # w-bar 累积收益、已实现波动率、价格相对均线偏离
        out[f"ENG_RET_{w}"] = close.pct_change(w)
        out[f"ENG_RV_{w}"] = rv
        out[f"ENG_PRICE_TO_MA_{w}"] = close / (ma + EPS) - 1.0
        # 量 / 持仓 / 成交额相对滚动均值的比率（量比族）
        out[f"ENG_VOLUME_RATIO_{w}"] = volume / (volume.rolling(w, min_periods=mp).mean() + EPS)
        out[f"ENG_POSITION_RATIO_{w}"] = position / (position.rolling(w, min_periods=mp).mean() + EPS)
        out[f"ENG_AMOUNT_RATIO_{w}"] = amount / (amount.rolling(w, min_periods=mp).mean() + EPS)

    # 中长窗口 ATR（true_range = max(H-L, |H-prevC|, |L-prevC|)）相对 close 归一
    tr = ctx["true_range"]
    for w in (med_w, long_w):
        mp = _min_periods(w)
        out[f"ENG_ATR_{w}"] = tr.rolling(w, min_periods=mp).mean() / (close + EPS)

    # 短/中波动率比（regime-change 代理） + 跨窗口动量差
    out[f"ENG_VOL_RATIO_{short_w}_{med_w}"] = (
        out[f"ENG_RV_{short_w}"] / (out[f"ENG_RV_{med_w}"] + EPS)
    )
    out[f"ENG_RET_DIFF_{short_w}_{med_w}"] = (
        out[f"ENG_RET_{short_w}"] - out[f"ENG_RET_{med_w}"]
    )

    # 突破 / 跌破触发（用 .shift(1) 排除当根 high/low，避免自指）
    mp_med = _min_periods(med_w)
    out[f"ENG_PRICE_BREAKOUT_{med_w}"] = close / (
        high.shift(1).rolling(med_w, min_periods=mp_med).max() + EPS
    ) - 1.0
    out[f"ENG_PRICE_BREAKDOWN_{med_w}"] = close / (
        low.shift(1).rolling(med_w, min_periods=mp_med).min() + EPS
    ) - 1.0


# ════════════════════════════════════════════════════════════════════
# E. 时间编码  —— 仅依赖 timestamp，无前视风险
# ════════════════════════════════════════════════════════════════════

def _add_time_encoding(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    ts = ctx["ts"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["ENG_TOD_SIN"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    out["ENG_TOD_COS"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    out["ENG_WEEKDAY"] = ts.dt.dayofweek.astype("float32")
    out["ENG_IS_DAY_SESSION"] = ts.dt.hour.between(9, 15).astype("float32")


# ════════════════════════════════════════════════════════════════════
# G. 长窗口动量 / 波动率制度  —— 120 / 240 / 480 bar
# ════════════════════════════════════════════════════════════════════

def _add_long_window_features(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, ret1 = ctx["close"], ctx["ret1"]
    volume = ctx["volume"]

    for w in (120, 240, 480):
        mp = _min_periods(w)
        ma = close.rolling(w, min_periods=mp).mean()
        out[f"ENG_RET_{w}"] = close.pct_change(w)
        out[f"ENG_RV_{w}"] = ret1.rolling(w, min_periods=mp).std()
        out[f"ENG_PRICE_TO_MA_{w}"] = close / (ma + EPS) - 1.0
        out[f"ENG_VOLUME_RATIO_{w}"] = volume / (volume.rolling(w, min_periods=mp).mean() + EPS)

    # 多窗口收益符号一致度：[5,20,60,120,240] 中已存在的窗口取符号均值 ∈ [-1,1]
    cons_windows = [w for w in (5, 20, 60, 120, 240) if f"ENG_RET_{w}" in out.columns]
    if len(cons_windows) >= 2:
        signs = pd.concat([np.sign(out[f"ENG_RET_{w}"]) for w in cons_windows], axis=1)
        out["ENG_MOMENTUM_CONSISTENCY"] = signs.mean(axis=1)


# ════════════════════════════════════════════════════════════════════
# F. 隔夜 + 日内运行收益  —— 依赖 yesterday 的日终 close (.shift(1))
# ════════════════════════════════════════════════════════════════════

def _add_overnight_intraday(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, open_ = ctx["close"], ctx["open_"]
    tdate = ctx["tdate"]

    # 每日最后一根 close → 昨日日终收盘 (shift(1) 保证只用 ≤ d-1 的信息)
    daily_last_close = out.groupby(tdate)["CLOSE"].last().shift(1)
    prev_close_mapped = tdate.map(daily_last_close)
    is_first_bar = out.groupby(tdate).cumcount() == 0
    day_open = out.groupby(tdate)["OPEN"].transform("first")

    # 隔夜跳空：仅在每日第一根 bar 才有非零值（非首根填 0），用昨日日终 close
    out["ENG_OVERNIGHT_GAP"] = np.where(
        is_first_bar, open_ / (prev_close_mapped + EPS) - 1.0, 0.0
    )
    out["ENG_OVERNIGHT_GAP_ABS"] = np.abs(out["ENG_OVERNIGHT_GAP"])
    # 日内累计简单收益（相对今日开盘）
    out["ENG_INTRADAY_RET"] = (close - day_open) / (day_open + EPS)
    # 把 prev_close_mapped 留到 ctx 给后续 F.A1 / F.A3 / N 复用
    ctx["prev_close_mapped"] = prev_close_mapped
    ctx["day_open_per_bar"] = day_open
    ctx["is_first_bar"] = is_first_bar


# ════════════════════════════════════════════════════════════════════
# H. VWAP 偏离 / 区间位置 / Z-score / 量价加权累积  —— 滚动窗口
# ════════════════════════════════════════════════════════════════════

def _add_vwap_zscore_signed(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, high, low = ctx["close"], ctx["high"], ctx["low"]
    open_ = ctx["open_"]
    volume, amount = ctx["volume"], ctx["amount"]
    ret1, high_low_range = ctx["ret1"], ctx["high_low_range"]

    # 滚动 VWAP 偏离 + 价格在 [rolling_low, rolling_high] 中的位置
    for w in (20, 60):
        mp = _min_periods(w)
        rvwap = amount.rolling(w, min_periods=mp).sum() / (
            volume.rolling(w, min_periods=mp).sum() + EPS
        )
        out[f"ENG_VWAP_DEV_{w}"] = close / (rvwap + EPS) - 1.0
        lo_w = low.rolling(w, min_periods=mp).min()
        hi_w = high.rolling(w, min_periods=mp).max()
        out[f"ENG_PRICE_POSITION_{w}"] = (close - lo_w) / (hi_w - lo_w + EPS)

    # 中长窗口 close Z-score
    for w in (60, 120):
        mp = _min_periods(w)
        rm = close.rolling(w, min_periods=mp).mean()
        rs = close.rolling(w, min_periods=mp).std()
        out[f"ENG_CLOSE_ZSCORE_{w}"] = (close - rm) / (rs + EPS)

    # 量加权 1-bar 收益符号累积（盘中买卖力量倾向）
    ret_sign = np.sign(ret1)
    for w in (10, 30, 60):
        mp = _min_periods(w)
        out[f"ENG_VOLUME_PRICE_TREND_{w}"] = (
            (ret_sign * volume).rolling(w, min_periods=mp).sum()
            / (volume.rolling(w, min_periods=mp).sum() + EPS)
        )

    # 买入压力：close-to-high 比例 × 成交量，再窗口内除以总成交量
    for w in (5, 20):
        mp = _min_periods(w)
        bp = (close - low) / (high_low_range + EPS) * volume
        out[f"ENG_BUYING_PRESSURE_{w}"] = (
            bp.rolling(w, min_periods=mp).sum()
            / (volume.rolling(w, min_periods=mp).sum() + EPS)
        )

    # Parkinson 高低估计量（年化 √240）—— 仅依赖当根 H/L
    log_hl_sq = np.log(high / (low + EPS)) ** 2
    for w in (20, 60):
        mp = _min_periods(w)
        out[f"ENG_PARKINSON_VOL_{w}"] = (
            np.sqrt(log_hl_sq.rolling(w, min_periods=mp).mean() / (4.0 * np.log(2)))
            * np.sqrt(240)
        )

    # tick 方向（sign(close-open)）量加权
    tick_dir = np.sign(close - open_)
    for w in (20, 60):
        mp = _min_periods(w)
        out[f"ENG_TICK_DIR_VW_{w}"] = (
            (tick_dir * volume).rolling(w, min_periods=mp).sum()
            / (volume.rolling(w, min_periods=mp).sum() + EPS)
        )

    # 波动率 trend / Z-score（依赖 runtime 因子 STD20 / STD60；缺失则跳过）
    if "STD20" in out.columns and "STD60" in out.columns:
        out["ENG_VOL_TREND"] = out["STD20"].astype("float64") / (
            out["STD60"].astype("float64") + EPS
        )
    if "STD20" in out.columns:
        std20 = out["STD20"].astype("float64")
        out["ENG_VOL_ZSCORE"] = (
            (std20 - std20.rolling(120, min_periods=40).mean())
            / (std20.rolling(120, min_periods=40).std() + EPS)
        )


# ════════════════════════════════════════════════════════════════════
# L. 持仓量 (OI = POSITION) 因子  —— pct_change，方向确认 / 背离
# ════════════════════════════════════════════════════════════════════

def _add_oi_features(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, position = ctx["close"], ctx["position"]
    for w in (5, 20):
        oi_chg = position.pct_change(w)
        ret_w = close.pct_change(w)
        out[f"ENG_OI_CHANGE_{w}"] = oi_chg
        # OI 与价格同向变动 → 趋势确认
        out[f"ENG_OI_PRICE_CONFIRM_{w}"] = (np.sign(oi_chg) == np.sign(ret_w)).astype("float32")
        # OI 与价格背离（量增价跌 / 量缩价涨）
        out[f"ENG_OI_DIVERGE_{w}"] = (
            ((oi_chg > 0) & (ret_w < 0)) | ((oi_chg < 0) & (ret_w > 0))
        ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# M. 合成 / 交互 / 效率比 / 极值距离  —— 主要为非线性条件式特征
# ════════════════════════════════════════════════════════════════════

def _add_synthetic_interactions(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, high, low = ctx["close"], ctx["high"], ctx["low"]
    open_ = ctx["open_"]
    volume = ctx["volume"]
    ret1, tr = ctx["ret1"], ctx["true_range"]
    high_low_range = ctx["high_low_range"]
    tdate = ctx["tdate"]

    # S1) 波动率归一化动量（minute 级 Sharpe like）
    for w in (5, 20, 60, 120):
        mp = _min_periods(w)
        ret_w = close.pct_change(w)
        rv_w = ret1.rolling(w, min_periods=mp).std()
        out[f"ENG_SHARPE_{w}"] = ret_w / (rv_w * np.sqrt(w) + EPS)

    # S2) 动量加速度：当前 w-bar 收益 - w-bar 前的 w-bar 收益
    for w in (20, 60):
        ret_now = close.pct_change(w)
        ret_lag = close.pct_change(w).shift(w)
        out[f"ENG_MOM_ACCEL_{w}"] = ret_now - ret_lag

    # S3) 动量 × 波动率制度（高低波下动量含义不同）
    if "STD20" in out.columns and "STD60" in out.columns:
        vol_regime = out["STD20"].astype("float64") / (out["STD60"].astype("float64") + EPS)
        for w in (20, 60):
            ret_w = close.pct_change(w)
            out[f"ENG_TREND_X_VOLREG_{w}"] = ret_w * vol_regime

    # S4) 区间扩张（瞬时振幅相对历史均值的偏离）
    for w in (20, 60):
        mp = _min_periods(w)
        hl = high_low_range
        hl_ma = hl.rolling(w, min_periods=mp).mean()
        out[f"ENG_RANGE_EXPAND_{w}"] = hl / (hl_ma + EPS) - 1.0

    # S5) 量价分歧：|vol_z| - |ret_z|（量异常但价格未跟上 → 潜在反转）
    for w in (20, 60):
        mp = _min_periods(w)
        vol_z = (volume - volume.rolling(w, min_periods=mp).mean()) / (
            volume.rolling(w, min_periods=mp).std() + EPS
        )
        ret_z = close.pct_change(w) / (
            ret1.rolling(w, min_periods=mp).std() * np.sqrt(w) + EPS
        )
        out[f"ENG_VOL_PRICE_DIVERGE_{w}"] = vol_z.abs() - ret_z.abs()

    # S6) Kaufman 效率比：直线位移 / 路径长度
    for w in (20, 60):
        mp = _min_periods(w)
        displacement = (close - close.shift(w)).abs()
        path = ret1.abs().rolling(w, min_periods=mp).sum() * close.shift(w).abs()
        out[f"ENG_EFFICIENCY_{w}"] = displacement / (path + EPS)

    # S7) 跨窗口分歧：sign(ret_short) × sign(ret_long)
    if all(f"ENG_RET_{w}" in out.columns for w in (5, 60)):
        out["ENG_DIVERGE_5_60"] = np.sign(out["ENG_RET_5"]) * np.sign(out["ENG_RET_60"])
    if all(f"ENG_RET_{w}" in out.columns for w in (20, 240)):
        out["ENG_DIVERGE_20_240"] = np.sign(out["ENG_RET_20"]) * np.sign(out["ENG_RET_240"])

    # S8) 距日内极值距离 / 日内区间位置（cummax / cummin 是因果累积）
    day_high = out.groupby(tdate)["HIGH"].cummax()
    day_low = out.groupby(tdate)["LOW"].cummin()
    atr_proxy = tr.rolling(20, min_periods=6).mean() + EPS
    out["ENG_DIST_DAY_HIGH"] = (close - day_high) / atr_proxy   # ≤ 0
    out["ENG_DIST_DAY_LOW"] = (close - day_low) / atr_proxy     # ≥ 0
    out["ENG_DAY_RANGE_POS"] = (close - day_low) / (day_high - day_low + EPS)
    ctx["day_high"] = day_high
    ctx["day_low"] = day_low

    # S9) 日内累计收益 × 20-bar 动量（方向 + 位置共振）
    if "ENG_INTRADAY_RET" in out.columns:
        out["ENG_INTRADAY_RET_X_RET20"] = out["ENG_INTRADAY_RET"] * close.pct_change(20)


# ════════════════════════════════════════════════════════════════════
# F.A 分钟级日内运行收益与速度  —— 替代旧 daily_ret 的"每根 bar 都更新"特征
# 历史教训：旧 daily_ret = log(today_close / yesterday_close) 广播到当日所有 bar，
# 9:00 的 bar 已含 15:00 的 close，是典型前视泄露 (top-1 importance)。
# 此处所有特征均"仅使用过去 + 当前 bar"，模型可学到等价信号但完全因果。
# ════════════════════════════════════════════════════════════════════

def _add_minute_level_running(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, ret1, high, low = ctx["close"], ctx["ret1"], ctx["high"], ctx["low"]
    volume = ctx["volume"]
    tdate, ts = ctx["tdate"], ctx["ts"]
    prev_close_mapped = ctx["prev_close_mapped"]
    day_open_per_bar = ctx["day_open_per_bar"]

    # F.A1) 当前 close 相对昨日日终 close 的对数收益（含跨夜跳空 + 今日已发生）
    out["ENG_PREV_CLOSE_LOG_RET"] = np.log(close / (prev_close_mapped + EPS)).astype("float32")

    # F.A2) 当日截至 bar t 的对数累计收益（相对今日开盘）
    out["ENG_INTRADAY_LOG_RET_CUMSUM"] = np.log(close / (day_open_per_bar + EPS)).astype("float32")

    # F.A3) 跨日延续 × 日内行进 的交互
    out["ENG_PREV_CLOSE_LOG_RET_X_INTRADAY"] = (
        out["ENG_PREV_CLOSE_LOG_RET"].astype("float64")
        * out["ENG_INTRADAY_RET"].astype("float64")
    ).astype("float32")

    # F.A4) "我相对昨收的位置"在分钟级上的短期变化（动量速度）
    for lookback in (5, 15, 30):
        out[f"ENG_PREV_CLOSE_LOG_RET_DELTA_{lookback}"] = (
            out["ENG_PREV_CLOSE_LOG_RET"].astype("float64")
            - out["ENG_PREV_CLOSE_LOG_RET"].shift(lookback).astype("float64")
        ).astype("float32")

    # F.A5) 日内 bar 进度 ∈ [0, 1]
    # ⚠️ 防前视：分母改用「昨日 bar 数」而非 groupby(today).transform("count")。
    # 后者会把当日总 bar 数（含未来 bar）暴露给当日早盘 bar，构成软前视。
    # 用昨日总 bar 数（中位 fallback）即可保证因果，同时几乎不影响数值（期货
    # 交易时段固定，相邻日 bar 数差异极小）。
    bar_idx_in_day = out.groupby(tdate).cumcount()
    total_per_day = out.groupby(tdate)[ts.name].count()
    prev_day_count = total_per_day.shift(1)
    median_count = float(total_per_day.median()) if len(total_per_day) else 1.0
    prev_day_count = prev_day_count.fillna(median_count)
    prev_day_count_mapped = tdate.map(prev_day_count).astype("float64")
    out["ENG_INTRADAY_BAR_PROGRESS"] = (
        bar_idx_in_day.astype("float64") / (prev_day_count_mapped + EPS)
    ).astype("float32")

    # F.A6) 分钟级近 N 根 1-bar 收益 std + 短/长比
    for w in (10, 20, 60):
        mp = _min_periods(w)
        out[f"ENG_INTRADAY_RV_{w}"] = ret1.rolling(w, min_periods=mp).std().astype("float32")
    out["ENG_INTRADAY_RV_RATIO_10_60"] = (
        out["ENG_INTRADAY_RV_10"].astype("float64")
        / (out["ENG_INTRADAY_RV_60"].astype("float64") + EPS)
    ).astype("float32")

    # F.A7) 日内 expanding 正收益占比（盘中"上行 bar 比例"，∈ [0, 1]）
    ret_pos_flag = (ret1 > 0).astype("float32")
    out["ENG_INTRADAY_POS_RATIO_CUM"] = (
        ret_pos_flag.groupby(tdate).expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True).astype("float32")
    )

    # F.A8) 日内 expanding 累计 tick 方向均值 (sign(ret1)) ∈ [-1, 1]
    tick_sign = np.sign(ret1).astype("float32")
    out["ENG_INTRADAY_TICK_DIR_CUM"] = (
        tick_sign.groupby(tdate).expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True).astype("float32")
    )

    # F.A9) 当前 close 在近 N 根 [low, high] 区间的位置（分钟级 RSV 视角）
    for w in (30, 60):
        mp = _min_periods(w)
        rolling_high = high.rolling(w, min_periods=mp).max()
        rolling_low = low.rolling(w, min_periods=mp).min()
        out[f"ENG_INTRADAY_RANGE_POS_{w}"] = (
            (close - rolling_low) / (rolling_high - rolling_low + EPS)
        ).astype("float32")

    # F.A10) 分钟级 1-bar 收益 60 根滚动偏度
    out["ENG_INTRADAY_RET1_SKEW_60"] = ret1.rolling(60, min_periods=20).skew().astype("float32")

    # F.A11) 当日截至 bar t 的成交量加权 1-bar 收益累计（VWAP 漂移指标）
    signed_vol_ret = (ret1 * volume).astype("float64")
    abs_vol_ret = (ret1.abs() * volume).astype("float64")
    out["ENG_INTRADAY_VOL_WEIGHTED_RET_CUM"] = (
        signed_vol_ret.groupby(tdate).expanding(min_periods=1).sum()
        / (abs_vol_ret.groupby(tdate).expanding(min_periods=1).sum() + EPS)
    ).reset_index(level=0, drop=True).astype("float32")


# ════════════════════════════════════════════════════════════════════
# N. 日级 lag 上下文（截止昨日）  —— daily_log_ret.shift(1) 后再 rolling
# ════════════════════════════════════════════════════════════════════

def _add_daily_lag_context(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    tdate = ctx["tdate"]

    # 日级 close 与日级对数收益（注：daily_log_ret[d] = log(close_d / close_{d-1})）
    daily_close = out.groupby(tdate)["CLOSE"].last()
    daily_log_ret = np.log(daily_close / daily_close.shift(1))

    def _bcast(daily: pd.Series) -> pd.Series:
        """日级 Series → bar 级 Series（每个 bar 取所属日的值）。"""
        return tdate.map(daily).astype("float32")

    # B1) 截止到昨日的 20d 日级波动率
    out["ENG_DAILY_VOL_20_LAG1"] = _bcast(
        daily_log_ret.shift(1).rolling(20, min_periods=10).std()
    )

    # B2) 昨日全天对数收益除以截止昨日 5d 日级波动率（标量化跨日动量强度）
    daily_rv_5_lag1 = daily_log_ret.shift(1).rolling(5, min_periods=3).std()
    out["ENG_DAILY_RET_LAG1_VOL_NORM"] = (
        _bcast(daily_log_ret.shift(1)).astype("float64")
        / (_bcast(daily_rv_5_lag1).astype("float64") + EPS)
    ).astype("float32")

    # B3) 截止到昨日 20d 累计对数收益（中期日级动量）
    out["ENG_DAILY_RET_20D_LAG1"] = _bcast(
        daily_log_ret.shift(1).rolling(20, min_periods=10).sum()
    )

    # H. 完整日级因子组（候选池；audit 后保留有用）
    out["ENG_DAILY_RET_LAG2"] = _bcast(daily_log_ret.shift(2))
    out["ENG_DAILY_RET_LAG3"] = _bcast(daily_log_ret.shift(3))
    out["ENG_DAILY_RET_LAG1_TO_5_MEAN"] = _bcast(
        daily_log_ret.shift(1).rolling(5, min_periods=3).mean()
    )
    out["ENG_DAILY_RET_3D_LAG1"] = _bcast(
        daily_log_ret.shift(1).rolling(3, min_periods=2).sum()
    )
    out["ENG_DAILY_RET_5D_LAG1"] = _bcast(
        daily_log_ret.shift(1).rolling(5, min_periods=3).sum()
    )
    out["ENG_DAILY_RET_10D_LAG1"] = _bcast(
        daily_log_ret.shift(1).rolling(10, min_periods=5).sum()
    )
    out["ENG_DAILY_RET_60D_LAG1"] = _bcast(
        daily_log_ret.shift(1).rolling(60, min_periods=20).sum()
    )
    out["ENG_DAILY_RV_5_LAG1"] = _bcast(daily_rv_5_lag1)
    daily_rv_60_lag1 = daily_log_ret.shift(1).rolling(60, min_periods=20).std()
    out["ENG_DAILY_RV_60_LAG1"] = _bcast(daily_rv_60_lag1)
    out["ENG_DAILY_VOL_RATIO_5_60_LAG1"] = (
        _bcast(daily_rv_5_lag1).astype("float64")
        / (_bcast(daily_rv_60_lag1).astype("float64") + EPS)
    ).astype("float32")

    # 昨日跳空 = 昨日开盘 / 前日收盘 - 1，再 .shift(1) 得"昨日跳空"
    daily_open = out.groupby(tdate)["OPEN"].first()
    daily_gap_lag1 = (daily_open / daily_close.shift(1) - 1.0).shift(1)
    out["ENG_DAILY_GAP_LAG1"] = _bcast(daily_gap_lag1)

    # D. 关键交互：分钟级"我相对昨收"的位置 × 日级 vol regime
    out["ENG_PREV_CLOSE_LOG_RET_X_DAILY_VOL"] = (
        out["ENG_PREV_CLOSE_LOG_RET"].astype("float64")
        * out["ENG_DAILY_VOL_20_LAG1"].astype("float64")
    ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# K. 半方差比 + 连胜计数
# ════════════════════════════════════════════════════════════════════

def _add_semivar_consec(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    ret1 = ctx["ret1"]
    pos_ret = ret1.clip(lower=0.0).astype("float64")
    neg_ret = (-ret1.clip(upper=0.0)).abs().astype("float64")

    # 上 / 下行半方差对数比：>0 偏多动量，<0 偏空动量
    for w in (30, 60, 120):
        mp = _min_periods(w)
        up_var = (pos_ret ** 2).rolling(w, min_periods=mp).sum()
        dn_var = (neg_ret ** 2).rolling(w, min_periods=mp).sum()
        out[f"ENG_SEMIVAR_LOG_RATIO_{w}"] = np.log(
            (up_var + EPS) / (dn_var + EPS)
        ).astype("float32")

    # 连续同向 bar 计数（截断 ±10）：当前价格趋势的"持续度"
    sign_r = np.sign(ret1).astype("float32")
    change_pt = (sign_r != sign_r.shift(1)).cumsum()
    consec = sign_r.groupby(change_pt).cumcount() + 1
    out["ENG_CONSECUTIVE_DIR_COUNT"] = (
        (consec.astype("float64") * sign_r.astype("float64")).clip(-10, 10)
    ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# G2-G5. 复合 alpha 信号（多源动量一致性 + 信号质量 + 突破触发）
# ════════════════════════════════════════════════════════════════════

def _add_composite_alpha(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, high, low = ctx["close"], ctx["high"], ctx["low"]
    tr = ctx["true_range"]

    # G1) 多窗口动量符号一致度 ∈ [-1, 1]
    signs = pd.concat(
        [np.sign(close.pct_change(w)) for w in (5, 20, 60)], axis=1
    )
    out["ENG_MULTI_HORIZON_SIGN_AGREE"] = signs.mean(axis=1).astype("float32")

    # G2) 动量 × 一致度（仅在多窗口共振时给出强信号）
    if "ENG_RET_20" in out.columns:
        out["ENG_RET20_X_SIGN_AGREE"] = (
            out["ENG_RET_20"].astype("float64")
            * out["ENG_MULTI_HORIZON_SIGN_AGREE"].astype("float64")
        ).astype("float32")

    # G3) 跨日 + 日内方向一致：sign(prev_close) × sign(intraday_cumsum)
    out["ENG_PREV_CLOSE_X_INTRADAY_SIGN_AGREE"] = (
        np.sign(out["ENG_PREV_CLOSE_LOG_RET"].astype("float64"))
        * np.sign(out["ENG_INTRADAY_LOG_RET_CUMSUM"].astype("float64"))
    ).astype("float32")

    # G4) 信号质量（minute 级 Sharpe）
    out["ENG_PREV_CLOSE_LOG_RET_SHARPE_60"] = (
        out["ENG_PREV_CLOSE_LOG_RET"].astype("float64")
        / (out["ENG_INTRADAY_RV_60"].astype("float64") * np.sqrt(60) + EPS)
    ).astype("float32")

    # G5) 突破 / 回落触发：
    #   close > 60-bar 历史高 → (close - 高) / ATR 60 > 0
    #   close < 60-bar 历史低 → (close - 低) / ATR 60 < 0
    #   横盘期接近 0；ATR 60 = true_range 60 滚动均值
    hi60 = high.rolling(60, min_periods=20).max()
    lo60 = low.rolling(60, min_periods=20).min()
    atr60 = tr.rolling(60, min_periods=20).mean()
    out["ENG_BREAKOUT_60"] = np.where(
        close > hi60.shift(1),
        (close - hi60.shift(1)) / (atr60 + EPS),
        np.where(
            close < lo60.shift(1),
            (close - lo60.shift(1)) / (atr60 + EPS),
            0.0,
        ),
    ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# J. 跨日同分钟历史先验  —— 在 minute 组内 shift(1) 后 rolling
# ════════════════════════════════════════════════════════════════════

def _hist_same_minute_stat(
    series: pd.Series, minute_key: pd.Series, window: int, fn: str
) -> pd.Series:
    """在 minute 组内：先 shift(1) 排除"今日同分钟自身"，再 rolling N 个历史样本。
    fn ∈ {"mean", "std"}。所有统计严格使用 ≤ 昨日同分钟的数据，无前视。"""
    shifted = series.groupby(minute_key).shift(1)
    if fn == "mean":
        return shifted.groupby(minute_key).transform(
            lambda x: x.rolling(window, min_periods=_min_periods(window)).mean()
        )
    if fn == "std":
        return shifted.groupby(minute_key).transform(
            lambda x: x.rolling(window, min_periods=_min_periods(window)).std()
        )
    raise ValueError(fn)


def _add_same_minute_history(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    ts = ctx["ts"]
    ret1 = ctx["ret1"]
    volume = ctx["volume"]
    minute_key = ts.dt.hour * 60 + ts.dt.minute

    # C1) 今日累计日内收益 vs 历史 30 天同分钟累计的标准化偏离（实测稳健 top-15）
    intra_cumret = out["ENG_INTRADAY_LOG_RET_CUMSUM"].astype("float64")
    typ_intra_mean = _hist_same_minute_stat(intra_cumret, minute_key, 30, "mean")
    typ_intra_std = _hist_same_minute_stat(intra_cumret, minute_key, 30, "std")
    out["ENG_TYPICAL_INTRADAY_DEV_30"] = (
        (intra_cumret - typ_intra_mean) / (typ_intra_std + EPS)
    ).astype("float32")

    # I1) "我相对昨收"的同分钟 Z-score
    prev_close_64 = out["ENG_PREV_CLOSE_LOG_RET"].astype("float64")
    typ_pcr_mean = _hist_same_minute_stat(prev_close_64, minute_key, 30, "mean")
    typ_pcr_std = _hist_same_minute_stat(prev_close_64, minute_key, 30, "std")
    out["ENG_TYPICAL_PREV_CLOSE_DEV_30"] = (
        (prev_close_64 - typ_pcr_mean) / (typ_pcr_std + EPS)
    ).astype("float32")

    # I2) 1-bar 收益的同分钟 Z-score + 历史均值（先验方向）
    ret1_64 = ret1.astype("float64")
    typ_r1_mean_60 = _hist_same_minute_stat(ret1_64, minute_key, 60, "mean")
    typ_r1_std_60 = _hist_same_minute_stat(ret1_64, minute_key, 60, "std")
    out["ENG_TYPICAL_RET1_Z_60"] = (
        (ret1_64 - typ_r1_mean_60) / (typ_r1_std_60 + EPS)
    ).astype("float32")
    out["ENG_HIST_SAME_MIN_RET_MEAN_60"] = typ_r1_mean_60.astype("float32")

    # I3) 成交量 log 同分钟偏离 + Z-score
    vol_log = np.log1p(volume.astype("float64"))
    typ_vol_mean = _hist_same_minute_stat(vol_log, minute_key, 30, "mean")
    typ_vol_std = _hist_same_minute_stat(vol_log, minute_key, 30, "std")
    out["ENG_TYPICAL_VOLUME_LOG_DEV_30"] = (vol_log - typ_vol_mean).astype("float32")
    out["ENG_TYPICAL_VOLUME_LOG_Z_30"] = (
        (vol_log - typ_vol_mean) / (typ_vol_std + EPS)
    ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# J2. 候选显式交互（audit 后裁剪保留）
# ════════════════════════════════════════════════════════════════════

def _add_candidate_interactions(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    out["ENG_INTRADAY_PROGRESS_X_CUMRET"] = (
        out["ENG_INTRADAY_BAR_PROGRESS"].astype("float64")
        * out["ENG_INTRADAY_LOG_RET_CUMSUM"].astype("float64")
    ).astype("float32")

    if "ENG_RET_5" in out.columns:
        out["ENG_RET5_OVER_RV60"] = (
            out["ENG_RET_5"].astype("float64")
            / (out["ENG_INTRADAY_RV_60"].astype("float64") * np.sqrt(5) + EPS)
        ).astype("float32")

    if "ENG_RET_20" in out.columns:
        out["ENG_RET20_OVER_RV60"] = (
            out["ENG_RET_20"].astype("float64")
            / (out["ENG_INTRADAY_RV_60"].astype("float64") * np.sqrt(20) + EPS)
        ).astype("float32")

    if "ENG_DAY_RANGE_POS" in out.columns:
        out["ENG_DAY_RANGE_POS_X_CUMRET_SIGN"] = (
            out["ENG_DAY_RANGE_POS"].astype("float64")
            * np.sign(out["ENG_INTRADAY_LOG_RET_CUMSUM"].astype("float64"))
        ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# O. 因子融合（同类簇内取均值 / 几何均值）
#    思想：把高度相关的同类因子合成为单一指标，降低噪声、突出共同信号。
#    同量纲组用算术均值，正值组用几何均值（对数空间求平均更稳健）。
# ════════════════════════════════════════════════════════════════════

def _add_factor_fusion(out: pd.DataFrame, ctx: dict[str, Any]) -> None:
    # K1) 日级历史动量融合（同量纲）
    daily_mom_components = [
        "ENG_DAILY_RET_LAG2", "ENG_DAILY_RET_LAG3",
        "ENG_DAILY_RET_LAG1_TO_5_MEAN",
        "ENG_DAILY_RET_3D_LAG1", "ENG_DAILY_RET_5D_LAG1",
    ]
    existing = [c for c in daily_mom_components if c in out.columns]
    if existing:
        out["ENG_FUSED_DAILY_MOMENTUM"] = (
            out[existing].astype("float64").mean(axis=1)
        ).astype("float32")

    # K2) 日级波动率融合（≥0 → 几何均值）
    daily_vol_components = ["ENG_DAILY_RV_5_LAG1", "ENG_DAILY_VOL_20_LAG1", "ENG_DAILY_RV_60_LAG1"]
    existing = [c for c in daily_vol_components if c in out.columns]
    if existing:
        log_vols = pd.concat(
            [np.log(out[c].astype("float64") + EPS) for c in existing], axis=1
        )
        out["ENG_FUSED_DAILY_VOL"] = np.exp(log_vols.mean(axis=1)).astype("float32")

    # K3) 分钟级运行收益融合（同量纲算术均值）
    intraday_run_components = ["ENG_PREV_CLOSE_LOG_RET", "ENG_INTRADAY_LOG_RET_CUMSUM"]
    existing = [c for c in intraday_run_components if c in out.columns]
    if len(existing) >= 2:
        out["ENG_FUSED_INTRADAY_RUN_RET"] = (
            out[existing].astype("float64").mean(axis=1)
        ).astype("float32")

    # K4) 分钟级方向倾向融合：tick_dir 与 (pos_ratio*2-1) 同尺度后取均值
    if (
        "ENG_INTRADAY_TICK_DIR_CUM" in out.columns
        and "ENG_INTRADAY_POS_RATIO_CUM" in out.columns
    ):
        pos_rescaled = out["ENG_INTRADAY_POS_RATIO_CUM"].astype("float64") * 2.0 - 1.0
        tick_dir = out["ENG_INTRADAY_TICK_DIR_CUM"].astype("float64")
        out["ENG_FUSED_INTRADAY_DIRECTION"] = (
            (pos_rescaled + tick_dir) / 2.0
        ).astype("float32")

    # K5) 分钟级波动率融合（几何均值）
    intra_vol_components = ["ENG_INTRADAY_RV_10", "ENG_INTRADAY_RV_20", "ENG_INTRADAY_RV_60"]
    existing = [c for c in intra_vol_components if c in out.columns]
    if existing:
        log_intvols = pd.concat(
            [np.log(out[c].astype("float64") + EPS) for c in existing], axis=1
        )
        out["ENG_FUSED_INTRADAY_VOL"] = np.exp(log_intvols.mean(axis=1)).astype("float32")

    # K6) 跨日 + 日内 同向一致性融合（实数加权版）
    if (
        "ENG_PREV_CLOSE_LOG_RET" in out.columns
        and "ENG_INTRADAY_LOG_RET_CUMSUM" in out.columns
    ):
        out["ENG_FUSED_PREV_CLOSE_X_INTRADAY"] = (
            out["ENG_PREV_CLOSE_LOG_RET"].astype("float64")
            * out["ENG_INTRADAY_LOG_RET_CUMSUM"].astype("float64")
        ).astype("float32")

    # K7) "信号 / 噪声"融合：分钟级运行收益 / 分钟级波动率 → Sharpe-like
    if (
        "ENG_FUSED_INTRADAY_RUN_RET" in out.columns
        and "ENG_FUSED_INTRADAY_VOL" in out.columns
    ):
        out["ENG_FUSED_INTRADAY_SHARPE"] = (
            out["ENG_FUSED_INTRADAY_RUN_RET"].astype("float64")
            / (out["ENG_FUSED_INTRADAY_VOL"].astype("float64") + EPS)
        ).astype("float32")


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════

def add_engineered_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    trade_date_col: str,
    engineered_windows: dict[str, Any],
    use_engineered_features: bool = True,
    use_synthetic_factors: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """计算所有 ENG_* 工程化特征。

    参数
    ----
    df : 已合并 runtime 量价因子（如 STD20）+ 中观因子之后的 DataFrame，
        必须包含列 OPEN/HIGH/LOW/CLOSE 及 timestamp_col/trade_date_col；
        VOLUME/AMOUNT/POSITION 缺失会回退为 0。
    timestamp_col : 1 分钟时间戳列名（如 "TDATE"）
    trade_date_col : 交易日列（如 "TRADE_DATE"）
    engineered_windows : {"short": int, "medium": int, "long": int}
    use_engineered_features : False 时直接返回原 df + 空列表（消融用）
    use_synthetic_factors : False 时只保留 _ORIGINAL_ENG_COLS 中的 35 项基础因子

    返回
    ----
    (engineered_df, eng_cols)
        engineered_df : 包含全部 ENG_* 列的副本，按 timestamp 升序
        eng_cols : 所有以 "ENG_" 开头的列名列表

    防前视：见模块 docstring。
    """
    if not use_engineered_features:
        return df, []

    out = df.copy().sort_values(timestamp_col).reset_index(drop=True)

    # 统一类型 + 派生中间量
    close = out["CLOSE"].astype("float64")
    open_ = out["OPEN"].astype("float64")
    high = out["HIGH"].astype("float64")
    low = out["LOW"].astype("float64")
    volume = _safe_col(out, "VOLUME")
    position = _safe_col(out, "POSITION")
    amount = _safe_col(out, "AMOUNT")

    high_low_range = high - low
    true_range = pd.concat(
        [
            high_low_range,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    ret1 = close.pct_change()

    short_w = int(engineered_windows.get("short", 5))
    med_w = int(engineered_windows.get("medium", 20))
    long_w = int(engineered_windows.get("long", 60))

    # 共享 ctx，内部各 _add_* 子函数读写同一份字典
    ctx: dict[str, Any] = {
        "close": close, "open_": open_, "high": high, "low": low,
        "volume": volume, "position": position, "amount": amount,
        "ret1": ret1, "high_low_range": high_low_range, "true_range": true_range,
        "short_w": short_w, "med_w": med_w, "long_w": long_w,
        "ts": out[timestamp_col],
        "tdate": out[trade_date_col],
    }

    # 按设计顺序依次计算（部分组依赖前序组的中间产物，必须按此顺序）。
    # 每个 helper 通过 out["X"] = ... 逐列插入；累计后 BlockManager 高度碎片化，
    # 既触发 pandas PerformanceWarning，也拖慢后续列访问。每次 helper 调用后
    # 做一次 copy() 把同 dtype 的 block 合并，消除告警并让 astype/取列更快。
    helpers = [
        _add_single_bar_features,        # A
        _add_window_features,            # B + C + D 部分
        _add_time_encoding,              # E
        _add_long_window_features,       # G long-window + 一致度
        _add_overnight_intraday,         # F overnight + intraday
        _add_vwap_zscore_signed,         # H VWAP/Z/signed/Parkinson/tick
        _add_oi_features,                # L OI / POSITION
        _add_synthetic_interactions,     # M S1-S9 合成交互
        _add_minute_level_running,       # F.A 因果版 daily_ret 替代
        _add_daily_lag_context,          # N 日级 lag 上下文
        _add_semivar_consec,             # K 半方差比 + 连胜
        _add_composite_alpha,            # G2-G5 复合 alpha + 突破触发
        _add_same_minute_history,        # J 跨日同分钟历史先验
        _add_candidate_interactions,     # J2 候选交互
        _add_factor_fusion,              # O 因子融合
    ]
    for fn in helpers:
        fn(out, ctx)
        out = out.copy()

    # 消融实验：仅保留 35 项基础 ENG_*
    if not use_synthetic_factors:
        drop = [
            c for c in out.columns
            if c.startswith("ENG_") and c not in _ORIGINAL_ENG_COLS
        ]
        out = out.drop(columns=drop)

    eng_cols = [c for c in out.columns if c.startswith("ENG_")]
    out[eng_cols] = out[eng_cols].astype("float32")
    return out, eng_cols


# ════════════════════════════════════════════════════════════════════
# 中观因子（mid_weekly）的派生计算 + 与微观因子的显式交互
# ════════════════════════════════════════════════════════════════════
#
# 这两个函数都假设输入数据的时间戳已在 dataset.py::_read_mid_weekly_xlsx 阶段
# 加过 publication_lag（按 freq 平移），因此 rolling/pct_change 默认 backward
# 即可保证因果性、无前视。
# ════════════════════════════════════════════════════════════════════


def compute_mid_weekly_derivatives(
    factor_df: pd.DataFrame,
    level_cols: list[str],
    *,
    timestamp_col: str,
    windows: list[int],
    transforms: list[str],
    extreme_high_quantile: float = 0.85,
    extreme_low_quantile: float = 0.15,
) -> tuple[pd.DataFrame | None, list[str]]:
    """对稀疏的中观因子观测序列做 per-column 派生计算。

    支持的 transforms：
      - ret          : pct_change(w)，clip [-1, 5]，过去 w 期变化幅度
      - zscore       : (s - mean) / std，过去 w 期标准化偏离
      - pct_rank     : 当前值在过去 w 期的百分位排名 [0, 1]
      - accel        : pct_change(w) - pct_change(w).shift(w)，二阶差分（变化的变化），
                       捕捉拐点 / 加速；clip [-2, 2]
      - extreme_flag : pct_rank >= extreme_high_quantile → HIGH_FLAG=1；
                       pct_rank <= extreme_low_quantile → LOW_FLAG=1。
                       把"基本面极端区位"事件显式化为 0/1 dummy

    所有 transform 仅基于 sparse 观测的过去 + 当前点（rolling/pct_change/shift 默认
    backward），无前视；前提是上游 ts 已加过发布滞后。

    返回 (derived_df, derived_cols)；当 level_cols 为空 / 数据不足时返回 (None, [])。
    """
    if not level_cols:
        return None, []
    if not windows or not transforms:
        return None, []

    pieces: list[pd.DataFrame] = []
    derived_cols: list[str] = []
    for col in level_cols:
        s = factor_df.sort_values(timestamp_col).set_index(timestamp_col)[col].dropna()
        min_required = max(2, min(windows))
        if len(s) < min_required:
            continue
        per_col = pd.DataFrame(index=s.index)
        for w in windows:
            mp = max(1, w // 2)
            rank_w = None        # pct_rank 与 extreme_flag 共用
            pct_change_w = None  # ret 与 accel 共用

            if "ret" in transforms:
                name = f"{col}_RET_{w}"
                pct_change_w = s.pct_change(w)
                per_col[name] = pct_change_w.clip(-1.0, 5.0)
                derived_cols.append(name)
            if "zscore" in transforms:
                name = f"{col}_ZSCORE_{w}"
                mean = s.rolling(w, min_periods=mp).mean()
                std = s.rolling(w, min_periods=mp).std().replace(0, np.nan)
                per_col[name] = (s - mean) / std
                derived_cols.append(name)
            if "pct_rank" in transforms:
                name = f"{col}_PCT_RANK_{w}"
                rank_w = s.rolling(w, min_periods=mp).rank(pct=True)
                per_col[name] = rank_w
                derived_cols.append(name)
            if "accel" in transforms:
                # 二阶差分：本期 w-pct_change 与 w 期前 w-pct_change 的差，仅过去信息
                name = f"{col}_ACCEL_{w}"
                if pct_change_w is None:
                    pct_change_w = s.pct_change(w)
                per_col[name] = (pct_change_w - pct_change_w.shift(w)).clip(-2.0, 2.0)
                derived_cols.append(name)
            if "extreme_flag" in transforms:
                if rank_w is None:
                    rank_w = s.rolling(w, min_periods=mp).rank(pct=True)
                high_name = f"{col}_HIGH_FLAG_{w}"
                low_name = f"{col}_LOW_FLAG_{w}"
                per_col[high_name] = (rank_w >= extreme_high_quantile).astype("float32")
                per_col[low_name] = (rank_w <= extreme_low_quantile).astype("float32")
                derived_cols.append(high_name)
                derived_cols.append(low_name)
        pieces.append(per_col)
    if not pieces:
        return None, []
    derived = pd.concat(pieces, axis=1).sort_index()
    derived.index.name = timestamp_col
    derived = derived.astype("float32")
    return derived.reset_index(), derived_cols


def add_mid_micro_interactions(
    merged_df: pd.DataFrame,
    mid_cols: list[str],
    *,
    micro_factors: list[str],
    mid_selector: str = "pct_rank_only",
    max_columns: int = 100,
) -> tuple[pd.DataFrame, list[str]]:
    """A1.2: mid 状态变量 × 微观核心因子 显式交互，输出 MIDxMICRO_<micro>_X_<mid> 列。
    （命名前缀 MIDxMICRO_ 明示该列是"中观 × 微观"乘积；旧前缀 XINT_ 已废弃。）

    因果性：micro 列由 add_engineered_features 计算（仅过去+当前 bar），mid 列在
    上游已加过 publication_lag；逐 bar 算术乘积不会跨 bar 引入额外信息 → 因果。

    参数
    ----
    merged_df : 已经合并 micro + mid 派生因子之后的 DataFrame（mid_cols 与 micro
        列必须都已存在）。
    mid_cols : 候选 mid 列名（通常是 _merge_mid_weekly_features 返回的所有 MID_* 列）
    micro_factors : 用户配置的微观核心因子列名列表（见 config）
    mid_selector : 选 mid 子集的策略
        - pct_rank_only         : 只选 PCT_RANK_* 列（量级 [0,1]，与 micro 乘积稳定）
        - pct_rank_and_extreme  : 加上 HIGH_FLAG / LOW_FLAG
        - all                   : 不过滤
    max_columns : 维度上限，超过则按 (micro 优先, 同 micro 内不同 mid) 顺序截断

    返回 (df_with_xint, xint_col_names)。如果没有可用列对，返回 (df, [])。
    """
    if not mid_cols or not micro_factors:
        return merged_df, []

    available_micro = [c for c in micro_factors if c in merged_df.columns]
    if not available_micro:
        return merged_df, []

    sel = str(mid_selector or "pct_rank_only").lower()
    if sel == "pct_rank_only":
        target_mid = [c for c in mid_cols if "_PCT_RANK_" in c]
    elif sel == "pct_rank_and_extreme":
        target_mid = [
            c for c in mid_cols
            if "_PCT_RANK_" in c or "_HIGH_FLAG_" in c or "_LOW_FLAG_" in c
        ]
    else:  # "all"
        target_mid = list(mid_cols)
    target_mid = [c for c in target_mid if c in merged_df.columns]
    if not target_mid:
        return merged_df, []

    pairs = [(m, n) for m in available_micro for n in target_mid][:int(max_columns)]
    if not pairs:
        return merged_df, []

    # 一次性 concat 避免多次 assign 触发 PerformanceWarning
    new_cols_data: dict[str, pd.Series] = {}
    for micro, mid in pairs:
        col_name = f"MIDxMICRO_{micro}_X_{mid}"
        new_cols_data[col_name] = (
            merged_df[micro].astype("float64") * merged_df[mid].astype("float64")
        ).astype("float32")
    new_block = pd.DataFrame(new_cols_data, index=merged_df.index)
    merged_df = pd.concat([merged_df, new_block], axis=1)
    new_cols = list(new_cols_data.keys())
    print(
        f"[mid_micro] generated {len(new_cols)} MIDxMICRO_* cols "
        f"({len(available_micro)} micro × {len(target_mid)} mid, capped at {max_columns})"
    )
    return merged_df, new_cols
