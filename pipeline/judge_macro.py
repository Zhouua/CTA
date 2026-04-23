"""
judge_macro.py
==============
月度宏观 regime 判断模块，与原有回测逻辑完全隔离。

对外接口：
    build_monthly_regime(macro_csv_path, lag_months=1) -> pd.DataFrame
        返回以月初日期为索引的 DataFrame，包含：
            is_macro_strong       : bool  - 宏观周期是否强
            is_inventory_strong   : bool  - 库存周期是否强
            is_demand_strong      : bool  - 工业需求是否强
            open_position         : bool  - 任一条件为 True 则允许开仓

判断逻辑（简洁、可读）：
    宏观周期强  = PPI同比 > 0  AND  制造业PMI >= 49.5
    即 ths_PPI_当月同比 > 0 AND nbs_制造业采购经理指数_pct >= 49.5
                   （PPI正区间说明商品价格处于通胀环境，是商品CTA最直接的宏观驱动因子；
                     PMI未跌入明显收缩区间提供景气度确认）

    库存周期强  = PMI新订单 > 50.0  AND  PMI产成品库存 < 50.0
    即 nbs_新订单指数_pct > 50 AND nbs_产成品库存指数_pct < 50
                   （订单扩张同时产成品库存偏低，是主动补库的双重确认信号，
                     比单纯新订单>50更能识别真正的补库周期，减少噪声）

    工业需求强  = 工业增加值同比 > 5.5  AND  固定资产投资累计增长 > 3.5
    即 ths_规模以上工业增加值_当月同比 > 5.5 AND nbs_固定资产投资额累计增长_pct > 3.5
                   （工业增加值同比反映实体生产强度，固投累计增速反映中游制造业、
                     基建等对黑色和有色链条的真实吸纳需求。两者同时不弱，
                     说明商品需求并非只靠价格反弹或库存博弈驱动，而是有实物工作量支撑）

阈值选择依据：
    - PMI >= 49.5 比 50 更宽容，适应 PMI 公布存在小幅噪声的情况
    - PPI > 0 是商品价格通胀/通缩的分界线，通胀环境下趋势策略成功率更高
    - 新订单 > 50 AND 产成品库存 < 50 共同构成"主动补库"条件：
      需求扩张（新订单强）+ 库存偏低（产成品库存弱）→ 企业被迫提高采购，
      驱动大宗商品价格持续上行趋势

滞后处理：
    lag_months=1 意味着用上月的数据信号驱动本月的仓位，
    规避因子公布时间的前视偏差。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── 宏观因子列名（来自 macro_monthly_features_core.csv）─────────────────────
_COL_PPI = "ths_PPI_当月同比"                                 # PPI 同比 %
_COL_PMI = "nbs_制造业采购经理指数_pct"                      # 制造业 PMI
_COL_NEW_ORDERS = "nbs_新订单指数_pct"                        # PMI 新订单分项
_COL_FINISHED_GOODS_INV = "nbs_产成品库存指数_pct"            # PMI 产成品库存分项
_COL_INDUSTRIAL_PRODUCTION = "ths_规模以上工业增加值_当月同比"   # 工业增加值同比
_COL_FIXED_ASSET_INVESTMENT = "nbs_固定资产投资额累计增长_pct"  # 固定资产投资累计同比

# 宏观判断阈值
_PPI_THRESHOLD = 0.0               # PPI > 0 为商品价格通胀环境，是商品CTA的宏观顺风
_PMI_THRESHOLD = 49.5              # PMI 不低于此值视为未明显收缩
_NEW_ORDERS_THRESHOLD = 50.0       # 新订单 PMI 高于此值视为订单扩张
_FINISHED_GOODS_INV_THRESHOLD = 50.0  # 产成品库存 PMI 低于此值视为库存偏低（补库压力大）
_INDUSTRIAL_PRODUCTION_THRESHOLD = 5.5   # 工业增加值同比高于 5.5% 才视为生产需求明确偏强
_FIXED_ASSET_INVESTMENT_THRESHOLD = 3.5  # 固投累计增速高于 3.5% 视为投资需求更扎实


def load_macro_data(macro_csv_path: str | Path) -> pd.DataFrame:
    """读取月度宏观因子 CSV，返回按 tdate 升序排列的 DataFrame。"""
    df = pd.read_csv(macro_csv_path, parse_dates=["tdate"])
    df = df.sort_values("tdate").reset_index(drop=True)
    return df


def _series_with_lag(raw: pd.Series, dates: pd.Series, lag_months: int) -> pd.Series:
    """
    将布尔序列按月数滞后，返回以 dates 为索引的新 Series。
    滞后意味着：本月信号 = lag_months 个月前的原始判断结果。
    """
    lagged = raw.astype(bool).shift(lag_months, fill_value=False)
    lagged.index = dates
    return lagged


def judge_macro_strong(df: pd.DataFrame, lag_months: int = 1) -> pd.Series:
    """
    宏观周期强判断：
        PPI同比 > 0  AND  制造业PMI >= 49.5

    PPI 突破零轴意味着商品价格处于通胀环境，是商品 CTA 趋势策略最直接的
    宏观顺风；PMI >= 49.5 作为景气度确认，排除价格短暂反弹但需求已明显收缩
    的情形。

    Returns: 以 tdate 为索引的 bool Series，name='is_macro_strong'
    """
    # NaN 参与比较时自动为 False（pandas 行为），不需要额外处理
    raw = (df[_COL_PPI] > _PPI_THRESHOLD) & (df[_COL_PMI] >= _PMI_THRESHOLD)
    return _series_with_lag(raw, df["tdate"], lag_months).rename("is_macro_strong")


def judge_inventory_cycle_strong(df: pd.DataFrame, lag_months: int = 1) -> pd.Series:
    """
    库存周期强判断：
        PMI新订单 > 50.0  AND  PMI产成品库存 < 50.0

    两个条件共同构成"主动补库"信号：
      - 新订单 > 50：需求端真正进入扩张区间
      - 产成品库存 < 50：企业手头库存偏低，需要采购补库
    两者同时满足意味着企业被迫加大原材料采购，是商品价格趋势性上涨最可靠
    的领先信号；避免"订单扩张但库存高企（被动补库）"这类商品需求较弱的假信号。

    Returns: 以 tdate 为索引的 bool Series，name='is_inventory_strong'
    """
    raw = (df[_COL_NEW_ORDERS] > _NEW_ORDERS_THRESHOLD) & (df[_COL_FINISHED_GOODS_INV] < _FINISHED_GOODS_INV_THRESHOLD)
    return _series_with_lag(raw, df["tdate"], lag_months).rename("is_inventory_strong")


def judge_industrial_demand_strong(df: pd.DataFrame, lag_months: int = 1) -> pd.Series:
    """
    工业需求强判断：
        工业增加值同比 > 5.5  AND  固定资产投资累计增长 > 3.5

    它和已有的价格/库存类条件不同，直接从实体工作量角度确认商品需求：
      - 工业增加值同比 > 5.5：生产端维持较强扩张
      - 固投累计增长 > 3.5：制造业、基建等中游投资并未明显走弱
    两者共同成立时，更容易出现黑色、有色、能化等品种的真实需求支撑，
    对 CTA 而言属于更独立的一类宏观顺风信号。

    Returns: 以 tdate 为索引的 bool Series，name='is_demand_strong'
    """
    raw = (
        (df[_COL_INDUSTRIAL_PRODUCTION] > _INDUSTRIAL_PRODUCTION_THRESHOLD)
        & (df[_COL_FIXED_ASSET_INVESTMENT] > _FIXED_ASSET_INVESTMENT_THRESHOLD)
    )
    return _series_with_lag(raw, df["tdate"], lag_months).rename("is_demand_strong")


def build_monthly_regime(
    macro_csv_path: str | Path,
    lag_months: int = 1,
) -> pd.DataFrame:
    """
    构建月度 regime 表，供 backtest_macro.py 使用。

    Parameters
    ----------
    macro_csv_path : 月度宏观因子 CSV 路径
    lag_months     : 数据滞后月数（默认 1，即用上月数据判断本月是否开仓）

    Returns
    -------
    pd.DataFrame
        索引 = tdate（月初日期），列：
            is_macro_strong     : bool
            is_inventory_strong : bool
            is_demand_strong    : bool
            open_position       : bool  (任意一个为 True 则允许开仓)
    """
    df = load_macro_data(macro_csv_path)
    macro_sig = judge_macro_strong(df, lag_months=lag_months)
    inv_sig = judge_inventory_cycle_strong(df, lag_months=lag_months)
    demand_sig = judge_industrial_demand_strong(df, lag_months=lag_months)

    regime = pd.DataFrame(
        {
            "is_macro_strong": macro_sig,
            "is_inventory_strong": inv_sig,
            "is_demand_strong": demand_sig,
        }
    )
    regime["open_position"] = (
        regime["is_macro_strong"]
        | regime["is_inventory_strong"]
        | regime["is_demand_strong"]
    )
    return regime
