# 因子设计思路

## 1. 因子层次结构

pipeline 的因子分三层：

| 层次 | 来源 | 代码位置 |
|---|---|---|
| **微观（5min）** | 价格/成交量工程特征 | `pipeline/cal_factors.py` |
| **中观（mid_weekly）** | 外部 xlsx（基本面/库存/持仓）| `pipeline/_merge_mid_weekly_features` |
| **运行时因子** | `factor_engine.py` 生成 | `factors.runtime` 配置块 |

---

## 2. 微观工程因子（cal_factors.py）

### 因子模块

| 模块 | 内容 | 备注 |
|---|---|---|
| A. 单 bar 特征 | 上下影线比、实体比、成交量突破 | intrabar shape |
| B. 滑动窗口 | 多窗口 return/std/ATR/MA deviation | 核心骨干 |
| C. 时间编码 | sin/cos 时间、竞价时段标记 | 非线性时间信号 |
| D. 长窗口 | ROC/CORR/OBV（大 window）| 中长期动量 |
| E. 隔夜/日内 | 隔夜 gap、日内累计 return | 跨日结构 |
| F. VWAP Z-score | 有符号 VWAP 偏离 + regime flag | 量价结合 |
| G. VOL trend | STD20/STD60 趋势 Z-score | 依赖 runtime 因子 |
| H. OI（持仓量） | pct_change + 方向确认/背离 | POSITION 列 |
| I. 合成交互 | price × vol × OI 的乘积组合 | 高阶信号 |
| J. 分钟级滚动 | 日内累计 + 当分钟 VWAP/return/vol | bar 级细节 |
| K. 日级滞后 | 昨日/前日日级因子广播到 5min 网格 | 日级背景 |
| L. 半方差/连续 | 上行/下行半方差、连续同向 bar | 不对称 vol |
| M. 复合 alpha | IC 加权的因子融合 | 实验性 |
| N. 同分钟历史 | 同一分钟在历史上的统计行为 | 日内周期性 |
| O. 因子融合 | 同类簇内取均值，降噪突出共同信号 | 高相关因子合成 |

### 关键原则

- 所有特征都是**无未来信息**的（shift 严格）。
- 骨干因子（`CORR30 / STD60 / ATR / ENG_POSITION_RATIO_60`）在大多数品种的 feature_importance 中排名靠前，不要随意改。
- 因子过多会造成 feature matrix 维度膨胀，LGB 的 gain 会被稀释，IC audit 用于裁剪低贡献因子。

---

## 3. 中观因子（mid_weekly）的关键教训

### v0 → v1 → v2 实验结论

| 版本 | 配置 | 中位 ΔSharpe vs 纯微观 | 退步品种数 |
|---|---|---|---|
| v1（含水平列）| `level_keep=true` | +0.098 | 5/25 |
| v2（去水平列）| `level_keep=false` | +0.235 | 4/25 |

**v2 是当前最优。**

### 水平列的问题

周频源列经 `merge_asof + ffill` 在 5min 网格上铺 ≈480 根 bar 的"分段常量"（同一数字）。LGB 把它当 regime flag 用（一个 split 点把数据切成"上次观测高 / 低"两段），挤占骨干微观因子的 gain。**本质上是伪 regime 信号**。

- **周频主导品种**（日频列 < 50%）：v2 去掉水平列获益，代表品种 CU（+1.72）、PB（+1.00）、BB（+0.76）。
- **日频主导品种**（日频列 > 50%）：v2 去掉日频水平列损失真信号，代表退步品种 C（−0.77）、SN（−0.65）、FU（−1.33）。

### 派生因子仍然有效

`_RET_{4/13/52}`, `_ZSCORE_{4/13/52}`, `_PCT_RANK_{4/13/52}` 虽然也走 ffill，但每次新观测到来时值会更新（相对过去 w 个观测的统计量），不是裸数字，LGB 不容易把 ffill 段当 regime flag。

---

## 4. 因子 IC 的上限认知

- RB 实测 val IC ≈ 0.14，对应年化净收益约 2%（已考虑成本）。
- IC 约束意味着：**模型的绝对预测能力有上限**，alpha 来源主要是捕捉极端 bar（大行情）而非平均精度。
- 这也是为什么在模型设计上选择重视极端情况（sample weight）而非追求全样本 RMSE 最小化。

---

## 5. IC Audit 机制

训练时 `pipeline/factor_audit.py` inline 计算每个因子在 train 集上的 IC、RankIC、分 regime 的 IC，输出到 `results/runs/<run_id>/<PID>/factor_registry.json`。

- 这替代了旧的全局 `data/factor_registry.json`，每次训练重新计算，不依赖历史文件。
- 低 IC 因子会被标记但不自动 drop（保留 LGB 自己选择的权利），避免强行过滤误伤交互效应。

---

## 6. 因子处理完整流程

```
原始 CSV（5min 行情）
  │
  ├─ [Step 1] _read_raw_data
  │    读 CSV，timeout-retry，返回原始 OHLCV + POSITION 宽表
  │
  ├─ [Step 2] generate_runtime_factors（factor_engine.py）
  │    按 factors.runtime 配置动态生成 runtime_factor_cols（STD/CORR/ATR 等）
  │    ← 默认路径；legacy CSV 因子路径已弃用
  │
  ├─ [Step 3] add_engineered_features（cal_factors.py）
  │    生成 ENG_* 列：A–O 模块（单bar / 滑动窗口 / 时间编码 / OI 等）
  │    → engineered_cols
  │
  ├─ [Step 4] _merge_mid_weekly_features(若 config.yaml::use_mid_weekly 为true)
  │    ├─ _read_mid_weekly_xlsx：读 4 行表头 xlsx → 稀疏观测表
  │    ├─ _apply_mid_weekly_quality_filter
  │    │    频率归一化稀疏度过滤（eff_ratio < 0.6 → drop）
  │    │    step-dummy 识别（首值后非空率 ≥ 0.9 → drop）
  │    ├─ _compute_mid_weekly_derivatives
  │    │    在稀疏观测网格（不是 5min 网格）上算：
  │    │    RET_{4/13/52}、ZSCORE_{4/13/52}、PCT_RANK_{4/13/52}
  │    ├─ merge_asof(direction="backward") + ffill → 铺到 5min 网格
  │    ├─ staleness clamp：ffill_max_bars=8064（≈4周）超期 → NaN
  │    └─ AVAILABLE 哑变量（clamp 之后生成）
  │    → mid_weekly_cols
  │
  ├─ [Step 5] _add_targets
  │    future_return = future_close / CLOSE - 1
  │    target_vol_scale = intraday_rolling_std(20) × √horizon
  │    target_vol_norm = future_return / (target_vol_scale + ε)   ← 训练目标
  │
  ├─ [Step 6] load_or_build_feature_frame（缓存层）
  │    签名 = hash(mtime + runtime_spec + horizon + mid_files + ...)
  │    命中 → 直接读 parquet；miss → 走 Step 1–5 并写 parquet
  │    路径：results/cache/products/<PID>/<signature>.parquet
  │
  └─ [Step 7] prepare()  ← 每次训练都跑，不缓存
       │
       ├─ split_by_vol → REGIME_LABEL（-1/+1）、DATA_SPLIT（train/val/test）
       │    cutoff 只从 train 行的 daily vol 分布计算，严禁 look-ahead
       │
       ├─ inf → NaN（全部候选列）
       │
       ├─ ffill（仅非 MID_* 列，fill_method="forward_fill"）
       │    MID_* 列跳过（staleness clamp 已在 Step 4 处理，不能再 ffill 覆盖）
       │
       ├─ [过滤 A] missing ratio（基于 train 行快照）
       │    非 MID 列：train 缺失率 > max_factor_missing_ratio=0.35 → drop
       │    MID_* 列：train 缺失率 > mid_weekly_missing_ratio_relax=0.65 → drop
       │
       ├─ [过滤 B] variance filter（基于 train 行快照）
       │    train std ≤ min_factor_std=1e-8 → drop（常数列）
       │
       ├─ MID_* NaN → fillna(0.0)
       │    过滤后残留 NaN 补 0；AVAILABLE 哑变量标记"此处为补零"
       │
       ├─ dropna(subset=[target_col, "future_return", "REGIME_LABEL", "DATA_SPLIT"])
       │    行级：target 或 regime 缺失的 bar 直接丢弃
       │
       └─ [Step 8] factor_audit.py（由 train_products.py 在 prepare 之后调用）
            ├─ compute_walk_forward_ic
            │    对 train_data 按月切窗口，每月算 Spearman IC
            │    聚合 → mean_ic / std_ic / ICIR / n_windows（月数）
            │
            ├─ 过滤规则（同时满足三条才进入训练）
            │    abs(mean_ic)  ≥  min_abs_ic   （config: factors.audit_thresholds）
            │    abs(icir)     ≥  min_icir      （config: factors.audit_thresholds）
            │    n_windows     ≥  3
            │    → 通过 → train_factor（selected_feature_cols 传给 modeling）
            │    → 不通过 → not_train_factor（附 reason 字段）
            │
            ├─ 写 results/runs/<run_id>/<PID>/factor_registry.json
            │    metadata: 品种/阈值/过滤规则/聚合统计
            │    train_factor:     [ {name, mean_ic, icir, ...} ]
            │    not_train_factor: [ {name, reason, ...} ]
            │
            └─ backfill_importance（训练完成后回填）
                 把 LightGBM importance_gain 写回 registry
                 仅作元数据 / 报告图，不影响过滤结果
```

### 关键设计约束

| 约束 | 原因 |
|---|---|
| 过滤阈值基于 **train 行快照** | 防止 val/test 信息泄露到特征选择 |
| MID_* 不走 prepare 的 ffill | Step 4 staleness clamp 已设上限；再 ffill 会把过期值延伸 |
| MID_* NaN 补 0 而非 dropna | 周频列大量 NaN 会杀掉太多行；AVAILABLE 哑变量保留"此处为空"的信息 |
| Regime cutoff 只从 train 计算 | split_by_vol 的核心不变量，任何新 regime 设计都必须遵守 |
| 工程特征（ENG_*）跳过 missing filter | 工程特征由代码保证生成，不走 factor_cols 的缺失率检查 |
