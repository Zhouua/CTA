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

## 6. 中观因子 IC 审计的频率错配问题（empirical）

### 问题发现

`baseline_roll240` vs `baseline_roll240_no_mid` 的 A/B 对比（29 个品种）显示：

- 中位数 ΔSharpe = **-0.159**，退步占比 51.7%（>1/3 → 整体无效）
- 最大退步：M −3.03、C −2.59、EG −2.27、AL −1.36、J −0.96

退步品种的共同特征（实测）：

| 现象 | 数据 |
|---|---|
| 中观+交互特征占模型权重 45-85% | M=62%, J=61%, EG=60%, C=67% |
| val IC 基本稳定，test IC 崩塌或符号反转 | M: val 0.016 → test **-0.048**；EG: val 0.017 → test **-0.028** |
| MIDxMICRO 交互特征爆炸 | J=53 个、C=78 个、AL=52 个通过审计 |
| 信号坍缩或过度交易 | M: 51→7 笔；J: 5→161 笔 |

改善品种（RU +2.61、FU +1.55、Y +1.42）的共同点是 test IC 与 val IC 方向一致甚至提升，说明问题是**中观因子失效而非中观因子无用**。

### 根本原因：两层

**层 1：1 分钟粒度计算周频因子的 IC，存在月间自相关膨胀**

周频因子每周只更新一次，在一个月窗口（~2400 根 bar）内只有 4-5 个不同值。月度 IC 估计中，相邻月份共享约 75% 的周观测，导致 32 个月的 IC 序列高度自相关。样本 `std_ic` 被低估 2-3×，ICIR = mean_ic / std_ic 因此被同等倍数虚高。结果：mean_IC≈0.03 的中观因子 ICIR 可达 0.7-1.2，轻松通过 0.30 门槛，但实际预测力接近零。

**层 2：IC 审计只用 train 数据，无法检测 factor decay**

中观因子（豆粕库存/基差/乙二醇成本等）在 2021-2023 训练期有稳定的负相关关系，通过审计。但 2024-2025 测试期宏观 regime 切换后，这些关系消失乃至逆转。val IC 不足以预警（val 期仍属于"旧 regime"），审计完全没有发现机制。

### MIDxMICRO 没有单一合适的 IC 频率

`MIDxMICRO = 微观因子（每分钟变） × 中观 PCT_RANK（每周变）`

这是**双频特征**：

- 按 1 分钟算 IC：会被微观分量本身的 IC 虚高带过，无法区分"交互有效"和"微观本来就有效"
- 按周度算 IC：丢掉微观的逐 bar 信息，问题变形
- 按条件 IC（高/低中观 regime 内分别算微观 IC）：理论正确，但不在现有一刀切阈值框架内

**结论**：MIDxMICRO 不适合通过 IC 审计做准入，应改为**继承父因子审计结论 + 硬上限**。

---

## 7. 分层审计设计（三频道框架）——设计规格

### 设计原则

> 每个因子按它真正产生独立观测的频率配对目标变量，用独立样本估计 IC。

周频因子每月只有 4-5 个独立观测（不是 2400 个），那就用 4-5 个独立观测来量化它的信息量。数量少本身就是真实——不应通过 1 分钟粒度制造虚假样本量。

### 三层配置

```
Tier A  微观因子 (ENG_*, runtime)
────────────────────────────────────────────────────────
样本粒度：1 分钟
窗口粒度：月度（~2400 obs/窗口，~33 窗口）
min_abs_ic = 0.005，min_icir = 0.30
不变。

Tier B  中观日度因子 (MID_*，频率=日)
────────────────────────────────────────────────────────
目标聚合：每个交易日的 target_vol_norm 均值 → 一个日度目标值
因子取值：当天 forward-fill 后的常数值（每天一个）
窗口粒度：月度（~22 obs/窗口，~33 窗口）
min_obs_per_window = 15
min_abs_ic = 0.015，min_icir = 0.40
+ val 一致性：sign(IC_val) == sign(IC_train)，|IC_val| ≥ 0.40×|IC_train|

Tier C  中观周度因子 (MID_*，频率=周)
────────────────────────────────────────────────────────
目标聚合：每个自然周的 target_vol_norm 均值 → 一个周度目标值
因子取值：该周第一个有效值（每周一个）
窗口粒度：季度（~13 obs/窗口，~12 窗口）
min_obs_per_window = 8
min_abs_ic = 0.030，min_icir = 0.40
+ val 一致性：sign(IC_val) == sign(IC_train)，|IC_val| ≥ 0.35×|IC_train|
+ 硬预算：按 |ICIR| 取 top-6 原始指标（避免弱信号堆叠）

MIDxMICRO 交互特征
────────────────────────────────────────────────────────
不做独立 IC 审计（无单一合适频率）
准入条件：中观父因子通过 Tier B/C + 微观父因子通过 Tier A
硬上限：每品种最多 15 个（从当前 100 降）
超出部分按父因子 |ICIR| 排序截断
```

### 阈值标定逻辑

每窗口单次 IC 估计的标准误 ≈ 1/sqrt(n_obs - 2)：

| 层次 | obs/窗口 | IC se/窗口 | 相对微观 | min_abs_ic 调整 |
|---|---|---|---|---|
| 微观（Tier A） | 2400 | 0.020 | 1× | 0.005 |
| 日度中观（Tier B） | 22 | 0.224 | ~11× | 0.015（×3） |
| 周度中观（Tier C） | 13 | 0.302 | ~15× | 0.030（×6） |

阈值相对于 IC 噪声水平近似等比放大，保持相同的"信噪比要求"。

### val 一致性检验的作用

val 数据量（~9-10 个月）不足以计算稳定 ICIR，因此 val 检验只做**方向检验**：IC 在 val 期是否和 train 期同号，且没有大幅衰减（保留率 ≥ 35-40%）。

这一门直接拦截"train 有信号但 val 已开始衰减"的因子，对 M（test_IC 0.015→-0.048）、EG（0.004→-0.028）、RB（-0.001→-0.041）这类 factor decay 情形有早期预警作用。val IC 在全段上算（不做 walk-forward），因为样本太少做滑窗会更噪。

### 实现链路

```
dataset.py::_read_mid_weekly_xlsx
  → 已有每列 freq（"日"/"周"/"月"）
  → PreparedData 新增 mid_feature_freq_map: dict[str, str]
    （原始列名 → 频率字符串；衍生列继承父列频率）

factor_audit.py::audit_and_filter
  → 接收 mid_feature_freq_map
  → 将 MID_* 按频率分 Tier B/Tier C 两组
  → 每组按对应粒度聚合 → compute_walk_forward_ic
  → val 一致性检验作第二门
  → 写入 registry 时附 audit_tier 字段

config.yaml
  → 新增 factors.mid_audit 子块，独立于 audit_thresholds
  → MIDxMICRO max_columns 从 100 → 15
```

---

## 8. 因子处理完整流程

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

---

## 9. 三层审计全批实验结果（tiered_mid_full，2026-05-06）

### 实验设置

- 基线A：`baseline_roll240_no_mid`（纯微观，无中观因子）
- 基线B：`baseline_roll240`（旧中观，未分层，所有 MID_* 走 1min 月度 IC）
- 实验：`tiered_mid_full`（三层审计 v3，含 max_icir=0.9 + max_mid_total=8）

config 关键参数：
```yaml
factors.mid_audit.weekly:
  min_abs_ic: 0.030
  min_icir:   0.40
  max_icir:   0.90   # 新增：防季度窗口过拟合
  max_raw_features: 6
factors.mid_audit:
  max_mid_total_per_product: 8  # 新增：Tier C + MIDxMICRO 合计上限
```

### 全批汇总（29 个可比品种，test 集 net）

| 指标 | 纯微观基线 | 旧中观 | **三层审计v3** |
|---|---|---|---|
| 均值 Sharpe | -0.172 | -0.248 | **+0.133** |
| 中位 Sharpe | -0.001 | 0.000 | **+0.227** |
| 正 Sharpe 品种 | 11/29 | 12/29 | **16/29** |
| Sharpe ≥ 1.0 | 4 | 4 | **6** |
| 均值年化收益 | +0.28% | +0.97% | +0.65% |
| 均值最大回撤 | -5.62% | -6.49% | **-4.85%** |
| 品种级改善率 vs 纯微观 | — | 41% | **62%** |
| 均值 ΔSharpe vs 纯微观 | — | -0.076 | **+0.305** |

三层审计是三者中**唯一均值 Sharpe 为正**的配置，最大回撤也最低。

### 大幅改善品种（ΔSharpe vs 纯微观 > 0.5，共 14 个）

| 品种 | 纯微观 | v3 | ΔSharpe |
|---|---|---|---|
| ZN | -0.979 | 1.776 | +2.755 |
| RU | -1.885 | 0.553 | +2.438 |
| CU | -0.964 | 0.974 | +1.938 |
| J  | -1.226 | 0.347 | +1.573 |
| P  | -0.461 | 1.068 | +1.530 |
| RB | 0.321  | 1.768 | +1.447 |
| FU | 0.046  | 1.021 | +0.976 |
| PG | -0.233 | 0.707 | +0.940 |
| EG | -1.678 | -0.907 | +0.770 |
| M  | 1.741  | 2.512 | +0.771 |

### 仍有退步的品种及根本原因

| 品种 | 纯微观 | v3 | ΔSharpe | 根本原因 |
|---|---|---|---|---|
| RR | -1.135 | -3.423 | -2.288 | **数据异常**：pred_std=56.83，目标 std=0.002，模型预测失去量纲（R²=-8.4亿）。推断 RR.xlsx 中含有极端值（如超长窗口 _RET_52/_ACCEL_52 在特定时段出现奇异值），需清洗源数据 |
| BU | 1.087 | -0.516 | -1.603 | **pred_std 膨胀**：7 个 Tier C 特征使 val pred_std 从 5.0e-5 → 2.8e-4（5.5×），entry 阈值下降，交易数从 13 → 138，方向无改善 |
| SS | -1.867 | -2.297 | -0.430 | pred_std 膨胀 + 交易数膨胀（145→252）|
| FB | -0.334 | -1.046 | -0.713 | pred_std=2.2e-3（远高于其他品种），旧中观 +1.084 → 改为 Tier C 筛选后选出了干扰因子 |
| JD | -0.425 | -1.178 | -0.753 | 旧中观 +0.995；Tier C 过滤掉了有效因子，改为选入新的弱因子 |
| B  | 0.372 | -0.317 | -0.689 | 有效 mid 因子被过滤后引入干扰 |
| JM | 0.102 | -0.541 | -0.643 | 同上 |

### 核心矛盾：分层过滤的"选集替换"问题

三层审计在 **native-frequency** 粒度上更诚实地估计 IC，但部分品种（BU、FB、JD、SS）在旧 1min 月度 IC 下有效的中观因子，**在周度季度粒度下未通过门槛**（样本太少，IC 估计噪声大），被过滤掉；但同时，另一批通过了 Tier C 门槛的因子被选入，这批因子在 1min 粒度上并无预测力，反而干扰模型。

典型特征：这类品种在旧中观下是"受益品种"（ΔSharpe > 0），但 v3 把它们带坏。

### v3 与旧版的对比迭代路径

| 版本 | 核心变化 | 均值 ΔSharpe vs 纯微观 | 退步品种数 |
|---|---|---|---|
| 旧中观（baseline_roll240） | 全部 MID_* 走 1min 月度 IC | -0.076 | 17/29 |
| v1（native-freq，无上限） | Tier C + val 一致性，无总量上限 | 未全批（仅8品） | — |
| v2（+总量上限=8） | max_mid_total=8 | 未全批 | — |
| **v3（+max_icir=0.9）** | max_icir 防季度过拟合 | **+0.305** | **11/29** |

### 下一步方向

1. **RR**：检查 `data/mid_weekly/RR.xlsx` 是否存在极端值或编码异常，需要清洗后重跑
2. **pred_std 膨胀类（BU/SS/FB/JD）**：增加"val pred_std 稳定性"门——若中观因子加入后 val pred_std > N × 纯微观 pred_std，拒绝该品种的中观因子；或进一步降低 `max_mid_total_per_product`（如从 8 → 5）
3. **选集替换类（JD/FB）**：这些品种在旧中观下"刚好"有好用的因子，Tier C 设计本质上改变了选集——可能需要接受"这类品种不适合加中观"的结论，或提供品种级 mid_weekly 开关（但需注意全局规则原则）
