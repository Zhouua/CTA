# 模型设计思路

## 1. 当前模型

双 regime LightGBM 回归：

- **Regime 分割**：每根 bar 的日内 vol 百分位（相对于 train 集分布的 cutoff）决定分入 `low_vol`（-1）或 `high_vol`（+1）。cutoff **只从 train 行计算**，不能泄露 val/test 信息。
- **两个独立模型**：各自学习本 regime 内 feature → return 的映射，参数互不共享。
- **特征缩放**：RobustScaler，fit 在 train，transform 到 val/test。
- 分好以后的预测目标为经过波动归一化后的回报率

训练目标与损失函数

```
target_vol_norm = future_return / (intraday_rolling_std_20bar × √horizon + ε)
```

- 损失函数：**L2（MSE）**，`objective: regression, metric: l2`
- 预测时反归一化：`pred_return = raw_pred × target_vol_scale`，backtest 使用 `pred_return`

## 2. 高低波分 Regime 的设计逻辑

**为什么分 regime**：高波和低波环境下，feature → return 的结构性关系不同（动量强度、反转周期、成交量信号的方向均有差异）。用一个模型拟合两种状态容易相互污染。

**Regime 是粗粒度的二值切割**，不是连续的。同一 regime 内部 vol 仍有 2–3 倍的范围差异。

**不变量**：Regime cutoff 只从 train 行的 daily vol 分布计算，然后应用到所有 split。新的 regime idea 也要遵循同样原则，不能引入 look-ahead。

### 是否分 Regime 的比较
代码位置 `scripts/check_Regime.py`

## 3. Vol 归一化

### 3.1 两种 target 的区别

| | 方案 A：不归一化 | 方案 B：Vol 归一化（当前）|
|---|---|---|
| **Target 定义** | `future_return` | `future_return / vol_scale` |
| **模型学的是** | 未来绝对收益 | 未来收益是当前 vol 的几倍（信号强度）|
| **预测输出** | 直接可用 | 需反归一化：`pred_return = raw_pred × vol_scale` |
| **梯度分布** | 高 vol bar 的 \|return\| 更大 → 主导 L2 梯度 | vol 均衡后梯度分布均匀 |
| **跨期稳定性** | target 尺度随时间漂移（牛市 vs 熊市 vol 不同）| target 分布相对稳定 |

### 3.2 Vol 归一化的潜在风险

反归一化 `pred_return = raw_pred × vol_scale` 意味着最终预测依赖 **vol_scale 的估计质量**：
- `target_vol_scale = intraday_rolling_std_20bar × √horizon`——用当前 20bar 内的波动估计未来 vol
- 若市场刚发生 regime change（vol 骤升或骤降），20bar 滞后的估计会低估/高估真实 vol
- 低估时：反归一化放大 pred_return → 信号过强 → 过度开仓
- 高估时：反归一化缩小 pred_return → 信号过弱 → 错过开仓

这是一个**工程风险**，不是方向错误。目前可以接受，后续可以考虑用更稳健的 vol 估计（如 Parkinson vol、EWMA）替换简单 rolling std。

### 3.3 是否 Vol 归一化的比较
代码位置 `scripts/check_VolNorm.py`

## 4. 数据 future return 的分布特征
代码位置 `scripts/generate_report_charts.py`

从实际数据观察：**target_vol_norm 均值约 0，方差较大，在 0 线附近无规律波动**。这意味着：

- 模型预测的信噪比本身极低，IC 上限受限（RB 实测 IC ≈ 0.14，对应年化净收益约 2%）。
- L2 损失下，极端 bar 的平方误差更大，梯度会向极端 bar 倾斜——这既是弊（过拟合尾部）也是潜在利（如果极端 bar 确实有信号）。
- 不应该尝试降低尾部敏感度（如 Huber），**应该顺着 L2 的特性进一步放大对极端 bar 的重视**。

## 5. 模型真正在做什么——与 backtest 的对齐分析

### backtest 的实际需求

backtest 的入场逻辑是：

```
entry: |pred_return| > Q88(|val pred_return|)   ← 仅 top 12% bar 开仓
exit:  |pred_return| < Q40(|val pred_return|)
```

**模型不需要精确预测 return 的幅度，只需要在 top 12% bar 上方向对**。这本质上是一个**尾部排序/方向分类问题**，而不是全样本回归问题。

L2 全样本最小化 ≠ 尾部方向准确率最大化。当前 L2 均等对待所有 bar，但真正决定 P&L 的只有预测最极端的那些 bar 是否方向正确。

### 当前参数的隐含约束

| 参数 | 值 | 含义 |
|---|---|---|
| `target_horizon` | 30 bars (30min) | 训练目标：未来 30min 收益 |
| `min_hold_bars` | 10 bars | 最短持仓 10min |
| `confirmation_bars` | 2 | 开仓前需连续 2 bar 信号 |
| `flatten_at_day_end` | true | 当日末强平，实际持仓 < 30min |
| `entry_quantile` | 0.88 | val 集 top 12% 才开仓 |

**错位**：模型预测 30min 后的收益，但因为日末强平，日末附近的仓位实际持有远少于 30 bar。这些 bar 的训练 label（30min 收益）包含了"今天收盘后才会发生的价格变动"，是无法实现的未来收益。这会污染 target，引入无法学习的噪声。

---

## 6. 重视极端情况（样本加权）

### 6.1 问题根源

§5 已指出：backtest 只有 top 12% 极端预测 bar 才开仓，P&L 完全由尾部方向准确率决定。但 L2 损失均等对待所有 bar，绝大多数"不会开仓"的中性 bar 的梯度在主导训练。

| | 优化对象 | 权重分布 |
|---|---|---|
| L2 损失（默认）| 全样本均方误差 | 均匀，隐式偏向数量多的中性 bar |
| backtest 需求 | top 12% 尾部方向准确率 | 极端 bar 全部重要，中性 bar 无关 |

### 6.2 设计：按 target 幅度加权

```
weight_i = |target_vol_norm_i|^alpha
weight_i /= mean(weight)
```

- `alpha=0`：等价于当前默认（均匀权重）
- `alpha=1`：权重线性正比于 |极端幅度|，大行情 bar 权重高
- `alpha=2`：平方强调，尾部更极端（但对噪声大 bar 过于敏感）

**关键约束**：val set 不传 weight，early stopping 保持对全样本过拟合的无偏检测。

### 6.3 与 target 定义的正交性

样本加权和 target 定义是两个**独立**的设计维度，可以任意组合：

| | `target_vol_norm`（终值，当前）| `target_extreme_norm`（路径极值，§7）|
|---|---|---|
| 均匀权重（`alpha=0`）| 当前默认 | 换 target，不改权重 |
| 加权（`alpha=1`）| 强调大终值 bar | 同时强调大极值 bar（双重改进）|

**样本加权：同样的问题，对大幅度 bar 更用功。§7 的 signed_extreme：换一个更对齐实际需求的问题。两者不能相互替代。**

### 6.4 实现方向

`modeling.py::train_single_regime_model`，在 `lgb.Dataset` 上传 `weight` 参数。config 键 `model.sample_weight_alpha`，默认 0，先跑 alpha=1 的 A/B 对比。

---

## 7. 预测极大极小值的模型设计

### 7.1 当前 target 的本质局限

当前 `target_vol_norm` 只看 t+N 时刻终值，不看中间路径：

| 场景 | 实际价格路径 | 当前 target | 极值 target |
|---|---|---|---|
| 先涨 3% 再回到 0 | ↑↑↑↓↓ | ≈ 0（无信号）| +3%（强多信号）|
| 持续上涨 3% | ↑↑↑↑↑ | +3% ✓ | +3% ✓ |
| 先跌 2% 再涨回 0 | ↓↓↑↑↑ | ≈ 0（无信号）| −2%（强空信号）|
| 震荡 ±1% | ↑↓↑↓↑ | ≈ 0 | ≈ ±1%（小信号）|

**关键发现**：CTA 策略的盈利来源是捕捉**日内大幅趋势段**，即使最终价格回到原位。当前 target 把这类机会标记为"无信号"，模型学不到。

---

### 7.2 Signed Extreme Target 设计

```
future_max_return = max(close[t+1:t+N+1]) / close[t] - 1
future_min_return = min(close[t+1:t+N+1]) / close[t] - 1

signed_extreme = future_max_return  if |future_max_return| ≥ |future_min_return|
                 future_min_return  otherwise

target_extreme_norm = signed_extreme / (vol_scale + ε)
```

含义：**horizon 窗口内，绝对幅度更大的那个方向，就是预测目标**。

---

### 7.3 与 backtest 的对齐

backtest 用动态止盈（`exit_quantile=0.40`）而非固定持仓到 t+N：

- **旧 target 的错位**：预测终值，但策略在价格反转时已退出
- **新 target 的对齐**：预测窗口内峰值方向 → 开仓 → 价格达峰时策略已持仓 → 动态止盈退出

---

### 7.4 实现方向

`dataset.py::_add_targets`：同日内逆序 rolling 取 max/min，shift(-1) 对齐未来窗口；不跨日避免日末前视。config 切换 `target_column: target_extreme_norm` 即可，无需改 modeling 代码。

建议先换 target（§7），再叠加样本加权（§6），逐步 A/B 验证。

---

### 7.5 三种 target 方案对比

| 方案 | 改什么 | 核心效果 |
|---|---|---|
| **样本加权** | 训练损失权重 | 对极端 bar 更用力拟合，目标仍是终值 |
| **Signed Extreme target**（推荐）| 预测目标定义 | 直接捕捉窗口内最大幅度方向 |
| **Quantile regression** | 损失函数 | 预测分位数终值，无法捕捉"涨后回归"型 |

**为什么不用 Huber**：截断大误差影响，方向相反。

---

## 8. 超参数 Horizon × Vol_cutoff Grid Search 实验结论

来源：`scripts/exp_pure_micro_horizon_cutoff.py`，结果记录于 `docs/exp_pure_micro_horizon_vs_cutoff.md`。

实验条件：禁用中观因子，仅用微观+工程因子，6 个品种（AU/AG/JD/M/CU/SN），横扫 horizon ∈ {1,5,10,30,60,120} × cutoff ∈ {0.60,0.65,0.70,0.75,0.80}。

### 8.1 Horizon 效应（cutoff=0.65 下的全品种均值）

| horizon | mean net ann | mean sharpe | mean test IC | 含义 |
|---:|---:|---:|---:|---|
| 1 bar (5min) | −27.95% | −5.30 | +0.114 | IC 最高但亏损最严重 |
| 5 bar (25min) | −35.64% | −3.73 | +0.047 | 最差 |
| 10 bar (50min) | −20.64% | −2.40 | +0.024 | 中等 |
| 30 bar (150min) | −11.40% | −1.13 | −0.006 | 当前配置 |
| 60 bar (300min) | −10.16% | −0.86 | −0.005 | 略优于 30 |
| **120 bar (600min)** | **−6.67%** | **−0.41** | +0.008 | **纯微观最优** |

**核心矛盾**：horizon=1 的 IC 最高（0.114），但净收益最差（−28%）。这是**IC-P&L 断层**。

### 8.2 为什么短 horizon 高 IC 却亏损？

短 horizon 的 IC 高是因为信号对 5min 内的价格扰动敏感，但：
- 开仓频率极高（horizon=1：约 1200–2000 trades/yr）
- 每笔 commission+slippage = 2bp，高频下成本累积吃掉全部毛利
- horizon=30 trades ≈ 300–500/yr，horizon=120 trades ≈ 100–250/yr，成本拖累大幅下降

**结论**：IC 衡量的是模型预测能力；能否盈利还取决于 `signal-to-cost ratio = gross_edge / round_trip_cost`。短 horizon 毛利薄，成本比例高。

### 8.3 Vol cutoff 效应

- **cutoff=0.65**（top 35% = high_vol）在 horizon ≥ 30 下几乎全面优于 cutoff=0.75
- 当前 `vol_percentage=0.65` 是正确选择，无需更改

### 8.4 品种类型与最优 horizon 的关系

| 品种类别 | 代表 | 最优 horizon | 最优 cutoff | 特点 |
|---|---|---:|---:|---|
| high-vol heavy | AU, AG | 60–120 | 65–75 | high_vol bar 多，较长 horizon 成本可控 |
| balanced | JD, M | 120 | 65 | 中等 high_vol，长 horizon 最优 |
| low-vol heavy | CU, SN | 60–120 | 65 | high_vol bar 极少，长 horizon 减少无效交易 |

### 8.5 对当前配置（horizon=30, cutoff=0.65）的评估

horizon=30 是 grid 中第 4 好，cutoff=0.65 是全局最优。

**当前不改 horizon 的理由**：grid search 是纯微观条件下的结论。中观因子加入后毛利提升，horizon=30 的 cost/return ratio 变好。如果要重新实验应在中观因子配置下重跑。

---

## 9. 进一步优化方向（按优先级）

### 优先级 1：样本加权
改动最小，详见 §6。

### 优先级 2：在中观因子配置下重跑 horizon grid search
当前 grid search 结论来自纯微观条件。中观因子提升了毛利，最优 horizon 可能前移。

### 优先级 3：vol_percentage 与 regime 内样本不均衡
high_vol 训练样本仅为 low_vol 的 54%，样本少 + 同等模型复杂度 → 过拟合风险更高。方向：high_vol 适当降低复杂度或增大最小叶节点样本数，当前已有差异但幅度不够。

### 优先级 4：Regime 内 vol_scale 作为特征
regime 是粗粒度二值，内部 vol 仍有 2–3x 差异。把 `target_vol_scale` 作为特征输入，让模型自己学习 vol 水平对信号强度的调节，比三分 regime 更灵活且不碎片化训练集。

### 优先级 5：尾部方向准确率作为辅助评估指标
当前 IC 是全样本的，但 P&L 只由 top/bottom 10% 预测的方向准确率（DA@tail）决定。DA@tail 是比 IC 更贴近 backtest 真实需求的度量。

---

## 10. 当前配置评估

| 参数 | 当前值 | 评估 |
|---|---|---|
| `num_leaves` | 63 | 适中；high_vol 可考虑降低 |
| `max_depth` | 6 | 合理 |
| `learning_rate` | 0.03 | 合理，配合 600 rounds |
| `num_boost_round` | 600 | 合理，early_stopping=50 保护 |
| `min_child_samples` | 120/150 | low/high vol 差异方向正确，幅度可加大 |
| `reg_lambda` | 1.0 | L2 正则已开；high_vol 可适当增大 |
| `bagging_fraction` | 0.8 | 合理 |
| `feature_fraction` | 0.8 | 合理 |
| `target_horizon` | 30 | 有日末错位问题，待实验 |
| `entry_quantile` | 0.88 | top 12% 开仓，交易频率合理 |
| `sample_weight_alpha` | 0（未开启）| 最近期改进点 |
