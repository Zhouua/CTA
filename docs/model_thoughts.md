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

### 2.1 Regime 模型质量门控

#### 问题背景

`check_Regime` 实验（15 品种）发现三个产品在 dual model 下严重跑输 single model：

| 产品 | dual net | single net | 差距 | 退化 regime |
|---|---:|---:|---:|---|
| FU | -13.42% | +23.19% | +36.6% | lv_val_ic=-0.019 |
| BU | -5.17% | -1.65% | +3.5% | hv_val_ic=-0.089 |
| AL | -8.96% | -4.53% | +4.4% | hv_val_ic=-0.041 |

共同特征：某个 regime 模型的 val IC 为**负值且幅度 > 0.01**，即模型预测方向与真实收益在验证集上系统性反向。

#### 诊断

`best_iteration=1` 的含义：第 1 轮即是 val 上最优点，之后每加一棵树都使 val RMSE 上升，early stopping 50 轮后停止。模型本质上只有一棵树，预测接近常数（`pred_std` ≈ target_std 的 2%）。

val_IC < 0 说明这唯一一棵树学到的方向在 val 上是错的——不是"无信号"，而是**系统性反向**。根因通常是 train/val 之间信号结构发生了转变（如 FU 能源类品种在 2024 年后整体波动率环境变化，low_vol 的内涵从"高波段内的平静期"变成"结构性低波环境"，特征-收益关系出现反转）。

超参数调整（`learning_rate`、`num_leaves`、正则化）不能修复此问题，因为问题在信号本身而非模型复杂度。

#### 设计决策

**检测条件**：任意 regime 模型的 `val_spearman_ic < min_regime_val_ic`（默认 -0.01，写入 `config.yaml`）

**动作**：该 regime 在 backtest 中强制输出 `position=0`，不产生任何仓位；另一 regime 照常交易

**不采用 fallback 到 single model 的原因**：fallback 引入不同架构混用，违反 dual model 的设计一致性；若某产品 single model 整体更优，应在 universe 筛选层面排除该产品，而不是在 regime 层面切换架构。

#### 验证（15 品种，阈值 -0.01）

阈值 -0.01 在全部 15 个已运行产品上零误触发：

| 触发产品 | 触发原因 | 结果 | 备注 |
|---|---|---|---|
| FU | lv_val_ic=-0.019 | ✓ | dual -13.42% vs single +23.19%，low_vol 不开仓恢复大部分损失 |
| BU | hv_val_ic=-0.089 | ✓ | dual -5.17% vs single -1.65% |
| AL | hv_val_ic=-0.041 | ✓ | dual -8.96% vs single -4.53% |

AG（lv_val_ic=-0.0075）、ZN（lv_val_ic=-0.0039）未触发，dual 对这两个品种仍优于 single，规则未误伤。

#### 实现（已完成）

- `pipeline/modeling.py::RegimeModelArtifact`（dataclass 末字段）：`val_spearman_ic: float = 0.0`，在 `train_single_regime_model` return 时从 `metrics["val_metrics"]["spearman_ic"]` 填入
- `pipeline/backtest.py::build_backtest_settings`：从 `config.model.min_regime_val_ic` 读入阈值（默认 -0.01）
- `pipeline/backtest.py::execute_backtest`：`build_signal_rule_map` 之后，遍历 `artifact_map`，将 `val_spearman_ic < min_regime_val_ic` 的 regime 在 `rule_map` 中标记 `"degenerate": True`
- `pipeline/backtest.py::generate_positions`：bar 循环最前，若 `rules["degenerate"]` 为 True，强制 position=0（持仓立即平）并 `continue`，action 记为 `"regime_quality_skip"`
- `config.yaml::model.min_regime_val_ic: -0.01`

---

### 比较结论

详见 [§ 一些结论](#一些结论)。**简要（15 品种）**：扩展到全品种后 single 均值略优（+0.98% vs -1.19%），但差距主要由 FU 一个品种贡献（dual -13.42% vs single +23.19%）。根因是 FU low_vol 模型 val_IC=-0.019，在 val 期系统性反向交易。加入 regime-level 质量门控（§ 2.1）后，FU/AL/BU 三个退化 regime 不开仓，其余品种 dual 优势保持不变。保留 dual_regime。

---

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

### 3.4 比较结论

详见 [§ 一些结论](#一些结论)。**简要（9 品种）**：vol_norm 胜 4 / raw_return 胜 4 / 平手 1。关键现象：raw_return IC 更高但 trades 系统性更少——低波品种上 raw_return 几乎不交易，等同空仓。暂定保留 vol_norm 作为默认。

---

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

## 6. 策略 A：重视极端情况（样本加权）

**适用场景**：沿用当前 backtest 结构——信号超阈值开仓，持仓 horizon 后由信号衰减退出。
模型预测的仍是 t+N 时刻的终值收益，只是让训练过程对大幅行情 bar 更用功。

### 6.1 问题根源

§5 已指出：backtest 只有 top 12% 极端预测 bar 才开仓，P&L 完全由尾部方向准确率决定。但 L2 均等对待所有 bar，绝大多数"不会开仓"的中性 bar 的梯度在主导训练。

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

### 6.3 backtest 结构不变

```
entry: |pred_return| > Q88(|val pred_return|)   ← 信号强度排名触发
exit:  |pred_return| < Q40(|val pred_return|)   ← 信号衰减触发
```

改的只是训练时各 bar 的权重，backtest 逻辑**完全不变**。

### 6.4 实现方向

`modeling.py::train_single_regime_model`，在 `lgb.Dataset` 上传 `weight` 参数。
config 键 `model.sample_weight_alpha`，默认 0，先跑 alpha=1 的 A/B 对比。

---

## 7. 策略 B：预测极大极小值（新 backtest 结构）

**与策略 A 的根本区别**：策略 B 需要同时改变训练目标 **和** backtest 结构。
模型预测的是"horizon 窗口内的价格极值"，backtest 的退出逻辑改为"价格触达预测目标后提前平仓"。

### 7.1 当前 target 的本质局限

当前 `target_vol_norm` 只看 t+N 时刻终值，不看中间路径：

| 场景 | 实际价格路径 | 当前 target | 极值 target |
|---|---|---|---|
| 先涨 3% 再回到 0 | ↑↑↑↓↓ | ≈ 0（无信号）| +3%（强多信号）|
| 持续上涨 3% | ↑↑↑↑↑ | +3% ✓ | +3% ✓ |
| 先跌 2% 再涨回 0 | ↓↓↑↑↑ | ≈ 0（无信号）| −2%（强空信号）|
| 震荡 ±1% | ↑↓↑↓↑ | ≈ 0 | ≈ ±1%（小信号）|

**关键发现**：CTA 策略的盈利来源是捕捉**日内大幅趋势段**，即使最终价格回到原位。当前 target 把这类机会标记为"无信号"，模型完全学不到。

---

### 7.2 Signed Extreme Target 设计

```
future_max_return = max(close[t+1:t+N+1]) / close[t] - 1
future_min_return = min(close[t+1:t+N+1]) / close[t] - 1

signed_extreme = future_max_return  if |future_max_return| ≥ |future_min_return|
                 future_min_return  otherwise

target_extreme_norm = signed_extreme / (vol_scale + ε)
```

含义：**horizon 窗口内，绝对幅度更大的那个方向就是预测目标。**

---

### 7.3 配套的新 backtest 结构

策略 B 的 `pred_return`（= raw_pred × vol_scale）代表的是**预测的价格极值目标**，不再是信号强度排名。
退出逻辑因此从"信号衰减"变为"价格触达目标"：

```
entry:
  entry_pred = pred_extreme_denorm[t]          ← 有符号
  if |entry_pred| > entry_threshold:
      direction = sign(entry_pred)             ← +1 做多，-1 做空
      target_abs = tp_fraction * |entry_pred|  ← 无符号目标距离
      entry_price = close[t]
      entry_bar = t

exit（三种情况，先触发者平仓）:

  current_return = close[now] / entry_price - 1
  pnl_return     = direction × current_return

  (1) 价格目标触达:
      pnl_return >= target_abs

  (2) 时间到期:
      hold_bars >= horizon

  (3) 反向强信号:
      sign(pred_extreme_denorm[now]) == -direction
      and |pred_extreme_denorm[now]| > entry_threshold
```

这与当前的 `exit_quantile` 逻辑完全不同：当前是看**当前信号是否衰减**，新逻辑是看**价格是否已经实现了预测的目标**。

---

### 7.4 实现方向

**训练侧**（`dataset.py::_add_targets`）：
同日内逆序 rolling 取 max/min，shift(-1) 对齐未来窗口；不跨日避免日末前视；在 train/val/test 切分边界附近，最好丢掉边界前后的 N 个 bar，或者至少做 embargo；所有 threshold 都只能用 train/val，不能用 test。
config 切换 `target_column: target_extreme_norm`，无需改 modeling 代码。

**回测侧**（`backtest.py::generate_positions`）：
需要新增 `exit_mode: target_hit`，在 position loop 内检查 `current_cumret >= pred_extreme_denorm`。
这是策略 B 与当前代码的**唯一结构性差异**。

---

### 7.5 两种策略的核心对比

| 维度 | 策略 A（样本加权）| 策略 B（Signed Extreme）|
|---|---|---|
| **改什么** | 训练损失权重 | 预测目标 + backtest 退出逻辑 |
| **pred_return 的含义** | 预测 t+N 终值收益（信号强度）| 预测窗口内的价格极值目标 |
| **退出触发** | 信号衰减低于 Q40 | 价格触达预测极值 OR 时间到期 |
| **能捕捉"涨后回归"** | ✗（target 仍是终值 ≈ 0）| ✓（极值 target = +3%）|
| **代码改动量** | 极小（modeling.py 一行）| 中等（dataset + backtest）|
| **可叠加** | ✓ | ✓（两者正交）|

**两条路线互不替代，互不依赖，可独立验证后叠加。**

---

## 8. 策略 A+B 融合：以极值预测作为止盈目标

**核心想法**：策略 A 的信号（model_A 预测 t+N 终值）做入场过滤，策略 B 的极值预测（model_B 预测窗口内最大幅度）替换掉当前的信号衰减退出，成为**止盈目标**。两个模型串联使用，而非二选一。

### 8.1 当前退出逻辑的局限

策略 A 的 Q40 退出是"信号软了就出"，不感知价格路径：
- 价格已上涨 2%、但 model_A 信号还没衰减 → 不提前锁利，等信号衰减后可能回吐
- 价格还没动、但 model_A 信号衰减了 → 提前出场，missed 后续行情
- Q40 阈值完全由 val 集 pred 分布决定，与实际价格目标脱节

策略 B 的极值预测正好提供了当前 Q40 所缺少的信息：**"价格最多能走多远"**。

### 8.2 融合后的 backtest 结构

```
entry: model_A 信号 > Q88,  direction = sign(pred_A)
       model_B 同步推断:    pred_extreme = model_B_pred × vol_scale

exit（三种情况，先触发者平仓）:
  (1) 价格触达目标（止盈）: direction × cumret >= pred_extreme   ← 替换 Q40 退出
  (2) 时间到期:              hold_bars >= horizon                 ← 兜底
  (3) 信号强烈反转:          model_A 预测反向且幅度 > Q88        ← 反手信号
```

相比纯策略 B，此结构保留了 model_A 对入场时机的过滤——不是每根 bar 都开仓，仍然只在 model_A top 12% bar 入场；退出则改由价格目标驱动，而不是信号衰减。

### 8.3 两种 target 的协同

| 模型 | 训练目标 | 推断输出 | 在融合策略中的角色 |
|---|---|---|---|
| model_A | `target_vol_norm`（终值）| `pred_return`（信号强度）| 入场过滤（Q88 阈值）|
| model_B | `target_extreme_norm`（极值）| `pred_extreme`（价格目标）| 止盈目标 |

两个模型可以共享同一次 prepare_data + factor_audit，训练独立，推断时串联。

### 8.4 实现方向

**训练侧**：`dataset.py::_add_targets` 同时生成 `target_vol_norm` 和 `target_extreme_norm`，两次调用 `train_dual_regime_models`，`target_col` 分别传不同列。

**回测侧**：`backtest.py::generate_positions` 接收两个 artifact_map（或在 prediction df 里新增 `pred_extreme` 列），在 position loop 内加入 `cumret >= pred_extreme` 的提前止盈判断。

**前提**：策略 A 和策略 B 各自独立验证后再实现融合，避免在未验证的基础上叠加复杂度。

---


## 一些结论

### 是否 Vol 归一化：check_VolNorm 实验结果

代码：`scripts/check_VolNorm.py`，10 个品种（HC 因 val 期高波段为空报错），共享 prepare_data + factor_audit。

| product | vol_norm net | raw_return net | vol_norm sharpe | raw_return sharpe | vol_norm IC | raw_return IC | vol_norm trades | raw_return trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RB** | +1.96% | -0.22% | +0.39 | -1.05 | -0.0019 | +0.0260 | 76 | 1 |
| **CU** | +3.54% | -2.04% | +0.59 | -0.82 | +0.0106 | +0.0168 | 66 | 15 |
| **AU** | -9.06% | +18.09% | -0.60 | +1.65 | -0.0105 | +0.0046 | 156 | 71 |
| **M** | -2.87% | -0.88% | -1.39 | -0.94 | -0.0236 | -0.0115 | 2 | 5 |
| **AG** | +30.74% | -1.23% | +0.92 | -0.80 | -0.0038 | -0.0358 | 202 | 8 |
| **AL** | -8.96% | +1.66% | -1.17 | +0.57 | +0.0019 | +0.0075 | 43 | 5 |
| **ZN** | +5.28% | +7.98% | +0.97 | +1.62 | +0.0171 | +0.0554 | 14 | 9 |
| **SN** | -12.49% | +7.72% | -1.09 | +1.25 | -0.0329 | +0.0087 | 188 | 42 |
| **JD** | +0.61% | -4.93% | +0.15 | -1.81 | -0.0005 | +0.0103 | 111 | 33 |
| **avg** | +0.97% | +2.91% | -0.14 | -0.04 | -0.0048 | +0.0091 | — | — |

> **HC failed**: `Regime 'high_vol' has empty split(s): val` — val 期高波段行为空，与 vol_norm/raw_return 无关，是数据本身的 regime 分布问题。

**结论**：9 品种中无法得出统一结论，两种方案各有胜负。

按 Sharpe 胜负统计：**vol_norm 胜 4 / raw_return 胜 4 / 平手 1**

| vol_norm 胜 | raw_return 胜 |
|---|---|
| RB (+0.39 vs -1.05) | AU (-0.60 vs **+1.65**) |
| CU (+0.59 vs -0.82) | AL (-1.17 vs **+0.57**) |
| AG (+0.92 vs -0.80) | ZN (+0.97 vs **+1.62**) |
| JD (+0.15 vs -1.81) | SN (-1.09 vs **+1.25**) |
| ↑ M 平手（均差）| |

**关键现象**：

1. **raw_return IC 系统性更高**（avg +0.0091 vs vol_norm -0.0048）——反归一化让预测和 future_return 的相关性下降，但 P&L 不一定更差。这再次印证 §4 的 IC-P&L 断层。

2. **raw_return 的 trades 系统性更少**——raw_return 模型在低波段产生的极端预测更少（梯度被高波主导），导致很少越过 Q88 阈值。RB 整个测试期仅 1 笔，AG 仅 8 笔，实际上等同于空仓。vol_norm 的梯度均衡让低波信号也能入场。

3. **vol_norm 胜出的品种**（RB/CU/AG/JD）恰好是 raw_return trades 极少的品种——raw_return 不是"更差"，而是几乎不交易。vol_norm 通过更多交易实现了更好的净收益。

4. **raw_return 胜出的品种**（AU/AL/ZN/SN）raw_return 仍有一定交易频率（5–71 笔），说明这些品种的高波段信号足够强，梯度不均衡不成问题；而 vol_norm 的反归一化对这些品种引入了额外噪声。

**暂定结论**：保留 vol_norm 作为默认，原因是 raw_return 在低波品种上近似空仓，这不是一个稳健的生产策略。但需要关注 AU/SN/AL 这类品种，后续可以考虑按品种类型选择 target。

---

### 是否分 Regime：check_Regime 实验结果

代码：`scripts/check_Regime.py`，20 个品种，15 个成功运行（HC/L/PP/V/P 因 val 期 high_vol 段为空报错），共享 prepare_data + factor_audit。

| product | dual net | single net | dual sharpe | single sharpe | dual IC | single IC | dual trades | single trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RB** | +1.96% | -8.87% | +0.39 | -1.45 | -0.0019 | +0.0242 | 76 | 170 |
| **CU** | +3.54% | +5.71% | +0.59 | +0.82 | +0.0106 | +0.0297 | 66 | 83 |
| **AU** | -9.06% | -3.63% | -0.60 | -0.31 | -0.0105 | -0.0143 | 156 | 254 |
| **M** | -2.87% | +0.00% | -1.39 | +0.00 | -0.0236 | +0.0169 | 2 | 0 |
| **AG** | +30.74% | +21.22% | +0.92 | +0.67 | -0.0038 | +0.0076 | 202 | 304 |
| **AL** | -8.96% | -4.53% | -1.17 | -0.54 | +0.0019 | +0.0048 | 43 | 50 |
| **ZN** | +5.28% | +2.12% | +0.97 | +0.48 | +0.0171 | +0.0024 | 14 | 6 |
| **SN** | -12.49% | -10.08% | -1.09 | -0.87 | -0.0329 | -0.0209 | 188 | 183 |
| **JD** | +0.61% | -3.68% | +0.15 | -0.95 | -0.0005 | +0.0031 | 111 | 35 |
| **RU** | -7.52% | -2.65% | -1.59 | -0.65 | +0.0015 | +0.0197 | 43 | 28 |
| **BU** | -5.17% | -1.65% | -1.75 | -1.25 | +0.0036 | -0.0016 | 10 | 4 |
| **FU** | -13.42% | +23.19% | -0.54 | +1.31 | +0.0049 | +0.0066 | 148 | 69 |
| **J** | -8.56% | -0.17% | -1.87 | -0.04 | -0.0036 | -0.0016 | 33 | 6 |
| **JM** | +5.71% | -2.00% | +0.27 | -0.19 | +0.0380 | +0.0196 | 195 | 19 |
| **Y** | +2.38% | -0.33% | +1.17 | -0.15 | +0.0284 | +0.0023 | 2 | 2 |
| **avg** | -1.19% | +0.98% | -0.37 | -0.21 | +0.0019 | +0.0066 | — | — |

> **HC/L/PP/V/P failed**: `Regime 'high_vol' has empty split(s): val` — val 期无 high_vol 数据，无法校准阈值。

**关键发现**：

1. **均值被 FU 单品种拉偏**：dual avg -1.19% 主要由 FU（-13.42%）拖累；剔除 FU 后 dual avg ≈ +0.36%，与 single 基本持平。FU 根因是 low_vol 模型 val_IC=-0.019（详见 § 2.1 质量门控）。

2. **single_model IC 系统性更高**（avg +0.0066 vs dual +0.0019），但净收益不一定更好——IC-P&L 断层再次得到印证。

3. **single_model 过度交易**：在 dual 胜出的品种（RB/AG/ZN/JD/JM/Y），single trades 系统性更多（RB: 170 vs 76，AG: 304 vs 202），高频交易成本吃掉 IC 带来的毛利。

4. **dual 下行保护更强**：RB 单模型崩溃至 -8.87%，dual 保持 +1.96%；AG dual +30.74% vs single +21.22%，差距显著。

**结论**：保留 dual_regime，并引入 regime-level 质量门控（§ 2.1）处理退化模型，预期恢复 FU/AL/BU 三个品种的损失。

---

### Horizon × Vol_cutoff Grid Search 实验结论

来源：`scripts/exp_pure_micro_horizon_cutoff.py`，结果记录于 `docs/exp_pure_micro_horizon_vs_cutoff.md`。

实验条件：禁用中观因子，仅用微观+工程因子，6 个品种（AU/AG/JD/M/CU/SN），横扫 horizon ∈ {1,5,10,30,60,120} × cutoff ∈ {0.60,0.65,0.70,0.75,0.80}。

#### Horizon 效应（cutoff=0.65 下的全品种均值）

| horizon | mean net ann | mean sharpe | mean test IC | 含义 |
|---:|---:|---:|---:|---|
| 1 bar (5min) | −27.95% | −5.30 | +0.114 | IC 最高但亏损最严重 |
| 5 bar (25min) | −35.64% | −3.73 | +0.047 | 最差 |
| 10 bar (50min) | −20.64% | −2.40 | +0.024 | 中等 |
| 30 bar (150min) | −11.40% | −1.13 | −0.006 | 当前配置 |
| 60 bar (300min) | −10.16% | −0.86 | −0.005 | 略优于 30 |
| **120 bar (600min)** | **−6.67%** | **−0.41** | +0.008 | **纯微观最优** |

**核心矛盾**：horizon=1 的 IC 最高（0.114），但净收益最差（−28%）。这是**IC-P&L 断层**。

#### 为什么短 horizon 高 IC 却亏损？

短 horizon 的 IC 高是因为信号对 5min 内的价格扰动敏感，但：
- 开仓频率极高（horizon=1：约 1200–2000 trades/yr）
- 每笔 commission+slippage = 2bp，高频下成本累积吃掉全部毛利
- horizon=30 trades ≈ 300–500/yr，horizon=120 trades ≈ 100–250/yr，成本拖累大幅下降

**结论**：IC 衡量的是模型预测能力；能否盈利还取决于 `signal-to-cost ratio = gross_edge / round_trip_cost`。短 horizon 毛利薄，成本比例高。

#### Vol cutoff 效应

- **cutoff=0.65**（top 35% = high_vol）在 horizon ≥ 30 下几乎全面优于 cutoff=0.75
- 当前 `vol_percentage=0.65` 是正确选择，无需更改

#### 品种类型与最优 horizon 的关系

| 品种类别 | 代表 | 最优 horizon | 最优 cutoff | 特点 |
|---|---|---:|---:|---|
| high-vol heavy | AU, AG | 60–120 | 65–75 | high_vol bar 多，较长 horizon 成本可控 |
| balanced | JD, M | 120 | 65 | 中等 high_vol，长 horizon 最优 |
| low-vol heavy | CU, SN | 60–120 | 65 | high_vol bar 极少，长 horizon 减少无效交易 |

#### 对当前配置（horizon=30, cutoff=0.65）的评估

horizon=30 是 grid 中第 4 好，cutoff=0.65 是全局最优。

**当前不改 horizon 的理由**：grid search 是纯微观条件下的结论。中观因子加入后毛利提升，horizon=30 的 cost/return ratio 变好。如果要重新实验应在中观因子配置下重跑。

---

## 9. Optuna 超参搜索（LightGBM common_params）

### 9.1 动机

`config.yaml::model.common_params` 中的 LightGBM 超参（learning_rate=0.03、num_leaves=63、max_depth=6、min_child_samples=120、feature_fraction=0.8、bagging_fraction=0.8、reg_alpha=0、reg_lambda=1.0）来自人工设定，未做系统性搜索。在中观因子框架稳定后，希望用 Optuna 对微观骨干层做一次系统调参，看是否能进一步压榨现有信号。

### 9.2 设计原则

- **全局参数，不做品种级调参**——所有品种共用一组超参（与 `feedback_no_per_product_tuning` 一致）。Optuna 找的是"对所有品种平均最优"，不是"在某个品种上最优"。
- **目标函数 = mean(val Spearman IC)** 跨所有品种、按 val 行数在两个 regime 内加权后再算品种均值。选 IC 而非 val Sharpe 的原因：(i) 与训练目标贴近；(ii) 不需要在每个 trial 内额外跑回测，速度快 5—10×；(iii) val Sharpe 与 val IC 高度相关。
- **测试集严格不参与**——Optuna 只看 train→val 流程，test 留作最终评估。
- **保留早停**——每个 trial 内 LightGBM 仍走 num_boost_round=600 + early_stopping_rounds=50，防止单 trial 过拟合，同时让搜索更快（多数 trial 在 100—300 轮就停）。
- **微观调参先行**——v1 以 `use_mid_weekly: false` 跑全品种调参，先把微观骨干层校准到位，再视情况打开中观重跑。

### 9.3 搜索空间

| 参数 | 当前值 | 搜索范围 | 类型 | 备注 |
|---|---|---|---|---|
| learning_rate | 0.03 | [0.005, 0.05] | log uniform | 用户指示窄化（避免过激 lr 把 ES 推早）|
| num_leaves | 63 | [31, 127] | int | 与 max_depth 联动 |
| max_depth | 6 | [4, 10] | int | 浅树→泛化、深树→拟合 |
| min_child_samples | 120 | [50, 300] | int | high_vol 域强制下限 150 |
| feature_fraction | 0.8 | [0.5, 1.0] | uniform | 列子采样 |
| bagging_fraction | 0.8 | [0.5, 1.0] | uniform | 行子采样（freq=5）|
| reg_alpha | 0 | [1e-8, 1.0] | log uniform | L1 |
| reg_lambda | 1.0 | [1e-8, 5.0] | log uniform | L2 |

### 9.4 工程实现要点

入口：`scripts/optuna_tune.py`

```
prepare 阶段（每品种一次）：
  prepare_data → audit_and_filter → 取 selected feature_cols
  按 regime 切片 + 预 fit RobustScaler → 缓存 (X_train, y_train, X_val, y_val) 到内存

trial 阶段（每 trial 重复）：
  for product in products:
    for regime in (-1, 1):
      lgb.train(params=trial_params, early_stopping=50)
      pred_val → spearman_ic(pred, y_val)
    weighted_avg(regime_ics) → product_ic
  mean(product_ics) → trial value
```

性能优化：

- **特征矩阵缓存到内存**：避免每个 trial 重做 prepare_data（每品种 1—2 分钟）
- **预先 fit scaler**：RobustScaler 只需 fit 一次，每 trial 直接 transform
- **MedianPruner**：n_startup_trials=5、n_warmup_steps=n_products//6，差的 trial 中途中止
- **sqlite 存储**（`results/optuna/study_*.db`）：支持中断续跑，同 study-name 启动会在已完成 trial 上接着跑

### 9.5 与 config 的耦合

- 关闭 parquet 缓存（`cache_merged_dataset: false`、`cache_generated_features: false`）——Optuna 跑期间不写盘
- regime overrides 保留：`high_vol.min_child_samples` 强制 ≥ 150（与 `config.yaml::model.high_vol_overrides` 一致）
- `early_stopping_rounds=50` 与生产配置同步

搜索完毕后，最优参数写回 `model.common_params`（不写 `high_vol_overrides`，让现有 +30 min_child_samples 的规则继续生效），并以全品种回测验证 Sharpe / 收益 / 回撤是否真有提升。

### 9.6 风险与边界

- **多品种均值目标的拉平效应**：mean(IC) 偏好"对所有品种都还行"的参数，可能错过"在 4 个稳健品种上极强、其他品种平庸"的局部最优。如果未来要专注 AL/M/PB/BU 的产品化，需要单独在这 4 个品种上做一次窄目标搜索作为对照。
- **超参与因子层耦合**：纯微观下找到的最优可能在加入中观后失效（特征维度从 ~210 涨到 ~250、IC 信噪比变化）。微观调参的结论需要在中观打开后做一次 sanity check。
- **IC 与 Sharpe 错位风险**：val IC 高 ≠ val Sharpe 高。极少数情况会出现 trial 提升 IC 但 backtest Sharpe 反而下降（信号方向更准但过度自信致开仓密度过大）。最终决策仍以全品种回测的 Sharpe / 净值为准，而非 Optuna 目标值。

### 9.7 v1 实测结果（2026-05-07）

**搜索结果：** 30 trials（19 完成 / 11 剪枝），39 品种参与，最优 mean(val Spearman IC) = **0.02998**（trial 20）。

| 参数 | 旧默认 | Optuna v1 |
|---|---:|---:|
| learning_rate | 0.030 | **0.0118** |
| num_leaves | 63 | 62 |
| max_depth | 6 | **7** |
| min_child_samples | 120 | **206** |
| feature_fraction | 0.80 | **0.583** |
| bagging_fraction | 0.80 | **0.563** |
| reg_alpha | 0 | **0.104** |
| reg_lambda | 1.0 | **~1.66e-7** |

参数已落 `config.yaml::model.common_params`。同步移除 `high_vol_overrides.min_child_samples=150`：旧 override 在 common=120 时是加强正则，但 common=206 后反向变成"降低高波域正则化"，因此清空。

**全品种 test 集回测对比（`optuna_micro_v1` vs `baseline_roll240_75c_nomid`，34 共同品种，纯微观）：**

| 指标 | baseline | Optuna v1 | Δ |
|---|---:|---:|---:|
| 平均 Sharpe | −0.285 | −0.277 | +0.008 |
| 中位 Sharpe | −0.218 | 0.000 | +0.062 |
| 平均年化 | −0.34% | **+0.60%** | +0.94pp |
| 平均回撤 | −4.98% | **−6.71%** | **−1.73pp（变差）** |
| 平均交易笔数 | 77 | 96 | +19 |
| 正 Sharpe 品种数 | 10 | **13** | +3 |
| 年化 > 0 品种数 | 8 | **15** | +7 |
| 改善 / 持平 / 退步 (ΔSharpe ±0.30) | — | 13 / 10 / 11 | — |

对比文件：`results/comparison/optuna_micro_v1_vs_baseline.{csv,md,png}`

**核心观察：**

1. **不是无脑加分，而是"换 portfolio"。** Optuna 把基线里 4 个 Sharpe ≥ 1.0 的强品种结构变了 3 个（AU +1.30→+0.16、SP +1.11→−0.20、JD +1.34→−0.71、J +0.85→**−1.34**），换来 5 个新的稳健品种（V / I / BU / NI 大幅提升 + AL 小幅提升）。
2. **Optuna 头部组合年化基底约 +8%、回撤 < 3%。** 5 品种等权（I / AL / V / NI / PB）平均年化 +8.2%、平均回撤 −2.7%，回撤显著优于 baseline 头部组合（−4.4%）。
3. **AG 是伪头部不可信。** 年化 +56% / 回撤 −37% / 交易暴增 11× 是 Optuna 把模型推到激进交易后碰上贵金属单边行情的结果，不能算 Optuna 真正的功劳。
4. **J 是最大灾难。** 年化从 +4.9% 跌到 −10.9%（−15.8pp），回撤从 −4.6% 加深到 −13.0%，交易 32→142，新参数把这个品种原本的清晰信号打散了。
5. **以 mean(val IC) 为目标不能控制回撤。** val IC 提升 ~15% 没有等比例传到 Sharpe，且平均回撤反而加深 1.73pp——目标函数缺少分布稳定性约束。

**决策：** 接受 Optuna v1 参数为新基线，理由：(i) 中位 Sharpe 与年化均改善；(ii) 头部组合的 Sharpe/回撤性价比更优，更适合产品化；(iii) 全样本年化 > 0 的品种数从 8 增至 15，覆盖面更广。

### 9.8 v2 计划

在 Optuna v1 微观参数基础上打开 `use_mid_weekly: true`，复用 §9 三层审计中观接入流程，跑全品种回测。验证两件事：

- 微观最优参数在中观加入后是否仍然适用（特征维度从 ~210 涨到 ~250、IC 信噪比变化）
- 中观 + Optuna 双重增强能否进一步提升头部品种的稳健性

对比基线：`tiered_mid_full`（v3 旧默认参数 + 中观）；如果 v2 显著优于 v1（纯微观）+ tiered_mid_full（中观+旧参），则形成新的多品种生产基线。



