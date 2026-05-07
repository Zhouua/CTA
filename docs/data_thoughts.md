# 数据处理思路

本文档记录**为什么这样处理数据**，而不是机械描述代码做了什么。
读这份文档是为了理解每个数据决策背后的取舍。
代码层细节看 `pipeline/dataset.py` 与 `dataloader/splitByVol.py`；
因子相关的设计看 `docs/factor_thoughts.md`。

---

## 数据源 — 单一来源主连 + 外挂中观

- **1min 主力连续行情** 来自 `data/分产品1min主连/`（`<PID>ZL.<EXCH>.csv`），
  这是已经主连展期的连续序列，不再做 contract 切换处理。
- **中观日 / 周指标** 是外部 xlsx，与行情时间轴解耦，由 pipeline 自己对齐。
- **品种注册表** `data/product_registry.json` 是数据层唯一"事实表"，列举品种、
  数据起止、流动性指标。

**取舍**：选注册表而不是 glob 文件系统的原因 —— 让"启用哪些品种"成为可审计、
可版本化的状态，避免新增/移除一个 CSV 就静默地改变批量训练范围。

---

## 原始 CSV 清洗

读到的脏数据（OHLC ≤ 0 但 VOLUME/AMOUNT 非零）选择 **遮罩为 NaN 后 ffill**，
不是 `dropna(OHLC)`。

**取舍**：`dropna` 会在时间序列里挖洞，下游 rolling 因子横跨这些洞会得到
不可解释的窗口；`ffill` 至少保留时间连续性，下游能正确滚动。这种脏数据本身
是数据源问题，不是模型该对付的边界条件。

不做的事：

- **不做"零成交量 bar"过滤**：bar 级零成交是品种级问题，由决策 9 的全局
  `zero_volume_ratio` 门槛在入口拦掉，不在 bar 级修。
- **不重采样、不补缺失 bar**：用原始时间戳，避免引入虚假 bar 让因子误解锁
  伪信号。

---

## 中观因子的对齐方法 —— 多频独立 lag + 频率感知过滤 + 有限 ffill

中观源是稀疏的（日/周/月观测），要对齐到 1min 网格，我做了三个非显然的选择：

### 3.1 按列频率独立 lag，而不是全局 lag

每个 xlsx 列读出"频率"元字段后单独偏移：默认 `日=1 / 周=3 / 月=5`。
同一 xlsx 内不同 freq 的列被 lag 到不同 ts，再 outer-merge 拼回 wide。

**取舍**：早期版本用单一 `publication_lag_days` 全局值，结果"日级数据被推迟
3 天才可用"显然过保守，"周级数据当天就能用"显然有前视。多频独立 lag 让"今天
能看到的数据"严格匹配真实发布节奏。

### 3.2 频率感知质量过滤：`eff_ratio = nn_ratio / freq_expected_ratio[freq]`

不能用裸的"非空率 < 阈值"过滤 —— 周列在每日网格上天然只有 ~20% 非空，
和"日列覆盖 20%"是两回事。把 `freq_expected_ratio = {日:1.0, 周:0.2}`
作为基准，看**有效覆盖率** `nn_ratio / expected`。

`min_active_ratio=0.6` 把那些"该有但实际也只有 60% 的列"剔除。

`drop_step_dummy=true` 单独处理"前段全空 + 后段稠密 ≥ 0.9"的 step-dummy
（数据源中后期才接入的列），避免 rolling 跨段被污染。

被剔除的列写进 `cache_meta.json::mid_weekly_dropped` 留审计。

### 3.3 ASOF backward + 有限 ffill（staleness clamp）

`merge_asof(direction="backward")` 把稀疏观测对齐到 1min 网格，但**不允许
无限 ffill**：从最近一次到达 ts 起，超过 `ffill_max_bars=8064`（≈ 5.6 天）
的行强制 NaN。

**取舍**：周/月数据如果一个观测能 ffill 几个月，模型会把这种"长期不变的常量"
当 regime flag 用，挤占真实信号 gain（这是 mid v1→v2 实验里观察到的退化）。
clamp 后 NaN，配合 `MID_*_AVAILABLE` int8 dummy 让模型显式知道"现在这个
指标过期了"。

---

splits + regime 标签 —— vol cutoff 仅由 train 决定

**不变量**：vol cutoff 必须由 train rows 算出，绝不能用 model 推。
任何"新的 regime 想法"都要照此规则：在 train 上算阈值，应用到所有 split。

切分顺序：

1. 月度（`split_granularity=month`）划 `train:val:test=0.70:0.15:0.15`。
2. 日级聚合算 `daily_vol_20`。
3. **rolling regime 默认开启，window=240 天**：每天用过去 240 天的
   `daily_vol_20` 在 `vol_percentage=0.65` 分位数上判定高/低波动。
4. `merged_data["DATA_SPLIT"]` 内部用 `valid` → `prepare()` 末端 rename 为 `val`。

**取舍：rolling 240 vs 固定 train 阈值**：
固定 train 阈值的问题是"训练期 vol 中枢和测试期严重不同"时，cutoff 失准。
rolling 240 自适应，22 个共同品种实测平均 +5.49pp。但**仍然只看过去 240 天**，
依然无前视。

`min_train_rows_per_regime=20000` 是硬地板：任一 regime train 行数低于此值
直接抛错。这个数字反映的是 LightGBM 在 1min 数据上"两段都能学到稳定的 IC"
的最小样本要求，低于这个就训不出来。

---
## 品种入场门槛

`pipeline/train_products.py::annotate_products_for_batch_skip` 在 batch
训练前一次性把 registry 标记为可训 / 跳过：

| 规则 | 配置键 | 失败状态 |
|---|---|---|
| `enabled=true` | `product_registry.json::enabled` | `skipped_disabled` |
| 数据日期覆盖 | `enforce_registry_coverage / required_data_start / required_data_end` | `skipped_insufficient_coverage` |
| **零成交量比** | `max_zero_volume_ratio`（默认 0.5）| `skipped_low_liquidity` |

`zero_volume_ratio` 由 `build_product_registry.py` 在扫 CSV 时一次算出
（`VOLUME==0` 的行数 / 总行数），存进 registry。

**取舍**：选**全局阈值** + 注册表预计算，而不是"WR 看着不行就 blacklist"。
理由：per-pid override 把判断散在代码里，新加品种容易遗漏；全局阈值是统一
规则，新品种自动用同一标尺。`max_zero_volume_ratio=0.5` 这个值反映"过半 bar
零量则 backtest 撮合不可信"，对应任何品种都成立。

`scripts/check_VolNorm.py / check_Regime.py / check_StrategyB.py` 共享同一
门槛逻辑（`_load_registry → annotate_products_for_batch_skip`），保持训练
与对比脚本视角一致。

**实现细节**：registry 扫描走 `pd.read_csv(usecols=[…])` list-based 列裁剪，
65 个品种全扫几十秒。先用 `csv.DictReader` 整文件迭代被卡死过 17 分钟才换掉。

---

## 训练矩阵装配 —— train-only 筛选 + 双轨缺失阈值

`prepare()` 末端的列裁剪只看 train rows，避免用 val/test 分布做选择：

1. 替换 `±inf → NaN`，对 *非* MID 列做 `ffill`（不覆盖 MID staleness clamp）。
2. **train-only 缺失率筛选**（双轨）：
   - 非 MID 列：缺失率 ≤ `max_factor_missing_ratio=0.35`
   - MID 列：缺失率 ≤ `mid_weekly_missing_ratio_relax=0.65`
3. **train-only 方差筛选**：std ≤ `min_factor_std=1e-8` 的列剔除。
4. MID 列剩余 NaN 用 `0.0` 填，配合 `MID_*_AVAILABLE` dummy。
5. `dropna(subset=feature_cols + [target_col, future_return, REGIME_LABEL,
   DATA_SPLIT])` 拿到最终 train/val/test。

**取舍：MID 单独走 0.65 阈值**：MID 列天然稀疏（周列只有 ~20% 非空 + lag/clamp
后更稀疏），用 0.35 全砍掉则中观信号根本进不来。0.65 让最稀疏的周指标也有
机会留下，配合 AVAILABLE dummy 标识其有效性。

**取舍：MID 用 0.0 填而不是 dropna**：dropna 严格的话，xlsx 发布前 / clamp
后窗口里所有 bar 都会被丢，相当于直接放弃这段时间的所有训练数据。0.0 + 
AVAILABLE dummy 让模型看到"这段时间没有外生信号"，由模型决定权重。

候选列里的 IC / ICIR 因子裁剪由 `pipeline/factor_audit` 在此之后调用，
详见 `docs/factor_thoughts.md`。

---

## 已踩过的坑（数据层）

- **OHLC=0 脏数据**：上游偶发，`_read_raw_data` 末段 ffill 修复，不要 dropna
  否则下游 rolling 会跨洞。
- **stale tick × raw_return**：raw_return 在低流动品种上学伪信号；vol_norm
  鲁棒，`max_zero_volume_ratio` 把品种本身拦掉，双重保险。
- **mid_weekly step-dummy**：xlsx 列前期全空、后期稠密，过去会污染 rolling。
  `drop_step_dummy=true` 直接 drop 这类列。
- **mid_weekly 单一 lag**：早期全局 lag 一刀切，日级数据被推迟过头、周级
  又有前视。已切成 per-freq lag。
- **缓存粒度漏字段**：`min_active_ratio` 不进签名导致改了阈值还读旧 parquet，
  调参看不到效果。已纳入签名。
- **DATA_SPLIT 命名**：`split_by_vol` 内部叫 `valid`，`prepare()` 末端 rename
  为 `val`；下游一律用 `val`。
- **regime 切分图硬编码 5min**：早期画 `5min_return`，与 `target_horizon`
  脱钩，看图判断高/低波分割效果是误导。改成 `plot_target_return_by_vol(...,
  return_col="future_return", target_horizon=h)` 后图与训练目标一致。
- **registry 扫描慢路径**：早期用 `csv.DictReader` 逐行迭代，65 文件卡 17
  分钟。改 `pd.read_csv(usecols=list)` 后几十秒搞定。
