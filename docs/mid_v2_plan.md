# G1 — 全局关闭 MID 原始水平列（`level_keep: false`）

本文写给用户在动手前过目：先搞清楚"现在的 MID 因子是什么、G1 会拿掉什么、会剩什么、为什么预期能修 FB/BB，风险在哪"。落地修改仅一处 config + 一处缓存签名，**不改算法、不做 per-pid**。

---

## 1. 当前（P2 之后）的 MID 因子完整画像

### 1.1 三层源头 → 最终特征矩阵

每个品种在 registry 里绑定一个或多个 `mid_weekly_files`（例如 `FB.xlsx`）。一个 xlsx 里每一列是一个**源指标**（`indicator_id`），携带元数据 `{名称, 频率(日/周), 单位, 来源}`。一列源指标在 pipeline 末端最多派生出下列 5 类特征：

| 类别 | 命名 | 生成位置 | 当前默认 | G1 后 |
|---|---|---|---|---|
| **① 原始水平列** | `MID_<PID>_<indicator_id>` | `_read_mid_weekly_xlsx` 读 xlsx 的原始值列 | 保留（`level_keep=true`）| **关闭** |
| **② 派生因子** | `MID_..._RET_{w}` / `_ZSCORE_{w}` / `_PCT_RANK_{w}` | `_compute_mid_weekly_derivatives`（`dataset.py:601`）| 保留（`derived.enabled=true`）| **保留** |
| **③ AVAILABLE 哑变量** | `MID_..._AVAILABLE` | `_merge_mid_weekly_features` 内部 | 保留（`available_dummy=true`）| **保留** |

每列源指标的派生数量 = `|windows| × |transforms|` = `3 × 3 = 9`（`windows=[4,13,52]`，`transforms=[ret, zscore, pct_rank]`）。

所以一条源指标**最多**产出 `1 (level) + 9 (derived) + 1 (AVAILABLE) = 11` 列。但受下面四道关卡影响实际会少。派生因子的**精确计算方式**在 §1.2 展开。

### 1.2 派生因子的精确计算（`_compute_mid_weekly_derivatives`, `dataset.py:601`）

**关键事实 1：派生在"稀疏观测网格"上算，不在 5min 网格上算。**
对每一条源列 `col` 做：`s = factor_df[[ts, col]].dropna().set_index(ts)[col]`。`s` 只含 xlsx 里真有观测的行（周频 → ≈ 1 行/周；日频 → ≈ 1 行/交易日）。后续 rolling / pct_change 全部在 `s` 的观测索引上进行。

**关键事实 2：`windows=[4, 13, 52]` 是 `s` 上的"观测数"，不是日历时间、不是 5min bar 数。**
对周频列 → 4/13/52 周；对日频列 → 4/13/52 个交易日。**同一个 `w` 在两种频率上语义完全不同。**这是当前代码的行为（见 `s.rolling(w, ...)` 和 `s.pct_change(w)`，w 作用在 `s` 的观测步长上），不是文档瑕疵 —— 周频 + 日频混用时，`_RET_52` 对周频 ≈ 年度收益率，对日频 ≈ 2 个月收益率。

**关键事实 3：`min_required = max(2, min(windows)) = 4`。** 源列观测数 < 4 则整条源列**跳过派生**，仅保留 level + AVAILABLE（若 level_keep）。

**三个 transform 的公式**（`w` 取自 `windows`，在 `s` 的观测索引上滚动）：

| Transform | 列名 | 公式 | min_periods | 额外处理 |
|---|---|---|---|---|
| `ret` | `<col>_RET_{w}` | `s.pct_change(w)` = `s[t] / s[t − w 观测] − 1` | 默认（w 个观测齐全后才产值）| `.clip(−1.0, 5.0)`（硬截到 [−100%, +500%]） |
| `zscore` | `<col>_ZSCORE_{w}` | `(s[t] − rolling_mean_w) / rolling_std_w` | `max(1, w//2)` = `2 / 6 / 26` | `rolling_std == 0` → NaN（避免除零） |
| `pct_rank` | `<col>_PCT_RANK_{w}` | 在最近 w 个观测里，当前值的百分位（`rolling(w).rank(pct=True)`）| `max(1, w//2)` | 无 |

**关键事实 4：派生计算完之后与 level 列一起走完全相同的下游路径：**`merge_asof(backward)` 进 5min 网格 → `ffill()` → staleness clamp 统一置 NaN（`dataset.py:711–729`，`carry_cols = file_level_cols + derived_cols_file`）。

**这意味着派生列也存在"5min 网格上分段常量"现象：**两个相邻观测之间（周频 ≈ 480 根 5min bar），派生值 = 上个观测时刻算出来的那个数字，ffill 保持不变。G1 的赌注是：虽然派生也 ffill，但**派生值本身已经是"相对过去 w 个观测的统计量"**（pct_change / 标准化 / 百分位），与"现货价 1234.56 这个裸数字"在 LGB 分裂时的表现不同 —— 裸数字会被当 regime flag 用，派生值的跨观测幅度大多在 [−1, +1] 量级（zscore / pct_rank）或 [−1, 5] 截断后的变动区间（ret），不容易变成"几乎分段常量"的伪开关。**这个赌注如果输了，就要上 G2 收紧 ffill_max_bars 或 §2.4 提到的关派生兜底。**

### 1.3 当前 pipeline 四道关卡（从源到特征矩阵）

```
xlsx (源指标 N 条) 
  → [关卡 A] _apply_mid_weekly_quality_filter  (§15 P2 频率归一化过滤)
          条件：eff_ratio = nonnull/expected_ratio < 0.6  → drop
                OR  step-dummy 形态 → drop
          影响：level/derived/AVAILABLE 一并不生成
  → _compute_mid_weekly_derivatives（只对 survive 的 N' 列生成派生）
  → pd.merge_asof(5min_bars, backward) + ffill（把稀疏观测铺到 5min 网格）
  → [关卡 B] staleness clamp  (ffill_max_bars=8064 ≈ 4周)
          超过 4 周没新观测 → level + derived 置 NaN；AVAILABLE 对应 0
  → AVAILABLE 哑变量（clamp 之后再生成，保证"观测已过期→AVAILABLE=0"）
  → 进 feature matrix 候选
  → [关卡 C] prepare 的前缀分流 missing_ratio_relax (dataset.py:1066)
          MID_* 列的 train-missing ratio > 0.65 → drop
          非 MID_* 列用默认 max_factor_missing_ratio=0.35
  → [关卡 D] 方差过滤 min_factor_std
          MID_* 列的 NaN 先 fillna(0.0) 再 dropna 全行（dataset.py:1086）
  → 实际 feature matrix
```

### 1.4 当前规模（来自 `results/runs/20260422_154414` 候选 run，即 full MID）

| PID | 源列数 | mid_weekly_feature_count | 估算构成 (level + AVAILABLE + derived) |
|---|---:|---:|---|
| FB | 5 | 39 | 5 + 5 + ~29 |
| BB | 5 | 39 | 5 + 5 + ~29 |
| RU | 8 | 66 | 8 + 8 + ~50 |
| FU | 8 | 87 | 8 + 8 + ~71 |
| M | 14 | 148 | 14 + 14 + ~120 |

派生列没达到上限（9×N）是因为 `windows=[4,13,52]` 的 52 周窗口需要至少 52 个观测，稀疏列凑不够就跳过（`dataset.py:625`）。

---

## 2. G1 的确切改动与预期效果

### 2.1 改动点（**仅 2 处**）

1. **`config.yaml::mid_weekly.level_keep`**：`true` → `false`。
2. **`pipeline/dataset.py::FactorDatasetBuilder`** 缓存签名（`_cache_signature` / `_build_signature` 构造处，具体行号落地时确认）把 `mid_weekly_level_keep` 纳入签名 dict，使 `true/false` 互不命中旧 parquet。

**不改**：
- `_compute_mid_weekly_derivatives`（派生逻辑）
- `_apply_mid_weekly_quality_filter`（P2 过滤保留，继续生效）
- `ffill_max_bars`（staleness 暂不动；如果 G1 不够，再做 G2）
- `missing_ratio_relax`（前缀分流不动）
- `modeling.py` / `backtest.py`（§10.1 不变量）

### 2.2 G1 之后特征矩阵里还剩什么

保留的 MID 特征类别（**全品种统一**）：

- `MID_<PID>_<ind>_AVAILABLE`：int8 二值，是否有过期内的新观测。
- `MID_<PID>_<ind>_RET_{4|13|52}`：源序列在观测网格上的 pct_change，clip 到 [-1, 5]。
- `MID_<PID>_<ind>_ZSCORE_{4|13|52}`：`(x − rolling_mean) / rolling_std`，窗口按观测数。
- `MID_<PID>_<ind>_PCT_RANK_{4|13|52}`：rolling percentile rank。

拿掉的：`MID_<PID>_<ind>` 原始水平列本身（即"xlsx 里那列数据直接 ffill 到 5min 网格"的那个特征）。

派生列仍然走 `merge_asof + ffill + staleness clamp`，但派生序列的**观测值本身每次更新都会变**（是 4/13/52 窗口的标准化/比值），而水平列在两次观测间值完全不动 → 派生列虽然也会 ffill，但它至少不是"长达几千根 bar 的同一数字"，LGB 不会把 ffill 段当成 regime 切换。

### 2.3 为什么 G1 期望能修 FB/BB 而不伤改善组

- **FB 现状**：5 条源列里 3 条周频（nonnull 0.16/0.17/0.17），2 条日频。P2 对这 3 条周频的 eff_ratio ≈ 0.85（> 0.6）所以全保留。`§14.1` 显示 FB/low_vol top-20 里水平列占 2 席，骨干列 `CORR30 / ENG_POSITION_RATIO_60 / daily_vol_20 / STD60 / ROC60` 全部落榜 → Sharpe 2.59→-0.04。
  G1 之后 FB 的 5 个水平列消失，派生列仍在但因为 `pct_change/rank/zscore` 不是近似常量，不会抢 top-20。
- **BB**：完全同构（5 条，3 周频）。同理。
- **改善组**（§14.4）：top-20 里水平列中位数 0（p75=1），即**改善组本来就不靠水平列吃饭**，拿掉水平列的代价 ≈ 0。derived 中位数 4 会保留。
- **RU**：G1 对 RU 的影响 = 拿掉 8 条水平列，保留所有派生和 AVAILABLE。P2 目前对 RU 的 2 条周频（`S022158717/18`，nonnull 0.11，eff=0.55<0.6）和 2 条日频稀疏（`M005028520/21`，nonnull 0.14）已经 drop，派生也随之丢失。G1 不额外 drop 源列，但拿掉水平维度 → 预期**不比 P2 坏**，是否修复 RU 需要 run 之后看。
- **FU**：P2 已经帮 FU drop 掉 4 个 step-dummy（§14.3 四个 ⚠ YES），剩 4 条源列（1 条日频 `S006870796` nonnull 0.97，3 条周频库存类 nonnull 0.20）。G1 拿掉这 4 条水平列 —— 尤其是 3 条周频库存列（在 5min 网格上 ffill ≈ 480 根 bar 分段常量，最容易被 LGB 当 regime flag），保留对应的派生和 AVAILABLE。G1 对 FU 的预期修复幅度应该不小于 FB/RU。

### 2.4 风险与已知不对的地方

- **派生列本身也会被 ffill 到 5min 网格**：观测之间派生值不变，仍是分段常量。G1 不治本。若跑完 FB/BB 没恢复或改善组 ΔSharpe 中位数反向，就说明派生列也是毒，要 G2（收紧 `ffill_max_bars`）或把派生列彻底关掉（`derived.enabled=false` = 只留 AVAILABLE 当有无信号）。
- **AVAILABLE 是 int8 常量近似物**：如果一个指标从未过期（`ffill_max_bars=8064` 意味着 4 周内几乎总有观测），AVAILABLE 在训练集上几乎恒为 1，方差过滤关卡 D 会把它删掉。所以 G1 不会额外放大 AVAILABLE 的存在感。
- **FU 独立画像**：§14.3 的 4 个 step-dummy 已经被 P2 drop 掉 → G1 对 FU 影响有限。FU 若仍不修复，另议。

---

## 3. 验证计划（follow §15.4 既有 rubric）

### 3.1 两个 A/B

1. `p2g1_vs_baseline` = G1 新 run vs `20260416_170652`（无 MID 的 baseline）
2. `p2g1_vs_p2` = G1 新 run vs P2 run（2026-04-23 那次，若已被清理则换成 candidate `20260422_154414`）

### 3.2 判收标准（与 §15.4 一致）

**G1 判定成功**需同时：
- `p2g1_vs_baseline` **9 个原回归品种**里退步（ΔSharpe < −0.30）个数从 P2 的 3（FB/FU/RU）降到 **≤ 1**。
- **改善组 9 个品种** ΔSharpe 中位数 ≥ −0.10（不伤改善组）。
- 25 品种整体 ΔSharpe 中位数 ≥ candidate（即 +0.098）。

三项任一不达，回到 §15.5 处置分支（不走 per-pid；考虑 G2 叠加或退回 candidate）。

### 3.3 执行步骤

```bash
# 前置（§13 安全闸）
git status --porcelain | wc -l          # = 0
df -h /Users/zhouzian/Desktop | awk 'NR==2 { if ($4+0 < 5) exit 1 }'

# 改 config.yaml: mid_weekly.level_keep: false
# 改 dataset.py: cache signature 纳入 mid_weekly_level_keep

python -m unittest tests.test_dataset_and_modeling -v    # 必须先绿
python pipeline/train_products.py --all --force-rebuild  # 全 25 品种

# A/B
python scripts/compare_runs.py --baseline 20260416_170652 --candidate <g1_run_id> --output results/comparison/p2g1_vs_baseline
python scripts/compare_runs.py --baseline 20260422_154414 --candidate <g1_run_id> --output results/comparison/p2g1_vs_candidate
```

### 3.4 单测追加

在 `tests/test_dataset_and_modeling.py` 新增 `test_mid_weekly_level_keep_false`：
- 合成 1 条日频源 + 1 条周频源 xlsx。
- 跑两次 `prepare`：`level_keep=true` 与 `level_keep=false`。
- 断言后者 `feature_cols` 不含 `MID_*_<ind_id>`（裸水平列），但仍含 `_RET_4 / _ZSCORE_4 / _PCT_RANK_4` 和 `_AVAILABLE`。
- 断言两次的 cache signature 不同（反向 sanity：flip 不该命中同一缓存）。

### 3.5 回滚

单行 config 翻回 `level_keep: true` + 清缓存；无 schema / migration 影响。

---

## 4. 与既有 §15 的关系

- G1 是 **§9 Q5 的变体 → 不是 P1/P2/P3 任何一支**，而是"全局关水平列"这条被 §14.5.4 未明确列出的路径。记为 **§9 Q8**（等 §15 收尾时补进 issue log）。
- P2 的过滤器 (`min_active_ratio`, `drop_step_dummy`) 保留 —— 它把 FU 的 4 个 step-dummy 已经正确 drop 了（这是 P2 唯一明确的胜利），G1 和它正交。
- G2（收紧 `ffill_max_bars`）作为 **条件后继**：仅在 G1 不满足 §3.2 判收条件时叠加，不预先执行。
