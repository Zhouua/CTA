# A1：mid_weekly 事件化 + 微观-中观显式交互（实施计划）

## 背景与动机

之前的 lag 实验显示：
- `lag=0`：含中观因子时 Y/RU sharpe 6.0/2.5（**前视污染**）
- `lag=1` 起：mid 因子单 IC 几乎 100% 衰减到 0

但 mid_weekly 不应该没用。可能性：
1. **数据 ts 仍残留前视**（vendor 把"事后修订"回填到原观测日，光 lag 无法恢复）
2. **mid 因子不是直接预测 5min 后向收益，而是修饰其他微观因子"何时该信任"**
3. **派生方式过于机械**：当前 RET / ZSCORE / PCT_RANK 三种 transform 丢失了"基本面事件"的信号

A1 的核心假设是 #2 + #3：**条件型 alpha** 需要把"基本面状态"显式作为修饰特征（而不是当独立 predictor）让模型见到。

## A1 整体目标

仍然要严格避免前视函数
把中观因子从"机械特征"升级为"事件状态 + 与微观因子的显式交互"，验证假设：
- 如果 MIDxMICRO_* 列在 LightGBM importance 中显著靠前 → mid 通过条件型 alpha 起作用，**有用**
- 如果 MIDxMICRO_* 列 IC 仍接近 0 → mid 真的无信号（可能是数据问题）

## 实施步骤

### A1.1 在 `_compute_mid_weekly_derivatives` 加 2 类 transform

**新增 transform**：

#### `accel`（二阶差分，捕捉拐点 / 加速）

```python
if "accel" in transforms:
    name = f"{col}_ACCEL_{w}"
    base = s.pct_change(w)              # 已计算（可与 ret 共用）
    per_col[name] = (base - base.shift(w)).clip(-2.0, 2.0)
    derived_cols.append(name)
```

含义：本期 w-pct_change 与 w 期前 w-pct_change 的差。
- 正值大 = 库存/价格变化在加速上行
- 负值大 = 变化在反转
- 接近 0 = 趋势平稳

#### `extreme_flag`（HIGH/LOW dummy）

```python
if "extreme_flag" in transforms:
    if rank_w is None:
        rank_w = s.rolling(w, min_periods=mp).rank(pct=True)
    high_name = f"{col}_HIGH_FLAG_{w}"
    low_name = f"{col}_LOW_FLAG_{w}"
    per_col[high_name] = (rank_w >= high_q).astype("float32")
    per_col[low_name] = (rank_w <= low_q).astype("float32")
    derived_cols.append(high_name)
    derived_cols.append(low_name)
```

含义：基于 pct_rank 阈值的 0/1 dummy。把"库存极端高 / 极端低"事件显式化，
不再让 LightGBM 自己挖到 0.85 / 0.15 这种阈值。

阈值默认 `extreme_high_quantile=0.85`、`extreme_low_quantile=0.15`，从 config 读取。

#### 实施细节

- 在 transform 循环中缓存 `rank_w`：`pct_rank` 和 `extreme_flag` 共用，避免重复 rolling
- `accel` 复用 `s.pct_change(w)` 的中间结果
- 遵循原代码风格：每个 window × transform 一列，仍走 `pieces.append(per_col)` → outer-merge

### A1.2 新增 `_add_mid_micro_interactions`（核心）

新增方法，由 `prepare` 在 `_merge_mid_weekly_features` 完成 + `_add_engineered_features` 完成后调用

**函数签名**：

```python
def _add_mid_micro_interactions(
    self,
    merged_data: pd.DataFrame,
    mid_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
```

**逻辑**：

```python
cfg = self.settings.get("mid_weekly_micro_interactions", {})
if not cfg.get("enabled"):
    return merged_data, []

micro_factors = list(cfg.get("micro_factors", []))           # 用户配置的微观核心
mid_selector = str(cfg.get("mid_selector", "pct_rank_only")) # 选哪类 mid
max_cols = int(cfg.get("max_columns", 100))                  # 维度封顶

# 1) 微观列：必须在 merged_data.columns 中
available_micro = [c for c in micro_factors if c in merged_data.columns]

# 2) 中观列：按 mid_selector 过滤
if mid_selector == "pct_rank_only":
    target_mid = [c for c in mid_cols if "_PCT_RANK_" in c]
elif mid_selector == "pct_rank_and_extreme":
    target_mid = [c for c in mid_cols
                  if "_PCT_RANK_" in c or "_HIGH_FLAG_" in c or "_LOW_FLAG_" in c]
else:  # "all"
    target_mid = list(mid_cols)

# 3) 笛卡尔积，命名 MIDxMICRO_<micro>_X_<mid>
pairs = [(m, n) for m in available_micro for n in target_mid][:max_cols]
new_cols = []
for micro, mid in pairs:
    col_name = f"MIDxMICRO_{micro}_X_{mid}"
    merged_data[col_name] = (
        merged_data[micro].astype("float32") * merged_data[mid].astype("float32")
    ).astype("float32")
    new_cols.append(col_name)

return merged_data, new_cols
```

**关键设计选择**：

1. **mid_selector 默认 `pct_rank_only`**：
   - PCT_RANK 已经是 [0, 1] 标准化的状态变量
   - 乘以 micro（已 z-score 化或对数收益级）量级合理
   - RET / ZSCORE 原始值量级不一致，乘起来会产生极端值

2. **微观因子默认选 5 个核心**（在 config 里配置）：
   ```yaml
   micro_factors:
     - ENG_PREV_CLOSE_LOG_RET     # 跨日运行收益
     - ENG_INTRADAY_LOG_RET_CUMSUM # 日内累积收益
     - ENG_INTRADAY_RV_60          # 60-bar 实际波动
     - ENG_DAY_RANGE_POS           # 日内 H/L 区位
     - ENG_RET_60                  # 60-bar 收益
   ```
   覆盖跨日动量 / 日内动量 / 波动 / 位置四类

3. **`max_columns` 默认 100**：5 个 micro × ~20 个 mid PCT_RANK 列（10 个指标 × 3 windows 但部分被 quality filter 砍）≈ 100。控制 MIDxMICRO 不超过 100 列避免维度爆炸

4. **NaN 处理**：micro 和 mid 都是 float32，乘积自动传播 NaN。下游 `dropna(subset=required_cols)` 会过滤掉 NaN 行 → 不需要特殊处理

### A1.3 在 `prepare` 中接入


```python
# 在 _add_engineered_features 调用之后：
# (现有代码已经写好 mid_weekly_cols 和 engineered_cols)

# 新增：mid × micro 显式交互
merged_data, mid_micro_int_cols = self._add_mid_micro_interactions(
    merged_data, mid_cols=mid_weekly_cols
)

# 加入 candidate_cols（与 factor_cols / engineered_cols / extra_cols 并列）
candidate_cols = list(dict.fromkeys(
    factor_cols + engineered_cols + extra_feature_cols + mid_micro_int_cols
))
```

**downstream 兼容**：
- variance filter（`std_series` 那段）会自动过滤掉常数 MIDxMICRO 列
- registry filter 会按 audit 阈值评估 MIDxMICRO_* 的 IC × ICIR
- diagnostics 把 MIDxMICRO_* 归到 "interaction" 类别（A1.3 任务）

### A1.4 settings 接入


新增三项：

```python
"mid_weekly_extreme_high_quantile": float(mid_weekly_cfg.get("extreme_high_quantile", 0.85)),
"mid_weekly_extreme_low_quantile":  float(mid_weekly_cfg.get("extreme_low_quantile", 0.15)),
"mid_weekly_micro_interactions": (
    dict(mid_weekly_cfg.get("micro_interactions", {}) or {})
    if isinstance(mid_weekly_cfg, dict) else {}
),
```

### A1.5 config.yaml

**位置**：`mid_weekly:` 段内

```yaml
mid_weekly:
  derived:
    enabled: true
    rolling_windows: [4, 13, 52]
    # 加入 accel + extreme_flag
    transforms: [ret, zscore, pct_rank, accel, extreme_flag]
  extreme_high_quantile: 0.85   # extreme_flag: pct_rank > 0.85 → HIGH_FLAG=1
  extreme_low_quantile: 0.15    # extreme_flag: pct_rank < 0.15 → LOW_FLAG=1
  # mid × micro 显式交互
  micro_interactions:
    enabled: true
    max_columns: 100
    mid_selector: pct_rank_only   # pct_rank_only / pct_rank_and_extreme / all
    micro_factors:
      - ENG_PREV_CLOSE_LOG_RET
      - ENG_INTRADAY_LOG_RET_CUMSUM
      - ENG_INTRADAY_RV_60
      - ENG_DAY_RANGE_POS
      - ENG_RET_60
  # ... 其他既有项保持不变
```

### A1.6 audit script + diagnostics 识别 MIDxMICRO 前缀

**位置 1**：`scripts/audit_factor_ic_importance.py::_classify_factor`

```python
if name.startswith("MIDxMICRO_"):
    return ("interaction", "engineered")  # 显式交互归 interaction
```

**位置 2**：`pipeline/diagnostics.py::_assign_group`

```python
def _assign_group(feat: str) -> str:
    if feat.startswith("MIDxMICRO_"):
        return "合成因子"   # 与"S1-S10"合成因子同类，紫色 / 橙色显示
    if feat.startswith("ENG_"):
        return "工程化特征" if feat in ORIGINAL_ENG_FEATURES else "合成因子"
    if feat.startswith("MID_"):
        return "中观因子"   # 如果之前没有这个分支，加上
    return "量价因子"
```

注：当前 `GROUPS` 只有"量价因子 / 工程化特征 / 合成因子"三类。如果想给 MIDxMICRO_ 单独颜色，加：
```python
GROUPS["显式交互"] = {"color": CICC_PURPLE}
```

但更简洁是把 MIDxMICRO_ 归到"合成因子"（橙色）—— 用户当前配色是 蓝/绿/橙。

## 验收标准

跑 RB / Y / RU 三品种，对比 A1 之前 vs 之后：

| 指标 | 期望变化 |
|---|---|
| TEST IC | A1 之后 ≥ A1 之前（即使加了 100 列也不该让 IC 下降）|
| LightGBM low_vol top-30 importance 中 MIDxMICRO_* 数量 | ≥ 5（说明显式交互真的被模型用上）|
| TEST net annual | 正向品种 ≥ 0%（mid 因子真贡献而不是噪音）|

如果 MIDxMICRO_* 仍占不到 top-30 + IC 没改善 → 说明假设 #2 也不成立，mid 数据本身有问题（结构性前视或 vendor 修订），需要换数据源。

## 实施工作量估计

| 子任务 | 文件 | 行数 | 难度 |
|---|---|---|---|
| A1.1 加 accel + extreme_flag | dataset.py | +30 | 小 |
| A1.2 _add_mid_micro_interactions | dataset.py | +60 | 中 |
| A1.3 prepare 接入 | dataset.py | +10 | 小 |
| A1.4 settings | dataset.py | +6 | 小 |
| A1.5 config.yaml | config.yaml | +20 | 小 |
| A1.6 audit + diagnostics 分类 | 2 个文件 | +6 | 小 |
| **总计** | | **~130 行** | |

## 实施顺序建议

1. 先做 A1.1（transform 扩展）+ A1.5（config）→ smoke test 单品种确认 ACCEL_* / HIGH_FLAG_* / LOW_FLAG_* 正确生成
2. 再做 A1.2 + A1.3 + A1.4（micro 交互核心）→ smoke test 看 MIDxMICRO_* 是否进入 feature_cols
3. 最后 A1.6（audit + diagnostics 分类）→ 视觉验证
4. 跑 RB / Y / RU batch 横向对比（A1.4 验收）

## 注意事项

1. **不要改 publication_lag 设置**：A1 是在防前视的基础上加因子工程，不是回退到 lag=0。如果发现 A1 后 IC 暴涨到 0.15+，要立即怀疑是不是 lag 机制被绕过了。

2. **registry 自动适配**：现有的 `factor_registry.json` audit 流程会自动给 MIDxMICRO_* 列计算 mean_ic / icir，按当前 `audit_thresholds` 决定 train_factor / not_train_factor。第一次跑训练后跑一次 audit 即可。

3. **维度控制**：`max_columns=100` 是建议起点。如果 MIDxMICRO_* 大部分 IC < 0.005 + importance=0，audit 会把它们打到 not_train_factor，下次训练自动过滤，不会拖累模型。

4. **量级警惕**：micro 因子（如 ENG_PREV_CLOSE_LOG_RET）量级 ~1e-3，mid PCT_RANK 量级 [0, 1]，乘积量级 ~1e-3。LightGBM 对量级不敏感（树分裂只看排序），但 RobustScaler 会处理。无需额外标准化。
