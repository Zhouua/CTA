# Mid-Weekly 因子集成方案与可恢复执行计划

## 0. 给后续 Claude 实例的 Bootstrap Prompt

> **这一节是给未来任意 Claude Code 会话使用的"恢复指令"。如果当前会话中断，把下面这段话作为新会话的第一条用户消息，就能从上次的进度继续。**

```
读取 docs/mid_weekly_integration_plan.md。按照下面的循环执行直到所有任务完成或必须停下问用户：

  loop:
    1) 读"§ 8 进度状态"表格,找出第一个 status != done 的任务行。
    2) 如果该任务的"前置条件"未满足,先完成依赖任务。
    3) 执行该任务"§ 步骤"列出的步骤;只调用文档中明确列出的命令。
    4) 跑该任务的"§ 验收脚本";输出存在 results/comparison/_audit/ 下,文件名带任务号。
    5) 若验收通过:把"§ 8 进度状态"中该行 status 改成 done、写入 timestamp 与产出路径,并把
       本次改动以一次提交落盘(commit message 前缀 "midweekly: ")。若失败:把 status 改成
       blocked、写入 failure 摘要,并停下问用户。
    6) 若有"§ 9 待用户决策"中尚未拍板的事项被本次任务触及,停下问用户。

  在以下情况停下并问用户:
    - 任务步骤涉及破坏性操作(删除文件、覆写非缓存文件、git push、reset --hard)。
    - 验收脚本结果落在文档里"需要人工裁决"的区间。
    - 决策点 1 (dummy var 有效性) 的验证产出已经齐全,需要用户确认是否保留 _AVAILABLE 列。

  禁止:
    - 跳过验收脚本直接 mark done。
    - 自己新增/扩展任务清单(如确实需要,写到"§ 9 待用户决策"等用户确认)。
    - 改动 backtest.py 与 modeling.py 的回测/损失逻辑(本计划只允许扩展 dataset.py 与新增脚本)。
```

> 如果您手动跑这个 prompt：直接复制上面三反引号内的全部内容当成下一条 user message 即可。

---

## 1. 背景

- 上一次批量训练 `results/runs/20260416_170652` 没有任何 mid_weekly 因子（`mid_weekly_feature_count=0` × 25 个 success 品种）。
- `data/mid_weekly/` 已手动放入 25 个品种的 `.xlsx`，但与现有 `dataset.py::_read_mid_weekly_factor` 不兼容（代码只读 CSV、且要求"1 时间戳 + 1 值列"，xlsx 是宽表 + 4 行表头）。
- `data/product_registry.json` 里所有品种 `mid_weekly_files=[]`。

## 2. 目标

把 25 个 xlsx 真正接进训练管道，跑出新的 batch run，做"加 mid_weekly vs 不加"的 A/B 对比。

## 3. 用户已锁定的决策

| 决策点 | 选择 | 备注 |
|---|---|---|
| 1. 缺失值策略 | **B**：宽容 + `_AVAILABLE` 哑变量；同时**必须验证哑变量是否真的有用**（见 § 6） | 用户对 dummy var 的有效性持保留态度，所以加专门的验证流程 |
| 2. xlsx 改造方向 | **a**：`dataset.py` 在线读 xlsx，原文件保留可追溯 | 不做"拆成单指标 CSV"的预处理 |
| 3. 重复列处理 | 仅删除"元数据完全相同"的硬重复；语义近似列**列入文档**让用户裁决 | audit 输出会在 § 9 列出待裁决项 |

---

## 4. 任务总览

| # | 任务 | 产出 | 是否阻塞下游 |
|---|---|---|---|
| T1 | 提取 baseline run 最优品种 | `docs/run_20260416_170652_top_performers.md` | 否（与 T2/T3 并行） |
| T2 | 审计 xlsx + 更新 recommendations | `docs/mid_weekly_recommendations.md` 增量 + `docs/mid_weekly_audit.md` | 阻塞 T3.2 |
| T3.1 | 改造 `dataset.py` 支持 xlsx 宽表 | `pipeline/dataset.py` patch + 新增测试 | 阻塞 T4 |
| T3.2 | 实现缺失值策略 B + 派生因子 | `pipeline/dataset.py` 续 + `config.yaml` `mid_weekly` 节 | 阻塞 T4 |
| T3.3 | 哑变量有效性验证机制 | `scripts/audit_mid_weekly_importance.py` | 阻塞 T4 完成态 |
| T4.1 | 更新 `product_registry.json` | `data/product_registry.json` patch | 阻塞 T4.2 |
| T4.2 | 跑新 batch | `results/runs/<new_run_id>/` | 阻塞 T4.3 |
| T4.3 | A/B 对比报告 | `results/comparison/midweekly_vs_baseline.{md,csv,png}` | 末端 |

---

## 5. 任务详解

### T1：Baseline Top Performers

**步骤**：
1. 读 `results/runs/20260416_170652/run_summary.csv`。
2. 过滤 `status==success`，按 `sharpe` 降序排序。
3. 写 `docs/run_20260416_170652_top_performers.md`：
   - 完整 25 行表（Sharpe / AnnRet / MaxDD / Trades）。
   - Top 10 摘要。
   - 标注"无 mid_weekly 因子"作为 baseline 身份。
   - 列出尾部 5 个（Sharpe < 3.5 或 < 0）作为 T4 重点观察对象。

**验收脚本** (`results/comparison/_audit/T1_check.txt`)：
```bash
test -f docs/run_20260416_170652_top_performers.md
grep -c "^| " docs/run_20260416_170652_top_performers.md  # 至少 27 (1 表头 + 1 分隔 + 25 数据)
```

---

### T2：Mid-Weekly Audit

**步骤**：
1. 写一次性脚本 `scripts/audit_mid_weekly_inputs.py`：
   - 入参：`--mid-weekly-dir data/mid_weekly --output docs/mid_weekly_audit.md`
   - 对每个 xlsx：
     - 解析 4 行表头（行 0 单位 / 行 1 指标名 / 行 2 频率 / 行 3 指标 ID）→ 数据 DataFrame（行 4+，第 0 列为日期）。
     - 抽取每个指标的 (start_date, end_date, non_null_ratio, frequency, indicator_id)。
     - 检测**硬重复**：(指标名, 指标 ID, 频率) 三元组完全相同 → 自动标记为可删。
     - 检测**软重复**：数值列两两 `corr > 0.99 且 NaN 重叠率 > 0.9` → 列入"待裁决"。
   - 输出 markdown：每个品种一节，列已有指标 / 命中推荐 / 缺失推荐 / 硬重复（脚本可删）/ 软重复（待用户确认）。
2. 在 `docs/mid_weekly_recommendations.md` 每个品种小节末尾追加一段 `**当前持有**` / `**缺失**` 子节，引用 audit 文档。
3. **脚本只移除硬重复**：在原 xlsx 旁边输出 `data/mid_weekly/_cleaned/<PID>.xlsx`，原文件不动。后续 T3 只读 cleaned 版本。

**验收脚本** (`results/comparison/_audit/T2_check.txt`)：
```bash
test -f docs/mid_weekly_audit.md
test -d data/mid_weekly/_cleaned
ls data/mid_weekly/_cleaned/*.xlsx | wc -l   # = 25
python -c "
import pandas as pd
from pathlib import Path
for p in Path('data/mid_weekly/_cleaned').glob('*.xlsx'):
    df = pd.read_excel(p, sheet_name=0, header=None, nrows=4)
    assert df.shape[0] == 4, f'{p} header rows != 4'
print('OK')
"
```

**缺失值策略备忘（写进 audit 文档"原则"段）**：
- 周频指标在分钟数据上必然有大段空 → `merge_asof(backward) + ffill` 是正确处理。
- 指标自身周内断更：先 `ffill`，**禁止 `bfill`**（会未来泄漏）。
- 起始时间晚于行情：交给 § 6 的策略 B 处理。

---

### T3：训练管道改造

#### T3.1 xlsx 读取

**改动文件**：`pipeline/dataset.py`（仅扩展 `_read_mid_weekly_factor` / `_merge_mid_weekly_features`）

- `_read_mid_weekly_factor` 接受 `.csv | .xlsx | .xls`：
  - xlsx 分支：读第 0 sheet，跳过 4 行表头，第 1 行作为指标名，频率/ID 元数据存进缓存 meta（用于 audit）。
  - 列重命名为 `MID_<PID>_<安全化指标名>`：把中文转成下划线 + ASCII，超长截断到 40 字符 + 6 位哈希后缀避免冲突。
- `_merge_mid_weekly_features` 一次接收所有 xlsx 派生的列，统一做 `merge_asof(backward) + ffill`。

**单测**：`tests/test_dataset_and_modeling.py` 加 `test_mid_weekly_xlsx_wide_format`：
- 合成一个 2 指标 xlsx（其中 1 列从某日期起为 NaN），跑完 `prepare`。
- 断言：两个 `MID_*` 列存在 / 时间戳对齐 / 没有越过周末 ffill 7 天以上。

#### T3.2 缺失策略 B + 派生因子

**新增 `config.yaml` 节**：
```yaml
mid_weekly:
  available_dummy: true            # 是否生成 MID_*_AVAILABLE 哑变量
  ffill_max_bars: 8064             # ≈ 4 周(5min×48×7×4),超出后回 NaN
  derived:
    enabled: true
    rolling_windows: [4, 13, 52]   # 周
    transforms: ["ret", "zscore", "pct_rank"]
  level_keep: true                 # 是否保留原始水平列
  missing_ratio_relax: 0.65        # 仅对 MID_* 列把 max_factor_missing_ratio 放宽到 0.65
```

**实现要点（在 `pipeline/dataset.py`）**：
1. 在 `_merge_mid_weekly_features` 之后:
   - 对每个 `MID_*` 列生成 `MID_*_AVAILABLE`（`notna().astype("int8")`）。
   - 应用 `ffill_max_bars` 截断：用 `where(notna_count_within_window <= ffill_max_bars)`。
2. 派生因子（在 `_add_engineered_features` 之后单独函数 `_add_mid_weekly_derived`）：
   - 按"周"重采样（取每日最后一个非空值），算变换，再 `merge_asof` 回 5min 网格。
   - 三种变换：
     - `MID_*_RET_<W>`：`pct_change(W)` + `clip(-1, 5)` 防爆。
     - `MID_*_ZSCORE_<W>`：`(x - rolling_mean(W)) / rolling_std(W)`，`min_periods = W//2`。
     - `MID_*_PCT_RANK_<W>`：`rolling.rank(pct=True)`。
3. **MID_* 列的过滤阈值放宽**：在 `prepare` 里只对前缀 `MID_` 的列用 `missing_ratio_relax` 而不是默认 0.35。

#### T3.3 哑变量有效性验证（回应用户的疑虑）

**目标**：客观证明 `MID_*_AVAILABLE` 是否真有信息，否则下一轮就关掉。

**新增脚本** `scripts/audit_mid_weekly_importance.py`：
- 读取一个 run 的 `results/models/<regime>/feature_importance.json`。
- 输出三件事：
  1. 所有 `MID_*_AVAILABLE` 列在两个 regime 中的 gain 重要性排名（top % 与绝对位次）。
  2. 把它们的 gain 总和 vs `MID_*` 原始列的 gain 总和 做 ratio（>5% 才算"有用"）。
  3. **可选 ablation**：传 `--ablation` 时，重新跑一遍训练（关掉 `available_dummy`），对比 val Sharpe 的差。

**裁决规则**（写在文档里，下次跑前由用户确认）：
- 若 `_AVAILABLE` 列 gain 总和 < 5% 且 ablation Δval_sharpe < 0.1 → 关闭 `available_dummy`。
- 若 ≥ 5% 或 Δval_sharpe ≥ 0.1 → 保留。
- 落地动作：在 § 9 列一条"决策点 1 复核"，等待用户确认。

---

### T4：跑新 run + 对比

#### T4.1 更新 registry

- 写 `scripts/update_registry_with_mid_weekly.py`：
  - 扫 `data/mid_weekly/_cleaned/*.xlsx`。
  - 文件名 `<PID>.xlsx` → product_id `<PID>`。
  - 在 registry 里把对应条目的 `mid_weekly_files` 设成 `["<PID>.xlsx"]`（路径相对 `mid_weekly_dir`，但读取时走 `_cleaned/` 子目录——通过 `paths.mid_weekly_dir` 临时指向 `_cleaned/` 实现，避免 registry 写死 `_cleaned/` 前缀）。
  - 没有对应 xlsx 的品种保持 `[]`。
- 输出 diff 给用户看：哪些品种被赋值了。

#### T4.2 跑新 batch

```bash
python pipeline/train_products.py --all --force-rebuild
```

- 自动新 run_id（建议保留默认时间戳命名，不强行注入）。
- `--force-rebuild` 必须有，旧 cache 不含 mid_weekly 列。

**预期产出**：`results/runs/<new_run_id>/run_summary.csv`，记录 `mid_weekly_feature_count > 0` 的品种数。

#### T4.3 A/B 对比

写 `scripts/compare_runs.py`：
- 入参：`--baseline 20260416_170652 --candidate <new_run_id> --output results/comparison/midweekly_vs_baseline`
- 输出：
  - `.csv`：每个品种 baseline / candidate 的 Sharpe / AnnRet / MaxDD / Trades / Win-Rate / ΔSharpe。
  - `.md`：分四节—— "全面提升的品种" / "持平品种(|ΔSharpe| < 0.3)" / "退步品种" / "结构变化(成交频率显著变)"。
  - `.png`：3 个面板——
    - (a) 散点 baseline_sharpe vs candidate_sharpe + 对角线。
    - (b) Bar：按 ΔSharpe 排序。
    - (c) Top10 baseline 品种的 NAV 曲线对比（叠加）。
- **裁决标准**：
  - 中位数 ΔSharpe > 0 → mid_weekly 整体加分。
  - > 1/3 品种退步 → 必须先做 T3.3 的 ablation 才能定论。

---

## 6. 哑变量有效性的额外说明（回应用户的疑虑）

用户指出"不太清楚 dummy variable 的有效性"。我的判断：

- **理论上为什么可能有效**：当某指标"从无到有"是政策/口径变更（例如 2018 年新指标上线），这个时间点的市场状态本身就有信息——用 `_AVAILABLE` 哑变量等价于让模型学到一个"换 regime"的事件。
- **实际上经常无效**：如果指标缺失只是"早期无人统计"，那么 `_AVAILABLE` 提供的信息和"日期本身"高度共线——LightGBM 已经能从时间相关因子里学到。
- **所以必须实测**：T3.3 的 audit 脚本就是为了让这件事不靠拍脑袋。第一次跑完后看 importance + ablation Δsharpe，再决定是否保留。
- **保险方案**：如果哑变量被证伪，关掉它的代价仅是 config 里 `available_dummy: false` 重跑——不需要回滚代码。

---

## 7. 执行顺序

```
T1  ─┐
T2  ─┤            (T1/T2 并行)
     │
     ↓
T3.1 → T3.2 → (T3.3 框架代码)
     ↓
T4.1 → T4.2 → T4.3
     ↓
T3.3 实跑 ablation (在 T4.2 产出后)
     ↓
回到 § 9 决策点 1 复核
```

---

## 8. 进度状态（每完成一项必须更新这一节）

| # | 任务 | status | 完成时间 | 产出路径 | 备注 |
|---|---|---|---|---|---|
| T1 | Baseline top performers | done | 2026-04-22T06:19:51Z | `docs/run_20260416_170652_top_performers.md` · 验收 `results/comparison/_audit/T1_check.txt` | 25 success 均 `mid_weekly_feature_count=0`，作为 baseline 身份确认 |
| T2 | xlsx audit + recs 更新 | done | 2026-04-22T06:25:10Z | `scripts/audit_mid_weekly_inputs.py` · `docs/mid_weekly_audit.md` · `data/mid_weekly/_cleaned/*.xlsx` · `docs/mid_weekly_recommendations.md` 每品种追加 · 验收 `results/comparison/_audit/T2_check.txt` | 25/25 xlsx 无硬重复；7 个软重复对触发 § 9 Q1（等待用户裁决） |
| T3.1 | xlsx 宽表读取 + 单测 | done | 2026-04-22T06:54:01Z | `pipeline/dataset.py` (`_read_mid_weekly_xlsx` / `_read_mid_weekly_csv` / `_build_mid_column_name`) · `tests/test_dataset_and_modeling.py::test_mid_weekly_xlsx_wide_format` · 验收 `results/comparison/_audit/T3_1_check.txt` | 列命名 `MID_<PID>_<ind_id>`，指标 meta 同步落 `cache_meta.json`，16/16 单测通过 |
| T3.2 | 策略 B + 派生因子 | done | 2026-04-22T07:12:02Z | `config.yaml` 新增 `mid_weekly:` 节 · `pipeline/dataset.py` (`_compute_mid_weekly_derivatives` / 扩展 `_merge_mid_weekly_features` / `prepare` 分流 MID_*) · `tests/test_dataset_and_modeling.py::test_mid_weekly_strategy_b_available_clamp_and_derived` · 验收 `results/comparison/_audit/T3_2_check.txt` | AVAILABLE 哑变量、`ffill_max_bars` 截断、RET/ZSCORE/PCT_RANK × [4,13,52] 派生因子、MID_* 单独 `missing_ratio_relax=0.65`、MID_* NaN→0 补齐以免 dropna 杀行；17/17 单测通过 |
| T3.3 | 哑变量验证脚本 | done | 2026-04-22T07:27:23Z | `scripts/audit_mid_weekly_importance.py` · 验收 `results/comparison/_audit/T3_3_check.txt` · 预览 `results/comparison/_audit/T3_3_preview.{md,json}` | 仅写脚本；骨架支持 full feature_importance.json 与 top_features fallback；`--ablation` 当前发射 skeleton 需后续在 T3.3* 接实际训练 |
| T4.1 | registry 更新 | done | 2026-04-22T07:40:04Z | `scripts/update_registry_with_mid_weekly.py` · `data/product_registry.json` 25 条被赋值 · `config.yaml` `paths.mid_weekly_dir` 指向 `_cleaned/` · 验收 `results/comparison/_audit/T4_1_check.txt` | 25 个 xlsx 品种全部映射为 `[<PID>.xlsx]`；脚本幂等；17/17 单测通过 |
| T4.2 | 新 batch run | done | 2026-04-22T09:22:03Z | run_id `20260422_154414` · `results/runs/20260422_154414/run_summary.csv` (25 success / 9 failed / 31 skipped) · 训练日志 `results/comparison/_audit/t4_2_run_logs/train.log` · 验收 `results/comparison/_audit/T4_2_check.txt` | 所有 25 个成功品种 `mid_weekly_feature_count ∈ [39, 201]`（baseline=0）；ZN 首轮因瞬时 IO 空读失败，`--resume-run` 后成功；success 集合与 baseline 完全一致 |
| T4.3 | A/B 对比报告 | todo | – | – | – |
| T3.3* | 实跑 ablation + 裁决 | todo | – | – | 依赖 T4.2 |

> status 取值：`todo` / `in_progress` / `blocked` / `done`。`blocked` 必须在备注里写阻塞原因 + 触发的 § 9 决策点。

---

## 9. 待用户裁决项

> 任何在执行中触发的"需要人决定"的事项都追加到这里，不要删除已完结条目（保留追溯）。

| 序号 | 触发任务 | 事项 | 状态 |
|---|---|---|---|
| Q1 | T2 | xlsx 软重复列裁决：7 对 B 选项已执行——为每对删一列，保留另一列。执行机制：`scripts/apply_soft_dup_decisions.py`（可重入，按 indicator_id 匹配）。当前 `data/mid_weekly/_cleaned/*.xlsx` 已应用。 | **decided (B)** 2026-04-22 |
| Q2 | T3.3* | 哑变量是否保留（依赖 ablation 结果） | 待 T4.2 完成 |

**Q1 软重复明细与裁决**（决定由用户确认，`scripts/apply_soft_dup_decisions.py` 为执行记录）：

| Product | 保留 (ID) | 删除 (ID) | \|corr\| | 理由 |
|---|---|---|---:|---|
| AU | 期货库存:黄金 (S000025741) | 库存:黄金 (S000025737) | 1.000 | 期货库存对应 SHFE 交割口径 |
| AU | COMEX:黄金:总持仓 (S002825231) | COMEX:黄金:期货(新版):总持仓:持仓数量 (S008523183) | 1.000 | 保留原版命名，历史口径一致 |
| JM | 市场价:焦煤(主焦煤):当旬值 (S005948590) | 生产资料价格:焦煤(1/3焦煤) (M003575153) | 1.000 | 主焦煤对应交割标的 |
| M | 现货价:豆粕:地区均价 (S002856699) | 现货价:豆粕 (S004242412) | 0.997 | 地区均价更稳健 |
| RB | 注册仓单:螺纹钢 (S002853768) | 期货库存:螺纹钢:小计 (S005476450) | 1.000 | 注册仓单直接反映交割能力 |
| RU | 仓单数量:天然橡胶:总计 (S004410360) | 仓单数量:天然橡胶 (S002842061) | 1.000 | 总计命名更明确 |
| SN | 期货库存:锡:总计 (S005580468) | 库存:锡:总计 (S005580375) | 0.995 | 与 AU #1 同逻辑 |

---

## 10. 不变量（任何任务都不能违反）

1. 不动 `pipeline/backtest.py` 与 `pipeline/modeling.py` 的回测/损失逻辑——本计划只允许扩展 `pipeline/dataset.py` 与新增脚本。
2. mid_weekly 数据**只允许 backward `merge_asof` + ffill**，禁止 `bfill` 与任何 future-looking 操作。
3. 所有改动落 git commit，message 前缀 `midweekly:`，方便 `git log --grep="^midweekly:"` 抽取整条线。
4. 原 xlsx 文件保留不动；清洗结果只写到 `data/mid_weekly/_cleaned/`。
5. 旧 run `results/runs/20260416_170652/` 保留为 baseline，不允许覆写。

---

## 11. 实际实现说明（中观因子端到端流程）

> 本节把 T3.1 / T3.2 落地的代码行为固化为文档，便于后续 Claude 会话或他人检视，不作为可执行步骤。

### 11.0 术语：`MID feat` 是什么

`run_summary.csv` 里的 `mid_weekly_feature_count` 字段等于该品种特征矩阵中以 `MID_` 开头的列数——**不是"原始中观指标条数"**。每个原始中观指标 `X` 最多贡献 11 列，按下表展开：

| 列类型 | 命名模板 | 数量 / 指标 |
|---|---|---:|
| 水平列（原始值） | `MID_<PID>_<ind_id>` | 1 |
| 派生列 | `..._RET_{4,13,52}` · `..._ZSCORE_{4,13,52}` · `..._PCT_RANK_{4,13,52}` | 9（3 变换 × 3 窗口）|
| 哑变量 | `..._AVAILABLE` | 1 |

所以 `mid_weekly_feature_count = 11 × n_原始指标`（若 `derived.enabled=true` 且 `available_dummy=true`）。`20260422_154414` run 里 `FB=39 (≈4 指标)`、`FU=87 (≈8 指标)`、`JD=201 (≈18 指标)` 的差异源头就在这里。

### 11.1 数据来源与前置处理

- 源文件：`data/mid_weekly/_cleaned/<PID>.xlsx`，25 个品种各一个。
- xlsx 形态：**宽表 + 4 行表头**（行 0 单位 / 行 1 指标名 / 行 2 频率 / 行 3 指标 ID），行 4+ 为数据。
- 前置清洗：T2 `audit_mid_weekly.py` 去硬重复；§9 Q1 裁决经 `apply_soft_dup_decisions.py` 去软重复。
- **原始 xlsx 保留不动**（§10 不变量 4），清洗结果只落在 `_cleaned/`；`config.yaml` 的 `paths.mid_weekly_dir` 指向 `_cleaned/` 目录。

### 11.2 核心代码位置

所有逻辑集中在 `pipeline/dataset.py::_merge_mid_weekly_features`（§10 不变量 1：不碰 `backtest.py` / `modeling.py`）。按文件循环、每品种独立处理：

```
pipeline/dataset.py
├── _read_mid_weekly_xlsx        L411–450   读 xlsx → 稀疏观测表
├── _read_mid_weekly_csv         L452–486   兜底 CSV 路径（legacy）
├── _read_mid_weekly_factor      L488–497   根据扩展名分发
├── _compute_mid_weekly_derivatives  L499–549  稀疏网格上的派生因子
├── _merge_mid_weekly_features   L551–647   主流程（merge + clamp + dummy）
└── _build_mid_column_name       L380–409   列命名与冲突消歧
```

### 11.3 步骤 1 — 读 xlsx → 稀疏观测表 (`_read_mid_weekly_xlsx`)

- 跳过 4 行表头，从第 5 行起读；第 0 列强制 `to_datetime`，落为 `NaT` 的行丢弃。
- 列命名规则（`_build_mid_column_name`）：
  - 优先 `MID_<PID>_<indicator_id>`（指标 ID 全局唯一）。
  - 兜底 `MID_<PID>_<ASCII 化指标名>[_<6 位 md5>]`，尾部超 40 字符截断。
  - 同名冲突加 `_2 / _3 / ...` 计数后缀。
- 数值列强制 `float32`，节省内存。
- 同时落一份 meta（指标名 / 频率 / 单位 / 源文件），后续写进 `cache_meta.json` 供 audit 引用。
- 时间戳去重：同一 ts 保留最后一条观测。

### 11.4 步骤 2 — 派生因子（在稀疏网格上算，`_compute_mid_weekly_derivatives`）

**关键不变量：先在周频稀疏网格上算，再 merge 到 5min**——这样 rolling 窗口的单位是"观测数 ≈ 周数"，而不是"分钟数"。

| 变换 | 公式 | min_periods |
|---|---|---|
| `_RET_w` | `pct_change(w).clip(-1.0, 5.0)` | — |
| `_ZSCORE_w` | `(x - rolling_mean(w)) / rolling_std(w)`，`std=0 → NaN` | `max(1, w//2)` |
| `_PCT_RANK_w` | `rolling(w).rank(pct=True)` | `max(1, w//2)` |

窗口 `[4, 13, 52]` 对应 ~1 月 / ~1 季 / ~1 年。`ret` 的 clip 是防 "除以 0 或极小分母" 造成爆值。

控制开关：`config.yaml::mid_weekly.derived.{enabled, rolling_windows, transforms}`。

### 11.5 步骤 3 — 合并到 5min 网格

```python
merged = pd.merge_asof(5min_bars, factor_df,
                       on=timestamp, direction="backward")
merged[level_cols + derived_cols] = merged[level_cols + derived_cols].ffill()
```

- **`direction="backward"` 是 §10 不变量 2 的硬约束**：严禁 `bfill` 或任何 forward-looking。
- `merge_asof(backward)` 在相邻观测之间已经给出"前值延续"的效果；显式 `ffill` 作为 defensive doc 保留。
- `data.mid_alignment` 只支持 `asof_forward_fill`，其它值抛错（`_merge_mid_weekly_features` L559）。

### 11.6 步骤 4 — Staleness clamp（"策略 B"的真正落点）

防止一个周频指标断更后被无限 ffill——这是用户决策点 1（"缺失值策略 B"）的核心机制。

```python
# 每次新观测到达时 __mid_arrival_ts__ 发生变化
# 同一 arrival 组内累加计数，超过 ffill_max_bars 强制回写 NaN
stale_mask = (staleness > ffill_max_bars) & arrival.notna()
merged.loc[stale_mask, level_cols + derived_cols] = np.nan
```

- `ffill_max_bars` 默认 `8064`（= 5 min × 48 × 7 × 4 ≈ 4 周）；超限后水平列 + 派生列一并置 NaN。
- arrival 辅助列为每个文件一份 `__mid_arrival_ts_{file_idx}__`，clamp 完成后立即 drop，不落入特征矩阵。

### 11.7 步骤 5 — `_AVAILABLE` 哑变量（§6 验证对象）

```python
MID_<X>_AVAILABLE = merged[MID_<X>].notna().astype("int8")
```

- **在 clamp 之后计算**：一旦某观测被 staleness clamp 置 NaN，其 AVAILABLE 也变 0。这是该哑变量"是否真的携带信息"的前提。
- 控制开关：`config.yaml::mid_weekly.available_dummy`（默认 `true`）。
- T3.3 `audit_mid_weekly_features.py` + T3.3\* ablation 专门验证它的 gain 占比是否 ≥ 5%；若否，关闭即可，不用回滚代码。

### 11.8 步骤 6 — MID_* 前缀分流（`prepare`）

- 全局 `data.max_factor_missing_ratio = 0.35`（默认），对 5min 频因子是合理门槛，但会把大量周频水平列误删。
- MID_* 列单独走 `mid_weekly.missing_ratio_relax = 0.65`（`prepare` 里按前缀分流），让周频水平列能活下来。
- MID_* 列在合并/filter 之后把剩余 NaN 补 0，避免 `dropna` 把整行杀掉——T3.2 验收里"以免 dropna 杀行"的落地动作。

### 11.9 缓存与恢复

- 特征矩阵 cache 落在 `results/cache/products/<PID>/<signature>.parquet`；签名包含 raw 文件 mtime/size、runtime factor spec、mid_weekly 文件列表等。
- 配套 `cache_meta.json` 里含 `mid_weekly_cols` 与 `mid_weekly_metadata`（L832-833 加载时回填），保证从 cache 读回来时 audit 依然可用。
- **改 MID 相关的 config（如 `available_dummy`、`derived.enabled`、`rolling_windows`）不会自动让 cache 失效**——需要显式 `--force-rebuild`，否则读到的是旧派生列。这是 T4.2 rerun 的必带 flag。

### 11.10 配置节（当前 baseline）

```yaml
# config.yaml
paths:
  mid_weekly_dir: data/mid_weekly/_cleaned   # 指向已清洗目录

data:
  use_mid_weekly: true
  mid_alignment: asof_forward_fill
  max_factor_missing_ratio: 0.35             # 全局门槛，MID_* 另走 relax

mid_weekly:
  available_dummy: true                       # MID_*_AVAILABLE
  ffill_max_bars: 8064                        # ≈ 4 周的 staleness clamp
  derived:
    enabled: true
    rolling_windows: [4, 13, 52]              # 周
    transforms: [ret, zscore, pct_rank]
  level_keep: true                            # 保留原始水平列
  missing_ratio_relax: 0.65                   # MID_* 专属门槛
```

### 11.11 Registry 绑定

- `data/product_registry.json` 里每个品种的 `mid_weekly_files: ["<PID>.xlsx"]`（相对 `paths.mid_weekly_dir`）。
- 写入由 `scripts/update_registry_with_mid_weekly.py` 幂等执行：扫 `_cleaned/*.xlsx`，匹配不到的品种保持空列表。
- baseline run `20260416_170652` 因 `mid_weekly_files=[]` 而全员 `mid_weekly_feature_count=0`；这是 A/B 对照成立的前提。
