# CTA趋势跟踪策略

双波动率域商品期货CTA研究框架，以1分钟K线为底层数据，沿"微观—中观—宏观"三层构建并评估择时增强方案。详细设计见 `docs/report.md`。

---

## 项目结构

```
.
├── config.yaml                         # 所有路径、超参数、信号与成本参数
├── pipeline/                           # 核心库与主入口
│   ├── train_products.py               # 批量训练主入口（单品种 / 全量 / 断点续跑）
│   ├── backtest.py                     # 回测引擎
│   ├── backtest_macro.py               # 宏观叠加回测
│   ├── dataset.py                      # 特征工程与数据集构建（FactorDatasetBuilder）
│   ├── modeling.py                     # 双域 LightGBM 训练
│   ├── factor_engine.py                # 运行时量价因子（154 个）
│   ├── cal_factors.py                  # 工程化与合成因子（35 + 118 个）
│   ├── factor_audit.py                 # 逐品种 walk-forward IC 审计
│   ├── diagnostics.py                  # 训练后图表生成（7 张/品种）
│   ├── judge_macro.py                  # 月频宏观信号判断
│   ├── config_utils.py                 # 配置加载与路径解析
│   └── build_product_registry.py       # 品种注册表扫描与写入
├── scripts/                            # 独立辅助脚本（不依赖 pipeline/）
│   ├── generate_report_charts.py       # 报告对比图表生成（ICIR、双域 vs 单域）
│   ├── compare_runs.py                 # 对比两次 run 的夏普 / 收益 / 回撤
│   ├── audit_mid_weekly_inputs.py      # 中观 xlsx 输入质量审计
│   ├── audit_mid_weekly_importance.py  # 中观因子在模型中的 Gain 重要性审计
│   └── exp_pure_micro_horizon_cutoff.py # horizon × cutoff 参数扫描实验
├── dataloader/
│   └── splitByVol.py                   # 波动率域划分与标签
├── data/
│   ├── 分产品1min主连/                  # 原始 1 分钟行情 CSV（按品种）
│   ├── mid_weekly/                     # 中观因子 xlsx（按品种）
│   └── product_registry.json           # 品种注册表（run 覆盖范围 / 中观文件绑定）
├── results/
│   ├── cache/products/                 # 特征缓存（按品种 + 签名哈希）
│   ├── models/                         # 单品种训练模型（persist_models=true 时写入）
│   ├── runs/                           # 批量训练输出（每次 run 独立子目录）
│   │   └── <run_id>/<PID>/             # 每品种：因子注册表 + 图表 + 汇总 JSON
│   └── backtest_macro/                 # 宏观叠加回测输出
├── docs/
│   └── report.md                       # 研究报告大纲（含图表引用）
└── tests/                              # 单元测试（从 repo 根运行）
```

---

## 快速开始

### 环境准备

```bash
pip install lightgbm pandas numpy scipy matplotlib
```

所有路径与超参均通过 `config.yaml` 统一控制。

### 单品种训练与回测

```bash
# 默认读取 config.yaml paths.raw_data 指定的品种
python pipeline/train_products.py --product RB

# 指定品种（CU = 铜）
python pipeline/train_products.py --product CU

# 强制重建特征缓存（更改因子代码后需加此参数）
python pipeline/train_products.py --product CU --force-rebuild

# 自定义本次 run 的输出目录名
python pipeline/train_products.py --product CU --run-name my_experiment
```

### 批量训练

```bash
# 训练 data/product_registry.json 中所有品种
python pipeline/train_products.py --all

# 指定多个品种
python pipeline/train_products.py --product RB --product CU --product AU

# 断点续跑（仅重试上次 non-success 的品种）
python pipeline/train_products.py --resume-run <run_id>
```

每个品种训练完成后，结果写入 `results/runs/<run_id>/<PID>/`，包括：

- `factor_registry.json` — 本品种因子 IC 审计结果
- `backtest_summary.json` — 测试集回测指标
- 7 张训练诊断图（数据分布、因子 IC、模型重要性、净值曲线等）

### 宏观叠加回测

```bash
python pipeline/backtest_macro.py
```

### 对比与报告工具

```bash
# 对比两次 run 的品种级结果
python scripts/compare_runs.py --baseline <run_id_1> --candidate <run_id_2>

# 生成报告 ICIR 图与双域 vs 单域对比图（结果写入 results/comparison/ 和 results/runs/micro_result/RB/）
python scripts/generate_report_charts.py

# 重建品种注册表（新增品种 CSV 后运行）
python pipeline/build_product_registry.py
```

### 测试

```bash
# 从 repo 根目录运行（pipeline/ 以包形式导入）
python -m unittest discover tests
```

---

## 核心配置说明

| 参数 | 位置 | 说明 |
|---|---|---|
| `data.target_horizon` | `config.yaml` | 预测视窗（1分钟K线根数） |
| `vol_split.vol_percentage` | `config.yaml` | 波动率域划分分位数（训练集） |
| `model.common_params` | `config.yaml` | LightGBM 公共超参 |
| `signal.entry_quantile` | `config.yaml` | 开仓阈值（验证集预测值分位） |
| `signal.exit_quantile` | `config.yaml` | 平仓阈值 |
| `backtest.commission_rate` | `config.yaml` | 手续费率 |
| `data.use_mid_weekly` | `config.yaml` | 是否启用中观因子 |

`--force-rebuild` 使特征缓存失效。不加时，训练与回测均复用 `results/cache/` 中已有的 parquet 文件，跳过耗时的因子工程步骤。

详细设计与实验结论见 `docs/report.md`。
