# Mid-Weekly A/B Report — 20260422_154414 vs 20260423_180440

- Baseline run: `results/runs/20260422_154414`（`mid_weekly_feature_count = 0`）
- Candidate run: `results/runs/20260423_180440`（`mid_weekly_feature_count > 0`）
- 比较口径：各品种 test 集 net 指标（`backtest_summary.json → test_backtest`）

## 摘要

- 中位数 ΔSharpe = **+0.157** （> 0 时整体加分）
- 退步 (ΔSharpe<0) 占比 = **11.1%** （> 33.3% 时必须先做 T3.3 ablation 才能定论）
- 比较品种数 = 9
- 结论：mid_weekly **整体加分**；退步占比 ≤ 1/3 → 可按 gain-share 结论决定 AVAILABLE 去留

### 全面提升的品种（ΔSharpe ≥ 0.30） (4)

| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |
|---|---:|---|---:|---:|---:|---:|
| M | +1.637 | 4.009 → 5.646 | +0.126 | +0.001 | +11 | +0.027 |
| FU | +1.097 | 4.773 → 5.869 | +2.132 | +0.159 | -82 | +0.010 |
| Y | +0.829 | 3.081 → 3.910 | +0.065 | +0.011 | +33 | +0.008 |
| B | +0.477 | 3.967 → 4.445 | +0.065 | +0.002 | +32 | +0.015 |

### 持平品种（|ΔSharpe| < 0.30） (4)

| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |
|---|---:|---|---:|---:|---:|---:|
| SN | +0.157 | 6.262 → 6.419 | +0.221 | +0.002 | -8 | +0.023 |
| JD | +0.136 | 6.256 → 6.392 | +0.042 | -0.002 | -25 | +0.013 |
| BB | +0.000 | 2.165 → 2.165 | +0.000 | +0.000 | +0 | +0.000 |
| FB | +0.000 | -0.036 → -0.036 | +0.000 | +0.000 | +0 | +0.000 |

### 退步品种（ΔSharpe ≤ -0.30） (1)

| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |
|---|---:|---|---:|---:|---:|---:|
| RU | -0.622 | 4.139 → 3.517 | -0.184 | +0.007 | +50 | -0.003 |

### 结构变化（|ΔTradeCount / baseline TradeCount| ≥ 25%）（0）

_无_
