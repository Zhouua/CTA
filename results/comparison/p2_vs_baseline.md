# Mid-Weekly A/B Report — 20260416_170652 vs 20260423_180440

- Baseline run: `results/runs/20260416_170652`（`mid_weekly_feature_count = 0`）
- Candidate run: `results/runs/20260423_180440`（`mid_weekly_feature_count > 0`）
- 比较口径：各品种 test 集 net 指标（`backtest_summary.json → test_backtest`）

## 摘要

- 中位数 ΔSharpe = **-0.228** （> 0 时整体加分）
- 退步 (ΔSharpe<0) 占比 = **77.8%** （> 33.3% 时必须先做 T3.3 ablation 才能定论）
- 比较品种数 = 9
- 结论：mid_weekly **整体未加分**；且退步占比超过 1/3 → 强制进入 T3.3 ablation 复核

### 全面提升的品种（ΔSharpe ≥ 0.30） (1)

| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |
|---|---:|---|---:|---:|---:|---:|
| M | +1.602 | 4.044 → 5.646 | +0.158 | -0.004 | +20 | +0.038 |

### 持平品种（|ΔSharpe| < 0.30） (5)

| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |
|---|---:|---|---:|---:|---:|---:|
| Y | +0.089 | 3.821 → 3.910 | +0.029 | +0.007 | +106 | -0.015 |
| B | -0.048 | 4.493 → 4.445 | +0.027 | +0.001 | +16 | -0.022 |
| JD | -0.054 | 6.446 → 6.392 | +0.041 | -0.002 | -1 | +0.038 |
| BB | -0.228 | 2.393 → 2.165 | -0.058 | -0.042 | +32 | -0.058 |
| SN | -0.241 | 6.661 → 6.419 | +0.156 | +0.000 | +8 | +0.005 |

### 退步品种（ΔSharpe ≤ -0.30） (3)

| Product | ΔSharpe | baseline → candidate Sharpe | ΔAnnRet | ΔMaxDD | ΔTradeCount | ΔTradeWinRate |
|---|---:|---|---:|---:|---:|---:|
| FU | -0.804 | 6.673 → 5.869 | -0.845 | +0.028 | -104 | +0.008 |
| RU | -0.964 | 4.482 → 3.517 | -0.182 | -0.019 | +38 | +0.001 |
| FB | -2.630 | 2.594 → -0.036 | -0.217 | -0.061 | -54 | -0.038 |

### 结构变化（|ΔTradeCount / baseline TradeCount| ≥ 25%）（2）

| Product | baseline trades | candidate trades | % change | ΔSharpe |
|---|---:|---:|---:|---:|
| Y | 255 | 361 | +41.6% | +0.089 |
| BB | 88 | 120 | +36.4% | -0.228 |
