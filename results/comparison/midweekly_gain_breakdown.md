# Mid-Weekly Feature-Importance Gain Breakdown

Run: `results/runs/20260422_154414`

## 数据完整性 / Coverage

- **警告：部分 (product, regime) 仅有 top-N 回退**；结果口径不等同于全特征 gain 份额。
- 触发 plan § 12.1 停下条件 → **§ 9 Q3**：是否允许扩展 `pipeline/modeling.py` 把全特征 importance 落到 `training_summary.json`。
- 缺口 regime 数：50
  示例：`AG/low_vol, AG/high_vol, AU/low_vol, AU/high_vol, B/low_vol, B/high_vol, BB/low_vol, BB/high_vol...`

## 回归品种（T4.3 ΔSharpe < 0）AVAILABLE gain share 分布

- 样本数 (pid, regime): 18
- median: 0.0000
- p90: 0.0000
- max: 0.0000
- threshold for keep: 0.05
- → 回归品种 AVAILABLE 中位数 gain share 低于 5%，倾向 drop（仍需 ablation 验证）。

## 每品种 × regime 明细

| Product | Regime | Full? | #Total | #MID-level | #MID-derived | #AVAILABLE | level share | derived share | AVAILABLE share | other share |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AG | low_vol | top-N | 20 | 0 | 4 | 0 | 0.000 | 0.087 | 0.000 | 0.913 |
| AG | high_vol | top-N | 20 | 0 | 5 | 0 | 0.000 | 0.050 | 0.000 | 0.950 |
| AU | low_vol | top-N | 20 | 1 | 4 | 0 | 0.012 | 0.083 | 0.000 | 0.905 |
| AU | high_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.019 | 0.000 | 0.981 |
| B | low_vol | top-N | 20 | 0 | 6 | 0 | 0.000 | 0.131 | 0.000 | 0.869 |
| B | high_vol | top-N | 20 | 0 | 5 | 0 | 0.000 | 0.124 | 0.000 | 0.876 |
| BB | low_vol | top-N | 20 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 1.000 |
| BB | high_vol | top-N | 20 | 1 | 3 | 0 | 0.041 | 0.064 | 0.000 | 0.895 |
| BU | low_vol | top-N | 20 | 1 | 4 | 0 | 0.018 | 0.095 | 0.000 | 0.887 |
| BU | high_vol | top-N | 20 | 1 | 5 | 0 | 0.011 | 0.110 | 0.000 | 0.880 |
| C | low_vol | top-N | 20 | 0 | 5 | 0 | 0.000 | 0.125 | 0.000 | 0.875 |
| C | high_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.048 | 0.000 | 0.952 |
| CS | low_vol | top-N | 20 | 0 | 1 | 0 | 0.000 | 0.041 | 0.000 | 0.959 |
| CS | high_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.048 | 0.000 | 0.952 |
| CU | low_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.041 | 0.000 | 0.959 |
| CU | high_vol | top-N | 20 | 1 | 2 | 0 | 0.024 | 0.027 | 0.000 | 0.949 |
| EG | low_vol | top-N | 20 | 0 | 4 | 0 | 0.000 | 0.113 | 0.000 | 0.887 |
| EG | high_vol | top-N | 20 | 1 | 1 | 0 | 0.102 | 0.008 | 0.000 | 0.890 |
| FB | low_vol | top-N | 20 | 2 | 1 | 0 | 0.076 | 0.079 | 0.000 | 0.845 |
| FB | high_vol | top-N | 20 | 0 | 1 | 0 | 0.000 | 0.029 | 0.000 | 0.971 |
| FU | low_vol | top-N | 20 | 1 | 5 | 0 | 0.017 | 0.139 | 0.000 | 0.844 |
| FU | high_vol | top-N | 20 | 0 | 6 | 0 | 0.000 | 0.247 | 0.000 | 0.753 |
| JD | low_vol | top-N | 20 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 1.000 |
| JD | high_vol | top-N | 20 | 0 | 3 | 0 | 0.000 | 0.078 | 0.000 | 0.922 |
| JM | low_vol | top-N | 20 | 1 | 4 | 0 | 0.015 | 0.190 | 0.000 | 0.795 |
| JM | high_vol | top-N | 20 | 0 | 1 | 0 | 0.000 | 0.021 | 0.000 | 0.979 |
| M | low_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.063 | 0.000 | 0.937 |
| M | high_vol | top-N | 20 | 1 | 3 | 0 | 0.033 | 0.140 | 0.000 | 0.827 |
| PB | low_vol | top-N | 20 | 0 | 1 | 0 | 0.000 | 0.053 | 0.000 | 0.947 |
| PB | high_vol | top-N | 20 | 1 | 3 | 0 | 0.029 | 0.059 | 0.000 | 0.911 |
| PG | low_vol | top-N | 20 | 0 | 5 | 0 | 0.000 | 0.126 | 0.000 | 0.874 |
| PG | high_vol | top-N | 20 | 1 | 5 | 0 | 0.014 | 0.056 | 0.000 | 0.930 |
| RB | low_vol | top-N | 20 | 0 | 1 | 0 | 0.000 | 0.032 | 0.000 | 0.968 |
| RB | high_vol | top-N | 20 | 1 | 0 | 0 | 0.020 | 0.000 | 0.000 | 0.980 |
| RR | low_vol | top-N | 20 | 0 | 7 | 0 | 0.000 | 0.460 | 0.000 | 0.540 |
| RR | high_vol | top-N | 20 | 0 | 18 | 0 | 0.000 | 0.000 | 0.000 | nan |
| RU | low_vol | top-N | 20 | 0 | 4 | 0 | 0.000 | 0.113 | 0.000 | 0.887 |
| RU | high_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.040 | 0.000 | 0.960 |
| SN | low_vol | top-N | 20 | 1 | 2 | 0 | 0.014 | 0.030 | 0.000 | 0.956 |
| SN | high_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.040 | 0.000 | 0.960 |
| SP | low_vol | top-N | 20 | 2 | 4 | 0 | 0.034 | 0.102 | 0.000 | 0.864 |
| SP | high_vol | top-N | 20 | 0 | 4 | 0 | 0.000 | 0.103 | 0.000 | 0.897 |
| V | low_vol | top-N | 20 | 2 | 1 | 0 | 0.042 | 0.018 | 0.000 | 0.940 |
| V | high_vol | top-N | 20 | 1 | 1 | 0 | 0.012 | 0.012 | 0.000 | 0.976 |
| WR | low_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.080 | 0.000 | 0.920 |
| WR | high_vol | top-N | 20 | 0 | 6 | 0 | 0.000 | 0.225 | 0.000 | 0.775 |
| Y | low_vol | top-N | 20 | 0 | 3 | 0 | 0.000 | 0.069 | 0.000 | 0.931 |
| Y | high_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.050 | 0.000 | 0.950 |
| ZN | low_vol | top-N | 20 | 0 | 2 | 0 | 0.000 | 0.036 | 0.000 | 0.964 |
| ZN | high_vol | top-N | 20 | 0 | 3 | 0 | 0.000 | 0.059 | 0.000 | 0.941 |

> 口径说明：`share` = 该类列 gain 之和 / 该 regime 中所有特征 gain 之和。只要 Full? 列出现 `top-N`，该行分母偏低（只覆盖前 20），share 会系统性偏高。
