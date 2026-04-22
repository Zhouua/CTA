# Mid-Weekly Dummy-Variable Audit

Run: `results/runs/20260416_170652`

## 裁决规则（plan § 9 Q2）

- 若 `sum(MID_*_AVAILABLE gain) / sum(MID_* gain) >= 5%` **或** 消融 |Δval_sharpe| >= 0.1 → **保留** `available_dummy`。
- 否则 → **关闭** `available_dummy`。

## 每 (product, regime) 的 AVAILABLE 贡献

| Product | Regime | #MID总 | #AVAILABLE | #LEVEL | AVAILABLE gain / MID gain | 通过 5% 阈值 | 数据来源 |
|---|---|---:|---:|---:|---:|---|---|

## 汇总建议

- (product, regime) 对中 `AVAILABLE` 通过 5% 阈值：0 / 0

> 最终由用户填入 `docs/mid_weekly_integration_plan.md § 9 Q2`。
