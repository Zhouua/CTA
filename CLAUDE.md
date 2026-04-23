# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dual-regime CTA research on 5-minute commodity futures bars. The pipeline labels each bar as `low_vol` / `high_vol` using a cutoff computed **only on the training split**, trains a separate LightGBM regressor per regime, then runs a rule-based backtest whose entry/exit thresholds come from quantiles of the validation-set predictions. All paths, hyperparameters, signal rules, and cost assumptions live in `config.yaml`.

The README (Chinese) is the most detailed reference for design intent and the meaning of each output plot — read it before changing modeling/backtest defaults.

## Common commands

```bash
# Single product (uses paths.raw_data, default RBZL.SHF.csv)
python code/train.py [--config config.yaml] [--force-rebuild]
python code/backtest.py [--config config.yaml] [--force-rebuild]

# Batch over every product in data/product_registry.json
python code/train_products.py --all
python code/train_products.py --product RB --product CU
python code/train_products.py --resume-run <run_id>   # only retries non-success entries

# Rebuild the registry by scanning data/分产品1min主连/
python code/build_product_registry.py

# Macro-filtered overlay backtest (independent of backtest.py)
python code/backtest_macro.py

# Tests — module imports are `from code.xxx import ...`, so run from repo root
python -m unittest discover tests
python -m unittest tests.test_dataset_and_modeling
```

`--force-rebuild` invalidates the merged-features parquet cache. Without it, both train and backtest will reuse `results/cache/...` and skip the slow factor-engineering step.

## Architecture

### Config-driven pipeline

`code/config_utils.py::load_project_config` reads `config.yaml` and supports `config_override` — a dict that is **deep-merged** on top of the base config. This is the only mechanism that makes per-product runs work: `train_products.py::build_product_config_override` injects a `product:` section plus per-product output paths, and every downstream `prepare_data` / `train_dual_regime_models` / `build_backtest_settings` call accepts the same `config_override`. When adding new config-driven behavior, plumb `config_override` through rather than reading the YAML again.

Path keys in `config.yaml` are resolved relative to the directory containing the config file (`resolve_path` in `config_utils.py`). Absolute paths are passed through unchanged.

### Data flow

1. **`code/dataset.py::FactorDatasetBuilder`** — single class that owns the entire feature pipeline:
   - `_read_raw_data` reads the per-product CSV (with timeout-retry around `pd.read_csv`).
   - `_merge_factor_features` calls `code/factor_engine.py::generate_runtime_factors` when `factors.runtime.enabled=true` (the default). Legacy CSV factors in `data/factors/` are only used when runtime factors are disabled.
   - `_merge_mid_weekly_features` merges mid-frequency factors via `pd.merge_asof(direction="backward")` then forward-fills. Files come from `product.mid_weekly_files` in the (possibly overridden) config.
   - `_add_engineered_features` adds `ENG_*` columns (multi-window returns, RV, ATR, MA deviation, intrabar shape, time-of-day sin/cos).
   - `_add_targets` computes `future_return`, `target_vol_scale` (floored intra-day rolling std × √horizon), and `target_vol_norm` (default training target — vol-normalized future return). Predictions are de-normalized in `modeling._convert_prediction`.
   - `prepare` then calls `dataloader/splitByVol.py::split_by_vol` to compute the vol cutoff on the train rows, label every row `low_vol` (-1) / `high_vol` (+1), and assign `DATA_SPLIT` (train/val/test) by month/week/day boundaries.

2. **`code/modeling.py::train_dual_regime_models`** — fits one LightGBM per regime. Hyperparameters come from `model.common_params`, optionally overridden by `model.low_vol_overrides` / `model.high_vol_overrides`. Each artifact bundles the booster, fitted scaler (`RobustScaler` by default), feature importance, and metrics. When `model.persist_models=true` they're written to `results/models/{low_vol,high_vol}/`. Batch runs disable persistence to avoid clobbering across products.

3. **`code/backtest.py::execute_backtest`**:
   - `predict_dual_regime` routes each row to its regime's booster.
   - `build_signal_rule_map` derives entry/exit thresholds **per regime** from absolute-value quantiles of the validation predictions, then floors entry by `(commission+slippage)*round_trip_turnover*cost_filter_multiple` if `enforce_cost_filter`.
   - `generate_positions` is a stateful loop (confirmation bars → open → min-hold → exit-quantile or reverse → cooldown). `flip_to_flat_first=true` forces a flat bar before reversing direction; `flatten_at_day_end=true` zeros the position at each day's last bar.
   - PnL is computed at bar level with `position.shift(1)` (next-bar execution) and aggregated daily for plotting.

### Regime labeling — important invariant

The vol cutoff **must** be computed from training rows only. `dataloader/splitByVol.py` enforces this; the regime label is never derived from a model. If you add a new "regime" idea, build it the same way: compute on train, apply to all splits.

### Caching keys

Per-product feature caches live at `results/cache/products/<PRODUCT_ID>/<signature>.parquet`. The signature hashes raw file mtime/size, runtime factor spec, `use_engineered_features`, `target_horizon`, and the mid-weekly file list. Changing any of those invalidates the cache automatically; changing engineered-feature code does not — use `--force-rebuild` then.

### Batch training session

`code/train_products.py::execute_training_session` runs products sequentially and rewrites `manifest.json`, `run_summary.{json,csv}`, and `failed_products.json` after **every** product so interrupted runs are resumable via `--resume-run`. `batch_training.required_data_start/end` in `config.yaml` is a hard gate — products with insufficient registry coverage are marked `skipped_insufficient_coverage` before any training is attempted.

### Macro overlay (separate from main pipeline)

`code/judge_macro.py` reads monthly macro factors from `data/macro/` and produces a boolean monthly mask. `code/backtest_macro.py` reuses the trained dual-regime models, then forces the position to zero in months where the macro mask is False. It does **not** modify `backtest.py` — keep the two paths independent.

## Conventions

- All entry-point scripts (`train.py`, `backtest.py`, `train_products.py`, `backtest_macro.py`, `dataset.py`) prepend both the repo root and `code/` to `sys.path` and set `MPLCONFIGDIR` to `.mplconfig/` before any matplotlib import. Preserve this when adding new entry points or matplotlib will write to `~/.config/matplotlib` and may collide.
- The pipeline uses `DATA_SPLIT` values `train` / `val` / `test` (note: `valid` is renamed to `val` after `split_by_vol`). `REGIME_LABEL` is `-1` for `low_vol` and `+1` for `high_vol` (`REGIME_NAME_MAP` in `dataset.py`).
- Tests import via `from code.dataset import ...` (the `code/` package), so they must be run from the repo root.
