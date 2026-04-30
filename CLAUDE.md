# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dual-regime CTA research on 1-minute commodity futures bars. The pipeline labels each bar as `low_vol` / `high_vol` using a daily-vol cutoff computed **only on the training split**, trains a separate LightGBM regressor per regime, then runs a rule-based backtest whose entry/exit thresholds come from quantiles of the validation-set predictions. All paths, hyperparameters, signal rules, and cost assumptions live in `config.yaml`.

`docs/report.md` (Chinese) is the primary reference for design intent and the meaning of each output plot — read it before changing modeling/backtest defaults.

## Repo layout

- `pipeline/` — core library and main training/backtest entry points. Everything here is either imported by other pipeline modules or is a primary CLI (`train_products.py`, `backtest.py`, `backtest_macro.py`). `judge_macro.py` stays here because `backtest_macro.py` imports it; `build_product_registry.py` stays because `tests/` imports it; `diagnostics.py` stays because `train_products.py` imports it for post-training chart generation.
- `scripts/` — one-off auxiliary scripts: mid-weekly input audits, A/B comparison runs, feature-importance audits, horizon/cutoff sweep experiments, report chart generation. These are self-contained (no imports from `pipeline/`) and are invoked from the command line only.
- `tests/` — import via `from pipeline.xxx import ...` (run from repo root with `python -m unittest discover tests`).
- `data/`, `docs/`, `results/` — data inputs, design notes, batch training / backtest outputs.

## Common commands

```bash
# Single product (default RB; product_id resolved from paths.raw_data)
python pipeline/train_products.py --product RB [--config config.yaml] [--force-rebuild]
python pipeline/backtest.py [--config config.yaml] [--force-rebuild]

# Named run (custom run directory under results/runs/)
python pipeline/train_products.py --product RB --run-name my_experiment

# Batch over every product in data/product_registry.json
python pipeline/train_products.py --all
python pipeline/train_products.py --product RB --product CU
python pipeline/train_products.py --resume-run <run_id>   # only retries non-success entries

# Factor IC audit is no longer a separate command — it's executed inline by
# train_products.py per product (see pipeline/factor_audit.py). The audit
# writes results/runs/<run_id>/<PID>/factor_registry.json, which replaces
# both the old global data/factor_registry.json and per-product feature_manifest.json.

# Rebuild the registry by scanning data/分产品1min主连/
python pipeline/build_product_registry.py

# Macro-filtered overlay backtest (independent of backtest.py)
python pipeline/backtest_macro.py

# One-off scripts (see scripts/ for the full list)
python scripts/audit_mid_weekly_inputs.py --mid-weekly-dir data/mid_weekly --output docs/mid_weekly_audit.md
python scripts/audit_mid_weekly_importance.py --run results/runs/<run_id>
python scripts/compare_runs.py --baseline <baseline_id> --candidate <candidate_id>
python scripts/exp_pure_micro_horizon_cutoff.py   # horizon × vol_cutoff sweep experiment
python scripts/generate_report_charts.py          # regenerate report-level comparison charts

# Tests — module imports are `from pipeline.xxx import ...`, so run from repo root
python -m unittest discover tests
python -m unittest tests.test_dataset_and_modeling
```

`--force-rebuild` invalidates the merged-features parquet cache. Without it, both train and backtest will reuse `results/cache/...` and skip the slow factor-engineering step.

## Architecture

### Config-driven pipeline

`pipeline/config_utils.py::load_project_config` reads `config.yaml` and supports `config_override` — a dict that is **deep-merged** on top of the base config. This is the only mechanism that makes per-product runs work: `train_products.py::build_product_config_override` injects a `product:` section plus per-product output paths, and every downstream `prepare_data` / `train_dual_regime_models` / `build_backtest_settings` call accepts the same `config_override`. When adding new config-driven behavior, plumb `config_override` through rather than reading the YAML again.

Path keys in `config.yaml` are resolved relative to the directory containing the config file (`resolve_path` in `config_utils.py`). Absolute paths are passed through unchanged.

### Data flow

Per-product training flow (executed by `train_products.py::run_single_product_training`):

**① Data + Features** (`prepare_data` / `FactorDatasetBuilder`):
1. **`pipeline/dataset.py::FactorDatasetBuilder`** — single class that owns the entire feature pipeline:
   - `_read_raw_data` reads the per-product CSV (with timeout-retry around `pd.read_csv`).
   - `_merge_factor_features` calls `pipeline/factor_engine.py::generate_runtime_factors` when `factors.runtime.enabled=true` (the default). Legacy CSV factors in `data/factors/` are only used when runtime factors are disabled.
   - `_merge_mid_weekly_features` reads xlsx/csv mid-weekly files, applies a frequency-aware quality filter (`_apply_mid_weekly_quality_filter` — drops step-dummies and low-coverage columns), computes derived factors (RET/ZSCORE/PCT_RANK/ACCEL/extreme_flag over rolling windows), then merges onto the 1-min grid via `pd.merge_asof(direction="backward")` + forward-fill + staleness clamp. Files come from `product.mid_weekly_files` in the (possibly overridden) config.
   - `_add_engineered_features` calls `pipeline/cal_factors.py::add_engineered_features` to compute all `ENG_*` columns: 35 base features (single-bar shape, multi-window returns/RV/MA/volume, time encoding) and ~118 synthetic features (long-window 120/240/480, overnight/intraday running returns, VWAP/Z-score/Parkinson/tick-direction, OI, daily-lag context, cross-day same-minute history, semi-variance, factor fusion, etc.).
   - `_add_mid_micro_interactions` calls `pipeline/cal_factors.py::add_mid_micro_interactions` to compute `MIDxMICRO_<micro>_X_<mid>` columns (mid PCT_RANK_* × 5 core micro factors, up to `max_columns=100`). Runs after `_add_engineered_features` so micro columns exist.
   - `_add_targets` computes `future_return` (simple return at `target_horizon` bars, same-day only), `target_vol_scale` (floored intra-day rolling std × √horizon), `target_vol_norm` (training target), and `5min_return` (fixed log return at 5 bars). `target_horizon` is set in `config.yaml::data.target_horizon`.
   - `prepare` then calls `dataloader/splitByVol.py::split_by_vol` to compute the vol cutoff on the train rows (source: `vol_split.regime_label_source`, default `daily`), label every row `low_vol` (-1) / `high_vol` (+1), and assign `DATA_SPLIT` (train/val/test) by `split_granularity` boundaries (default `month`).

**② Factor Audit** (`pipeline/factor_audit.py::audit_and_filter`):
- Computed inline against the current product's `train_data` — never reads a pre-existing registry file.
- Walk-forward monthly Spearman IC on all candidate features; filters by `abs(mean_ic) >= min_abs_ic AND abs(icir) >= min_icir AND n_windows >= 3`.
- Writes `results/runs/<run_id>/<PID>/factor_registry.json` (per-product, replaces old global `data/factor_registry.json` + `feature_manifest.json`).
- After training, `backfill_importance` patches each record's `importance_gain` from the LightGBM booster (metadata only, does not re-filter).

**③ Model Training**:
2. **`pipeline/modeling.py::train_dual_regime_models`** — fits one LightGBM per regime on the audit-selected feature set. Hyperparameters come from `model.common_params`, optionally overridden by `model.low_vol_overrides` / `model.high_vol_overrides`. Each artifact bundles the booster, fitted scaler (`RobustScaler` by default), feature importance, and metrics. When `model.persist_models=true` they're written to `results/models/{low_vol,high_vol}/`. Batch runs disable persistence to avoid clobbering across products.

**④ Backtest + Diagnostics**:
3. **`pipeline/backtest.py::execute_backtest`**:
   - `predict_dual_regime` routes each row to its regime's booster.
   - `build_signal_rule_map` derives entry/exit thresholds **per regime** from absolute-value quantiles of the validation predictions, then floors entry by `(commission+slippage)*round_trip_turnover*cost_filter_multiple` if `enforce_cost_filter`.
   - `generate_positions` is a stateful loop (confirmation bars → open → min-hold → exit-quantile or reverse → cooldown). `flip_to_flat_first=true` forces a flat bar before reversing direction; `flatten_at_day_end=true` zeros the position at each day's last bar.
   - PnL is computed at bar level with `position.shift(1)` (next-bar execution) and aggregated daily for plotting.
   - After backtest, `pipeline/diagnostics.py::generate_diagnostic_charts` generates 8 charts to `product_dir` (distribution plots, factor IC plots, regime importance plots). Failures are caught and logged — they do not abort the training run.

### Regime labeling — important invariant

The vol cutoff **must** be computed from training rows only. `dataloader/splitByVol.py` enforces this; the regime label is never derived from a model. If you add a new "regime" idea, build it the same way: compute on train, apply to all splits.

### Caching keys

Per-product feature caches live at `results/cache/products/<PRODUCT_ID>/<signature>.parquet`. The signature hashes raw file mtime/size, runtime factor spec, `use_engineered_features`, `target_horizon`, the mid-weekly file list, and the mid-weekly quality-filter spec (`min_active_ratio`, `drop_step_dummy`, `freq_expected_ratio`). Changing any of those invalidates the cache automatically; changing engineered-feature code or `cal_factors.py` does not — use `--force-rebuild` then.

### Batch training session

`pipeline/train_products.py::execute_training_session` runs products sequentially and rewrites `run_summary.json` and `failed_products.json` after **every** product so interrupted runs are resumable via `--resume-run`. At the end of a completed run it also writes `run_summary.md` (human-readable markdown table sorted by test net annual return). `manifest.json` and `run_summary.csv` are no longer written to disk. `batch_training.required_data_start/end` in `config.yaml` is a hard gate — products with insufficient registry coverage are marked `skipped_insufficient_coverage` before any training is attempted.

### Macro overlay (separate from main pipeline)

`pipeline/judge_macro.py` reads monthly macro factors from `data/macro/` and produces a boolean monthly mask. `pipeline/backtest_macro.py` reuses the trained dual-regime models, then forces the position to zero in months where the macro mask is False. It does **not** modify `backtest.py` — keep the two paths independent.

## Conventions

- All entry-point scripts in `pipeline/` (`backtest.py`, `train_products.py`, `backtest_macro.py`, `dataset.py`, `diagnostics.py`) prepend both the repo root and their own directory to `sys.path` and set `MPLCONFIGDIR` to `.mplconfig/` before any matplotlib import. Preserve this when adding new entry points or matplotlib will write to `~/.config/matplotlib` and may collide.
- The pipeline uses `DATA_SPLIT` values `train` / `val` / `test` (note: `valid` is renamed to `val` after `split_by_vol`). `REGIME_LABEL` is `-1` for `low_vol` and `+1` for `high_vol` (`REGIME_NAME_MAP` in `dataset.py`).
- Tests import via `from pipeline.dataset import ...` (the `pipeline/` package), so they must be run from the repo root.
- Scripts under `scripts/` must remain self-contained — do not introduce imports from `pipeline/`. If a script starts needing pipeline library functions, promote it to `pipeline/` instead.
