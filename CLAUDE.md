# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Databricks-native pipeline that ingests stock, FX, macroeconomic, and news data for C20 and STOXX50 companies, engineers features through a medallion architecture, trains an XGBoost classifier to predict next-day stock direction, and stores predictions in Delta Lake.

All code runs on Databricks — notebooks cannot be executed locally. There are no local build, lint, or test commands.

## Deployment

Pipelines are defined as Databricks Asset Bundles (YAML in `pipeline/`). Deploy via the Databricks CLI:

```bash
databricks bundle deploy   # deploy all pipelines
databricks bundle run combined_pipeline   # trigger a run
```

Notebook paths in the YAMLs are hardcoded to the workspace path `/Workspace/Users/marcus.egelund-muller@devoteam.com/databricks-stocks/...` — update these if deploying under a different user or workspace.

`combined-pipeline.yml` has placeholder `job_id: 0` entries for sub-pipelines that must be replaced with actual job IDs from the Databricks UI before use.

## Architecture

### Delta catalog layout

All tables live in the `stocks` Unity Catalog:

| Schema | Purpose |
|--------|---------|
| `stocks.reference` | Ticker reference table (`stocks.reference.tickers`) — source of truth for tracked companies |
| `stocks.stage` | Raw ingested data per ticker (one Delta table per stock ticker, plus `news_raw`) |
| `stocks.bronze` | Cleaned/unioned data (`stocks_all`, `news`) |
| `stocks.silver` | Feature-engineered data (`stocks_w_features`, `news_sentiment`, `fx_exchange_rates`, `macro_rates`) |
| `stocks.gold` | Model-ready feature sets (`stocks_w_prev_returns`, `stocks_combined_features`) |
| `stocks.models` | MLflow-registered models (Unity Catalog) |
| `stocks.reporting` | Prediction outputs (`predictions_combined`) |

### Data domains and pipeline flow

Four independent domain pipelines each run stage → bronze → silver, then the combined pipeline merges them:

```
stocks-pipeline  (18:17 CET)  stage → bronze → silver
fx-pipeline      (18:20 CET)  stage → bronze → silver
macro-pipeline   (18:25 CET)  stage → bronze → silver
news-pipeline    (18:00 CET)  stage → bronze → silver
                                         ↓
combined-pipeline (19:00 CET)
  gold_stocks   ← silver stocks
  gold_combined ← gold_stocks + silver fx + silver macro + silver news
  predict_combined ← gold_combined  (writes to stocks.reporting.predictions_combined)
```

The stocks domain also has its own standalone predict notebook (`predict_tomorrow`) using only stock features.

### Medallion layer responsibilities

- **Stage**: Fetches from external sources (yfinance for OHLCV and news; other sources for FX/macro). Merges into Delta tables using `uuid` or `Date` as merge keys — idempotent on re-run.
- **Bronze**: Unions per-ticker stage tables into a single `stocks_all` table; deduplicates news.
- **Silver (stocks)**: Computes technical features (`feat_daily_return`, `feat_volatility_ratio`, `feat_relative_Volume`, candlestick shadows, pivot points, etc.) and the binary `label` (1 if `feat_daily_return > 0`). Also creates a forward-filled calendar grid to handle weekends/holidays.
- **Silver (news)**: Calls Databricks built-in `ai_analyze_sentiment()` on article summaries (falling back to title), then aggregates to a daily `sentiment_score` per ticker (range −1 to +1).
- **Gold (stocks)**: Adds 1/2/3-day lag features for all silver features (`prev_feat_*_1`, `_2`, `_3`).
- **Gold (combined)**: Pivots FX from long to wide (24 columns: `{CURRENCY}_{rate|chg|avg5d|avg20d}`), then left-joins stocks gold + FX wide + macro + news sentiment on `(company, Date)`.
- **Predict**: Loads the `@champion` model alias from Unity Catalog (`stocks.models.combined_predictor@champion`), scores via a Pandas UDF broadcast pattern, and merges results into the reporting table.

### ML model lifecycle

- Training: `notebooks/models/xgBoost.py` (single-stock features) and `notebooks/models/train_combined_model.ipynb` (combined features)
- Hyperparameter tuning with Hyperopt + MLflow nested runs
- Registered to Unity Catalog as `stocks.models.xgb_direction_predictor` or `stocks.models.combined_predictor`
- After training, manually set the `champion` alias in the Databricks Models UI to promote a version to production

### Ticker reference

`notebooks/00_setup/setup_ticker_reference.ipynb` must be run once to populate `stocks.reference.tickers` before any pipeline can execute. It tracks 20 C20 (Danish) and 50 STOXX50 (European) tickers. The STOXX50 composition changes quarterly — verify at stoxx.com when updating.

Ticker symbols use Yahoo Finance format (e.g., `NOVO-B.CO`, `ASML.AS`). The company identifier used in Delta tables is derived by replacing `-` and `.` with `_` (e.g., `NOVO_B_CO`).

## Key conventions

- Silver features are prefixed `feat_`; gold lag features are prefixed `prev_feat_`
- All pipelines use `whenNotMatchedInsertAll` (or `whenMatchedUpdateAll`) merges — safe to re-run
- `spark.conf.set("spark.sql.ansi.enabled", "false")` is set in silver_stocks to allow division by zero in feature calculations
- The `combined_pipeline` runs 42 minutes after the last domain pipeline finishes, allowing time for completion
