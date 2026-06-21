# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the model

```bash
# Activate the virtual environment first
source venv/bin/activate

# Run with defaults (beauty use case, today's date, train mode)
python -m core.propensity

# Run with specific options
python -m core.propensity --use-case food --anchor-date 2026-03-01 --mode train
python -m core.propensity --use-case beauty --anchor-date 2026-01-15 --mode evaluate
python -m core.propensity --use-case shops --anchor-date 2025-12-01 --mode train
```

Arguments:
- `--use-case`: `food`, `beauty`, or `shops`
- `--anchor-date`: `YYYY-MM-DD` (defaults to today)
- `--mode`: `train` (fits and saves model) or `evaluate` (loads saved model from `output/logistic.pkl`)

## Install dependencies

```bash
pip install -r requirements.txt
```

## Architecture

The pipeline runs in three stages:

1. **Data loading** (`core/render.py`): Renders a Jinja2 SQL template and runs it against BigQuery. Results are cached as a BQ table named `{BQ_DATASET}.{use_case}_{anchor_date}` — on re-runs with the same date/use-case, the cached table is used directly without re-querying.

2. **Model** (`models/logistic_regression.py`): Trains a `LogisticRegression` (with median imputation + standard scaling via sklearn `Pipeline`) or loads the last saved model from `output/logistic.pkl`. Writes three BQ tables: feature coefficients, permutation importances, and full propensity scores for all customers.

3. **Scoring** (`core/accuracy.py`): Computes accuracy, AUC, precision, recall, capture rate at top 10%, and precision@K, then appends results to `BQ_DATASET.model_results`.

## SQL templates

- `templates/query` — used for `beauty` and `shops`. Targets customers with 0 orders in the vertical in the past 90 days who placed at least 1 general order. Conversion window is 14 days after anchor date.
- `templates/food_query` — used for `food`. Shorter 30-day lookback, adds food-specific features (coffee/RTE ratios, per-FP filtering), and supports filtering by `primary_fp` (the fulfillment point a customer orders from most).

Both templates use `{{anchor_date}}` and `{{use_case_filter}}` as Jinja2 variables. `food_query` also accepts `{{fps}}` for FP filtering.

## Configuration (`configs/lookup.py`)

- `BQ_PROJECT` / `BQ_DATASET`: BigQuery project and dataset for all reads/writes.
- `VERTICAL_LOOKUP`: maps use-case names to BigQuery WHERE clause fragments.
- `FOOD_FPS`: list of New Cairo fulfillment points used to filter food predictions.
- `FEATURES`: per-use-case feature allowlist. Currently commented out in `models/logistic_regression.py` — set to `None` to use all available columns; set to a list to restrict features.

## Data

`input/` contains locally cached CSV exports per vertical and date (same schema as the BQ cached tables). These are not used by the pipeline directly — BQ is the source of truth — but are useful for offline inspection.

`output/logistic.pkl` is the serialized sklearn pipeline saved during `train` mode and loaded during `evaluate` mode.
