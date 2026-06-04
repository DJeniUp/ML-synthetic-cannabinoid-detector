# modeling — CB1 Synthetic Cannabinoid Detector

Trains an XGBoost regressor to predict pChEMBL activity on the CB1 receptor (CHEMBL218).

## Install

```bash
cd modeling
uv sync
```

## Run training

```bash
uv run python train.py
```

The first run fetches data from ChEMBL and caches it under `../data/`.  
Subsequent runs reuse the cached CSVs.  
The trained model is saved to `models/xgb_model.json`.  
MLflow metrics are logged under `mlruns/` (open with `uv run mlflow ui`).

## Walk-forward retraining

```bash
uv run python retrain.py
```

Iterates cutoff years 2010–2024, training on `year ≤ cutoff` and testing on `year == cutoff+1`.  
Saves a results table and plot to `models/`.

## Key configuration

All parameters live in `config.yaml`:

| Key | Default | Meaning |
|-----|---------|---------|
| `cutoff_year` | 2017 | Temporal train/test split |
| `threshold` | 6.5 | pChEMBL threshold for "active" |
| `penalty` | 5.0 | Multiplier for dangerous-miss squared error |
| `xgboost.*` | see file | XGBoost hyperparameters |

## Metrics

- **RMSE / MAE / R²** — standard regression metrics
- **Dangerous miss rate** — % of truly active compounds (≥ 6.5) predicted below threshold
- **Danger-aware RMSE** — RMSE with dangerous misses penalised `penalty`× heavier
