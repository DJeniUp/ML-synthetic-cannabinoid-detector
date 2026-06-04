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

## Hyperparameter tuning (Optuna)

```bash
uv run python tune.py                    # 30 trials (default)
uv run python tune.py --n-trials 60     # more trials
uv run python tune.py --study-name my-study --seed 123
```

The script:
- Loads the same temporal train/test split used by `train.py` (never a random split)
- Featurises molecules once, then runs Optuna's TPE sampler for `--n-trials` trials
- Searches over: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`
- Logs every trial to MLflow as a nested run under a parent `optuna_*` run
- Writes the best parameters to `best_params.yaml`

After tuning, copy the `xgboost:` block from `best_params.yaml` into `config.yaml`, then run `train.py`:

```bash
uv run python tune.py --n-trials 30
# inspect best_params.yaml, then:
# manually copy the xgboost: block into config.yaml
uv run python train.py
```

To inspect all trial results in the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
# open http://localhost:5000
```

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
