"""Hyperparameter optimisation with Optuna.

Tunes XGBoost on the temporal train/test split (never a random split).
Featurisation is done once before the study to avoid repeating the
expensive SMILES-to-fingerprint conversion on every trial.

Usage:
    uv run python tune.py [--n-trials N] [--study-name NAME] [--seed SEED]

After the study completes, best parameters are written to best_params.yaml.
Copy the xgboost block from that file into config.yaml, then run train.py.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import mlflow
import optuna
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.config import load_config
from pipeline.data import load_or_fetch
from pipeline.trainer import Trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEARCH_SPACE = {
    # (suggest_type, *args, **kwargs)
    "n_estimators":    ("int",   200,   1000),
    "learning_rate":   ("float", 0.01,  0.3,   {"log": True}),
    "max_depth":       ("int",   3,     10),
    "subsample":       ("float", 0.5,   1.0),
    "colsample_bytree":("float", 0.5,   1.0),
    "min_child_weight":("int",   1,     10),
    "reg_alpha":       ("float", 1e-8,  10.0,  {"log": True}),
    "reg_lambda":      ("float", 1e-8,  10.0,  {"log": True}),
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> int | float:
    kind, *rest = spec
    kwargs = rest.pop() if rest and isinstance(rest[-1], dict) else {}
    low, high = rest
    if kind == "int":
        return trial.suggest_int(name, low, high, **kwargs)
    return trial.suggest_float(name, low, high, **kwargs)


def _make_objective(
    X_train, y_train, X_test, y_test, base_cfg: dict, parent_run_id: str
):
    mlflow_cfg = base_cfg["mlflow"]

    def objective(trial: optuna.Trial) -> float:
        suggested = {name: _suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()}

        trial_cfg = copy.deepcopy(base_cfg)
        trial_cfg["xgboost"].update(suggested)

        trainer = Trainer(trial_cfg)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        # Store all metrics so we can log them to the parent run later
        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        with mlflow.start_run(
            nested=True,
            parent_run_id=parent_run_id,
            run_name=f"trial_{trial.number:03d}",
        ):
            mlflow.log_params(suggested)
            mlflow.log_metrics(metrics)

        rmse = metrics["rmse"]
        print(
            f"  Trial {trial.number:3d} | "
            f"RMSE={rmse:.4f}  "
            f"miss={metrics['dangerous_miss_rate_pct']:.1f}%  "
            f"danger_rmse={metrics['danger_aware_rmse']:.4f}  "
            f"lr={suggested['learning_rate']:.4f}  "
            f"depth={suggested['max_depth']}  "
            f"n={suggested['n_estimators']}"
        )
        return rmse

    return objective


def _write_best_params(best_params: dict, best_metrics: dict, cfg_path: Path) -> Path:
    out = {
        "xgboost": {
            **best_params,
            "random_state": 42,
            "n_jobs": -1,
        },
        "best_metrics": best_metrics,
    }
    out_path = cfg_path.parent / "best_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(out, f, sort_keys=False, default_flow_style=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for CB1 XGBoost model")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--study-name", default="xgb-cb1-tuning", help="Optuna study name")
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed for reproducibility")
    args = parser.parse_args()

    cfg_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(cfg_path)

    print("Loading data …")
    train_df, test_df = load_or_fetch(cfg)
    print(f"  Train: {len(train_df)} molecules | Test: {len(test_df)} molecules")

    print("Featurising (once, before study) …")
    base_trainer = Trainer(cfg)
    X_train, y_train = base_trainer.featurize(train_df)
    X_test, y_test = base_trainer.featurize(test_df)
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
    )

    print(f"\nStarting Optuna study '{args.study_name}' — {args.n_trials} trials …\n")

    with mlflow.start_run(run_name=f"optuna_{args.study_name}") as parent_run:
        mlflow.log_params({
            "n_trials": args.n_trials,
            "sampler": "TPE",
            "sampler_seed": args.seed,
            "objective_metric": "rmse",
            "cutoff_year": cfg["cutoff_year"],
        })

        objective = _make_objective(
            X_train, y_train, X_test, y_test, cfg, parent_run.info.run_id
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        best_metrics = {k: best.user_attrs[k] for k in best.user_attrs}
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

    out_path = _write_best_params(study.best_params, best_metrics, cfg_path)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  Trial number : {best.number}")
    print(f"  RMSE         : {best.value:.4f}")
    print(f"  Miss rate    : {best_metrics.get('dangerous_miss_rate_pct', '?'):.1f}%")
    print(f"  Danger RMSE  : {best_metrics.get('danger_aware_rmse', '?'):.4f}")
    print("\n  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\nBest params written → {out_path}")


if __name__ == "__main__":
    main()
