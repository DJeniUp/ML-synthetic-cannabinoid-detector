"""OOP Trainer: featurise → train XGBoost → evaluate → log to MLflow → save."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


class Trainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        fp_cfg = cfg["fingerprint"]
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_cfg["radius"], fpSize=fp_cfg["n_bits"]
        )
        self.model: XGBRegressor | None = None

    # ------------------------------------------------------------------
    # Featurisation
    # ------------------------------------------------------------------

    def featurize(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute count Morgan fingerprints. Returns (X, y)."""
        fps, ys = [], []
        for smiles, target in zip(df["canonical_smiles"], df["pchembl_value"]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = self._gen.GetCountFingerprintAsNumPy(mol).astype(np.int16)
            fps.append(fp)
            ys.append(target)
        return np.array(fps), np.array(ys)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        xgb_cfg = self.cfg["xgboost"]
        self.model = XGBRegressor(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            min_child_weight=xgb_cfg.get("min_child_weight", 1),
            reg_alpha=xgb_cfg.get("reg_alpha", 0.0),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
            random_state=xgb_cfg["random_state"],
            n_jobs=xgb_cfg["n_jobs"],
        )
        self.model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Call train() before evaluate().")
        pred = self.model.predict(X_test)
        threshold = self.cfg["threshold"]
        penalty = self.cfg["penalty"]

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        dangerous_miss = (y_test >= threshold) & (pred < threshold)
        n_active = int((y_test >= threshold).sum())
        miss_count = int(dangerous_miss.sum())
        miss_rate = miss_count / n_active * 100 if n_active > 0 else 0.0

        base_sq = (y_test - pred) ** 2
        weighted = np.where(dangerous_miss, base_sq * penalty, base_sq)
        danger_rmse = float(np.sqrt(weighted.mean()))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "dangerous_miss_count": miss_count,
            "n_active": n_active,
            "dangerous_miss_rate_pct": miss_rate,
            "danger_aware_rmse": danger_rmse,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")
        save_path = path or self.cfg["model"]["save_path"]
        os.makedirs(Path(save_path).parent, exist_ok=True)
        self.model.save_model(save_path)
        return save_path

    # ------------------------------------------------------------------
    # Full pipeline with MLflow
    # ------------------------------------------------------------------

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        log_mlflow: bool = True,
    ) -> dict[str, float]:
        mlflow_cfg = self.cfg["mlflow"]
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run():
            # Log all config params
            flat_params: dict[str, Any] = {}
            for section, val in self.cfg.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        flat_params[f"{section}.{k}"] = v
                else:
                    flat_params[section] = val
            mlflow.log_params(flat_params)

            print("Featurising …")
            X_train, y_train = self.featurize(train_df)
            X_test, y_test = self.featurize(test_df)
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

            print("Training XGBoost …")
            self.train(X_train, y_train)

            print("Evaluating …")
            metrics = self.evaluate(X_test, y_test)
            mlflow.log_metrics(metrics)

            save_path = self.save()
            mlflow.log_artifact(save_path)
            print(f"Model saved → {save_path}")

        return metrics
