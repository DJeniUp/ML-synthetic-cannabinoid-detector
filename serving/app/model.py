"""Model loading and inference.

The model is an XGBoost regressor trained on count Morgan fingerprints
(radius=2, 2048 bits). Featurisation is replicated here so this service
has no dependency on the modeling package.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

THRESHOLD: float = float(os.environ.get("ACTIVITY_THRESHOLD", "6.5"))
_MODEL_PATH: str = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent.parent / "models" / "xgb_model.json"),
)

_RADIUS = 2
_FP_SIZE = 2048

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=_RADIUS, fpSize=_FP_SIZE)

_model: Optional[xgb.Booster] = None


def get_model() -> xgb.Booster:
    global _model
    if _model is None:
        model_path = Path(_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run `python train.py` in the modeling/ directory first."
            )
        _model = xgb.Booster()
        _model.load_model(str(model_path))
    return _model


def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _gen.GetCountFingerprintAsNumPy(mol).astype(np.int16)


def _sigmoid_confidence(pred: float, threshold: float = THRESHOLD) -> float:
    return 1.0 / (1.0 + math.exp(-(pred - threshold)))


def predict_single(smiles: str) -> Optional[dict]:
    """Return prediction dict or None if SMILES is unparseable."""
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None
    model = get_model()
    dmat = xgb.DMatrix(fp.reshape(1, -1).astype(np.float32))
    pred = float(model.predict(dmat)[0])
    return {
        "smiles": smiles,
        "pchembl_value": pred,
        "is_active": pred >= THRESHOLD,
        "confidence": _sigmoid_confidence(pred),
    }


def predict_batch(smiles_list: list[str]) -> tuple[list[dict], int]:
    """Return (list of prediction dicts, count of invalid SMILES)."""
    fps, valid_smiles, n_invalid = [], [], 0
    for smi in smiles_list:
        fp = smiles_to_fingerprint(smi)
        if fp is None:
            n_invalid += 1
        else:
            fps.append(fp)
            valid_smiles.append(smi)

    if not fps:
        return [], n_invalid

    model = get_model()
    dmat = xgb.DMatrix(np.array(fps, dtype=np.float32))
    preds = model.predict(dmat).tolist()

    results = [
        {
            "smiles": smi,
            "pchembl_value": pred,
            "is_active": pred >= THRESHOLD,
            "confidence": _sigmoid_confidence(pred),
        }
        for smi, pred in zip(valid_smiles, preds)
    ]
    return results, n_invalid
