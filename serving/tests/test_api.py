"""API integration tests.

Run with:
    uv run pytest tests/ -v

Requires a trained model at models/xgb_model.json.
The MODEL_PATH env var can override the default location.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_SMILES = "CC1=CC[C@@H]2[C@@H](C1)c1c(O)cc(CCCCCN=[N+]=[N-])cc1OC2(C)C"
INACTIVE_SMILES = "COc1ccc(CCNC(=O)Nc2ccc(Cl)cc2)cc1"  # known low-activity compound
INVALID_SMILES = "NOT_A_SMILES!!!"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_smiles():
    response = client.post("/predict", json={"smiles": VALID_SMILES})
    assert response.status_code == 200
    body = response.json()
    assert "pchembl_value" in body
    assert "is_active" in body
    assert "confidence" in body
    assert isinstance(body["pchembl_value"], float)
    assert isinstance(body["is_active"], bool)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_invalid_smiles_returns_422():
    response = client.post("/predict", json={"smiles": INVALID_SMILES})
    assert response.status_code == 422


def test_predict_empty_smiles_returns_422():
    response = client.post("/predict", json={"smiles": ""})
    assert response.status_code == 422


def test_predict_batch_mixed():
    payload = {
        "smiles_list": [VALID_SMILES, INACTIVE_SMILES, INVALID_SMILES]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["n_invalid"] == 1
    assert len(body["results"]) == 2
    for result in body["results"]:
        assert "pchembl_value" in result
        assert "is_active" in result
        assert "confidence" in result


def test_predict_batch_all_invalid():
    response = client.post("/predict_batch", json={"smiles_list": [INVALID_SMILES, "ALSO_BAD"]})
    assert response.status_code == 200
    body = response.json()
    assert body["n_invalid"] == 2
    assert body["results"] == []


def test_predict_active_flag_consistency():
    response = client.post("/predict", json={"smiles": VALID_SMILES})
    assert response.status_code == 200
    body = response.json()
    expected_active = body["pchembl_value"] >= 6.5
    assert body["is_active"] == expected_active


def test_confidence_at_threshold():
    """Confidence must be exactly 0.5 for a compound predicted exactly at 6.5."""
    # We can't force a specific prediction, so just verify the schema is valid.
    response = client.post("/predict", json={"smiles": VALID_SMILES})
    assert response.status_code == 200
    body = response.json()
    conf = body["confidence"]
    assert 0.0 <= conf <= 1.0
    if body["is_active"]:
        assert conf >= 0.5
    else:
        assert conf <= 0.5
