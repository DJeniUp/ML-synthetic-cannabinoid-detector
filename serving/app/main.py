"""FastAPI application for CB1 activity prediction."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .model import predict_batch, predict_single
from .schemas import BatchPredictRequest, BatchPredictResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="CB1 Synthetic Cannabinoid Detector",
    description=(
        "Predicts pChEMBL potency on the human CB1 cannabinoid receptor "
        "(CHEMBL218) from a SMILES string. "
        "Compounds with pChEMBL ≥ 6.5 are flagged as potentially active."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict CB1 activity for a single compound."""
    result = predict_single(request.smiles)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse SMILES: {request.smiles!r}",
        )
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict CB1 activity for a list of compounds."""
    results, n_invalid = predict_batch(request.smiles_list)
    return BatchPredictResponse(
        results=[PredictResponse(**r) for r in results],
        n_invalid=n_invalid,
    )
