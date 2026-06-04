"""Pydantic request / response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound", min_length=1)


class PredictResponse(BaseModel):
    smiles: str
    pchembl_value: float = Field(..., description="Predicted pChEMBL activity")
    is_active: bool = Field(..., description="True if predicted pChEMBL >= 6.5")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sigmoid of signed distance from the 6.5 threshold (0.5 = at threshold)",
    )


class BatchPredictRequest(BaseModel):
    smiles_list: list[str] = Field(..., min_length=1, description="List of SMILES strings")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    n_invalid: int = Field(..., description="Number of SMILES that could not be parsed")
