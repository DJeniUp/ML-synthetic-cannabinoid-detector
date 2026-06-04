# serving — CB1 Cannabinoid Detector API

FastAPI service that predicts CB1 receptor potency (pChEMBL) from SMILES strings.

## Prerequisites

Train the model first from the `modeling/` sub-project:

```bash
cd ../modeling
uv run python train.py          # produces models/xgb_model.json
cp models/xgb_model.json ../serving/models/
```

## Install and run locally

```bash
cd serving
uv sync
uv run uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single SMILES → pChEMBL + activity flag + confidence |
| POST | `/predict_batch` | List of SMILES → batch predictions |

### Example — single prediction

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @examples/input_example.json
```

### Example — batch prediction

```bash
curl -X POST http://localhost:8000/predict_batch \
     -H "Content-Type: application/json" \
     -d @examples/batch_example.json
```

## Run tests

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

## Docker

```bash
# Build (from serving/ directory)
docker build -t cb1-serving .

# Run — mount the trained model
docker run -p 8000:8000 \
  -v "$(pwd)/../modeling/models/xgb_model.json:/app/models/xgb_model.json:ro" \
  cb1-serving

# Or set MODEL_PATH via env
docker run -p 8000:8000 \
  -e MODEL_PATH=/models/xgb_model.json \
  -v "$(pwd)/../modeling/models:/models:ro" \
  cb1-serving
```

## Response schema

```json
{
  "smiles": "CC1=CC[C@@H]2...",
  "pchembl_value": 7.42,
  "is_active": true,
  "confidence": 0.83
}
```

- `is_active`: `true` if `pchembl_value >= 6.5`
- `confidence`: sigmoid of `(pred − 6.5)`, in [0, 1]; 0.5 means prediction is exactly at the threshold
