FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml ./

# Install dependencies (sync without dev extras)
RUN uv sync --no-dev

# Copy application code
COPY app/ ./app/

# Model is mounted at runtime; provide a default location
# Override with -e MODEL_PATH=/path/to/xgb_model.json
RUN mkdir -p models

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
