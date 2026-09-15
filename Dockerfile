# Multi-stage lightweight Python container for ConnectTel Churn API
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifact
COPY app/ app/
COPY model/ model/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=model/churn_pipeline.joblib

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
