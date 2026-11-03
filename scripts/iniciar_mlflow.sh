#!/usr/bin/env bash
# Levanta la interfaz web de MLflow (tracking + Model Registry) en http://127.0.0.1:5000
cd "$(dirname "$0")/.."
exec .venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000
