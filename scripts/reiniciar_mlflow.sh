#!/usr/bin/env bash
# Borra TODO el historial de MLflow (runs, registry, artefactos) para empezar limpio.
cd "$(dirname "$0")/.."
pkill -f "mlflow server" 2>/dev/null; sleep 2
rm -rf mlflow.db mlartifacts mlruns
echo "MLflow reiniciado. Levantalo con scripts/iniciar_mlflow.sh"
