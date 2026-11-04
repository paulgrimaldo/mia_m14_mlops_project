#!/usr/bin/env bash
# Reproduce TODO el flujo de datos + modelo desde cero (MLflow debe poder arrancar en :5000).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
scripts/reiniciar_mlflow.sh
(nohup scripts/iniciar_mlflow.sh > mlflow_server.log 2>&1 &)
for i in $(seq 1 30); do curl -sf http://127.0.0.1:5000/health >/dev/null && break; sleep 1; done
$PY -m src.data.generar_datos
$PY -m src.data.limpiar_ofuscar > /dev/null
$PY -m src.training.train
$PY -m src.serving.exportar_modelo > /dev/null
$PY -m src.drift.calibrar_umbrales > /dev/null
$PY -m src.drift.concepto
$PY -m src.training.reentrenar
