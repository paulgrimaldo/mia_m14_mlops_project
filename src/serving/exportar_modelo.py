"""Descarga el modelo referenciado por el ALIAS `champion` del Model Registry y deja
junto a él un metadata.json (versión, run_id, umbral, métricas) para empaquetarlo en la imagen.

El servicio NO usa una ruta suelta: la imagen se construye siempre a partir de
`models:/talento-sc@champion`, y la API informa la versión y el run exactos que sirve.
"""
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

NOMBRE = "talento-sc"
ALIAS = "champion"
DESTINO = Path(__file__).resolve().parent / "model"


def exportar():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    c = MlflowClient()
    mv = c.get_model_version_by_alias(NOMBRE, ALIAS)
    run = c.get_run(mv.run_id)
    DESTINO.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{NOMBRE}@{ALIAS}", dst_path=str(DESTINO / "artefacto"))
    meta = {
        "modelo": NOMBRE, "alias": ALIAS, "version": mv.version, "run_id": mv.run_id,
        "run_name": run.info.run_name, "experimento_id": run.info.experiment_id,
        "umbral": float(run.data.params["umbral_decision"]),
        "metricas": {k: round(v, 4) for k, v in run.data.metrics.items()},
        "mlflow_version": mlflow.__version__,
    }
    (DESTINO / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(exportar(), indent=2))
