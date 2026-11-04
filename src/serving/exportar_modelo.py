"""Descarga un modelo del Model Registry y deja junto a él un metadata.json
(versión, run_id, versión de datos, umbral, métricas) para empaquetarlo en la imagen.

Por defecto exporta `models:/talento-sc@champion` a src/serving/model (lo que va a la imagen).
El servicio NO usa una ruta suelta: la imagen se construye siempre a partir del alias, y la API
informa la versión, el run y la versión de datos exactos que sirve.

    python -m src.serving.exportar_modelo                                   # champion -> imagen
    python -m src.serving.exportar_modelo --version 2 --destino artifacts/modelos/v2
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

NOMBRE = "talento-sc"
ALIAS = "champion"
DESTINO = Path(__file__).resolve().parent / "model"


def exportar(alias=ALIAS, version=None, destino=DESTINO):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    c = MlflowClient()
    mv = c.get_model_version(NOMBRE, str(version)) if version else c.get_model_version_by_alias(NOMBRE, alias)
    uri = f"models:/{NOMBRE}/{mv.version}" if version else f"models:/{NOMBRE}@{alias}"
    run = c.get_run(mv.run_id)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(destino / "artefacto"))
    meta = {
        "modelo": NOMBRE, "alias": None if version else alias, "uri": uri, "version": mv.version,
        "run_id": mv.run_id, "run_name": run.info.run_name, "experimento_id": run.info.experiment_id,
        "version_datos": run.data.tags.get("version_datos", "desconocida"),
        "umbral": float(run.data.params["umbral_decision"]),
        "metricas": {k: round(v, 4) for k, v in run.data.metrics.items()},
        "mlflow_version": mlflow.__version__,
    }
    (destino / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default=ALIAS)
    ap.add_argument("--version", type=int, help="exportar una versión concreta en lugar del alias")
    ap.add_argument("--destino", default=str(DESTINO))
    a = ap.parse_args()
    print(json.dumps(exportar(a.alias, a.version, a.destino), indent=2))
