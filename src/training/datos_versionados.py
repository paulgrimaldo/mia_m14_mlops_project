"""Versionado de datos: cada versión es un conjunto de archivos con hash SHA-256.

* `data/versiones.json` (versionado en git) es el manifiesto: versión -> archivos, hashes, filas, descripción.
* Cada run de MLflow registra con `mlflow.log_input` los splits exactos que usó (pestaña *Datasets*),
  más el tag `version_datos`, los hashes y el commit de git: run -> datos queda trazable.
* El tag de git `datos-vN` marca el commit en que se generó esa versión.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from src.common.features import TARGET

RAIZ = Path(__file__).resolve().parents[2]
MANIFIESTO = RAIZ / "data" / "versiones.json"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, text=True).strip()
    except Exception:
        return "desconocido"


def registrar_version(version: str, rutas: dict, descripcion: str) -> dict:
    """Agrega/actualiza la versión en el manifiesto y devuelve su entrada."""
    entrada = {
        "descripcion": descripcion,
        "archivos": {k: {"ruta": str(Path(r).relative_to(RAIZ)), "sha256": sha256(Path(r)),
                         "filas": int(len(pd.read_csv(r)))} for k, r in rutas.items()},
    }
    man = json.loads(MANIFIESTO.read_text()) if MANIFIESTO.exists() else {}
    man[version] = entrada
    MANIFIESTO.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    return entrada


def loguear_en_run(version: str, partes: dict, rutas: dict):
    """Dentro de un run activo: datasets (pestaña Datasets de MLflow) + tags de trazabilidad."""
    for split, df in partes.items():
        ds = mlflow.data.from_pandas(df, source=str(rutas[split]), targets=TARGET, name=f"{version}-{split}")
        mlflow.log_input(ds, context=split)
    mlflow.set_tags({"version_datos": version, "git_commit": git_commit(),
                     **{f"sha256_{s}": sha256(Path(r))[:16] for s, r in rutas.items()}})
