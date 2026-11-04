"""Promueve una versión del registry a `champion`. El champion saliente queda con alias `anterior`
(rollback inmediato: `--version <anterior>`). Deja constancia en tags de la versión promovida.

    python -m src.training.promover --version 3 --motivo "concept drift: criterio nuevo de scouting"
"""
import argparse
from datetime import date

from mlflow.tracking import MlflowClient

from src.training.train import NOMBRE_MODELO, TRACKING


def promover(version: str, motivo: str):
    import mlflow
    mlflow.set_tracking_uri(TRACKING)
    c = MlflowClient()
    actual = c.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    if actual.version == version:
        print(f"v{version} ya es champion")
        return
    aliases = c.get_registered_model(NOMBRE_MODELO).aliases
    if aliases.get("challenger") == version:
        c.delete_registered_model_alias(NOMBRE_MODELO, "challenger")
    c.set_registered_model_alias(NOMBRE_MODELO, "anterior", actual.version)
    c.set_registered_model_alias(NOMBRE_MODELO, "champion", version)
    c.set_model_version_tag(NOMBRE_MODELO, version, "promovido", f"{date.today()} - {motivo}")
    c.set_model_version_tag(NOMBRE_MODELO, actual.version, "retirado", f"{date.today()} - reemplazado por v{version}")
    print(f"champion: v{actual.version} -> v{version}   (v{actual.version} queda como 'anterior')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--motivo", default="promoción manual")
    a = ap.parse_args()
    promover(a.version, a.motivo)
