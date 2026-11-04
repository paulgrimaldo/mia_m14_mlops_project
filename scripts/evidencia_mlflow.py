"""Vuelca a docs/evidencias el estado de MLflow: runs ordenados por la métrica principal y Model Registry."""
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
OUT = Path(__file__).resolve().parents[1] / "docs" / "evidencias"
c = MlflowClient()
exp = c.get_experiment_by_name("talento-sc-deteccion")
runs = mlflow.search_runs([exp.experiment_id], order_by=["metrics.val_f1 DESC"])
cols = {"run_id": "run_id", "tags.mlflow.runName": "run", "tags.version_datos": "version_datos", "tags.familia": "familia", "tags.imputacion": "imputacion",
        "metrics.val_f1": "val_f1", "metrics.test_f1": "test_f1", "metrics.test_recall": "test_recall",
        "metrics.test_precision": "test_precision", "metrics.test_roc_auc": "test_auc",
        "metrics.val_nrmse_imputacion": "nrmse_imputacion", "params.umbral_decision": "umbral"}
t = runs[list(cols)].rename(columns=cols).round(4)
t.to_csv(OUT / "mlflow_runs.csv", index=False)
lines = ["EXPERIMENTO talento-sc-deteccion (ordenado por val_f1, la métrica principal)", t.to_string(index=False), "",
         "MODEL REGISTRY: talento-sc"]
for mv in c.search_model_versions("name='talento-sc'"):
    al = [a for a, v in c.get_registered_model("talento-sc").aliases.items() if v == mv.version]
    ins = c.get_run(mv.run_id).inputs.dataset_inputs
    ds = ", ".join(f"{i.dataset.name}(digest {i.dataset.digest})" for i in ins)
    lines.append(f"  v{mv.version}  alias={al}  run_id={mv.run_id}  tags={dict(mv.tags)}")
    lines.append(f"       datasets: {ds}")
champ = c.get_model_version_by_alias("talento-sc", "champion")
lines += ["", f"VERSIÓN DESPLEGADA: talento-sc v{champ.version} (alias champion) <- run {champ.run_id} "
              f"({c.get_run(champ.run_id).info.run_name})"]
(OUT / "mlflow_registry.txt").write_text("\n".join(lines))
print("\n".join(lines))
