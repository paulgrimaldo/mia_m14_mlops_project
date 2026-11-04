"""Reentrenamiento tras la alarma de concept drift -> datos-v2 -> modelo v3 (challenger).

datos-v2 = ventana reciente del flujo 2026, ya etiquetada con el criterio NUEVO de scouting:
  train = lotes 7-9 · val = lote 10 · test = lotes 11-12
El histórico 2021-2025 NO se incluye: sus etiquetas siguen la regla vieja y contradicen el concepto
nuevo (se midió: sumarlo baja el F1 en datos del criterio nuevo de ~0.53 a ~0.45).

El nuevo modelo se registra como `challenger`. Se promueve a `champion` solo con --promover,
después de compararlo con el champion vigente sobre el MISMO test de datos-v2.
"""
import argparse
import json

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

from src.common.features import FEATURES, TARGET
from src.data.generar_datos import generar_stream
from src.drift.concepto import LOTE_CAMBIO
from src.training.datos_versionados import RAIZ, registrar_version
from src.training.train import EXPERIMENTO, NOMBRE_MODELO, TRACKING, entrenar_run, lgbm, xgb

VERSION = "datos-v2"
DIR = RAIZ / "data" / "versiones" / VERSION
LOTES = {"train": [7, 8, 9], "val": [10], "test": [11, 12]}
RUNS = [
    ("xgb_mice_d3_reentreno", "xgboost", "mice", xgb, dict(max_depth=3, n_estimators=200, learning_rate=0.10, subsample=0.9)),
    ("lgbm_mice_l7_reentreno", "lightgbm", "mice", lgbm, dict(num_leaves=7, n_estimators=150, learning_rate=0.10)),
]


def construir_datos_v2():
    stream = generar_stream(lote_cambio=LOTE_CAMBIO)
    assert min(LOTES["train"]) >= LOTE_CAMBIO, "datos-v2 debe contener solo lotes del criterio nuevo"
    DIR.mkdir(parents=True, exist_ok=True)
    partes, rutas = {}, {}
    for split, lotes in LOTES.items():
        partes[split] = stream[stream["lote"].isin(lotes)].reset_index(drop=True)
        rutas[split] = DIR / f"{split}.csv"
        partes[split].to_csv(rutas[split], index=False)
    registrar_version(VERSION, rutas, f"Flujo 2026, lotes {LOTES['train']}/{LOTES['val']}/{LOTES['test']} "
                                      "(train/val/test) etiquetados con el criterio nuevo de scouting")
    return partes, rutas


def main(promover: bool):
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENTO)
    c = MlflowClient()
    partes, rutas = construir_datos_v2()
    X = {k: v[FEATURES] for k, v in partes.items()}
    y = {k: v[TARGET].astype(int) for k, v in partes.items()}
    d = (X["train"], y["train"], X["val"], y["val"], X["test"], y["test"])

    # champion vigente evaluado en el mismo test de datos-v2 (línea de comparación)
    vig = c.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    modelo_vig = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    u_vig = float(c.get_run(vig.run_id).data.params["umbral_decision"])
    f1_vig = f1_score(y["test"], (modelo_vig.predict_proba(X["test"])[:, 1] >= u_vig).astype(int))
    print(f"champion vigente v{vig.version} en test {VERSION}: F1={f1_vig:.3f}")

    resultados = []
    for r in RUNS:
        run_id, val_f1, fam, _ = entrenar_run(*r, d, (VERSION, partes, rutas))
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_f1_champion_vigente", f1_vig)
            mlflow.set_tag("reentrenamiento", f"concept drift: reemplazo candidato de v{vig.version}")
        resultados.append((run_id, val_f1, fam))

    run_id, val_f1, fam = max(resultados, key=lambda r: r[1])
    mv = mlflow.register_model(f"runs:/{run_id}/modelo", NOMBRE_MODELO)
    test_f1 = c.get_run(run_id).data.metrics["test_f1"]
    for k, v in {"familia": fam, "val_f1": f"{val_f1:.4f}", "version_datos": VERSION,
                 "reemplaza_a": f"v{vig.version}"}.items():
        c.set_model_version_tag(NOMBRE_MODELO, mv.version, k, v)
    c.set_registered_model_alias(NOMBRE_MODELO, "challenger", mv.version)
    decision = "promovido a champion" if promover and test_f1 > f1_vig else "queda como challenger"
    if promover and test_f1 > f1_vig:
        c.set_registered_model_alias(NOMBRE_MODELO, "champion", mv.version)

    res = dict(version_datos=VERSION, modelo_nuevo=f"v{mv.version}", run_id=run_id, familia=fam,
               test_f1_nuevo=round(test_f1, 3), champion_vigente=f"v{vig.version}",
               test_f1_champion_vigente=round(f1_vig, 3), decision=decision)
    (RAIZ / "docs" / "evidencias" / "reentrenamiento.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--promover", action="store_true", help="mover el alias champion si el nuevo modelo es mejor")
    main(ap.parse_args().promover)
