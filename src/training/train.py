"""Entrenamiento reproducible con trazabilidad total en MLflow.

* Semilla fija (SEED=42). División train/val/test = 70/15/15 AGRUPADA por jugador
  (un mismo niño aparece en varias temporadas: separarlo evitaría fuga de información).
* 8 ejecuciones comparables (mismo split, misma métrica principal: val_f1):
  6 con MICE (XGBoost x3, LightGBM x3) + 2 líneas base (mediana, sin imputar).
* El mejor modelo de cada familia con MICE se registra en el Model Registry
  ('talento-sc'); el mejor global recibe el alias `champion` (el que se despliega).
"""
import argparse
import os
import warnings
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.common.features import CATEGORICAS, FEATURES, NUMERICAS, TARGET

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

SEED = 42
RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "data" / "processed"
NOMBRE_MODELO = "talento-sc"
EXPERIMENTO = "talento-sc-deteccion"
TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def preparar_splits():
    df = pd.read_csv(PROC / "jugador_temporada.csv")
    g = df["jugador_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, rest = next(gss.split(df, groups=g))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va, te = next(gss2.split(rest_df, groups=rest_df["jugador_id"]))
    parts = {"train": df.iloc[tr], "val": rest_df.iloc[va], "test": rest_df.iloc[te]}
    for k, v in parts.items():
        v.to_csv(PROC / f"{k}.csv", index=False)
    return parts


def construir_pipeline(clasificador, imputacion):
    if imputacion == "mice":
        num = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED,
                               add_indicator=True, skip_complete=True)
    elif imputacion == "mediana":
        num = SimpleImputer(strategy="median", add_indicator=True)
    else:
        num = "passthrough"  # los árboles manejan NaN de forma nativa
    prep = ColumnTransformer([
        ("num", num, NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    return Pipeline([("prep", prep), ("clf", clasificador)])


def lgbm(**kw):
    return LGBMClassifier(random_state=SEED, verbose=-1, scale_pos_weight=3, n_jobs=2, **kw)


def xgb(**kw):
    return XGBClassifier(random_state=SEED, scale_pos_weight=3, n_jobs=2, eval_metric="logloss", **kw)


RUNS = [
    # nombre, familia, imputación, constructor, hiperparámetros
    ("xgb_mice_d3", "xgboost", "mice", xgb, dict(max_depth=3, n_estimators=200, learning_rate=0.10, subsample=0.9)),
    ("xgb_mice_d4", "xgboost", "mice", xgb, dict(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.9)),
    ("xgb_mice_d6", "xgboost", "mice", xgb, dict(max_depth=6, n_estimators=500, learning_rate=0.03, subsample=0.8)),
    ("lgbm_mice_l7", "lightgbm", "mice", lgbm, dict(num_leaves=7, n_estimators=150, learning_rate=0.10)),
    ("lgbm_mice_l15", "lightgbm", "mice", lgbm, dict(num_leaves=15, n_estimators=200, learning_rate=0.05)),
    ("lgbm_mice_l31", "lightgbm", "mice", lgbm, dict(num_leaves=31, n_estimators=400, learning_rate=0.03, min_child_samples=30)),
    ("xgb_mediana_d4", "xgboost", "mediana", xgb, dict(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.9)),
    ("lgbm_sin_imputar_l15", "lightgbm", "ninguna", lgbm, dict(num_leaves=15, n_estimators=200, learning_rate=0.05)),
]


def mejor_umbral(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(th[i])


def metricas(y, p, umbral, prefijo):
    yhat = (p >= umbral).astype(int)
    return {f"{prefijo}_f1": f1_score(y, yhat), f"{prefijo}_precision": precision_score(y, yhat, zero_division=0),
            f"{prefijo}_recall": recall_score(y, yhat), f"{prefijo}_roc_auc": roc_auc_score(y, p),
            f"{prefijo}_pr_auc": average_precision_score(y, p)}


def nrmse_imputacion(pipe, Xv, frac=0.25):
    """Error de reconstrucción (RMSE normalizado): se ocultan valores CONOCIDOS y se comparan con lo imputado."""
    cols = ["sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s", "altura_cm", "peso_kg"]
    completo = Xv.dropna(subset=cols)
    if len(completo) < 50:
        return np.nan
    rng = np.random.default_rng(SEED)
    X = completo.copy()
    ocultas = {c: rng.random(len(X)) < frac for c in cols}
    for c, m in ocultas.items():
        X.loc[m, c] = np.nan
    num = pipe.named_steps["prep"].named_transformers_["num"]
    out = num.transform(X[NUMERICAS])
    errs = []
    for c, m in ocultas.items():
        j = NUMERICAS.index(c)
        errs.append(np.sqrt(np.mean((out[m, j] - completo[c].to_numpy()[m]) ** 2)) / completo[c].std())
    return float(np.mean(errs))


def entrenar_run(nombre, familia, imput, ctor, hp, d):
    Xtr, ytr, Xva, yva, Xte, yte = d
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags({"familia": familia, "imputacion": imput, "split": "70/15/15 agrupado por jugador",
                         "registrable": "si" if imput == "mice" else "no", "datos": "sintetico"})
        mlflow.log_params({**hp, "imputacion": imput, "seed": SEED, "scale_pos_weight": 3,
                           "n_features": len(FEATURES), "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte)})
        pipe = construir_pipeline(ctor(**hp), imput)
        pipe.fit(Xtr, ytr)
        pv, pt = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]
        u = mejor_umbral(yva, pv)  # el umbral se elige SOLO con validación
        m = {**metricas(yva, pv, u, "val"), **metricas(yte, pt, u, "test")}
        mlflow.log_param("umbral_decision", round(u, 4))
        mlflow.log_metrics(m)
        if imput != "ninguna":
            nr = nrmse_imputacion(pipe, Xva)
            if nr == nr:
                mlflow.log_metric("val_nrmse_imputacion", nr)
        sig = infer_signature(Xtr.head(50), pipe.predict_proba(Xtr.head(50))[:, 1])
        mlflow.sklearn.log_model(pipe, name="modelo", signature=sig, input_example=Xtr.head(3),
                                 serialization_format="cloudpickle")
        _guardar_importancias(pipe)
        print(f"{nombre:24s} val_f1={m['val_f1']:.3f} test_f1={m['test_f1']:.3f} "
              f"test_auc={m['test_roc_auc']:.3f} umbral={u:.2f}")
        return run.info.run_id, m["val_f1"], familia, imput


def _guardar_importancias(pipe):
    try:
        prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
        nombres = prep.get_feature_names_out()
        imp = pd.Series(clf.feature_importances_, index=nombres).sort_values(ascending=False)
        p = PROC / "importancias_tmp.csv"
        imp.rename("importancia").to_csv(p)
        mlflow.log_artifact(str(p), "analisis")
    except Exception:
        pass


def registrar_mejores(resultados):
    """Registra el mejor run MICE de cada familia y marca el alias `champion`."""
    client = MlflowClient()
    mice = [r for r in resultados if r[3] == "mice"]
    mejores = {}
    for r in mice:
        if r[2] not in mejores or r[1] > mejores[r[2]][1]:
            mejores[r[2]] = r
    ordenados = sorted(mejores.values(), key=lambda r: r[1])  # el peor primero => versión 1
    versiones = []
    for run_id, f1, fam, _ in ordenados:
        mv = mlflow.register_model(f"runs:/{run_id}/modelo", NOMBRE_MODELO)
        client.set_model_version_tag(NOMBRE_MODELO, mv.version, "familia", fam)
        client.set_model_version_tag(NOMBRE_MODELO, mv.version, "val_f1", f"{f1:.4f}")
        versiones.append((mv.version, f1, fam))
    campeon = max(versiones, key=lambda v: v[1])
    client.set_registered_model_alias(NOMBRE_MODELO, "champion", campeon[0])
    for v in versiones:
        if v[0] != campeon[0]:
            client.set_registered_model_alias(NOMBRE_MODELO, "challenger", v[0])
    print(f"\nRegistrado '{NOMBRE_MODELO}': " + ", ".join(f"v{v[0]}={v[2]} (val_f1={v[1]:.3f})" for v in versiones))
    print(f"Alias champion -> v{campeon[0]}")
    return versiones, campeon


def main():
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENTO)
    parts = preparar_splits()
    Xs = {k: v[FEATURES] for k, v in parts.items()}
    ys = {k: v[TARGET].astype(int) for k, v in parts.items()}
    d = (Xs["train"], ys["train"], Xs["val"], ys["val"], Xs["test"], ys["test"])
    print(f"train={len(d[0])} val={len(d[2])} test={len(d[4])}  tasa talento train={d[1].mean():.3f}")
    resultados = [entrenar_run(*r, d) for r in RUNS]
    registrar_mejores(resultados)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
