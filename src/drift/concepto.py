"""Concept drift: la relación entradas -> 'talento' cambia aunque las entradas luzcan igual.

Escenario (lote 7 en adelante): los scouts adoptan un criterio nuevo (valoran técnica y visión de juego,
dejan de premiar el físico/maduración temprana/edad relativa). Los features siguen con la misma
distribución, pero la regla que genera la etiqueta cambió.

Criterio de reentrenamiento (justificado con simulación de falsas alarmas):
    ALARMA si F1_lote < F1_base - CAIDA durante K lotes CONSECUTIVOS.
Retraso de etiquetas: la etiqueta real (promoción a selección/preselección) llega L lotes después;
se muestra la alarma "con retraso" que es la que realmente se vería en producción.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.features import FEATURES  # noqa: E402
from src.data.generar_datos import generar_stream  # noqa: E402
from src.drift.detectores import evaluar_lote  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "data" / "processed"
EVID = RAIZ / "docs" / "evidencias"
MODELO_DIR = RAIZ / "src" / "serving" / "model"

CAIDA = 0.10      # caída absoluta de F1 respecto al baseline
K_LOTES = 3       # lotes consecutivos
RETRASO = 3       # lotes que tarda en llegar la etiqueta real
LOTE_CAMBIO = 7


def cargar_modelo():
    import mlflow.sklearn
    meta = json.loads((MODELO_DIR / "metadata.json").read_text())
    return mlflow.sklearn.load_model(str(MODELO_DIR / "artefacto")), meta


def primera_alarma(f1s, base, caida=CAIDA, k=K_LOTES):
    """Índice (1-based) del lote en que se cumple el criterio, o None."""
    run = 0
    for i, f in enumerate(f1s, start=1):
        run = run + 1 if f < base - caida else 0
        if run >= k:
            return i
    return None


def f1_por_lote(modelo, umbral, df):
    out = []
    for l, g in df.groupby("lote"):
        yhat = (modelo.predict_proba(g[FEATURES])[:, 1] >= umbral).astype(int)
        out.append(dict(lote=l, f1=f1_score(g["talento"], yhat), recall=recall_score(g["talento"], yhat),
                        tasa_pred_positiva=yhat.mean(), tasa_real_positiva=g["talento"].mean()))
    return pd.DataFrame(out)


def falsas_alarmas(modelo, umbral, base, n_sim=300, n_lotes=12, tam=400):
    """Secuencias SIN cambio de regla (muestras del pool val+test): ¿con qué frecuencia dispara el criterio?"""
    pool = pd.concat([pd.read_csv(PROC / "val.csv"), pd.read_csv(PROC / "test.csv")], ignore_index=True)
    p = (modelo.predict_proba(pool[FEATURES])[:, 1] >= umbral).astype(int)
    y = pool["talento"].to_numpy()
    rng = np.random.default_rng(0)
    cuenta = 0
    for _ in range(n_sim):
        f1s = []
        for _ in range(n_lotes):
            idx = rng.choice(len(pool), size=tam, replace=False)
            f1s.append(f1_score(y[idx], p[idx]))
        cuenta += primera_alarma(f1s, base) is not None
    return cuenta / n_sim


def ejecutar():
    modelo, meta = cargar_modelo()
    umbral, base = meta["umbral"], meta["metricas"]["test_f1"]
    stream = generar_stream(lote_cambio=LOTE_CAMBIO)
    res = f1_por_lote(modelo, umbral, stream)
    ref = pd.read_csv(PROC / "train.csv")
    res["data_drift"] = [evaluar_lote(ref, g)[1]["estado"] for _, g in stream.groupby("lote")]
    alarma = primera_alarma(res["f1"].tolist(), base)
    disponible = None if alarma is None else alarma + RETRASO
    fa = falsas_alarmas(modelo, umbral, base)

    EVID.mkdir(parents=True, exist_ok=True)
    res.to_csv(EVID / "concept_drift_lotes.csv", index=False)
    resumen = dict(f1_base=round(base, 3), umbral_alarma=round(base - CAIDA, 3), k_lotes=K_LOTES,
                   lote_cambio_real=LOTE_CAMBIO, lote_alarma=alarma, lote_alarma_con_retraso_etiquetas=disponible,
                   retraso_etiquetas_lotes=RETRASO, falsas_alarmas_sin_deriva_pct=round(fa * 100, 1),
                   f1_antes=round(res.loc[res.lote < LOTE_CAMBIO, "f1"].mean(), 3),
                   f1_despues=round(res.loc[res.lote >= LOTE_CAMBIO, "f1"].mean(), 3),
                   data_drift_en_verde_todos_los_lotes=bool((res["data_drift"] == "VERDE").all()))
    (EVID / "concept_drift_resumen.json").write_text(json.dumps(resumen, indent=2))
    graficar(res, base, alarma, disponible)
    return res, resumen


def graficar(res, base, alarma, disponible):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(res["lote"], res["f1"], "o-", color="#1f77b4", label="F1 por lote (etiqueta real)")
    ax.plot(res["lote"], res["recall"], "s--", color="#7f7f7f", alpha=.7, label="Recall por lote")
    ax.axhline(base, color="green", ls=":", label=f"F1 base (test) = {base:.2f}")
    ax.axhline(base - CAIDA, color="red", ls="--", label=f"Umbral de alarma = {base - CAIDA:.2f} ({K_LOTES} lotes seguidos)")
    ax.axvspan(LOTE_CAMBIO - .5, res["lote"].max() + .5, color="orange", alpha=.12, label="Nuevo criterio de scouting")
    if alarma:
        ax.axvline(alarma, color="red", lw=2)
        ax.annotate(f"alarma (lote {alarma})", (alarma, 0.05), color="red", rotation=90, va="bottom", ha="right")
    if disponible and disponible <= res["lote"].max() + 2:
        ax.axvline(disponible, color="purple", lw=2, ls="-.")
        ax.annotate(f"alarma visible con retraso de etiquetas (lote {disponible})", (disponible, 0.05),
                    color="purple", rotation=90, va="bottom", ha="right")
    ax.set_xlabel("Lote de producción 2026 (400 jugadores por lote)")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1)
    ax.set_title("Concept drift: degradación del F1 con entradas sin data drift")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(EVID / "concept_drift.png", dpi=130)


if __name__ == "__main__":
    r, s = ejecutar()
    print(r.round(3).to_string(index=False))
    print(json.dumps(s, indent=2))
