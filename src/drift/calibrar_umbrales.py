"""Calibra empíricamente los umbrales: tasa de falsas alarmas con lotes del MISMO origen
y tasa de detección con deriva inyectada. Produce docs/evidencias/calibracion_data_drift.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.drift import detectores as D
from src.drift.escenarios import deriva_fuerte, deriva_solo_faltantes

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
SALIDA = Path(__file__).resolve().parents[2] / "docs" / "evidencias" / "calibracion_data_drift.json"


def cargar():
    ref = pd.read_csv(PROC / "train.csv")
    pool = pd.concat([pd.read_csv(PROC / "val.csv"), pd.read_csv(PROC / "test.csv")], ignore_index=True)
    return ref, pool


def calibrar(n_sim=200, tam=500):
    ref, pool = cargar()
    res = {"mismo_origen": [], "deriva_fuerte": [], "solo_faltantes": []}
    alertas_var = pd.Series(0.0, index=D.FEATURES)
    for i in range(n_sim):
        lote = pool.sample(n=tam, random_state=i)
        t, r = D.evaluar_lote(ref, lote)
        res["mismo_origen"].append(r["fraccion_en_alerta"])
        alertas_var += t.set_index("variable")["alerta"].astype(float)
        res["deriva_fuerte"].append(D.evaluar_lote(ref, deriva_fuerte(lote, i))[1]["fraccion_en_alerta"])
        res["solo_faltantes"].append(D.evaluar_lote(ref, deriva_solo_faltantes(lote, i))[1]["fraccion_en_alerta"])
    f = D.FRACCION_ROJO
    out = {
        "n_simulaciones": n_sim, "tam_lote": tam, "umbrales": dict(alfa=D.ALFA, d_min=D.D_MIN, psi=D.PSI_MAX,
                                                                       delta_nulos=D.DELTA_NULOS, fraccion_rojo=f),
        "falsa_alarma_variable_individual_pct": (alertas_var / n_sim * 100).round(2).to_dict(),
        "mismo_origen": {"fraccion_media": round(float(np.mean(res["mismo_origen"])), 3),
                         "fraccion_max": round(float(np.max(res["mismo_origen"])), 3),
                         "lotes_rojos_pct": round(float(np.mean(np.array(res["mismo_origen"]) >= f) * 100), 2)},
        "deriva_fuerte": {"fraccion_media": round(float(np.mean(res["deriva_fuerte"])), 3),
                          "lotes_rojos_pct": round(float(np.mean(np.array(res["deriva_fuerte"]) >= f) * 100), 2)},
        "solo_faltantes": {"fraccion_media": round(float(np.mean(res["solo_faltantes"])), 3),
                           "lotes_rojos_pct": round(float(np.mean(np.array(res["solo_faltantes"]) >= f) * 100), 2)},
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(calibrar(), indent=2))
