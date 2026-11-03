"""Detección de data drift: lote nuevo vs. baseline (conjunto de entrenamiento).

Pruebas por tipo de variable (y por qué):
  * NUMÉRICAS -> Kolmogorov-Smirnov de dos muestras. No paramétrica (no asume normalidad;
    nuestras variables son sesgadas y acotadas), sensible a cambios de forma, ubicación y escala.
  * CATEGÓRICAS -> PSI (Population Stability Index): sum((p_nuevo - p_ref) * ln(p_nuevo / p_ref)).
    Estándar de la industria para monitoreo de modelos; interpretable con umbrales conocidos.
  * FALTANTES (específico de esta tesis) -> prueba z de dos proporciones sobre la tasa de nulos de
    cada variable: un cambio en el patrón de datos incompletos altera lo que MICE tiene que imputar.

UMBRALES (ver docs/DRIFT.md, se calibran con simulación de lotes del mismo origen):
  * KS: variable en alerta si p < ALFA/k (Bonferroni, k = nº de variables) Y D >= D_MIN.
        D_MIN=0.10 evita alertar por diferencias estadísticamente significativas pero irrelevantes.
  * PSI: alerta si PSI >= 0.25 (convención: <0.10 estable, 0.10-0.25 moderado, >=0.25 cambio significativo).
  * Nulos: alerta si la tasa cambia >= 0.10 en valor absoluto y p < ALFA/k.
  * LOTE EN ROJO si la fracción de variables en alerta es >= FRACCION_ROJO (0.20).
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.common.features import CATEGORICAS, FEATURES, NUMERICAS

ALFA = 0.01
D_MIN = 0.10
PSI_MAX = 0.25
DELTA_NULOS = 0.10
FRACCION_ROJO = 0.20
EPS = 1e-4


def psi(ref: pd.Series, nuevo: pd.Series) -> float:
    cats = sorted(set(ref.astype(str)) | set(nuevo.astype(str)))
    pr = ref.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=EPS)
    pn = nuevo.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=EPS)
    return float(((pn - pr) * np.log(pn / pr)).sum())


def ks(ref: pd.Series, nuevo: pd.Series):
    a, b = ref.dropna(), nuevo.dropna()
    if len(a) < 20 or len(b) < 20:
        return np.nan, 1.0
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def _z_nulos(ref: pd.Series, nuevo: pd.Series):
    p1, p2, n1, n2 = ref.isna().mean(), nuevo.isna().mean(), len(ref), len(nuevo)
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    z = (p2 - p1) / se
    return float(p2 - p1), float(2 * (1 - stats.norm.cdf(abs(z))))


def evaluar_lote(ref: pd.DataFrame, lote: pd.DataFrame, features=FEATURES):
    """Devuelve (tabla por variable, resumen con estado VERDE/ROJO)."""
    k = len(features)
    filas = []
    for f in features:
        d_null, p_null = _z_nulos(ref[f], lote[f])
        alerta_null = abs(d_null) >= DELTA_NULOS and p_null < ALFA / k
        if f in NUMERICAS:
            d, p = ks(ref[f], lote[f])
            alerta = (p < ALFA / k) and (d >= D_MIN)
            filas.append(dict(variable=f, prueba="KS", estadistico=d, p_valor=p, d_nulos=d_null,
                              alerta_distribucion=alerta, alerta_nulos=alerta_null))
        else:
            v = psi(ref[f], lote[f])
            filas.append(dict(variable=f, prueba="PSI", estadistico=v, p_valor=np.nan, d_nulos=d_null,
                              alerta_distribucion=v >= PSI_MAX, alerta_nulos=alerta_null))
    t = pd.DataFrame(filas)
    t["alerta"] = t["alerta_distribucion"] | t["alerta_nulos"]
    frac = float(t["alerta"].mean())
    resumen = dict(estado="ROJO" if frac >= FRACCION_ROJO else "VERDE", fraccion_en_alerta=round(frac, 3),
                   variables_en_alerta=t.loc[t["alerta"], "variable"].tolist(), n_lote=len(lote))
    return t, resumen
