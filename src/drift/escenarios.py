"""Escenarios de data drift inyectados deliberadamente sobre un lote."""
import numpy as np
import pandas as pd


def deriva_fuerte(lote: pd.DataFrame, seed=0) -> pd.DataFrame:
    """Nueva academia/campeonato: categorías mayores, sensor de velocidad distinto,
    salto con otro protocolo, casi nadie corre el Yo-Yo y casi todos figuran como delanteros."""
    rng = np.random.default_rng(seed)
    d = lote.copy()
    d["categoria_sub"] = (d["categoria_sub"] + 3).clip(upper=17)
    d["edad_anios"] = d["edad_anios"] + 3
    d["sprint_30m_s"] = d["sprint_30m_s"] * 1.12
    d["salto_cmj_cm"] = d["salto_cmj_cm"] + 6
    d.loc[rng.random(len(d)) < 0.60, "yoyo_m"] = np.nan
    d["posicion"] = np.where(rng.random(len(d)) < 0.7, "DELANTERO", d["posicion"])
    return d


def deriva_solo_faltantes(lote: pd.DataFrame, seed=0) -> pd.DataFrame:
    """Una academia deja de medir la batería física: sube la tasa de datos incompletos."""
    rng = np.random.default_rng(seed)
    d = lote.copy()
    sin_bateria = rng.random(len(d)) < 0.55
    d.loc[sin_bateria, ["sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s"]] = np.nan
    return d
