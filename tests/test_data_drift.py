"""Pruebas automatizadas de DATA DRIFT: VERDE con datos del mismo origen, ROJO con deriva inyectada."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.drift import detectores as D
from src.drift.escenarios import deriva_fuerte, deriva_solo_faltantes

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(PROC / "train.csv")


@pytest.fixture(scope="module")
def pool():
    return pd.concat([pd.read_csv(PROC / "val.csv"), pd.read_csv(PROC / "test.csv")], ignore_index=True)


def test_psi_identicas_es_cero():
    s = pd.Series(["A", "B", "A", "C"] * 50)
    assert D.psi(s, s) == pytest.approx(0, abs=1e-9)


def test_psi_detecta_cambio_de_categorias():
    a = pd.Series(["M"] * 500 + ["F"] * 500)
    b = pd.Series(["M"] * 900 + ["F"] * 100)
    assert D.psi(a, b) > D.PSI_MAX


def test_ks_detecta_desplazamiento():
    rng = np.random.default_rng(0)
    d, p = D.ks(pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(0.5, 1, 500)))
    assert d >= D.D_MIN and p < 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_lote_mismo_origen_es_verde(ref, pool, seed):
    lote = pool.sample(n=500, random_state=seed)
    _, r = D.evaluar_lote(ref, lote)
    assert r["estado"] == "VERDE", r


@pytest.mark.parametrize("seed", range(10))
def test_lote_con_deriva_fuerte_es_rojo(ref, pool, seed):
    lote = deriva_fuerte(pool.sample(n=500, random_state=seed), seed)
    _, r = D.evaluar_lote(ref, lote)
    assert r["estado"] == "ROJO", r
    assert "sprint_30m_s" in r["variables_en_alerta"]


@pytest.mark.parametrize("seed", range(10))
def test_aumento_de_faltantes_es_rojo(ref, pool, seed):
    lote = deriva_solo_faltantes(pool.sample(n=500, random_state=seed), seed)
    t, r = D.evaluar_lote(ref, lote)
    assert r["estado"] == "ROJO", r
    assert t.set_index("variable").loc["yoyo_m", "alerta_nulos"]
