"""Pruebas de CONCEPT DRIFT: el criterio de reentrenamiento dispara tras el cambio de regla y no antes."""
from pathlib import Path

import pytest

from src.drift import concepto as C

MODELO = C.MODELO_DIR / "metadata.json"


def test_criterio_no_dispara_con_una_caida_aislada():
    assert C.primera_alarma([0.62, 0.40, 0.62, 0.60, 0.41, 0.62], base=0.62) is None


def test_criterio_dispara_con_caida_sostenida():
    assert C.primera_alarma([0.62, 0.60, 0.40, 0.39, 0.41, 0.40], base=0.62) == 5


@pytest.mark.skipif(not MODELO.exists(), reason="falta exportar el modelo (src.serving.exportar_modelo)")
def test_escenario_completo_alarma_despues_del_cambio():
    res, s = C.ejecutar()
    assert s["lote_alarma"] is not None and s["lote_alarma"] >= C.LOTE_CAMBIO
    assert s["f1_despues"] < s["f1_base"] - C.CAIDA
    assert s["falsas_alarmas_sin_deriva_pct"] <= 5
    assert s["data_drift_en_verde_todos_los_lotes"], "los inputs no deben mostrar data drift: es concept drift puro"
