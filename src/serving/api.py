"""API de inferencia: probabilidad de que un jugador infanto-juvenil sea 'talento'.

El modelo se carga desde el artefacto exportado del alias `champion` del Model Registry
(models:/talento-sc@champion). Si MLFLOW_MODEL_URI está definida (p. ej. el servidor MLflow
es alcanzable desde el pod) se carga directamente de allí.
Cada respuesta incluye el nombre del POD que la atendió, la versión del registro y el run_id.
"""
import json
import os
import socket
from pathlib import Path
from typing import Literal, Optional

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.common.features import FEATURES

DIR_MODELO = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))
META = json.loads((DIR_MODELO / "metadata.json").read_text())
URI = os.environ.get("MLFLOW_MODEL_URI") or str(DIR_MODELO / "artefacto")
MODELO = mlflow.sklearn.load_model(URI)
POD = os.environ.get("POD_NAME") or socket.gethostname()

app = FastAPI(title="Detección de talento infanto-juvenil (Santa Cruz)", version="1.0")


class Jugador(BaseModel):
    """Los campos opcionales pueden venir vacíos: el pipeline los imputa con MICE."""
    categoria_sub: int = Field(ge=5, le=17, description="Categoría Sub-N")
    edad_anios: float = Field(gt=4, lt=19)
    genero: Literal["M", "F"]
    trimestre_nacimiento: int = Field(ge=1, le=4)
    posicion: Literal["ARQUERO", "DEFENSA", "MEDIOCAMPISTA", "DELANTERO", "SIN_DATO"] = "SIN_DATO"
    altura_cm: Optional[float] = None
    peso_kg: Optional[float] = None
    imc: Optional[float] = None
    sprint_30m_s: Optional[float] = None
    salto_cmj_cm: Optional[float] = None
    yoyo_m: Optional[float] = None
    agilidad_s: Optional[float] = None
    tecnica_score: Optional[float] = Field(default=None, ge=0, le=100)
    partidos: Optional[float] = None
    goles: Optional[float] = None
    asistencias: Optional[float] = None
    ppm_equipo: Optional[float] = None
    anios_en_academia: Optional[float] = None


class Lote(BaseModel):
    jugadores: list[Jugador] = Field(min_length=1, max_length=1000)


class Resultado(BaseModel):
    probabilidad_talento: float
    talento: int


class Respuesta(BaseModel):
    pod: str
    modelo: str
    version_registry: str
    run_id: str
    umbral: float
    resultados: list[Resultado]


@app.get("/health")
def health():
    return {"status": "ok", "pod": POD}


@app.get("/info")
def info():
    return {"pod": POD, **{k: META[k] for k in ("modelo", "alias", "version", "run_id", "run_name", "umbral",
                                                  "mlflow_version")}, "metricas": META["metricas"]}


@app.post("/predecir", response_model=Respuesta)
def predecir(lote: Lote):
    X = pd.DataFrame([j.model_dump() for j in lote.jugadores])[FEATURES]
    X = X.astype({c: float for c in X.columns if X[c].dtype != object and c not in
                  ("genero", "posicion", "trimestre_nacimiento")})
    proba = MODELO.predict_proba(X)[:, 1]
    u = META["umbral"]
    return Respuesta(pod=POD, modelo=META["modelo"], version_registry=str(META["version"]), run_id=META["run_id"],
                     umbral=u, resultados=[Resultado(probabilidad_talento=float(p), talento=int(p >= u)) for p in proba])
