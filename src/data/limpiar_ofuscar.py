"""Limpieza, deduplicación y ofuscación: data/raw -> data/processed/jugador_temporada.csv

1. Detecta perfiles duplicados (misma fecha de nacimiento + nombres casi iguales).
2. Consolida a un registro por (jugador, temporada): suma partidos/goles/asistencias
   (niño en dos equipos), promedia mediciones, conserva los NaN cuando nadie midió.
3. Ofusca: seudónimo HMAC, sin nombre ni fecha de nacimiento.
4. Reporta calidad de datos (faltantes y precisión de la deduplicación).
"""
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ofuscar import seudonimo

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
OUT = RAIZ / "data" / "processed"
VACIAS = {"DE", "DEL", "LA", "LOS"}


def _tokens(nombre: str) -> frozenset:
    s = unicodedata.normalize("NFKD", nombre.upper()).encode("ascii", "ignore").decode()
    return frozenset(t for t in s.split() if t not in VACIAS)


def deduplicar(perfiles: pd.DataFrame, umbral=0.6):
    """Devuelve dict profile_id -> id canónico. Jaccard de tokens dentro de misma fecha de nacimiento."""
    padre = {p: p for p in perfiles["profile_id"]}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for _, g in perfiles.groupby(["fecha_nacimiento", "genero"]):
        filas = list(g.itertuples())
        tk = {f.profile_id: _tokens(f.nombre_completo) for f in filas}
        for i in range(len(filas)):
            for j in range(i + 1, len(filas)):
                a, b = tk[filas[i].profile_id], tk[filas[j].profile_id]
                if len(a & b) / len(a | b) >= umbral:
                    padre[raiz(filas[j].profile_id)] = raiz(filas[i].profile_id)
    largo = perfiles.set_index("profile_id")["nombre_completo"].str.len()
    grupos = {}
    for p in padre:
        grupos.setdefault(raiz(p), []).append(p)
    mapa = {}
    for miembros in grupos.values():
        canon = max(miembros, key=lambda m: (largo[m], m))  # el nombre más completo
        for m in miembros:
            mapa[m] = canon
    return mapa


def procesar():
    perf = pd.read_csv(RAW / "perfiles_raw.csv")
    med = pd.read_csv(RAW / "mediciones_raw.csv")
    part = pd.read_csv(RAW / "participaciones_raw.csv")
    ev = pd.read_csv(RAW / "eventos_raw.csv")
    etq = pd.read_csv(RAW / "etiquetas_raw.csv")
    tr = pd.read_csv(RAW / "transferencias_raw.csv")

    mapa = deduplicar(perf)
    for t in (med, part, ev, etq, tr):
        t["profile_id"] = t["profile_id"].map(mapa)
    perf["canon"] = perf["profile_id"].map(mapa)
    base = perf[perf["profile_id"] == perf["canon"]].set_index("canon")

    # ---- consolidación (jugador, temporada)
    k = ["profile_id", "temporada"]
    mnum = ["altura_cm", "peso_kg", "sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s", "tecnica_score"]
    m = med.groupby(k).agg({**{c: "mean" for c in mnum},
                            "posicion": lambda s: s.dropna().replace("", np.nan).dropna().iloc[0]
                            if s.replace("", np.nan).notna().any() else "SIN_DATO"})
    p = part.groupby(k).agg(partidos=("partidos", lambda s: s.sum(min_count=1)),
                            ppm_equipo=("ppm_equipo", "mean"), n_clubes_temporada=("club_id", "nunique"))
    e = ev.groupby(k + ["tipo"]).size().unstack(fill_value=0).rename(columns={"Goal": "goles", "Assist": "asistencias"})
    df = p.join(m, how="left").join(e, how="left").reset_index()
    for c in ("goles", "asistencias"):
        df[c] = df[c].fillna(0).astype(float)
        df.loc[df["partidos"].isna(), c] = np.nan  # no registrado != cero goles

    # ---- datos del perfil canónico (sin PII en la salida)
    df["fecha_nacimiento"] = df["profile_id"].map(base["fecha_nacimiento"])
    df["genero"] = df["profile_id"].map(base["genero"])
    f = pd.to_datetime(df["fecha_nacimiento"])
    df["categoria_sub"] = df["temporada"] - f.dt.year
    df["trimestre_nacimiento"] = (f.dt.month - 1) // 3 + 1
    ref = pd.to_datetime(df["temporada"].astype(str) + "-07-01")
    df["edad_anios"] = ((ref - f).dt.days / 365.25).round(2)
    ini = pd.to_datetime(tr.groupby("profile_id")["fecha_inicio"].min())
    df["anios_en_academia"] = (df["temporada"] - df["profile_id"].map(ini).dt.year).clip(lower=0)
    df["imc"] = df["peso_kg"] / (df["altura_cm"] / 100) ** 2
    df["posicion"] = df["posicion"].fillna("SIN_DATO")
    df = df.merge(etq.groupby(k)[["talento", "talento_regimen_b"]].max().reset_index(), on=k, how="left")
    df = df.rename(columns={"talento_regimen_b": "talento_b"})

    # ---- ofuscación
    df["jugador_id"] = df["profile_id"].map(seudonimo)
    cols = ["jugador_id", "temporada", "categoria_sub", "genero", "trimestre_nacimiento", "edad_anios", "posicion",
            "altura_cm", "peso_kg", "imc", "sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s",
            "tecnica_score", "partidos", "goles", "asistencias", "ppm_equipo", "anios_en_academia",
            "n_clubes_temporada", "talento", "talento_b"]
    sin_etiqueta = int(df["talento"].isna().sum())  # filas de perfiles duplicados no detectados
    out = df[cols].dropna(subset=["talento"]).astype({"talento": int, "talento_b": int})
    out = out.sort_values(["jugador_id", "temporada"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "jugador_temporada.csv", index=False)

    # ---- reporte de calidad
    verdad = pd.read_csv(RAW / "verdad_duplicados.csv")
    real = set(zip(verdad.profile_dup, verdad.profile_canonico))
    hallados = {(a, b) for a, b in mapa.items() if a != b}
    real_par = {frozenset(x) for x in real}
    hall_par = {frozenset(x) for x in hallados}
    tp = len(real_par & hall_par)
    rep = {
        "perfiles_raw": len(perf), "perfiles_unicos": len(set(mapa.values())),
        "duplicados_reales": len(real_par), "duplicados_detectados": len(hall_par),
        "precision_dedup": round(tp / max(len(hall_par), 1), 3), "recall_dedup": round(tp / max(len(real_par), 1), 3),
        "filas_jugador_temporada": len(out), "filas_descartadas_sin_etiqueta": sin_etiqueta,
        "jugadores_con_2_equipos_misma_temporada": int((out["n_clubes_temporada"] > 1).sum()),
        "pct_faltantes": (out.drop(columns=["jugador_id"]).isna().mean() * 100).round(1).to_dict(),
        "tasa_talento": round(float(out["talento"].mean()), 3),
        "pct_faltantes_por_temporada_yoyo": (out.groupby("temporada")["yoyo_m"].apply(lambda s: s.isna().mean() * 100)).round(1).to_dict(),
    }
    (OUT / "reporte_calidad.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    return rep


if __name__ == "__main__":
    print(json.dumps(procesar(), indent=2, ensure_ascii=False))
