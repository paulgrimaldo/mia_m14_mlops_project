"""Generador de datos SINTÉTICOS que imita la estructura de Copa Tahuichi.

No hay datos reales de menores: se simula una población de jugadores
infanto-juveniles (Sub-05 a Sub-17) de academias de Santa Cruz, 2021-2025
(+ 2026 como flujo de "producción" para las pruebas de drift).

Se reproducen los problemas de calidad descritos en la tesis:
  * perfiles duplicados (mismo niño, dos profileId, nombre con variantes),
  * niños inscritos en dos equipos en la misma temporada,
  * temporadas en que el niño no figura en la copa,
  * valores faltantes (MAR y MNAR) que dependen del nivel de digitalización
    del club y que disminuyen con los años (digitalización post-pandemia).

Salida (data/raw/, ignorada por git): tablas con la misma forma que los JSON de
la API (perfiles, transferencias, eventos, campeonatos) + mediciones físicas.
"""
import argparse
import json
import unicodedata
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
TEMPORADAS = list(range(2021, 2026))
TEMPORADA_STREAM = 2026
CATEGORIAS = list(range(5, 18))
CATEGORIAS_DAMAS = {10, 12, 14, 16}  # supuesto: damas compiten en estas categorías
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

CLUBES = [
    "TAHUICHI", "FLORIDA", "ARGENTINA", "BRASIL", "ACADEMIA CRUZ DEL SUR",
    "ESCUELA DE FUTBOL SERRANITO EL TORNO", "TILUCHI", "REAL CAMBA - B",
    "ACADEMIA LOS PINOS", "ESCUELA EQUIPETROL SUR", "CLUB PIRAI FC",
    "ACADEMIA PLAN 3000", "ESCUELA LA GUARDIA", "CLUB VILLA 1RO DE MAYO",
    "ACADEMIA URBARI", "ESCUELA DE FUTBOL COTOCA", "CLUB MONTERO JUNIOR",
    "ACADEMIA NUEVA ESPERANZA", "ESCUELA WARNES FC", "CLUB EL FORTIN",
    "ACADEMIA RIO GRANDE", "ESCUELA SANTA ROSITA", "CLUB LOS CHACOS",
    "ACADEMIA PARAGUA", "ESCUELA CIUDAD SATELITE", "CLUB RADIAL 17",
    "ACADEMIA LA PAMPA", "ESCUELA PORONGO FC",
]
NOMBRES = ["JUAN", "MIGUEL", "LUCAS", "DYLAN", "NICOLAS", "KYLIAN", "KALEB", "JOSE",
           "ANGEL", "ABDIEL", "EITAM", "HEINRICH", "KENT", "THIAGO", "BAHIR", "MATEO",
           "SANTIAGO", "DIEGO", "ALEJANDRO", "GABRIEL", "DANIEL", "SEBASTIAN", "ADRIAN",
           "EMILIANO", "BRAYAN", "JHOEL", "RODRIGO", "ANDRES", "FABIAN", "ISAAC",
           "VALENTINA", "SOFIA", "CAMILA", "LUCIANA", "ISABELLA", "MARIANA", "ARIANA",
           "DANIELA", "JIMENA", "ABIGAIL"]
APELLIDOS = ["MEDINA", "ARIAS", "CHOQUE", "AGUILAR", "CASTRO", "LEON", "JANCO", "BOZO",
             "TERRAZAS", "LIMON", "MAMANI", "CHAVEZ", "ROJAS", "JALDIN", "MOLINA",
             "CORREA", "ALVAREZ", "TERCEROS", "LOPEZ", "ORELLANA", "ARTEAGA", "SUAREZ",
             "VACA", "ROCA", "PEREZ", "FLORES", "GUTIERREZ", "SANCHEZ", "CUELLAR",
             "SEGOVIA", "PEÑA", "VARGAS", "RIOS", "SALVATIERRA", "ANTELO", "HURTADO"]
POSICIONES = ["ARQUERO", "DEFENSA", "MEDIOCAMPISTA", "DELANTERO"]
P_POSICION = [0.08, 0.30, 0.35, 0.27]
NIVEL_P = {"alto": 0.05, "medio": 0.15, "bajo": 0.38}
FACTOR_ANIO = {2021: 1.5, 2022: 1.3, 2023: 1.1, 2024: 1.0, 2025: 0.85, 2026: 1.1}  # 2026 ~ promedio del periodo de entrenamiento (aísla el concept drift)
PESOS = {  # peso de cada latente en la regla que define "talento"
    "A": dict(tec=0.85, fis=0.55, mad=0.30, rel=0.20, con=0.35),  # scouting tradicional (sesgo físico)
    "B": dict(tec=1.00, fis=-0.60, mad=-0.45, rel=-0.20, con=0.35),  # nuevo criterio: corrige sesgo de maduración (concept drift)
}
TASA_TALENTO = 0.15


def _curva(edad, xs, ys):
    return np.interp(edad, xs, ys)


def _nombre(rng, damas):
    n = rng.choice(NOMBRES[30:] if damas else NOMBRES[:30], size=rng.integers(1, 3), replace=False)
    a = rng.choice(APELLIDOS, size=2, replace=False)
    return " ".join(list(a) + list(n))  # formato "APELLIDOS NOMBRES"


def crear_clubes(rng):
    niveles = rng.choice(["alto", "medio", "bajo"], size=len(CLUBES), p=[0.25, 0.40, 0.35])
    return pd.DataFrame({
        "club_id": [f"club-{i:03d}" for i in range(len(CLUBES))],
        "nombre": CLUBES,
        "nivel_digital": niveles,
        "calidad": rng.normal(0, 1, len(CLUBES)),
    })


def simular_filas(rng, temporadas, clubes, n_varones=190, n_damas=55):
    """Una fila por (jugador, temporada) con rasgos COMPLETOS y latentes."""
    filas, jugadores = [], []
    for b in range(min(temporadas) - 17, max(temporadas) - 5 + 1):
        for genero, n in (("M", n_varones), ("F", n_damas)):
            for _ in range(n):
                ed_ing = rng.choice(np.arange(5, 15), p=_p_ingreso())
                inicio = b + ed_ing
                if inicio > max(temporadas):
                    continue
                pid = str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))))
                lat_t = rng.normal()
                lat_f = 0.2 * lat_t + np.sqrt(1 - 0.04) * rng.normal()
                p = dict(
                    pid=pid, genero=genero, b=b, mes=int(rng.integers(1, 13)), dia=int(rng.integers(1, 29)),
                    lat_t=lat_t, lat_f=lat_f, mad=rng.normal(0, 0.7),
                    off_h=rng.normal(0, 3.5), off_bmi=rng.normal(0, 1.0),
                    posicion=rng.choice(POSICIONES, p=P_POSICION), anio_ingreso=int(inicio),
                    club=int(rng.integers(0, len(clubes))),
                    nombre=_nombre(rng, genero == "F"),
                    sin_transf=bool(rng.random() < 0.10),
                )
                jugadores.append(p)
                club = p["club"]
                for s in range(max(inicio, min(temporadas)), min(max(temporadas), b + 17) + 1):
                    edad = s - b
                    if genero == "F" and edad not in CATEGORIAS_DAMAS:
                        continue                      # damas: solo compiten en ciertas categorías
                    if rng.random() < _p_abandono(edad):
                        break
                    if rng.random() < 0.08:
                        continue                      # temporada en que no figura en la copa
                    if rng.random() < 0.12:
                        club = int(rng.integers(0, len(clubes)))
                    filas.append(_fila(rng, p, s, club, clubes))
    df = pd.DataFrame(filas)
    return df, pd.DataFrame(jugadores)


def _p_ingreso():
    w = np.exp(-0.28 * np.arange(0, 10))
    return w / w.sum()


def _p_abandono(edad):
    return 0.10 if edad < 12 else 0.18 if edad < 15 else 0.28


def _fila(rng, p, s, club, clubes):
    cl = clubes.iloc[club]
    edad = (s + 0.5) - (p["b"] + (p["mes"] - 0.5) / 12)
    am = edad + p["mad"]  # edad biológica
    alt = _curva(am, [5, 7, 9, 11, 13, 15, 17], [109, 121, 133, 144, 156, 170, 175]) + p["off_h"] + rng.normal(0, 2.5)
    bmi = _curva(am, [5, 17], [15.3, 20.5]) + p["off_bmi"] + rng.normal(0, 0.8)
    peso = bmi * (alt / 100) ** 2
    sprint = _curva(am, [5, 17], [6.3, 4.45]) - 0.12 * p["lat_f"] + rng.normal(0, 0.12)
    cmj = _curva(am, [5, 17], [17, 40]) + 2.2 * p["lat_f"] + rng.normal(0, 2.5)
    yoyo = max(120, _curva(am, [5, 17], [350, 2000]) + 180 * p["lat_f"] + rng.normal(0, 150))
    agil = _curva(am, [5, 17], [14.2, 10.2]) - 0.25 * p["lat_f"] + rng.normal(0, 0.35)
    anios = max(0, s - p["anio_ingreso"])
    tec = float(np.clip(50 + 11 * p["lat_t"] + 1.2 * anios + rng.normal(0, 7), 0, 100))
    fuerza = rng.normal() * 0.7 + 0.6 * cl["calidad"]
    ppm = float(np.clip(1.45 + 0.35 * fuerza, 0, 3))
    partidos = int(np.clip(4 + rng.poisson(5), 3, 12))
    base = {"ARQUERO": 0.0, "DEFENSA": 0.07, "MEDIOCAMPISTA": 0.22, "DELANTERO": 0.55}[p["posicion"]]
    esc = 2.0 if edad < 9 else 1.0
    goles = int(rng.poisson(partidos * base * esc * np.exp(0.45 * p["lat_t"] + 0.2 * p["lat_f"] + 0.15 * fuerza)))
    ba = {"ARQUERO": 0.01, "DEFENSA": 0.08, "MEDIOCAMPISTA": 0.28, "DELANTERO": 0.20}[p["posicion"]]
    asis = int(rng.poisson(partidos * ba * esc * np.exp(0.4 * p["lat_t"])))
    club2 = None
    if rng.random() < 0.06:  # dos equipos a la vez
        club2 = int(rng.choice([c for c in range(len(clubes)) if c != club]))
    return dict(
        pid=p["pid"], temporada=s, genero=p["genero"], categoria_sub=s - p["b"], mes=p["mes"],
        trimestre_nacimiento=(p["mes"] - 1) // 3 + 1, edad_anios=round(edad, 2),
        posicion=p["posicion"], altura_cm=alt, peso_kg=peso, sprint_30m_s=sprint,
        salto_cmj_cm=cmj, yoyo_m=yoyo, agilidad_s=agil, tecnica_score=tec,
        partidos=partidos, goles=goles, asistencias=asis, ppm_equipo=ppm,
        anios_en_academia=anios, club=club, club2=club2, nivel=cl["nivel_digital"],
        lat_t=p["lat_t"], lat_f=p["lat_f"], mad_n=p["mad"] / 0.7, rel=(6.5 - p["mes"]) / 6,
    )


# ---------------------------------------------------------------- etiquetas
def _contribucion(df):
    return np.log1p((df["goles"] + df["asistencias"]) / df["partidos"].clip(lower=1))


def calibrar_etiquetas(df):
    """Media/desv. de la contribución por edad y umbrales por régimen (solo con la población de entrenamiento)."""
    c = _contribucion(df)
    g = c.groupby(df["categoria_sub"].astype(int))
    stats = {"mu": g.mean().to_dict(), "sd": g.std().fillna(1).to_dict()}
    scores = {r: _score(df, stats, r, np.random.default_rng(7)) for r in PESOS}
    umbrales = {r: float(np.quantile(s, 1 - TASA_TALENTO)) for r, s in scores.items()}
    return {"stats": stats, "umbrales": umbrales}


def _score(df, stats, regimen, rng):
    w = PESOS[regimen]
    mu = df["categoria_sub"].astype(int).map(stats["mu"])
    sd = df["categoria_sub"].astype(int).map(stats["sd"])
    z = (_contribucion(df) - mu) / sd
    return (w["tec"] * df["lat_t"] + w["fis"] * df["lat_f"] + w["mad"] * df["mad_n"]
            + w["rel"] * df["rel"] + w["con"] * z + rng.normal(0, 0.5, len(df)))


def etiquetar(df, calib, regimen, rng):
    s = _score(df, calib["stats"], regimen, rng)
    return (s > calib["umbrales"][regimen]).astype(int)


# ---------------------------------------------------------------- faltantes
TESTS = ["sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s"]
MULT = {"altura_cm": 0.6, "peso_kg": 0.6, "sprint_30m_s": 1.0, "salto_cmj_cm": 1.1, "yoyo_m": 1.4,
        "agilidad_s": 1.0, "tecnica_score": 0.8, "posicion": 1.2, "ppm_equipo": 0.3, "anios_en_academia": 0.5}


def mascara_faltantes(df, rng):
    """True = valor faltante. Depende del club (nivel digital), del año y, en parte, del valor mismo (MNAR)."""
    n = len(df)
    p = df["nivel"].map(NIVEL_P).to_numpy() * df["temporada"].map(FACTOR_ANIO).to_numpy()
    lento = (df["lat_f"] < -0.5).to_numpy()
    bateria = rng.random(n) < p * 0.6  # ese día no hubo jornada de pruebas físicas
    m = {}
    for v, mult in MULT.items():
        pv = p * mult * 0.7
        if v == "sprint_30m_s":
            pv = pv * (1 + 0.8 * lento)
        if v == "yoyo_m":
            pv = pv * (1 + 0.6 * lento)
        m[v] = rng.random(n) < np.clip(pv, 0, 0.9)
        if v in TESTS:
            m[v] = m[v] | bateria
    ev = rng.random(n) < p * 0.5  # el club no registró estadísticas de partidos
    for v in ("partidos", "goles", "asistencias"):
        m[v] = ev
    return pd.DataFrame(m, index=df.index)


def a_dataset_directo(df, calib, rng, regimen="A"):
    """Formato 'procesado' (post-limpieza) con faltantes; se usa para el flujo 2026."""
    m = mascara_faltantes(df, rng)
    out = df.drop(columns=["pid", "mes", "club", "club2", "nivel", "lat_t", "lat_f", "mad_n", "rel"]).copy()
    for c in m.columns:
        out[c] = out[c].astype(float) if c != "posicion" else out[c]
        out.loc[m[c].to_numpy(), c] = np.nan if c != "posicion" else "SIN_DATO"
    out["posicion"] = out["posicion"].fillna("SIN_DATO")
    out["imc"] = out["peso_kg"] / (out["altura_cm"] / 100) ** 2
    out["talento"] = etiquetar(df, calib, regimen, rng).to_numpy()
    out["talento_b"] = etiquetar(df, calib, "B", np.random.default_rng(rng.integers(10**9))).to_numpy()
    return out


# ---------------------------------------------------------------- tablas raw
def _variante_nombre(rng, nombre):
    t = nombre.split()
    k = rng.integers(0, 3)
    if k == 0 and len(t) >= 4:
        return " ".join(t[:2] + t[2:3])                     # falta el segundo nombre
    if k == 1:
        i = int(rng.integers(0, len(t)))
        s = t[i]
        t[i] = s[:-1] if len(s) > 4 else s + s[-1]            # error de tipeo
        return " ".join(t)
    return " ".join(t[2:] + t[:2]) if len(t) >= 4 else " ".join(reversed(t))  # orden invertido


def generar_raw(seed=SEMILLA):
    rng = np.random.default_rng(seed)
    clubes = crear_clubes(rng)
    df, jug = simular_filas(rng, TEMPORADAS, clubes)
    calib = calibrar_etiquetas(df)
    df["talento"] = etiquetar(df, calib, "A", rng).to_numpy()
    df["talento_b"] = etiquetar(df, calib, "B", rng).to_numpy()
    mask = mascara_faltantes(df, rng)
    cid = clubes["club_id"].to_numpy()

    # --- perfiles (con duplicados)
    perfiles = [dict(profile_id=r.pid, nombre_completo=r.nombre,
                     fecha_nacimiento=f"{r.b}-{r.mes:02d}-{r.dia:02d}", genero=r.genero) for r in jug.itertuples()]
    dup = {}
    for r in jug.sample(frac=0.05, random_state=seed).itertuples():
        nid = str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))))
        dup[r.pid] = nid
        perfiles.append(dict(profile_id=nid, nombre_completo=_variante_nombre(rng, r.nombre),
                             fecha_nacimiento=f"{r.b}-{r.mes:02d}-{r.dia:02d}", genero=r.genero))
    pid_fila = lambda pid: dup[pid] if pid in dup and rng.random() < 0.5 else pid

    med, part, ev, etq, camp = [], [], [], [], {}
    for i, r in enumerate(df.itertuples(index=False)):
        pf = pid_fila(r.pid)
        clubs = [int(r.club)] + ([int(r.club2)] if r.club2 == r.club2 and r.club2 is not None else [])
        n_cl = len(clubs)
        gen = "Damas" if r.genero == "F" else "Varones"
        camp_n = f"Copa Tahuichi {'Invierno ' if r.temporada % 2 else ''}Sub-{int(r.categoria_sub):02d} {gen}"
        camp[(camp_n, r.temporada)] = 1
        m = mask.iloc[i]
        vals = {k: (np.nan if m.get(k, False) else getattr(r, k)) for k in
                ["altura_cm", "peso_kg", "sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s", "tecnica_score"]}
        pos = "" if m["posicion"] else r.posicion
        part_split = _repartir(rng, r.partidos, n_cl)
        gol_split = _repartir(rng, r.goles, n_cl)
        asi_split = _repartir(rng, r.asistencias, n_cl)
        for j, c in enumerate(clubs):
            ruido = 1.0 if j == 0 else 1 + rng.normal(0, 0.01)  # segunda medición en el otro club
            med.append(dict(profile_id=pf, temporada=r.temporada, club_id=cid[c], posicion=pos,
                            **{k: (v * ruido if v == v else np.nan) for k, v in vals.items()}))
            part.append(dict(profile_id=pf, temporada=r.temporada, club_id=cid[c], campeonato=camp_n,
                             partidos=(np.nan if m["partidos"] else part_split[j]),
                             ppm_equipo=(np.nan if m["ppm_equipo"] else r.ppm_equipo)))
            if not m["partidos"]:
                for tipo, k in (("Goal", gol_split[j]), ("Assist", asi_split[j])):
                    for _ in range(k):
                        ev.append(dict(event_id=str(uuid.uuid4()), profile_id=pf, temporada=r.temporada,
                                       campeonato=camp_n, club_id=cid[c], tipo=tipo, minuto=int(rng.integers(1, 46))))
        etq.append(dict(profile_id=r.pid, temporada=r.temporada, talento=r.talento, talento_regimen_b=r.talento_b))

    transf = []
    for p in jug.itertuples():
        if p.sin_transf:
            continue
        pf = dup.get(p.pid, p.pid) if rng.random() < 0.3 else p.pid
        transf.append(dict(profile_id=pf, club_id=cid[p.club], fecha_inicio=f"{p.anio_ingreso}-01-01"))

    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perfiles).to_csv(RAW / "perfiles_raw.csv", index=False)
    pd.DataFrame(med).to_csv(RAW / "mediciones_raw.csv", index=False)
    pd.DataFrame(part).to_csv(RAW / "participaciones_raw.csv", index=False)
    pd.DataFrame(ev).to_csv(RAW / "eventos_raw.csv", index=False)
    pd.DataFrame(etq).to_csv(RAW / "etiquetas_raw.csv", index=False)
    pd.DataFrame(transf).to_csv(RAW / "transferencias_raw.csv", index=False)
    clubes.drop(columns=["calidad"]).to_csv(RAW / "clubes_raw.csv", index=False)
    pd.DataFrame([dict(campeonato=k[0], temporada=k[1]) for k in camp]).to_csv(RAW / "campeonatos.csv", index=False)
    pd.DataFrame(dict(profile_dup=list(dup.values()), profile_canonico=list(dup.keys()))) \
        .to_csv(RAW / "verdad_duplicados.csv", index=False)
    (RAW / "calibracion_etiquetas.json").write_text(json.dumps(
        {"umbrales": calib["umbrales"],
         "mu": {str(k): v for k, v in calib["stats"]["mu"].items()},
         "sd": {str(k): v for k, v in calib["stats"]["sd"].items()}}))
    return dict(filas=len(df), jugadores=len(jug), duplicados=len(dup), eventos=len(ev))


def _repartir(rng, total, k):
    if k == 1:
        return [int(total)]
    a = int(rng.integers(0, int(total) + 1))
    return [a, int(total) - a]


def cargar_calibracion():
    d = json.loads((RAW / "calibracion_etiquetas.json").read_text())
    return {"umbrales": d["umbrales"],
            "stats": {"mu": {int(k): v for k, v in d["mu"].items()}, "sd": {int(k): v for k, v in d["sd"].items()}}}


def generar_stream(seed=2026, n_lotes=12, tam_lote=400, lote_cambio=7):
    """Flujo 'producción 2026' en lotes. A partir de `lote_cambio` el scouting cambia de criterio (concept drift)."""
    rng = np.random.default_rng(seed)
    clubes = crear_clubes(np.random.default_rng(SEMILLA))
    calib = cargar_calibracion()
    pool = []
    while sum(len(x) for x in pool) < 4 * n_lotes * tam_lote:
        df, _ = simular_filas(rng, [TEMPORADA_STREAM], clubes, n_varones=300, n_damas=80)
        pool.append(df)
    df = pd.concat(pool, ignore_index=True)
    # La copa tiene una mezcla estable de categorías/género: los lotes la respetan (muestreo ponderado)
    ref = pd.read_csv(RAW.parent / "processed" / "jugador_temporada.csv")
    comp_ref = ref.groupby(["categoria_sub", "genero"]).size() / len(ref)
    comp_df = df.groupby(["categoria_sub", "genero"]).size() / len(df)
    w = pd.Series(list(zip(df["categoria_sub"], df["genero"]))).map((comp_ref / comp_df).to_dict()).fillna(0)
    df = df.sample(n=n_lotes * tam_lote, random_state=seed, weights=w.to_numpy()).reset_index(drop=True)
    out = []
    for k in range(n_lotes):
        sub = df.iloc[k * tam_lote:(k + 1) * tam_lote]
        d = a_dataset_directo(sub, calib, rng, "A" if (k + 1) < lote_cambio else "B")
        d.insert(0, "lote", k + 1)
        out.append(d)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEMILLA)
    print(generar_raw(ap.parse_args().seed))
