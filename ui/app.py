"""UI (puntos extra): consume la API REAL desplegada en Kubernetes (Service talento-api, :8080).
Ejecutar:  API_URL=http://localhost:8080 streamlit run ui/app.py
"""
import os
from collections import Counter

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Detección de talento - Santa Cruz", layout="wide")
st.title("Detección temprana de talento infanto-juvenil")
st.caption(f"Cliente de la API desplegada en Kubernetes · {API_URL} · datos de demostración SINTÉTICOS")

try:
    info = requests.get(f"{API_URL}/info", timeout=5).json()
    st.success(f"Modelo **{info['modelo']}** v{info['version']} (alias `{info['alias']}`) · run `{info['run_id'][:8]}` "
               f"· umbral {info['umbral']:.2f} · atendido por pod `{info['pod']}`")
except Exception as e:
    st.error(f"No se pudo contactar la API en {API_URL}: {e}")
    st.stop()

with st.form("jugador"):
    c1, c2, c3 = st.columns(3)
    cat = c1.slider("Categoría Sub-", 5, 17, 12)
    edad = c1.number_input("Edad (años)", 5.0, 18.9, float(cat) + 0.3, 0.1)
    genero = c1.selectbox("Género", ["M", "F"])
    tri = c1.selectbox("Trimestre de nacimiento", [1, 2, 3, 4])
    pos = c1.selectbox("Posición", ["SIN_DATO", "ARQUERO", "DEFENSA", "MEDIOCAMPISTA", "DELANTERO"])
    campos = {
        "altura_cm": (c2, 100.0, 195.0, 150.0), "peso_kg": (c2, 15.0, 90.0, 40.0),
        "sprint_30m_s": (c2, 3.5, 7.0, 5.0), "salto_cmj_cm": (c2, 10.0, 60.0, 28.0),
        "yoyo_m": (c3, 100.0, 3000.0, 1000.0), "agilidad_s": (c3, 8.0, 16.0, 11.5),
        "tecnica_score": (c3, 0.0, 100.0, 60.0), "goles": (c3, 0.0, 40.0, 3.0),
    }
    valores = {}
    for k, (col, lo, hi, d) in campos.items():
        usar = col.checkbox(f"Tengo {k}", value=k in ("tecnica_score", "altura_cm"))
        v = col.number_input(k, lo, hi, d, disabled=not usar)
        valores[k] = float(v) if usar else None
    enviar = st.form_submit_button("Predecir")

if enviar:
    j = dict(categoria_sub=cat, edad_anios=edad, genero=genero, trimestre_nacimiento=tri, posicion=pos, **valores)
    r = requests.post(f"{API_URL}/predecir", json={"jugadores": [j]}, timeout=10).json()
    res = r["resultados"][0]
    st.metric("Probabilidad de talento", f"{res['probabilidad_talento']:.1%}",
              "TALENTO" if res["talento"] else "no talento")
    st.caption(f"pod `{r['pod']}` · modelo v{r['version_registry']} · run `{r['run_id'][:8]}`. "
               "Los campos vacíos los imputa el pipeline con MICE.")

st.divider()
if st.button("Enviar 12 peticiones y ver qué pods responden (balanceo de carga)"):
    pods = []
    for _ in range(12):
        pods.append(requests.get(f"{API_URL}/health", timeout=5, headers={"Connection": "close"}).json()["pod"])
    st.bar_chart(pd.Series(Counter(pods)))
    st.write(dict(Counter(pods)))
