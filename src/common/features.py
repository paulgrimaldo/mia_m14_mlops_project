"""Contrato de variables compartido por entrenamiento, API y pruebas de drift."""

NUMERICAS = [
    "categoria_sub", "edad_anios", "altura_cm", "peso_kg", "imc",
    "sprint_30m_s", "salto_cmj_cm", "yoyo_m", "agilidad_s", "tecnica_score",
    "partidos", "goles", "asistencias", "ppm_equipo", "anios_en_academia",
]
# 'trimestre_nacimiento' se trata como categórica (efecto de edad relativa)
CATEGORICAS = ["genero", "posicion", "trimestre_nacimiento"]
FEATURES = NUMERICAS + CATEGORICAS
TARGET = "talento"
POSICIONES = ["ARQUERO", "DEFENSA", "MEDIOCAMPISTA", "DELANTERO", "SIN_DATO"]
