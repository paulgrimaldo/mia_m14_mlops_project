"""Ofuscación de datos de menores de edad.

Principios (minimización de datos):
  * el nombre, la foto, el CI, el FIFA-ID y el userId NO salen del paso de limpieza;
  * el identificador se reemplaza por un seudónimo HMAC-SHA256 con sal secreta
    (no reversible sin la sal; estable entre temporadas para poder seguir la proyección);
  * la fecha de nacimiento se generaliza a (edad en años, trimestre de nacimiento):
    el trimestre basta para modelar el efecto de edad relativa.

La sal real debe venir de la variable de entorno ANONIMIZAR_SAL y no versionarse.
"""
import hashlib
import hmac
import os

SAL = os.environ.get("ANONIMIZAR_SAL", "sal-de-desarrollo-NO-usar-con-datos-reales").encode()

CAMPOS_PROHIBIDOS = {"fullName", "shortName", "profilePictureUrl", "ci", "fifaId", "userId",
                     "cometId", "description", "externalProfileId", "dateOfBirth"}


def seudonimo(identificador: str) -> str:
    return hmac.new(SAL, str(identificador).encode(), hashlib.sha256).hexdigest()[:12]


def ofuscar_perfil_api(perfil: dict, temporada: int) -> dict:
    """Aplica la ofuscación a un perfil con el formato JSON de la API de Copa Tahuichi."""
    nac = perfil["dateOfBirth"]  # 'YYYY-MM-DD'
    anio, mes = int(nac[:4]), int(nac[5:7])
    limpio = {k: v for k, v in perfil.items() if k not in CAMPOS_PROHIBIDOS}
    limpio["jugador_id"] = seudonimo(perfil["profileId"])
    limpio.pop("profileId", None)
    limpio["categoria_sub"] = temporada - anio
    limpio["trimestre_nacimiento"] = (mes - 1) // 3 + 1
    return limpio
