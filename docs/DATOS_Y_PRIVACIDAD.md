# Datos sintéticos y protección de menores

## Por qué sintéticos
No existe un dataset consolidado (los registros están fragmentados por academia) y se trata de **menores de edad**. Se generaron datos
que replican la **estructura** de la plataforma de Copa Tahuichi (campeonatos *Copa Tahuichi Invierno Sub-05…Sub-17 Varones/Damas*,
calendarios, eventos de gol, perfiles con rol 1 = jugador, transferencias, goles por campeonato) más **asistencias simuladas**.
El enunciado admite datasets simulados si se justifica y se generan escenarios de deriva (§3.1): aquí el foco es la ingeniería, no la exactitud.

## Qué simula el generador ([generar_datos.py](../src/data/generar_datos.py), seed 42)
* 28 clubes (con nivel de digitalización alto/medio/bajo), cohortes 2004-2021, temporadas 2021-2025 (+2026 como flujo de producción).
* Trayectorias: ingreso a distintas edades, abandono creciente con la edad ("fuga de talento"), temporadas en que el niño no figura, cambios de club.
* Mediciones: altura/peso (curva de crecimiento + maduración biológica), sprint 30 m, salto CMJ, Yo-Yo, agilidad, técnica (scout), partidos, goles, asistencias, PPM del equipo.
* **Problemas de calidad de la tesis**: 5 % de jugadores con **perfil duplicado** (nombre con variante: falta un nombre, tipeo, orden invertido), 6 % de filas con el niño en **dos equipos a la vez**, y faltantes que dependen del club y del año (digitalización: Yo-Yo faltante 51 % en 2021 → 30 % en 2025) con componente MNAR (a los más lentos se les mide menos).
* Etiqueta `talento` (top ≈15 %): combinación de técnica, físico, maduración, edad relativa y contribución de gol. Un segundo régimen (`talento_b`) se usa para el concept drift.

## Limpieza ([limpiar_ofuscar.py](../src/data/limpiar_ofuscar.py))
Deduplicación por **misma fecha de nacimiento + Jaccard de tokens del nombre ≥ 0.6** → precisión 1.00, recall 0.83 contra la verdad simulada (147 de 178).
Consolidación por (jugador, temporada): se **suman** partidos/goles/asistencias de los dos equipos, se **promedian** las mediciones y se conserva el NaN cuando nadie midió ("no registrado" ≠ "0 goles").

## Ofuscación ([ofuscar.py](../src/data/ofuscar.py))
| Dato | Tratamiento |
|---|---|
| Nombre, foto, CI, FIFA-ID, userId | Eliminados antes de guardar el dataset procesado |
| `profileId` | Seudónimo **HMAC-SHA256** con sal secreta (`ANONIMIZAR_SAL`), 12 hex; estable entre temporadas (permite seguir la proyección), irreversible sin la sal |
| Fecha de nacimiento | Generalizada a **edad en años + trimestre de nacimiento** (basta para el efecto de edad relativa) |
| Datos crudos (`data/raw/`) | Fuera de git (`.gitignore`) |

`ofuscar_perfil_api(perfil, temporada)` aplica la misma política a un perfil con el **formato JSON real de la API**, de modo que cuando existan datos
reales el pipeline cambia solo la fuente (los `profiles`, `schedules`, `events` y `transfers` mapean a `perfiles/participaciones/eventos/transferencias`).
