# Arquitectura y decisiones

## Flujo

```
generar_datos ─► limpiar_ofuscar ─► train.py ──► MLflow (tracking + Model Registry 'talento-sc')
 (sintético)      (dedup + HMAC)     8 runs               │  alias champion → v2
                                                           ▼
                                          exportar_modelo.py  (models:/talento-sc@champion)
                                                           ▼
                                   Docker image (API FastAPI + artefacto + metadata.json)
                                                           ▼
                        Kubernetes: Deployment ×3 réplicas ─► Service LoadBalancer :8080 ─► UI Streamlit
                                                           ▲
                 pruebas de drift (KS/PSI/nulos + F1 por lotes) vigilan al modelo en el tiempo
```

## Versión desplegada ↔ run

| Campo | Valor |
|---|---|
| Modelo registrado | `talento-sc` |
| **Versión desplegada** | **v3** (alias `champion`, promovida el 2026-11-04 tras la alarma de concept drift) |
| **Run de origen** | `07939995e4c1458f9f947a583af64542` (`xgb_mice_d3_reentreno`, experimento `talento-sc-deteccion`) |
| **Versión de datos** | `datos-v2` (train/val/test con SHA-256 en [data/versiones.json](../data/versiones.json), registrados en la pestaña *Datasets* del run) |
| **Imagen** | `talento-sc-api:modelo-v3` (el tag repite la versión del registry; revisión 3 del Deployment) |
| Hiperparámetros | XGBoost max_depth=3, 200 árboles, lr=0.10, subsample=0.9, scale_pos_weight=3, seed=42, imputación MICE |
| Métricas (test datos-v2) | val_f1 = 0.656 · test_f1 = 0.519 · test_recall = 0.472 · test_precision = 0.577 · test_ROC-AUC = 0.843 · umbral = 0.538. El champion anterior (v2) logra F1 = 0.401 sobre el mismo test |
| Versión anterior | **v2 (alias `anterior`)**: `xgb_mice_d3`, run `845a8313…`, datos-v1, test F1 0.622 en datos-v1. Rollback: `kubectl -n talento-sc rollout undo deploy/talento-api` (imagen `modelo-v2`) + `python -m src.training.promover --version 2` |
| Otras versiones | v1: LightGBM `lgbm_mice_l31` (datos-v1), val_f1 = 0.619 |
| MLflow | 3.17.2 (servidor y cliente de la imagen, misma versión) |

Trazabilidad verificable: `curl localhost:8080/info` devuelve `version`, `run_id` y `run_name` del modelo que sirve el pod;
en la UI de MLflow ese `run_id` abre el run exacto. Cada respuesta de `/predecir` repite versión, run y **nombre del pod**.

## Versiones de datos

| Versión | Contenido | Filas (train/val/test) | Modelos | Tag de git |
|---|---|---|---|---|
| `datos-v1` | Temporadas 2021-2025, criterio de scouting tradicional, split agrupado por jugador | 4.841 / 988 / 1.062 | v1, v2 (`anterior`) | `datos-v1` |
| `datos-v2` | Flujo 2026, lotes 7-9 / 10 / 11-12, criterio nuevo (post concept drift) | 1.200 / 400 / 800 | **v3 (`champion`, desplegada)** | `datos-v2` |

Cadena de trazabilidad completa: **pod** → `/info` (versión v3, run, `version_datos`) → **run** en MLflow → pestaña *Datasets* (`datos-v2-train`, digest) →
**manifiesto** `data/versiones.json` (SHA-256 del archivo) → **commit** etiquetado `datos-v2` en git.

## Ciclo de reentrenamiento

```
alarma concept drift (lote 9) ─► llegan etiquetas ─► datos-v2 ─► reentrenar.py ─► v3 @challenger
                                                                       │
                       comparar con el champion en el MISMO test ──────┘   (0.519 vs 0.401)
       promover.py --version 3 ─► champion → v3, anterior → v2 ─► construir_y_desplegar.sh ─► rolling update (modelo-v3)
```

Historial del Deployment ([rollout_history.txt](evidencias/rollout_history.txt)): revisión 2 = `modelo-v2`, revisión 3 = `modelo-v3`.
El análisis de concept drift sigue evaluando a **v2** (copia congelada en `artifacts/modelos/v2`), porque fue el modelo que estuvo en producción durante el flujo 2026.

## Decisiones y justificación

| Decisión | Por qué |
|---|---|
| Problema: clasificación tabular `talento` (≈14 % de positivos) | Es el objetivo de la tesis; 18 variables, 6.891 filas jugador-temporada (>1.000 y >4 variables, sin juguetes). |
| Pipeline `ColumnTransformer → MICE (IterativeImputer + BayesianRidge) → XGBoost/LightGBM` dentro de **un solo objeto** sklearn | El imputador se entrena solo con *train* y viaja con el modelo: la API recibe datos incompletos tal cual llegan, sin lógica duplicada. `add_indicator=True` conserva la señal de "dato faltante" (MNAR). |
| División 70/15/15 **agrupada por jugador** (`GroupShuffleSplit`, seed 42) | Un niño aparece en varias temporadas; si sus filas cayeran en train y test habría fuga de información. |
| Umbral de decisión elegido **solo con validación** (máx. F1) y guardado en el run | Clases desbalanceadas; el test queda intacto para reportar. La API usa el mismo umbral del run. |
| 8 runs: 6 MICE (3 XGB + 3 LGBM) + línea base *mediana* + línea base *sin imputar* | Comparables entre sí (mismo split y métrica) y permiten discutir el aporte de MICE con honestidad. Reconstrucción de valores ocultos: NRMSE 0.30 (MICE) vs 1.00 (mediana). Con los mismos hiperparámetros (XGB d4) el AUC en test sube de 0.906 (mediana) a 0.916 (MICE) y el F1 es equivalente (0.632 vs 0.628, dentro del ruido). Frente a no imputar (LGBM l15): val_f1 0.574 vs 0.607 y AUC 0.899 vs 0.918. |
| Registry: se registra el mejor de **cada familia** (v1, v2); `champion` = mejor val_f1. El reentrenamiento registra v3 como `challenger` | Demuestra versionado y promoción por alias; la API nunca apunta a una ruta suelta. |
| Versionado de datos con **hash SHA-256 + manifiesto en git + `mlflow.log_input`**, sin DVC | Con ~3 MB de CSV, git alcanza para guardar las versiones; MLflow registra qué versión exacta usó cada run. DVC se justificaría con datos grandes o binarios. |
| datos-v2 = **solo la ventana reciente** (lotes 7-9) | Las etiquetas históricas siguen el criterio viejo y contradicen el nuevo: sumarlas bajó el F1 en datos del criterio nuevo de ~0.53 a ~0.45. |
| La promoción de v3 es un paso **explícito** (`src/training/promover.py`), no automático | Queda como decisión humana auditable (tags `promovido`/`retirado` en el registry) con rollback inmediato al alias `anterior`. Riesgo asumido: el test de v3 tiene solo 800 filas; se reevalúa cuando lleguen más etiquetas. |
| Imagen construida desde `models:/talento-sc@champion` (artefacto + `metadata.json` horneados) | Reproducible y sin depender del servidor MLflow en tiempo de ejecución (si el servidor cae, los pods siguen sirviendo). Opcional: `MLFLOW_MODEL_URI` para cargar directo del registry. |
| `python:3.13-slim` (misma versión que entrena), `mlflow-skinny==3.17.2`, `xgboost-cpu==3.4.1`, todo fijado | Mismo intérprete que serializó el modelo; skinny = cliente sin UI; xgboost-cpu evita ~300 MB de librerías NVIDIA (imagen de 189 MB comprimida). Usuario sin privilegios (uid 1000). |
| Kubernetes: Deployment con 3 réplicas, probes `/health`, `POD_NAME` por Downward API, Service `LoadBalancer` | Alta disponibilidad, autorreparación y balanceo demostrables; el nombre del pod en la respuesta evidencia el reparto. Docker Desktop publica el Service en `localhost:8080` (sin `port-forward`, que fijaría un solo pod). |
| Serialización `cloudpickle` en MLflow | MLflow 3.17 usa `skops` por defecto y rechaza los tipos de XGBoost/IterativeImputer; el modelo es propio y confiable. |

## Limitaciones declaradas

* Los datos son sintéticos: las métricas absolutas **no** son evidencia sobre el talento real; demuestran la ingeniería.
* La etiqueta `talento` se simula con latentes (técnica, físico, maduración, edad relativa); su regla es el objeto del concept drift.
* La deduplicación de perfiles tiene precisión 1.00 y recall 0.83: ~17 % de duplicados sobreviven (22 filas sin etiqueta descartadas).
* El clúster es local (Docker Desktop, un nodo): las réplicas comparten nodo; la alta disponibilidad frente a caída de nodo requiere un clúster multinodo.
