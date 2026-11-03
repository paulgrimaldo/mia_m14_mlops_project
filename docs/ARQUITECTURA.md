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
| **Versión desplegada** | **v2** (alias `champion`) |
| **Run de origen** | `e94d8168b64b4c128eda37c93a0894cd` (`xgb_mice_d3`, experimento `talento-sc-deteccion`) |
| Hiperparámetros | XGBoost max_depth=3, 200 árboles, lr=0.10, subsample=0.9, scale_pos_weight=3, seed=42, imputación MICE |
| Métricas | val_f1 = 0.631 · test_f1 = 0.622 · test_recall = 0.756 · test_ROC-AUC = 0.916 · umbral = 0.387 |
| Otra versión | v1 (alias `challenger`): LightGBM `lgbm_mice_l31`, val_f1 = 0.619 |
| MLflow | 3.17.2 (servidor y cliente de la imagen, misma versión) |

Trazabilidad verificable: `curl localhost:8080/info` devuelve `version`, `run_id` y `run_name` del modelo que sirve el pod;
en la UI de MLflow ese `run_id` abre el run exacto. Cada respuesta de `/predecir` repite versión, run y **nombre del pod**.

## Decisiones y justificación

| Decisión | Por qué |
|---|---|
| Problema: clasificación tabular `talento` (≈14 % de positivos) | Es el objetivo de la tesis; 18 variables, 6.891 filas jugador-temporada (>1.000 y >4 variables, sin juguetes). |
| Pipeline `ColumnTransformer → MICE (IterativeImputer + BayesianRidge) → XGBoost/LightGBM` dentro de **un solo objeto** sklearn | El imputador se entrena solo con *train* y viaja con el modelo: la API recibe datos incompletos tal cual llegan, sin lógica duplicada. `add_indicator=True` conserva la señal de "dato faltante" (MNAR). |
| División 70/15/15 **agrupada por jugador** (`GroupShuffleSplit`, seed 42) | Un niño aparece en varias temporadas; si sus filas cayeran en train y test habría fuga de información. |
| Umbral de decisión elegido **solo con validación** (máx. F1) y guardado en el run | Clases desbalanceadas; el test queda intacto para reportar. La API usa el mismo umbral del run. |
| 8 runs: 6 MICE (3 XGB + 3 LGBM) + línea base *mediana* + línea base *sin imputar* | Comparables entre sí (mismo split y métrica) y permiten discutir el aporte de MICE con honestidad. Reconstrucción de valores ocultos: NRMSE 0.30 (MICE) vs 1.00 (mediana). Con los mismos hiperparámetros (XGB d4) el AUC en test sube de 0.906 (mediana) a 0.916 (MICE) y el F1 es equivalente (0.632 vs 0.628, dentro del ruido). Frente a no imputar (LGBM l15): val_f1 0.574 vs 0.607 y AUC 0.899 vs 0.918. |
| Registry: se registra el mejor de **cada familia** (v1, v2); `champion` = mejor val_f1, `challenger` = el otro | Demuestra versionado y promoción por alias; la API nunca apunta a una ruta suelta. |
| Imagen construida desde `models:/talento-sc@champion` (artefacto + `metadata.json` horneados) | Reproducible y sin depender del servidor MLflow en tiempo de ejecución (si el servidor cae, los pods siguen sirviendo). Opcional: `MLFLOW_MODEL_URI` para cargar directo del registry. |
| `python:3.13-slim` (misma versión que entrena), `mlflow-skinny==3.17.2`, `xgboost-cpu==3.4.1`, todo fijado | Mismo intérprete que serializó el modelo; skinny = cliente sin UI; xgboost-cpu evita ~300 MB de librerías NVIDIA (imagen de 189 MB comprimida). Usuario sin privilegios (uid 1000). |
| Kubernetes: Deployment con 3 réplicas, probes `/health`, `POD_NAME` por Downward API, Service `LoadBalancer` | Alta disponibilidad, autorreparación y balanceo demostrables; el nombre del pod en la respuesta evidencia el reparto. Docker Desktop publica el Service en `localhost:8080` (sin `port-forward`, que fijaría un solo pod). |
| Serialización `cloudpickle` en MLflow | MLflow 3.17 usa `skops` por defecto y rechaza los tipos de XGBoost/IterativeImputer; el modelo es propio y confiable. |

## Limitaciones declaradas

* Los datos son sintéticos: las métricas absolutas **no** son evidencia sobre el talento real; demuestran la ingeniería.
* La etiqueta `talento` se simula con latentes (técnica, físico, maduración, edad relativa); su regla es el objeto del concept drift.
* La deduplicación de perfiles tiene precisión 1.00 y recall 0.83: ~17 % de duplicados sobreviven (22 filas sin etiqueta descartadas).
* El clúster es local (Docker Desktop, un nodo): las réplicas comparten nodo; la alta disponibilidad frente a caída de nodo requiere un clúster multinodo.
