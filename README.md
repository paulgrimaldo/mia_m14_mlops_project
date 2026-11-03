# Proyecto final · MLOps — Detección de talento infanto-juvenil (Santa Cruz)

Modelo predictivo con **MICE + XGBoost/LightGBM** (tesis *"Modelo predictivo mediante algoritmos de ensamble para la
detección de talento y gestión de datos incompletos en jugadores infanto-juveniles de Santa Cruz (2021-2025)"*)
llevado desde el entrenamiento hasta un despliegue **reproducible y observable**: MLflow → Docker → Kubernetes (3 réplicas) → pruebas de drift.

> **Datos 100 % sintéticos** (no se usan datos reales de menores). Imitan la estructura de Copa Tahuichi
> (campeonatos Sub-05…Sub-17, equipos, transferencias, goles/asistencias) e incluyen duplicados, niños en dos equipos y faltantes.
> Ver [docs/DATOS_Y_PRIVACIDAD.md](docs/DATOS_Y_PRIVACIDAD.md).

## Mapa de requisitos del enunciado

| Requisito | Dónde está | Evidencia |
|---|---|---|
| Fase 1 · entrenamiento reproducible (seed, split documentado), ≥5 runs | [src/training/train.py](src/training/train.py) (8 runs) | [mlflow_runs.csv](docs/evidencias/mlflow_runs.csv) |
| MLflow: tracking + Model Registry, >1 versión, alias de despliegue | `talento-sc` v1 (LightGBM) y v2 (XGBoost), alias `champion` → v2 | [mlflow_registry.txt](docs/evidencias/mlflow_registry.txt) |
| Servicio consume el registry por alias, no por ruta suelta | [src/serving/exportar_modelo.py](src/serving/exportar_modelo.py) (`models:/talento-sc@champion`) + [api.py](src/serving/api.py) | `/info` devuelve versión y run |
| Fase 2 · Dockerfile propio, deps fijadas con versión de MLflow | [Dockerfile](Dockerfile), [requirements-serving.txt](requirements-serving.txt) | [docker_build.txt](docs/evidencias/docker_build.txt) |
| Fase 3 · K8s ≥3 réplicas + 4 demostraciones | [k8s/](k8s/), [scripts/demo_k8s.sh](scripts/demo_k8s.sh) | [k8s_demos.txt](docs/evidencias/k8s_demos.txt) |
| Data drift (KS, PSI, umbral, verde/rojo) | [src/drift/detectores.py](src/drift/detectores.py), [tests/test_data_drift.py](tests/test_data_drift.py) | [data_drift_reporte.txt](docs/evidencias/data_drift_reporte.txt), [png](docs/evidencias/data_drift_verde_vs_rojo.png), [calibración](docs/evidencias/calibracion_data_drift.json) |
| Concept drift (gráfica temporal, criterio de reentrenamiento, retraso de etiquetas) | [src/drift/concepto.py](src/drift/concepto.py), [tests/test_concept_drift.py](tests/test_concept_drift.py) | [concept_drift.png](docs/evidencias/concept_drift.png) |
| Documento de arquitectura + versión desplegada ↔ run | [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md) | — |
| Tabla de reparto de trabajo | [docs/REPARTO_TRABAJO.md](docs/REPARTO_TRABAJO.md) | `git log` |
| Extra · UI contra el servicio en K8s | [ui/app.py](ui/app.py) (Streamlit) | — |

## Reproducir todo desde cero

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/pipeline_completo.sh        # datos -> limpieza/ofuscación -> MLflow -> registro -> exportar champion -> drift
.venv/bin/pytest -q                 # 36 pruebas (data drift + concept drift)

docker build -t talento-sc-api:1.0 .
docker save talento-sc-api:1.0 | docker exec -i desktop-control-plane ctr -n k8s.io images import -   # cargar imagen al nodo
kubectl apply -f k8s/namespace.yaml -f k8s/deployment.yaml -f k8s/service.yaml
curl localhost:8080/info
scripts/demo_k8s.sh                 # las 4 demostraciones (+ guarda evidencia)

pip install -r ui/requirements.txt && API_URL=http://localhost:8080 streamlit run ui/app.py   # extra
```

Interfaz de MLflow (para la defensa): `scripts/iniciar_mlflow.sh` → <http://127.0.0.1:5000>.
Guion de la defensa y preguntas probables: [docs/GUIA_DEFENSA.md](docs/GUIA_DEFENSA.md).

## Estructura

```
src/common/     contrato de variables (compartido por entrenamiento, API y drift)
src/data/       generador sintético · limpieza/deduplicación · ofuscación
src/training/   entrenamiento + registro en MLflow
src/serving/    exportación del champion por alias + API FastAPI
src/drift/      detectores, escenarios, concept drift, calibración
k8s/            namespace, deployment (3 réplicas), service
tests/          pruebas automatizadas de drift
scripts/        pipeline, MLflow, demos de Kubernetes
docs/           arquitectura, drift, datos/privacidad, reparto, guía de defensa, evidencias/
```
