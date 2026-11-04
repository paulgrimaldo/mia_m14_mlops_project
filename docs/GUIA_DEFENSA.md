# Guía para la defensa

## Antes de entrar (5 min)
```bash
scripts/iniciar_mlflow.sh &                       # UI en http://127.0.0.1:5000
kubectl -n talento-sc get pods                    # 3 pods Running
curl localhost:8080/info                          # versión y run que sirve el clúster
```
Si el clúster se reinició: cargar la imagen (`docker save … | docker exec -i desktop-control-plane ctr -n k8s.io images import -`) y `kubectl apply -f k8s/`.

## Guion en vivo
1. **MLflow UI** → experimento `talento-sc-deteccion`: cada run = una combinación (imputación, familia, hiperparámetros). Ordenar por `val_f1`, filtrar `tags.imputacion = "mice"`, seleccionar 3-4 runs → *Compare* (parallel coordinates: `xgb_mice_d3` gana en val_f1 y tiene el mejor recall). Columna `version_datos`: filtrar `tags.version_datos = "datos-v1"`. Abrir un run → pestaña **Datasets** (train/val/test con digest). **Models → talento-sc**: v1, **v2 `champion`** (run `845a8313…`, datos-v1) y **v3 `challenger`** (datos-v2, reentrenamiento).
2. `curl localhost:8080/info` → misma versión/run que el registry (trazabilidad modelo ↔ experimento).
3. **Balanceo**: `for i in $(seq 8); do curl -s --no-keepalive localhost:8080/health; echo; done` → pods distintos.
4. **Autorreparación**: en otra terminal `watch -n1 kubectl -n talento-sc get pods`; `kubectl -n talento-sc delete pod <uno>` mientras corre un bucle de `curl` → siempre 200, nace un pod nuevo.
5. **Escalado**: `kubectl -n talento-sc scale deploy/talento-api --replicas=5` → 5 pods; volver a 3.
6. **Drift**: `.venv/bin/pytest -v tests/` (verde/rojo); abrir `docs/evidencias/concept_drift.png`.
7. **UI** (extra): `API_URL=http://localhost:8080 streamlit run ui/app.py`.

## Preguntas probables
* **¿Por qué el split es agrupado por jugador?** Un niño aparece en varias temporadas; sin agrupar, sus filas caen en train y test (fuga).
* **¿Por qué MICE si XGBoost/LightGBM aceptan NaN?** Se comparó: sin imputar F1 0.574 vs 0.607 (LGBM), AUC 0.899 vs 0.918; vs mediana el AUC sube 0.906→0.916 y el error de reconstrucción baja de 1.00 a 0.30. El F1 vs mediana es equivalente (honesto: el beneficio en F1 es pequeño en estos datos sintéticos).
* **¿Cómo versionan los datos?** Manifiesto `data/versiones.json` con SHA-256 de cada archivo, tags de git `datos-v1`/`datos-v2`, y `mlflow.log_input` en cada run. DVC no se usó: con 3 MB git alcanza.
* **¿Por qué v3 no reemplazó a v2?** Mejora sobre el criterio nuevo (0.519 vs 0.401), pero el test tiene 800 filas; la promoción es una decisión explícita (`--promover`).
* **¿Cómo sabe el servicio qué versión sirve?** La imagen se construye de `models:/talento-sc@champion`; `metadata.json` (versión, run_id, umbral) viaja en la imagen y `/info` lo expone. Promover un challenger = mover el alias + `scripts/construir_y_desplegar.sh` (imagen `modelo-vN` + rolling update; `kubectl rollout history` muestra qué versión hubo en cada revisión).
* **¿Por qué el umbral no es 0.5?** Se maximiza F1 en validación (clases 14 %); queda guardado en el run y lo usa la API.
* **¿Por qué estos umbrales de drift?** Ver [DRIFT.md](DRIFT.md): Bonferroni + D≥0.10, PSI 0.25 estándar, 20 % de variables; calibrados con 200 lotes (0 falsas alarmas, 100 % detección).
* **¿Por qué data drift no detecta el concept drift?** Las entradas no cambian; solo la relación X→y. Por eso se monitorea además F1 con etiquetas y proxies (ver retraso de etiquetas).
* **¿Qué pasa si cae un nodo?** El clúster local tiene un nodo; con varios, el scheduler redistribuye las réplicas (limitación declarada).
* **¿Por qué `LoadBalancer` y no `port-forward`?** `port-forward` fija un único pod y oculta el balanceo.
* **¿Datos reales?** Sintéticos por privacidad de menores; `ofuscar_perfil_api` ya acepta el JSON real de la plataforma.
