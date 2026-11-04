#!/usr/bin/env bash
# Exporta el champion del registry, construye la imagen etiquetada con SU versión
# (talento-sc-api:modelo-vN), la carga en el nodo y hace rolling update del Deployment.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m src.serving.exportar_modelo > /dev/null
V=$(python3 -c "import json;print(json.load(open('src/serving/model/metadata.json'))['version'])")
IMG="talento-sc-api:modelo-v$V"
echo ">> construyendo $IMG"
docker build -q -t "$IMG" .
docker save "$IMG" | docker exec -i desktop-control-plane ctr -n k8s.io images import - > /dev/null
kubectl -n talento-sc set image deploy/talento-api api="$IMG"
kubectl -n talento-sc annotate deploy/talento-api kubernetes.io/change-cause="modelo talento-sc v$V ($IMG)" --overwrite
kubectl -n talento-sc rollout status deploy/talento-api --timeout=180s
kubectl -n talento-sc rollout history deploy/talento-api
