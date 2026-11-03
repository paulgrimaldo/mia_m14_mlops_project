#!/usr/bin/env bash
# Ejecuta las 4 demostraciones exigidas de Kubernetes y guarda la evidencia en docs/evidencias/k8s_demos.txt
cd "$(dirname "$0")/.."
NS=talento-sc
URL=${URL:-http://localhost:8080}
OUT=docs/evidencias/k8s_demos.txt
BODY='{"jugadores":[{"categoria_sub":12,"edad_anios":12.3,"genero":"M","trimestre_nacimiento":1,"posicion":"DELANTERO","tecnica_score":78}]}'
sec() { echo; echo "=================== $* ==================="; }
{
echo "Evidencia generada el $(date '+%Y-%m-%d %H:%M:%S')  ·  contexto: $(kubectl config current-context)"

sec "DEMO 1 - 3+ réplicas en estado Running simultáneamente"
kubectl -n $NS get deploy talento-api
kubectl -n $NS get pods -o wide

sec "DEMO 2 - El tráfico se reparte entre pods (20 peticiones POST /predecir a $URL)"
for i in $(seq 1 20); do
  curl -s --no-keepalive -X POST $URL/predecir -H 'content-type: application/json' -d "$BODY" \
   | python3 -c "import sys,json; r=json.load(sys.stdin); print('pod=%s  modelo=v%s run=%s' % (r['pod'], r['version_registry'], r['run_id'][:8]))"
done | tee /tmp/_pods.txt
echo "--- resumen: peticiones atendidas por pod"
awk '{print $1}' /tmp/_pods.txt | sort | uniq -c

sec "DEMO 3 - Autorreparación: se elimina un pod mientras se sigue consultando el servicio"
( for i in $(seq 1 60); do
    code=$(curl -s --no-keepalive -o /dev/null -m 2 -w '%{http_code}' $URL/health); echo "$(date +%T) HTTP $code"; sleep 0.25
  done ) > /tmp/_heal.txt &
BG=$!
sleep 3
VICTIMA=$(kubectl -n $NS get pods -o jsonpath='{.items[0].metadata.name}')
echo ">> $(date +%T) eliminando pod: $VICTIMA"
kubectl -n $NS delete pod $VICTIMA --wait=false
for t in 1 2 3 4; do sleep 2; echo "--- $(date +%T)"; kubectl -n $NS get pods; done
wait $BG
echo "--- códigos HTTP recibidos durante la prueba (60 peticiones):"
awk '{print $3}' /tmp/_heal.txt | sort | uniq -c
echo "--- peticiones fallidas (si las hubo):"; grep -v "HTTP 200" /tmp/_heal.txt || echo "ninguna"
kubectl -n $NS get pods

sec "DEMO 4 - Escalado: 3 -> 5 -> 2 -> 3 réplicas"
for n in 5 2 3; do
  echo ">> kubectl scale --replicas=$n"; kubectl -n $NS scale deploy/talento-api --replicas=$n
  kubectl -n $NS rollout status deploy/talento-api --timeout=120s >/dev/null; sleep 4
  kubectl -n $NS get deploy talento-api; kubectl -n $NS get pods --no-headers | awk '{print "   "$1, $3}'
  echo "   pods distintos que responden a 20 peticiones:"
  for i in $(seq 1 20); do curl -s --no-keepalive $URL/health | python3 -c "import sys,json;print(json.load(sys.stdin)['pod'])"; done | sort | uniq -c
done
} 2>&1 | tee $OUT
