# Observabilidad: data drift y concept drift

Referencia (baseline) = partición de entrenamiento (`data/processed/train.csv`, 4.841 filas).
Lotes nuevos = 500 jugadores (data drift) o 400 (concept drift).

## 1. Data drift

| Tipo de variable | Prueba | Por qué |
|---|---|---|
| Numéricas (15) | **Kolmogorov-Smirnov** de dos muestras | No paramétrica: nuestras variables son sesgadas y acotadas (no normales); detecta cambios de ubicación, escala y forma sin discretizar. |
| Categóricas (género, posición, trimestre de nacimiento) | **PSI** | Estándar de monitoreo de modelos; interpretable con umbrales conocidos. |
| Tasa de faltantes (todas) | **z de dos proporciones** | Específica de la tesis: si cambia cuánto falta, cambia lo que MICE debe imputar. |

### Umbrales y su origen

| Umbral | Valor | De dónde sale |
|---|---|---|
| KS | p < 0.01/18 (Bonferroni, 18 variables) **y** D ≥ 0.10 | α=1 % corregido por comparaciones múltiples; D≥0.10 evita alertar por diferencias significativas pero irrelevantes (con n grande cualquier diferencia lo es). |
| PSI | ≥ 0.25 | Convención de la industria: <0.10 estable, 0.10–0.25 moderado, ≥0.25 cambio significativo. |
| Faltantes | Δ ≥ 0.10 absoluto y p < 0.01/18 | Un cambio de 10 puntos en la tasa de nulos altera materialmente la imputación. |
| **Lote ROJO** | ≥ 20 % de las variables en alerta (≥4 de 18) | Una variable aislada es ruido o cambio local; 20 % indica un cambio de población. Se **calibró por simulación**: |

Calibración (`python -m src.drift.calibrar_umbrales`, 200 lotes de 500 por escenario, [calibracion_data_drift.json](evidencias/calibracion_data_drift.json)):

| Escenario | Fracción media de variables en alerta | Lotes en ROJO |
|---|---|---|
| Mismo origen que el entrenamiento | 0.0 % (máx. 0.0 %) | **0 / 200** (verde) |
| Deriva inyectada fuerte | 33 % | **200 / 200** (rojo) |
| Solo aumento de faltantes (batería física) | 22 % | **200 / 200** (rojo) |

**Deriva inyectada** ([escenarios.py](../src/drift/escenarios.py)): categorías +3 años, sensor de velocidad con +12 % de tiempo,
salto con otro protocolo (+6 cm), 60 % sin Yo-Yo, 70 % "delanteros". Pruebas: [tests/test_data_drift.py](../tests/test_data_drift.py)
(verde con 10 lotes de mismo origen, rojo con 10 de cada escenario). Reporte visible: [data_drift_reporte.txt](evidencias/data_drift_reporte.txt).

## 2. Concept drift

**Escenario.** A partir del lote 7 (de 12) los scouts cambian de criterio: dejan de premiar el físico, la maduración temprana y la
edad relativa y valoran más la técnica (corrige el sesgo que la tesis describe). Las **entradas siguen igual** (el detector de
data drift permanece en VERDE los 12 lotes) pero la regla que genera la etiqueta cambió.

**Resultado** ([concept_drift.png](evidencias/concept_drift.png), [lotes.csv](evidencias/concept_drift_lotes.csv)):
F1 medio 0.59 antes → 0.40 después del cambio (baseline de test 0.62).

### Criterio de reentrenamiento

> **Alarma si F1_lote < F1_base − 0.10 durante 3 lotes consecutivos.**

* *Caída de 0.10*: el F1 de un lote de 400 jugadores (≈55 positivos) fluctúa ±0.06 solo por muestreo; 0.10 queda fuera del ruido.
* *3 lotes seguidos*: un bajón aislado (el lote 2 cae a 0.47 sin que nada cambie) no dispara la alarma.
* Calibración: en 300 secuencias de 12 lotes **sin** cambio de regla → **0 % de falsas alarmas**.
* Detección: el cambio ocurre en el lote 7; la alarma salta en el **lote 9**.

### Retraso de etiquetas

En producción la etiqueta de talento (preselección, convocatoria, rendimiento posterior) llega **semanas o temporadas después**.
Con un retraso de 3 lotes, la alarma del lote 9 sería visible recién en el **lote 12**: durante ese tiempo el modelo sigue
decidiendo con la regla vieja. Qué haría el equipo mientras tanto:

1. **No confiar en el data drift**: es ciego a un concept drift puro (lo demostramos: VERDE en los 12 lotes).
2. **Vigilar la salida del modelo**: tasa de positivos predichos y distribución de scores (cambian si los scouts empiezan a contradecir al modelo).
3. **Etiquetado muestral rápido**: los scouts validan una muestra aleatoria del 5–10 % de cada lote al instante (F1 ruidoso, pero con ventana de varios lotes sirve de alerta temprana).
4. **Señales proxy**: tasa de sobreescritura del modelo por scouts, convocatorias tempranas, resultados de la siguiente copa.
5. **Champion-challenger en sombra**: entrenar un challenger con las etiquetas más recientes y compararlo en paralelo; promoverlo (alias `champion`) cuando supere al actual. **Implementado**: `python -m src.training.reentrenar` arma `datos-v2` (lotes 7-9 / 10 / 11-12) y registra **v3 `challenger`**: F1 0.519 vs 0.401 del champion sobre el mismo test ([reentrenamiento.json](evidencias/reentrenamiento.json)). Se **promovió** a `champion` (`src/training/promover.py`) y se desplegó como `modelo-v3`.

> Con v3 en producción, la **referencia del data drift pasa a `data/versiones/datos-v2/train.csv`**. Las pruebas y la calibración documentadas arriba corresponden al período de v2 (referencia datos-v1), y el análisis de concept drift evalúa la copia congelada de v2 (`artifacts/modelos/v2`).
6. **Decisiones reversibles**: usar el modelo para *priorizar* la lista de scouting, no para descartar jugadores, mientras no haya etiquetas frescas.
