# Imagen de inferencia: modelo champion + API. Sin nada instalado en el anfitrión.
FROM python:3.13-slim

# libgomp: requerida por LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero: capa cacheada mientras requirements-serving.txt no cambie
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Código + artefacto del modelo exportado desde models:/talento-sc@champion
COPY src/__init__.py src/__init__.py
COPY src/common src/common
COPY src/serving/__init__.py src/serving/__init__.py
COPY src/serving/api.py src/serving/api.py
COPY src/serving/model src/serving/model

RUN useradd --create-home --uid 1000 app
USER app

ENV MLFLOW_DISABLE_TELEMETRY=true
EXPOSE 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
