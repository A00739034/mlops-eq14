# =============================================================================
# Dockerfile para Pipeline de Análisis de Riesgo Crediticio
# Optimizado para reproducibilidad y consistencia entre entornos
# =============================================================================

# Usar imagen oficial de Python 3.11 (versión específica para reproducibilidad)
FROM python:3.11-slim

# Metadata
LABEL maintainer="MLOps Team"
LABEL description="Pipeline de análisis de riesgo crediticio con reproducibilidad garantizada"
LABEL version="1.0"

# Variables de entorno para reproducibilidad
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/root/.cache/pip

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias (mínimas para compilación)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip, setuptools y wheel
RUN pip install --upgrade pip setuptools wheel

# Copiar requirements primero (optimización de cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python (versiones exactas para reproducibilidad)
# Usar cache para acelerar builds subsecuentes
RUN pip install -r requirements.txt

# Verificar instalación de dependencias clave
RUN python -c "import pandas; import numpy; import sklearn; print('✓ Dependencias instaladas correctamente')"

# Copiar código de la aplicación
COPY . .

# Instalar paquete local en modo editable
RUN pip install -e .

# Crear directorios necesarios
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/data/interim \
             /app/models \
             /app/models/transformers \
             /app/reports \
             /app/reports/figures \
             /app/reports/reproducibility

# Hacer ejecutables los scripts
RUN chmod +x run_reproducibility_test.py \
            compare_reproducibility_results.py \
            quick_reproducibility_check.sh 2>/dev/null || true

# Exponer puerto para API (si se necesita)
EXPOSE 8000

# Script por defecto: ejecutar pipeline de reproducibilidad
# Puede ser sobrescrito con docker run o docker-compose
CMD ["python", "run_reproducibility_test.py", "--seed", "42"]
