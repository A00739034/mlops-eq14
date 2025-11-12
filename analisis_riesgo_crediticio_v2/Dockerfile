# Usar imagen oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/best_model.joblib

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
