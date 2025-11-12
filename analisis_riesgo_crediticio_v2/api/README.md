# 🚀 API de Análisis de Riesgo Crediticio

API REST desarrollada con FastAPI para predicción de riesgo crediticio usando Machine Learning.

## 📋 Características

- ✅ Predicciones en tiempo real
- ✅ Predicciones por lote (batch)
- ✅ Health checks y monitoreo
- ✅ Documentación automática (Swagger/ReDoc)
- ✅ Validación de datos con Pydantic
- ✅ Manejo robusto de errores
- ✅ Logging completo
- ✅ Soporte para CORS
- ✅ Dockerizado y listo para producción

## 🛠️ Instalación

### Opción 1: Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/A00739034/mlops-eq14.git
cd analisis_riesgo_crediticio_v2

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Docker

```bash
# Construir y ejecutar con Docker Compose
docker-compose up -d

# O construir imagen manualmente
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## 🚀 Uso

### Iniciar el servidor

```bash
# Desarrollo
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Producción
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Acceder a la documentación

Una vez iniciado el servidor:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoints

### 1. Health Check

```bash
GET /health
```

Respuesta:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2025-11-09T10:30:00"
}
```

### 2. Información del Modelo

```bash
GET /model/info
```

### 3. Predicción Individual

```bash
POST /predict
Content-Type: application/json

{
  "age": 35,
  "gender": 1,
  "income": 5000.0,
  "employment_type": 2,
  "credit_history": 3,
  "loan_amount": 15000.0,
  "loan_term": 36,
  "existing_loans": 1,
  "debt_to_income_ratio": 0.3
}
```

Respuesta:
```json
{
  "prediction": 1,
  "probability": 0.8542,
  "risk_level": "Bajo",
  "recommendation": "Aprobación recomendada con condiciones estándar",
  "timestamp": "2025-11-09T10:30:00",
  "model_version": "1.0.0"
}
```

### 4. Predicción por Lote

```bash
POST /predict/batch
Content-Type: application/json

[
  {
    "age": 35,
    "gender": 1,
    "income": 5000.0,
    ...
  },
  {
    "age": 28,
    "gender": 0,
    "income": 3500.0,
    ...
  }
]
```

### 5. Recargar Modelo

```bash
POST /model/reload
```

## 🧪 Ejemplos de Uso

### Python

```python
import requests

url = "http://localhost:8000/predict"

data = {
    "age": 35,
    "gender": 1,
    "income": 5000.0,
    "employment_type": 2,
    "credit_history": 3,
    "loan_amount": 15000.0,
    "loan_term": 36,
    "existing_loans": 1,
    "debt_to_income_ratio": 0.3
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": 1,
    "income": 5000.0,
    "employment_type": 2,
    "credit_history": 3,
    "loan_amount": 15000.0,
    "loan_term": 36,
    "existing_loans": 1,
    "debt_to_income_ratio": 0.3
  }'
```

### JavaScript

```javascript
const data = {
  age: 35,
  gender: 1,
  income: 5000.0,
  employment_type: 2,
  credit_history: 3,
  loan_amount: 15000.0,
  loan_term: 36,
  existing_loans: 1,
  debt_to_income_ratio: 0.3
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
})
  .then(response => response.json())
  .then(data => console.log(data));
```

## 📊 Campos de Entrada

| Campo | Tipo | Descripción | Rango |
|-------|------|-------------|-------|
| age | int | Edad del solicitante | 18-100 |
| gender | int | Género (0=Femenino, 1=Masculino) | 0-1 |
| income | float | Ingreso mensual | > 0 |
| employment_type | int | Tipo de empleo | 0-5 |
| credit_history | int | Historial crediticio | 0-4 |
| loan_amount | float | Monto del préstamo | > 0 |
| loan_term | int | Plazo en meses | 1-360 |
| existing_loans | int | Préstamos existentes | ≥ 0 |
| debt_to_income_ratio | float | Ratio deuda-ingreso | 0-1 |

## 📊 Niveles de Riesgo

| Probabilidad | Nivel de Riesgo | Descripción |
|--------------|-----------------|-------------|
| ≥ 0.8 | Bajo | Alta probabilidad de aprobación |
| 0.6 - 0.8 | Medio-Bajo | Buena probabilidad de aprobación |
| 0.4 - 0.6 | Medio | Probabilidad moderada |
| 0.2 - 0.4 | Medio-Alto | Baja probabilidad de aprobación |
| < 0.2 | Alto | Muy baja probabilidad de aprobación |

## 🔧 Configuración

### Variables de Entorno

```bash
# Archivo .env
MODEL_PATH=models/best_model.joblib
LOG_LEVEL=info
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=api tests/
```

## 📦 Deployment

### AWS EC2

```bash
# Conectar a instancia
ssh -i key.pem ubuntu@your-ec2-instance

# Clonar repositorio
git clone https://github.com/A00739034/mlops-eq14.git
cd analisis_riesgo_crediticio_v2

# Ejecutar con Docker
docker-compose up -d

# Verificar
curl http://localhost:8000/health
```

### AWS ECS/Fargate

```bash
# Construir y subir imagen a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t credit-risk-api .
docker tag credit-risk-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest
```

### Heroku

```bash
# Login a Heroku
heroku login
heroku container:login

# Crear app
heroku create credit-risk-api

# Push imagen
heroku container:push web --app credit-risk-api
heroku container:release web --app credit-risk-api

# Abrir app
heroku open --app credit-risk-api
```

## 🔒 Seguridad

- ✅ Validación de entrada con Pydantic
- ✅ Manejo de errores robusto
- ✅ Rate limiting (configurar con Nginx/CloudFlare)
- ✅ CORS configurado
- ⚠️ En producción: usar HTTPS
- ⚠️ En producción: implementar autenticación (JWT, API Keys)

## 📈 Monitoreo

### Prometheus Metrics (Opcional)

```bash
# Instalar prometheus-fastapi-instrumentator
pip install prometheus-fastapi-instrumentator

# Agregar a main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Logging

Los logs se guardan automáticamente. Para ver en tiempo real:

```bash
# Con Docker
docker logs -f credit_risk_api

# Local
tail -f logs/api.log
```

## 🐛 Troubleshooting

### Error: Modelo no encontrado

```bash
# Verificar que el modelo existe
ls -la models/best_model.joblib

# Si no existe, entrenar el modelo primero
python src/models/train_model.py
```

### Error: Puerto 8000 en uso

```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## 📚 Documentación Adicional

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Documentation](https://docs.docker.com/)

## 👥 Equipo

- **Equipo 14** - MLOps
- **Proyecto**: Análisis de Riesgo Crediticio

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Para soporte, por favor abre un issue en GitHub o contacta al equipo.

---

**Hecho con ❤️ por Equipo 14 - MLOps**
