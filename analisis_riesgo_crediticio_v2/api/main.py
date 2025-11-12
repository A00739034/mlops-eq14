"""
API FastAPI para predicción de riesgo crediticio
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="API de Análisis de Riesgo Crediticio",
    description="API para predicción de riesgo crediticio usando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
MODEL = None
MODEL_VERSION = None
MODEL_LOADED_AT = None

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class CreditApplicationInput(BaseModel):
    """
    Modelo Pydantic para validar los datos de entrada
    """
    # Información personal
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    gender: int = Field(..., ge=0, le=1, description="Género (0=Femenino, 1=Masculino)")
    
    # Información laboral
    income: float = Field(..., gt=0, description="Ingreso mensual")
    employment_type: int = Field(..., ge=0, le=5, description="Tipo de empleo")
    
    # Historial crediticio
    credit_history: int = Field(..., ge=0, le=4, description="Historial crediticio")
    loan_amount: float = Field(..., gt=0, description="Monto del préstamo solicitado")
    loan_term: int = Field(..., gt=0, le=360, description="Plazo del préstamo en meses")
    
    # Información financiera
    existing_loans: int = Field(..., ge=0, description="Número de préstamos existentes")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Ratio deuda-ingreso")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """
    Modelo Pydantic para la respuesta de predicción
    """
    prediction: int = Field(..., description="Predicción (0=Rechazado, 1=Aprobado)")
    probability: float = Field(..., description="Probabilidad de aprobación")
    risk_level: str = Field(..., description="Nivel de riesgo")
    recommendation: str = Field(..., description="Recomendación")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    model_version: Optional[str] = Field(None, description="Versión del modelo")


class HealthResponse(BaseModel):
    """
    Modelo para respuesta de health check
    """
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


def load_model():
    """
    Cargar el modelo entrenado
    """
    global MODEL, MODEL_VERSION, MODEL_LOADED_AT
    
    try:
        model_path = MODELS_DIR / "best_model.joblib"
        
        if not model_path.exists():
            logger.error(f"Modelo no encontrado en: {model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        MODEL = joblib.load(model_path)
        MODEL_VERSION = "1.0.0"
        MODEL_LOADED_AT = datetime.now().isoformat()
        
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
        logger.info(f"Versión del modelo: {MODEL_VERSION}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error cargando el modelo: {str(e)}")
        return False


def get_risk_level(probability: float) -> str:
    """
    Determinar el nivel de riesgo basado en la probabilidad
    """
    if probability >= 0.8:
        return "Bajo"
    elif probability >= 0.6:
        return "Medio-Bajo"
    elif probability >= 0.4:
        return "Medio"
    elif probability >= 0.2:
        return "Medio-Alto"
    else:
        return "Alto"


def get_recommendation(prediction: int, probability: float, risk_level: str) -> str:
    """
    Generar recomendación basada en la predicción
    """
    if prediction == 1 and probability >= 0.8:
        return "Aprobación recomendada con condiciones estándar"
    elif prediction == 1 and probability >= 0.6:
        return "Aprobación recomendada con análisis adicional"
    elif prediction == 1:
        return "Aprobación condicional - Requiere garantías adicionales"
    elif prediction == 0 and probability < 0.3:
        return "Rechazar - Alto riesgo de incumplimiento"
    else:
        return "Rechazar - Riesgo moderado, considerar en el futuro"


@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio - Cargar el modelo al iniciar la API
    """
    logger.info("Iniciando API de Análisis de Riesgo Crediticio...")
    
    if not load_model():
        logger.warning("La API se inició pero el modelo no pudo cargarse")
    else:
        logger.info("API iniciada correctamente con modelo cargado")


@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz - Información básica de la API
    """
    return {
        "name": "API de Análisis de Riesgo Crediticio",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check - Verificar estado de la API y el modelo
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Obtener información sobre el modelo cargado
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado"
        )
    
    return {
        "model_version": MODEL_VERSION,
        "model_loaded_at": MODEL_LOADED_AT,
        "model_type": str(type(MODEL)),
        "features_required": [
            "age", "gender", "income", "employment_type",
            "credit_history", "loan_amount", "loan_term",
            "existing_loans", "debt_to_income_ratio"
        ]
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_credit_risk(application: CreditApplicationInput):
    """
    Realizar predicción de riesgo crediticio
    
    Args:
        application: Datos de la solicitud de crédito
        
    Returns:
        Predicción con probabilidad y recomendación
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. El servicio está iniciando o hubo un error al cargar el modelo."
        )
    
    try:
        # Preparar datos para predicción
        input_data = pd.DataFrame([application.dict()])
        
        logger.info(f"Realizando predicción para solicitud: {application.dict()}")
        
        # Realizar predicción
        prediction = int(MODEL.predict(input_data)[0])
        probability = float(MODEL.predict_proba(input_data)[0][1])
        
        # Determinar nivel de riesgo y recomendación
        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(prediction, probability, risk_level)
        
        logger.info(f"Predicción completada: prediction={prediction}, probability={probability:.4f}")
        
        return PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
            model_version=MODEL_VERSION
        )
    
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(applications: List[CreditApplicationInput]):
    """
    Realizar predicciones por lote
    
    Args:
        applications: Lista de solicitudes de crédito
        
    Returns:
        Lista de predicciones
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    if len(applications) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Máximo 100 solicitudes por lote"
        )
    
    try:
        predictions = []
        
        for app in applications:
            input_data = pd.DataFrame([app.dict()])
            prediction = int(MODEL.predict(input_data)[0])
            probability = float(MODEL.predict_proba(input_data)[0][1])
            risk_level = get_risk_level(probability)
            recommendation = get_recommendation(prediction, probability, risk_level)
            
            predictions.append({
                "input": app.dict(),
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": risk_level,
                "recommendation": recommendation
            })
        
        return {
            "total_predictions": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions
        }
    
    except Exception as e:
        logger.error(f"Error en predicción por lote: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar predicciones: {str(e)}"
        )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Recargar el modelo (útil después de actualizaciones)
    """
    logger.info("Intentando recargar el modelo...")
    
    if load_model():
        return {
            "status": "success",
            "message": "Modelo recargado exitosamente",
            "model_version": MODEL_VERSION,
            "loaded_at": MODEL_LOADED_AT
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al recargar el modelo"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Manejador global de excepciones
    """
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
