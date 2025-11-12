"""
Configuración de la API
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Configuración de la aplicación
    """
    # API Settings
    API_TITLE: str = "API de Análisis de Riesgo Crediticio"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API para predicción de riesgo crediticio usando Machine Learning"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: str = "info"
    
    # Model Settings
    MODEL_PATH: str = "models/best_model.joblib"
    MODEL_VERSION: str = "1.0.0"
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    
    # Limits
    MAX_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Instancia global de configuración
settings = Settings()
