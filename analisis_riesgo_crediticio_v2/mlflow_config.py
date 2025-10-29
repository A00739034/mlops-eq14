# -*- coding: utf-8 -*-
"""
Configuración de MLflow para el proyecto de análisis de riesgo crediticio.

Este archivo contiene configuraciones para diferentes entornos de MLflow:
- Local (desarrollo)
- Remoto (staging/producción)
- Cloud (AWS, Azure, GCP)
"""

import os
from pathlib import Path
from typing import Dict, Any


class MLflowConfig:
    """Configuración centralizada para MLflow."""
    
    # Configuraciones por defecto
    DEFAULT_CONFIG = {
        'experiment_name': 'german_credit_risk',
        'artifact_location': './mlruns',
        'run_name_prefix': 'german_credit',
        'tags': {
            'project': 'german_credit_risk',
            'team': 'mlops_team',
            'version': '1.0.0'
        }
    }
    
    # Configuraciones por entorno
    ENVIRONMENTS = {
        'local': {
            'tracking_uri': 'file:./mlruns',
            'artifact_location': './mlruns',
            'registry_uri': None
        },
        'remote': {
            'tracking_uri': 'http://localhost:5000',
            'artifact_location': './mlruns',
            'registry_uri': 'http://localhost:5000'
        },
        'aws': {
            'tracking_uri': 'https://your-mlflow-server.amazonaws.com',
            'artifact_location': 's3://your-mlflow-bucket/mlruns',
            'registry_uri': 'https://your-mlflow-server.amazonaws.com'
        },
        'azure': {
            'tracking_uri': 'https://your-mlflow-server.azurewebsites.net',
            'artifact_location': 'wasbs://your-container@your-storage.blob.core.windows.net/mlruns',
            'registry_uri': 'https://your-mlflow-server.azurewebsites.net'
        },
        'gcp': {
            'tracking_uri': 'https://your-mlflow-server.run.app',
            'artifact_location': 'gs://your-bucket/mlruns',
            'registry_uri': 'https://your-mlflow-server.run.app'
        }
    }
    
    @classmethod
    def get_config(cls, environment: str = 'local') -> Dict[str, Any]:
        """
        Obtiene la configuración para un entorno específico.
        
        Args:
            environment: Entorno ('local', 'remote', 'aws', 'azure', 'gcp')
            
        Returns:
            Diccionario con la configuración del entorno
        """
        if environment not in cls.ENVIRONMENTS:
            raise ValueError(f"Entorno no soportado: {environment}")
        
        config = cls.DEFAULT_CONFIG.copy()
        config.update(cls.ENVIRONMENTS[environment])
        
        return config
    
    @classmethod
    def setup_environment(cls, environment: str = 'local'):
        """
        Configura las variables de entorno para MLflow.
        
        Args:
            environment: Entorno a configurar
        """
        config = cls.get_config(environment)
        
        # Configurar variables de entorno
        os.environ['MLFLOW_TRACKING_URI'] = config['tracking_uri']
        
        if config.get('registry_uri'):
            os.environ['MLFLOW_REGISTRY_URI'] = config['registry_uri']
        
        # Configurar credenciales para cloud (si es necesario)
        if environment in ['aws', 'azure', 'gcp']:
            cls._setup_cloud_credentials(environment)
        
        print(f"✅ MLflow configurado para entorno: {environment}")
        print(f"📊 Tracking URI: {config['tracking_uri']}")
    
    @classmethod
    def _setup_cloud_credentials(cls, environment: str):
        """
        Configura credenciales para servicios cloud.
        
        Args:
            environment: Entorno cloud
        """
        if environment == 'aws':
            # Configurar credenciales AWS
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if not aws_access_key or not aws_secret_key:
                print("⚠️  Advertencia: Credenciales AWS no configuradas")
                print("   Configura AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY")
        
        elif environment == 'azure':
            # Configurar credenciales Azure
            azure_client_id = os.getenv('AZURE_CLIENT_ID')
            azure_client_secret = os.getenv('AZURE_CLIENT_SECRET')
            azure_tenant_id = os.getenv('AZURE_TENANT_ID')
            
            if not all([azure_client_id, azure_client_secret, azure_tenant_id]):
                print("⚠️  Advertencia: Credenciales Azure no configuradas")
                print("   Configura AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID")
        
        elif environment == 'gcp':
            # Configurar credenciales GCP
            gcp_service_account = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if not gcp_service_account:
                print("⚠️  Advertencia: Credenciales GCP no configuradas")
                print("   Configura GOOGLE_APPLICATION_CREDENTIALS")


# Configuraciones específicas del proyecto
PROJECT_CONFIG = {
    'model_registry': {
        'staging_model_name': 'german_credit_staging',
        'production_model_name': 'german_credit_production',
        'description': 'Modelo de predicción de riesgo crediticio basado en German Credit dataset'
    },
    'experiments': {
        'baseline': 'german_credit_baseline',
        'feature_engineering': 'german_credit_features',
        'hyperparameter_tuning': 'german_credit_tuning',
        'model_comparison': 'german_credit_comparison'
    },
    'metrics': {
        'primary_metric': 'roc_auc',
        'secondary_metrics': ['precision', 'recall', 'f1', 'average_precision'],
        'threshold_metrics': ['accuracy', 'specificity', 'sensitivity']
    },
    'tags': {
        'project': 'german_credit_risk',
        'team': 'mlops_team',
        'version': '1.0.0',
        'dataset': 'german_credit',
        'task': 'binary_classification',
        'domain': 'finance'
    }
}


def get_project_config() -> Dict[str, Any]:
    """
    Obtiene la configuración específica del proyecto.
    
    Returns:
        Diccionario con la configuración del proyecto
    """
    return PROJECT_CONFIG


def setup_mlflow_for_project(environment: str = 'local'):
    """
    Configura MLflow para el proyecto completo.
    
    Args:
        environment: Entorno a configurar
    """
    # Configurar entorno MLflow
    MLflowConfig.setup_environment(environment)
    
    # Obtener configuración del proyecto
    project_config = get_project_config()
    
    print(f"📊 Configuración del proyecto:")
    print(f"   - Experimento principal: {project_config['experiments']['model_comparison']}")
    print(f"   - Métrica principal: {project_config['metrics']['primary_metric']}")
    print(f"   - Modelos: {project_config['model_registry']['staging_model_name']}, {project_config['model_registry']['production_model_name']}")
    
    return project_config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configurar MLflow para German Credit Risk")
    parser.add_argument("--environment", choices=['local', 'remote', 'aws', 'azure', 'gcp'],
                       help="Entorno a configurar", default='local')
    parser.add_argument("--show-config", action='store_true',
                       help="Mostrar configuración actual")
    
    args = parser.parse_args()
    
    if args.show_config:
        config = MLflowConfig.get_config(args.environment)
        print(f"\nConfiguración para entorno '{args.environment}':")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        setup_mlflow_for_project(args.environment)
