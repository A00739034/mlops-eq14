# -*- coding: utf-8 -*-
"""
Script para subir el proyecto completo a MLflow.

Este script ejecuta el pipeline completo y registra todo en MLflow,
incluyendo modelos, métricas, parámetros y artefactos.
"""

import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import argparse

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from mlflow_config import setup_mlflow_for_project, get_project_config
from main_pipeline import MLPipeline


class MLflowUploader:
    """
    Clase para subir el proyecto completo a MLflow.
    """
    
    def __init__(self, environment: str = 'local'):
        """
        Inicializa el uploader de MLflow.
        
        Args:
            environment: Entorno MLflow a usar
        """
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Configurar MLflow
        self.project_config = setup_mlflow_for_project(environment)
        
        # Cliente MLflow
        self.client = MlflowClient()
        
        # Pipeline
        self.pipeline = MLPipeline()
    
    def upload_complete_project(self, input_file: str = None):
        """
        Sube el proyecto completo a MLflow.
        
        Args:
            input_file: Archivo de datos de entrada
        """
        self.logger.info("=== INICIANDO SUBIDA COMPLETA A MLFLOW ===")
        
        input_file = input_file or "data/raw/german_credit_modified.csv"
        
        # Ejecutar pipeline completo
        pipeline_results = self.pipeline.run_full_pipeline(input_file)
        
        # Subir resultados a MLflow
        self._upload_pipeline_results(pipeline_results)
        
        # Registrar mejores modelos
        self._register_best_models()
        
        # Crear reporte final
        self._create_final_report()
        
        self.logger.info("=== SUBIDA COMPLETA A MLFLOW FINALIZADA ===")
    
    def _upload_pipeline_results(self, results: dict):
        """
        Sube los resultados del pipeline a MLflow.
        
        Args:
            results: Resultados del pipeline
        """
        self.logger.info("Subiendo resultados del pipeline a MLflow")
        
        with mlflow.start_run(run_name="Complete_Pipeline_Run"):
            # Log de parámetros generales
            mlflow.log_param("pipeline_version", "1.0.0")
            mlflow.log_param("total_time_minutes", results.get('total_time_minutes', 0))
            mlflow.log_param("environment", self.environment)
            
            # Log de tags del proyecto
            project_tags = self.project_config['tags']
            for key, value in project_tags.items():
                mlflow.set_tag(key, value)
            
            # Log de resultados de procesamiento de datos
            if 'data_processing' in results['results']:
                data_info = results['results']['data_processing']
                mlflow.log_param("data_shape", f"{data_info['output_shape'][0]}x{data_info['output_shape'][1]}")
                mlflow.log_param("missing_values", data_info['summary']['missing_values'])
                mlflow.log_param("duplicates", data_info['summary']['duplicates'])
            
            # Log de resultados de ingeniería de características
            if 'feature_engineering' in results['results']:
                feature_info = results['results']['feature_engineering']
                mlflow.log_param("original_features", feature_info['original_features'])
                mlflow.log_param("transformed_features", feature_info['transformed_features'])
                mlflow.log_param("selected_features", feature_info['selected_features'])
            
            # Log de resultados de modelos
            if 'model_evaluation' in results['results']:
                model_results = results['results']['model_evaluation']
                
                # Encontrar el mejor modelo
                best_model_name = None
                best_roc_auc = 0
                
                for model_name, model_data in model_results.items():
                    metrics = model_data['metrics']
                    roc_auc = metrics.get('roc_auc', 0)
                    
                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_model_name = model_name
                    
                    # Log de métricas de cada modelo
                    mlflow.log_metric(f"{model_name}_roc_auc", roc_auc)
                    mlflow.log_metric(f"{model_name}_precision", metrics.get('precision', 0))
                    mlflow.log_metric(f"{model_name}_recall", metrics.get('recall', 0))
                    mlflow.log_metric(f"{model_name}_f1", metrics.get('f1', 0))
                
                # Log del mejor modelo
                if best_model_name:
                    mlflow.log_param("best_model", best_model_name)
                    mlflow.log_metric("best_roc_auc", best_roc_auc)
            
            # Log de artefactos
            self._log_artifacts()
            
            # Log de archivos de visualización
            if 'visualization' in results['results']:
                viz_info = results['results']['visualization']
                for file_path in viz_info['visualization_files']:
                    if Path(file_path).exists():
                        mlflow.log_artifact(file_path)
    
    def _log_artifacts(self):
        """
        Registra artefactos importantes en MLflow.
        """
        # Log de archivos de datos procesados
        data_files = [
            "data/processed/processed_data.csv",
            "data/processed/features_data.csv"
        ]
        
        for file_path in data_files:
            if Path(file_path).exists():
                mlflow.log_artifact(file_path)
        
        # Log de modelos
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.joblib"):
                mlflow.log_artifact(str(model_file))
        
        # Log de reportes
        reports_dir = Path("reports")
        if reports_dir.exists():
            for report_file in reports_dir.glob("*.json"):
                mlflow.log_artifact(str(report_file))
    
    def _register_best_models(self):
        """
        Registra los mejores modelos en el Model Registry.
        """
        self.logger.info("Registrando mejores modelos en Model Registry")
        
        # Obtener el mejor modelo
        best_run = self._get_best_run()
        
        if best_run:
            # Registrar modelo de staging
            staging_model_name = self.project_config['model_registry']['staging_model_name']
            staging_uri = mlflow.register_model(
                model_uri=f"runs:/{best_run['run_id']}/model",
                name=staging_model_name,
                description=self.project_config['model_registry']['description']
            )
            
            self.logger.info(f"Modelo registrado en staging: {staging_model_name} (versión {staging_uri.version})")
            
            # Promover a staging
            self.client.transition_model_version_stage(
                name=staging_model_name,
                version=staging_uri.version,
                stage="Staging"
            )
            
            self.logger.info(f"Modelo promovido a Staging: {staging_model_name}")
    
    def _get_best_run(self) -> dict:
        """
        Obtiene la mejor ejecución basada en ROC-AUC.
        
        Returns:
            Diccionario con información de la mejor ejecución
        """
        primary_metric = self.project_config['metrics']['primary_metric']
        
        # Obtener el experimento actual
        experiment = mlflow.get_experiment_by_name(self.project_config['experiments']['model_comparison'])
        if not experiment:
            self.logger.warning("No se encontró el experimento")
            return {}
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{primary_metric} DESC"]
        )
        
        if runs:
            best_run = runs[0]
            return {
                'run_id': best_run.info.run_id,
                'run_name': best_run.data.tags.get('mlflow.runName', ''),
                'best_metric': best_run.data.metrics.get(primary_metric, 0),
                'metrics': best_run.data.metrics
            }
        
        return {}
    
    def _create_final_report(self):
        """
        Crea un reporte final del proyecto en MLflow.
        """
        self.logger.info("Creando reporte final")
        
        # Crear reporte Markdown
        report_content = f"""# Reporte Final - Proyecto German Credit Risk

## Información del Proyecto
- **Fecha de ejecución**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Entorno MLflow**: {self.environment}
- **Experimento**: {self.project_config['experiments']['model_comparison']}

## Configuración del Proyecto
- **Métrica principal**: {self.project_config['metrics']['primary_metric']}
- **Métricas secundarias**: {', '.join(self.project_config['metrics']['secondary_metrics'])}
- **Modelos registrados**: 
  - Staging: {self.project_config['model_registry']['staging_model_name']}
  - Production: {self.project_config['model_registry']['production_model_name']}

## Tags del Proyecto
"""
        
        for key, value in self.project_config['tags'].items():
            report_content += f"- **{key}**: {value}\n"
        
        report_content += """
## Próximos Pasos
1. Revisar métricas en MLflow UI
2. Evaluar modelo en staging
3. Promover a producción si cumple criterios
4. Configurar monitoreo en producción
"""
        
        # Guardar reporte
        report_path = Path("reports/mlflow_project_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Log del reporte
        mlflow.log_artifact(str(report_path))
        
        self.logger.info(f"Reporte final creado: {report_path}")
    
    def list_experiments(self):
        """
        Lista todos los experimentos disponibles.
        """
        experiments = self.client.search_experiments()
        
        print("\n📋 Experimentos disponibles en MLflow:")
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
            print(f"    Ubicación: {exp.artifact_location}")
            print(f"    Estado: {exp.lifecycle_stage}")
            print()
    
    def list_models(self):
        """
        Lista todos los modelos registrados.
        """
        try:
            models = self.client.search_registered_models()
            
            print("\n🤖 Modelos registrados en MLflow:")
            for model in models:
                print(f"  - {model.name}")
                print(f"    Descripción: {model.description}")
                print(f"    Versiones: {len(model.latest_versions)}")
                
                for version in model.latest_versions:
                    print(f"      * Versión {version.version} ({version.current_stage})")
                print()
        except Exception as e:
            print(f"❌ Error listando modelos: {str(e)}")
    
    def promote_model_to_production(self, model_name: str = None):
        """
        Promueve un modelo a producción.
        
        Args:
            model_name: Nombre del modelo a promover
        """
        model_name = model_name or self.project_config['model_registry']['production_model_name']
        
        try:
            # Obtener la última versión del modelo
            latest_version = self.client.get_latest_versions(model_name)[0]
            
            # Promover a producción
            self.client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production"
            )
            
            self.logger.info(f"Modelo promovido a Production: {model_name} (versión {latest_version.version})")
            
        except Exception as e:
            self.logger.error(f"Error promoviendo modelo: {str(e)}")


def main():
    """Función principal para subir proyecto a MLflow."""
    parser = argparse.ArgumentParser(description="Subir proyecto German Credit Risk a MLflow")
    parser.add_argument("--environment", choices=['local', 'remote', 'aws', 'azure', 'gcp'],
                       help="Entorno MLflow", default='local')
    parser.add_argument("--input", help="Archivo de datos de entrada",
                       default="data/raw/german_credit_modified.csv")
    parser.add_argument("--action", choices=["upload", "list", "models", "promote"],
                       help="Acción a ejecutar", default="upload")
    parser.add_argument("--model-name", help="Nombre del modelo para promover",
                       default="german_credit_production")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear uploader
    uploader = MLflowUploader(args.environment)
    
    try:
        if args.action == "upload":
            uploader.upload_complete_project(args.input)
            print("\n✅ Proyecto subido exitosamente a MLflow!")
            print(f"🌐 Accede a MLflow UI en: {mlflow.get_tracking_uri()}")
        
        elif args.action == "list":
            uploader.list_experiments()
        
        elif args.action == "models":
            uploader.list_models()
        
        elif args.action == "promote":
            uploader.promote_model_to_production(args.model_name)
            print(f"✅ Modelo {args.model_name} promovido a producción")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logging.error(f"Error ejecutando acción {args.action}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
