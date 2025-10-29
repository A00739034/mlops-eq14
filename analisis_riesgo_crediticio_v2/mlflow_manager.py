# -*- coding: utf-8 -*-
"""
MLflow Configuration: Script para configurar y gestionar MLflow con servidor remoto.

Este script permite configurar MLflow para usar un servidor remoto,
subir experimentos y gestionar el tracking de modelos.
"""

import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.tracking import MlflowClient
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse


class MLflowManager:
    """
    Gestor de MLflow para configurar tracking remoto y gestionar experimentos.
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "german_credit_risk"):
        """
        Inicializa el gestor de MLflow.
        
        Args:
            tracking_uri: URI del servidor de MLflow (ej: http://localhost:5000)
            experiment_name: Nombre del experimento
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Configurar MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Crear o obtener experimento
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            self.logger.info(f"Experimento creado: {self.experiment_name} (ID: {self.experiment_id})")
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            self.logger.info(f"Experimento existente encontrado: {self.experiment_name} (ID: {self.experiment_id})")
        
        mlflow.set_experiment(self.experiment_name)
        
        # Cliente MLflow
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Lista todos los experimentos disponibles.
        
        Returns:
            Lista de experimentos con sus metadatos
        """
        experiments = self.client.search_experiments()
        experiment_list = []
        
        for exp in experiments:
            experiment_list.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
                'creation_time': exp.creation_time
            })
        
        return experiment_list
    
    def list_runs(self, experiment_id: str = None) -> List[Dict[str, Any]]:
        """
        Lista todas las ejecuciones de un experimento.
        
        Args:
            experiment_id: ID del experimento (opcional)
            
        Returns:
            Lista de ejecuciones con sus metadatos
        """
        experiment_id = experiment_id or self.experiment_id
        runs = self.client.search_runs(experiment_ids=[experiment_id])
        
        run_list = []
        for run in runs:
            run_list.append({
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', ''),
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags
            })
        
        return run_list
    
    def get_best_run(self, metric_name: str = "roc_auc", experiment_id: str = None) -> Dict[str, Any]:
        """
        Obtiene la mejor ejecución basada en una métrica.
        
        Args:
            metric_name: Nombre de la métrica para comparar
            experiment_id: ID del experimento
            
        Returns:
            Diccionario con información de la mejor ejecución
        """
        experiment_id = experiment_id or self.experiment_id
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} DESC"]
        )
        
        if runs:
            best_run = runs[0]
            return {
                'run_id': best_run.info.run_id,
                'run_name': best_run.data.tags.get('mlflow.runName', ''),
                'best_metric': best_run.data.metrics.get(metric_name, 0),
                'metrics': best_run.data.metrics,
                'params': best_run.data.params,
                'model_uri': f"runs:/{best_run.info.run_id}/model"
            }
        
        return {}
    
    def register_model(self, run_id: str, model_name: str, description: str = None) -> str:
        """
        Registra un modelo en el Model Registry de MLflow.
        
        Args:
            run_id: ID de la ejecución que contiene el modelo
            model_name: Nombre del modelo a registrar
            description: Descripción del modelo
            
        Returns:
            URI del modelo registrado
        """
        model_uri = f"runs:/{run_id}/model"
        
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )
            
            self.logger.info(f"Modelo registrado: {registered_model.name} (versión {registered_model.version})")
            return f"models:/{model_name}/latest"
            
        except Exception as e:
            self.logger.error(f"Error registrando modelo: {str(e)}")
            raise
    
    def promote_model_to_staging(self, model_name: str, version: str = None) -> bool:
        """
        Promueve un modelo a staging.
        
        Args:
            model_name: Nombre del modelo
            version: Versión específica (opcional)
            
        Returns:
            True si se promovió exitosamente
        """
        try:
            if version:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Staging"
                )
            else:
                # Obtener la última versión
                latest_version = self.client.get_latest_versions(model_name)[0]
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging"
                )
            
            self.logger.info(f"Modelo {model_name} promovido a Staging")
            return True
            
        except Exception as e:
            self.logger.error(f"Error promoviendo modelo: {str(e)}")
            return False
    
    def promote_model_to_production(self, model_name: str, version: str = None) -> bool:
        """
        Promueve un modelo a producción.
        
        Args:
            model_name: Nombre del modelo
            version: Versión específica (opcional)
            
        Returns:
            True si se promovió exitosamente
        """
        try:
            if version:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production"
                )
            else:
                # Obtener la última versión
                latest_version = self.client.get_latest_versions(model_name)[0]
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production"
                )
            
            self.logger.info(f"Modelo {model_name} promovido a Production")
            return True
            
        except Exception as e:
            self.logger.error(f"Error promoviendo modelo: {str(e)}")
            return False
    
    def export_experiment(self, output_file: str = "mlflow_export.json") -> str:
        """
        Exporta los metadatos del experimento a un archivo JSON.
        
        Args:
            output_file: Nombre del archivo de salida
            
        Returns:
            Ruta del archivo exportado
        """
        export_data = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'tracking_uri': self.tracking_uri,
            'runs': self.list_runs(),
            'best_run': self.get_best_run()
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Experimento exportado a: {output_path}")
        return str(output_path)
    
    def create_model_comparison_report(self, output_file: str = "model_comparison_report.md") -> str:
        """
        Crea un reporte de comparación de modelos.
        
        Args:
            output_file: Nombre del archivo de salida
            
        Returns:
            Ruta del archivo de reporte
        """
        runs = self.list_runs()
        
        if not runs:
            self.logger.warning("No hay ejecuciones para comparar")
            return ""
        
        # Crear reporte Markdown
        report_content = f"""# Reporte de Comparación de Modelos - {self.experiment_name}

## Resumen del Experimento
- **Experimento ID**: {self.experiment_id}
- **Total de ejecuciones**: {len(runs)}
- **Tracking URI**: {self.tracking_uri}

## Comparación de Modelos

| Modelo | ROC-AUC | Precision | Recall | F1-Score | Parámetros |
|--------|---------|-----------|--------|----------|------------|
"""
        
        for run in runs:
            metrics = run.get('metrics', {})
            params = run.get('params', {})
            model_name = run.get('run_name', run.get('run_id', 'Unknown'))
            
            report_content += f"| {model_name} | "
            report_content += f"{metrics.get('roc_auc', 'N/A'):.3f} | "
            report_content += f"{metrics.get('precision', 'N/A'):.3f} | "
            report_content += f"{metrics.get('recall', 'N/A'):.3f} | "
            report_content += f"{metrics.get('f1', 'N/A'):.3f} | "
            report_content += f"{params.get('model_type', 'N/A')} |\n"
        
        # Mejor modelo
        best_run = self.get_best_run()
        if best_run:
            report_content += f"""
## Mejor Modelo
- **Run ID**: {best_run.get('run_id', 'N/A')}
- **Modelo**: {best_run.get('run_name', 'N/A')}
- **Mejor ROC-AUC**: {best_run.get('best_metric', 'N/A'):.3f}

## Recomendaciones
1. El modelo con mejor rendimiento es: {best_run.get('run_name', 'N/A')}
2. Considerar promover este modelo a staging para pruebas adicionales
3. Monitorear el rendimiento en datos de producción
"""
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Reporte de comparación creado en: {output_path}")
        return str(output_path)


def setup_remote_mlflow(tracking_uri: str, experiment_name: str = "german_credit_risk"):
    """
    Configura MLflow para usar un servidor remoto.
    
    Args:
        tracking_uri: URI del servidor MLflow (ej: http://localhost:5000)
        experiment_name: Nombre del experimento
    """
    # Configurar variables de entorno
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    
    # Crear gestor
    mlflow_manager = MLflowManager(tracking_uri, experiment_name)
    
    print(f"✅ MLflow configurado para servidor remoto: {tracking_uri}")
    print(f"📊 Experimento: {experiment_name}")
    
    return mlflow_manager


def main():
    """Función principal para gestionar MLflow desde línea de comandos."""
    parser = argparse.ArgumentParser(description="Gestor de MLflow para German Credit Risk")
    parser.add_argument("--tracking-uri", help="URI del servidor MLflow", 
                       default="http://localhost:5000")
    parser.add_argument("--experiment", help="Nombre del experimento", 
                       default="german_credit_risk")
    parser.add_argument("--action", choices=["list", "best", "export", "report", "register"],
                       help="Acción a ejecutar", default="list")
    parser.add_argument("--model-name", help="Nombre del modelo para registrar", 
                       default="german_credit_model")
    parser.add_argument("--output", help="Archivo de salida", 
                       default="mlflow_output")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear gestor MLflow
    mlflow_manager = MLflowManager(args.tracking_uri, args.experiment)
    
    try:
        if args.action == "list":
            print("\n📋 Experimentos disponibles:")
            experiments = mlflow_manager.list_experiments()
            for exp in experiments:
                print(f"  - {exp['name']} (ID: {exp['experiment_id']})")
            
            print(f"\n🏃 Ejecuciones en {args.experiment}:")
            runs = mlflow_manager.list_runs()
            for run in runs[:5]:  # Mostrar solo las primeras 5
                print(f"  - {run['run_name']} (ROC-AUC: {run['metrics'].get('roc_auc', 'N/A'):.3f})")
        
        elif args.action == "best":
            best_run = mlflow_manager.get_best_run()
            if best_run:
                print(f"\n🏆 Mejor modelo:")
                print(f"  - Run ID: {best_run['run_id']}")
                print(f"  - Modelo: {best_run['run_name']}")
                print(f"  - ROC-AUC: {best_run['best_metric']:.3f}")
            else:
                print("❌ No se encontraron ejecuciones")
        
        elif args.action == "export":
            output_file = f"{args.output}.json"
            mlflow_manager.export_experiment(output_file)
            print(f"✅ Experimento exportado a: {output_file}")
        
        elif args.action == "report":
            output_file = f"{args.output}.md"
            mlflow_manager.create_model_comparison_report(output_file)
            print(f"✅ Reporte creado en: {output_file}")
        
        elif args.action == "register":
            best_run = mlflow_manager.get_best_run()
            if best_run:
                model_uri = mlflow_manager.register_model(
                    best_run['run_id'], 
                    args.model_name,
                    "Mejor modelo de riesgo crediticio basado en German Credit dataset"
                )
                print(f"✅ Modelo registrado: {model_uri}")
            else:
                print("❌ No se encontró el mejor modelo para registrar")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logging.error(f"Error ejecutando acción {args.action}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
