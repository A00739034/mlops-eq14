# -*- coding: utf-8 -*-
"""
Pipeline Principal: Script para ejecutar el pipeline completo de análisis de riesgo crediticio.

Este script orquesta todas las etapas del pipeline: procesamiento de datos,
ingeniería de características, entrenamiento de modelos y generación de reportes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse
import json
from datetime import datetime
import sys
import os

# Agregar el directorio src al path para importar módulos
sys.path.append(str(Path(__file__).parent / "src"))

# Importar módulo de reproducibilidad PRIMERO para configurar semillas
from utils.reproducibility import set_seed, DEFAULT_RANDOM_SEED, ReproducibilityContext

from data.data_processor import DataProcessor, DataConfig
from features.feature_engineer import FeatureEngineer, FeatureConfig
from models.model_trainer import ModelTrainer, ModelConfig
from models.model_predictor import ModelPredictor, PredictionConfig
from visualization.data_visualizer import DataVisualizer, VisualizationConfig


class MLPipeline:
    """
    Pipeline principal para el análisis de riesgo crediticio.
    
    Esta clase orquesta todas las etapas del pipeline de machine learning
    siguiendo las mejores prácticas de MLOps.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el pipeline con configuración.
        
        Args:
            config: Diccionario con configuración del pipeline
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Configurar semillas aleatorias para reproducibilidad
        # Obtener la semilla de la configuración o usar la por defecto
        random_state = (
            self.config.get('models', {}).get('random_state') or
            self.config.get('features', {}).get('random_state') or
            self.config.get('data', {}).get('random_state') or
            DEFAULT_RANDOM_SEED
        )
        set_seed(random_state, verbose=False)
        self.logger.info(f"Semillas aleatorias configuradas a: {random_state}")
        
        # Inicializar componentes
        self.data_processor = DataProcessor(DataConfig(**self.config.get('data', {})))
        self.feature_engineer = FeatureEngineer(FeatureConfig(**self.config.get('features', {})))
        self.model_trainer = ModelTrainer(ModelConfig(**self.config.get('models', {})))
        self.predictor = ModelPredictor(PredictionConfig(**self.config.get('prediction', {})))
        self.visualizer = DataVisualizer(VisualizationConfig(**self.config.get('visualization', {})))
        
        # Crear directorios de salida
        self._create_output_directories()
        
        # Resultados del pipeline
        self.results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Obtiene configuración por defecto."""
        return {
            'data': {
                'random_state': 42,
                'target_column': 'target_bad'
            },
            'features': {
                'n_features_select': 15,
                'apply_pca': False,
                'create_interaction_features': True,
                'create_polynomial_features': False,
                'random_state': 42
            },
            'models': {
                'test_size': 0.25,
                'cv_folds': 5,
                'use_mlflow': True,
                'experiment_name': 'german_credit_risk',
                'random_state': 42
            },
            'prediction': {
                'decision_threshold': 0.5,
                'include_explanations': True,
                'output_format': 'csv'
            },
            'visualization': {
                'output_dir': 'reports/figures',
                'interactive': True,
                'output_format': 'png'
            },
            'paths': {
                'raw_data': 'data/raw/german_credit_modified.csv',
                'processed_data': 'data/processed/processed_data.csv',
                'features_data': 'data/processed/features_data.csv',
                'models_dir': 'models',
                'reports_dir': 'reports'
            }
        }
    
    def _create_output_directories(self) -> None:
        """Crea directorios de salida necesarios."""
        directories = [
            'data/processed',
            'data/interim',
            'models',
            'reports/figures',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directorio creado/verificado: {directory}")
    
    def run_data_processing(self, input_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta la etapa de procesamiento de datos.
        
        Args:
            input_path: Ruta del archivo de entrada
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        self.logger.info("=== INICIANDO PROCESAMIENTO DE DATOS ===")
        
        input_path = input_path or self.config['paths']['raw_data']
        output_path = self.config['paths']['processed_data']
        
        # Ejecutar pipeline de procesamiento
        processing_results = self.data_processor.process_pipeline(input_path, output_path)
        
        # Guardar información del procesamiento
        self.results['data_processing'] = processing_results
        
        self.logger.info("Procesamiento de datos completado exitosamente")
        return processing_results
    
    def run_feature_engineering(self, input_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta la etapa de ingeniería de características.
        
        Args:
            input_path: Ruta del archivo de datos procesados
            
        Returns:
            Diccionario con resultados de ingeniería de características
        """
        self.logger.info("=== INICIANDO INGENIERÍA DE CARACTERÍSTICAS ===")
        
        input_path = input_path or self.config['paths']['processed_data']
        output_path = self.config['paths']['features_data']
        
        # Cargar datos procesados
        df = pd.read_csv(input_path)
        X = df.drop(columns=[self.config['data']['target_column']])
        y = df[self.config['data']['target_column']]
        
        # Aplicar ingeniería de características
        X_transformed = self.feature_engineer.fit_transform(X, y)
        
        # Guardar datos con características transformadas
        output_df = X_transformed.copy()
        output_df[self.config['data']['target_column']] = y
        output_df.to_csv(output_path, index=False)
        
        # Guardar transformadores
        transformers_dir = Path(self.config['paths']['models_dir']) / 'transformers'
        self.feature_engineer.save_transformers(str(transformers_dir))
        
        # Información de características
        feature_info = {
            'original_features': len(X.columns),
            'transformed_features': len(X_transformed.columns),
            'selected_features': len(self.feature_engineer.selected_features_) if self.feature_engineer.selected_features_ else len(X_transformed.columns),
            'feature_names': X_transformed.columns.tolist()
        }
        
        self.results['feature_engineering'] = feature_info
        
        self.logger.info(f"Ingeniería de características completada: {feature_info['original_features']} -> {feature_info['transformed_features']} características")
        return feature_info
    
    def run_model_training(self, input_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta la etapa de entrenamiento de modelos.
        
        Args:
            input_path: Ruta del archivo con características transformadas
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        self.logger.info("=== INICIANDO ENTRENAMIENTO DE MODELOS ===")
        
        input_path = input_path or self.config['paths']['features_data']
        
        # Cargar datos con características transformadas
        df = pd.read_csv(input_path)
        X = df.drop(columns=[self.config['data']['target_column']])
        y = df[self.config['data']['target_column']]
        
        # Entrenar modelos
        training_results = self.model_trainer.train_all_models(X, y)
        
        # Guardar modelos
        models_dir = self.config['paths']['models_dir']
        self.model_trainer.save_models(models_dir)
        
        # Guardar información de modelos
        model_info = {
            'models_trained': len(self.model_trainer.trained_models),
            'best_model': self.model_trainer.best_model_name,
            'training_results': training_results
        }
        
        self.results['model_training'] = model_info
        
        self.logger.info(f"Entrenamiento de modelos completado: {model_info['models_trained']} modelos entrenados")
        self.logger.info(f"Mejor modelo: {model_info['best_model']}")
        return model_info
    
    def run_model_evaluation(self, input_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta la evaluación de modelos.
        
        Args:
            input_path: Ruta del archivo con características transformadas
            
        Returns:
            Diccionario con resultados de evaluación
        """
        self.logger.info("=== INICIANDO EVALUACIÓN DE MODELOS ===")
        
        input_path = input_path or self.config['paths']['features_data']
        
        # Cargar datos
        df = pd.read_csv(input_path)
        X = df.drop(columns=[self.config['data']['target_column']])
        y = df[self.config['data']['target_column']]
        
        # Dividir datos para evaluación
        X_train, X_test, y_train, y_test = self.model_trainer.split_data(X, y)
        
        # Evaluar cada modelo
        evaluation_results = {}
        for model_name, model in self.model_trainer.trained_models.items():
            try:
                # Hacer predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Evaluar predicciones
                metrics = self.predictor.evaluate_predictions(y_test, y_pred, y_pred_proba)
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'predictions': {
                        'y_true': y_test.tolist(),
                        'y_pred': y_pred.tolist(),
                        'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluando modelo {model_name}: {str(e)}")
                continue
        
        self.results['model_evaluation'] = evaluation_results
        
        self.logger.info("Evaluación de modelos completada")
        return evaluation_results
    
    def run_visualization(self, input_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta la generación de visualizaciones y reportes.
        
        Args:
            input_path: Ruta del archivo con datos procesados
            
        Returns:
            Diccionario con resultados de visualización
        """
        self.logger.info("=== INICIANDO GENERACIÓN DE VISUALIZACIONES ===")
        
        input_path = input_path or self.config['paths']['processed_data']
        
        # Cargar datos
        df = pd.read_csv(input_path)
        
        # Generar visualizaciones básicas
        visualization_files = []
        
        # Distribución del target
        target_file = self.visualizer.plot_target_distribution(df, self.config['data']['target_column'])
        if target_file:
            visualization_files.append(target_file)
        
        # Valores faltantes
        missing_file = self.visualizer.plot_missing_values(df)
        if missing_file:
            visualization_files.append(missing_file)
        
        # Matriz de correlación
        corr_file = self.visualizer.plot_correlation_matrix(df)
        if corr_file:
            visualization_files.append(corr_file)
        
        # Distribuciones de características
        dist_file = self.visualizer.plot_feature_distributions(df, self.config['data']['target_column'])
        if dist_file:
            visualization_files.append(dist_file)
        
        # Visualizaciones de modelos si están disponibles
        if 'model_evaluation' in self.results:
            for model_name, results in self.results['model_evaluation'].items():
                predictions = results['predictions']
                
                # Matriz de confusión
                cm_file = self.visualizer.plot_confusion_matrix(
                    np.array(predictions['y_true']),
                    np.array(predictions['y_pred']),
                    model_name
                )
                if cm_file:
                    visualization_files.append(cm_file)
                
                # Curvas ROC y Precision-Recall
                if predictions['y_pred_proba'] is not None:
                    roc_file = self.visualizer.plot_roc_curve(
                        np.array(predictions['y_true']),
                        np.array(predictions['y_pred_proba']),
                        model_name
                    )
                    if roc_file:
                        visualization_files.append(roc_file)
                    
                    pr_file = self.visualizer.plot_precision_recall_curve(
                        np.array(predictions['y_true']),
                        np.array(predictions['y_pred_proba']),
                        model_name
                    )
                    if pr_file:
                        visualization_files.append(pr_file)
        
        # Comparación de modelos
        if 'model_evaluation' in self.results:
            model_metrics = {}
            for model_name, results in self.results['model_evaluation'].items():
                model_metrics[model_name] = results['metrics']
            
            comparison_file = self.visualizer.plot_model_comparison(model_metrics)
            if comparison_file:
                visualization_files.append(comparison_file)
        
        # Dashboard interactivo
        if self.config['visualization']['interactive']:
            dashboard_file = self.visualizer.create_interactive_dashboard(df, self.config['data']['target_column'])
            if dashboard_file:
                visualization_files.append(dashboard_file)
        
        # Generar reporte completo
        report_path = self.visualizer.generate_report(df, self.results.get('model_evaluation'), self.config['data']['target_column'])
        
        visualization_info = {
            'files_generated': len(visualization_files),
            'visualization_files': visualization_files,
            'report_path': report_path
        }
        
        self.results['visualization'] = visualization_info
        
        self.logger.info(f"Visualizaciones generadas: {len(visualization_files)} archivos")
        return visualization_info
    
    def run_full_pipeline(self, input_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo.
        
        Args:
            input_path: Ruta del archivo de entrada
            
        Returns:
            Diccionario con resultados de todo el pipeline
        """
        self.logger.info("=== INICIANDO PIPELINE COMPLETO ===")
        
        start_time = datetime.now()
        
        try:
            # 1. Procesamiento de datos
            self.run_data_processing(input_path)
            
            # 2. Ingeniería de características
            self.run_feature_engineering()
            
            # 3. Entrenamiento de modelos
            self.run_model_training()
            
            # 4. Evaluación de modelos
            self.run_model_evaluation()
            
            # 5. Generación de visualizaciones
            self.run_visualization()
            
            # Calcular tiempo total
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Resumen final
            pipeline_summary = {
                'status': 'success',
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar resumen
            summary_path = Path(self.config['paths']['reports_dir']) / 'pipeline_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            self.logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
            self.logger.info(f"Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
            
            return pipeline_summary
            
        except Exception as e:
            self.logger.error(f"Error en el pipeline: {str(e)}")
            raise
    
    def generate_final_report(self) -> str:
        """
        Genera un reporte final con todos los resultados.
        
        Returns:
            Ruta del archivo de reporte generado
        """
        self.logger.info("Generando reporte final")
        
        report_path = Path(self.config['paths']['reports_dir']) / 'final_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte Final - Análisis de Riesgo Crediticio\n\n")
            f.write(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumen ejecutivo
            f.write("## Resumen Ejecutivo\n\n")
            if 'data_processing' in self.results:
                data_info = self.results['data_processing']
                f.write(f"- **Datos procesados:** {data_info['output_shape'][0]} filas, {data_info['output_shape'][1]} columnas\n")
                f.write(f"- **Valores faltantes:** {data_info['summary']['missing_values']}\n")
                f.write(f"- **Duplicados:** {data_info['summary']['duplicates']}\n")
            
            if 'feature_engineering' in self.results:
                feature_info = self.results['feature_engineering']
                f.write(f"- **Características originales:** {feature_info['original_features']}\n")
                f.write(f"- **Características transformadas:** {feature_info['transformed_features']}\n")
                f.write(f"- **Características seleccionadas:** {feature_info['selected_features']}\n")
            
            if 'model_training' in self.results:
                model_info = self.results['model_training']
                f.write(f"- **Modelos entrenados:** {model_info['models_trained']}\n")
                f.write(f"- **Mejor modelo:** {model_info['best_model']}\n")
            
            f.write("\n## Resultados Detallados\n\n")
            
            # Resultados de modelos
            if 'model_evaluation' in self.results:
                f.write("### Métricas de Modelos\n\n")
                f.write("| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision |\n")
                f.write("|--------|----------|-----------|--------|----|---------|---------------|\n")
                
                for model_name, results in self.results['model_evaluation'].items():
                    metrics = results['metrics']
                    f.write(f"| {model_name} | {metrics.get('accuracy', 0):.3f} | "
                           f"{metrics.get('precision', 0):.3f} | {metrics.get('recall', 0):.3f} | "
                           f"{metrics.get('f1', 0):.3f} | {metrics.get('roc_auc', 0):.3f} | "
                           f"{metrics.get('average_precision', 0):.3f} |\n")
            
            # Archivos generados
            if 'visualization' in self.results:
                f.write("\n### Archivos Generados\n\n")
                for file_path in self.results['visualization']['visualization_files']:
                    f.write(f"- {Path(file_path).name}\n")
        
        self.logger.info(f"Reporte final generado en: {report_path}")
        return str(report_path)


def main():
    """Función principal para ejecutar el pipeline desde línea de comandos."""
    parser = argparse.ArgumentParser(description="Pipeline de Análisis de Riesgo Crediticio")
    parser.add_argument("--input", help="Archivo CSV de entrada", 
                       default="data/raw/german_credit_modified.csv")
    parser.add_argument("--config", help="Archivo JSON con configuración")
    parser.add_argument("--stage", choices=["data", "features", "models", "eval", "viz", "all"],
                       help="Etapa específica a ejecutar", default="all")
    parser.add_argument("--output-dir", help="Directorio de salida", default=".")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logging detallado")
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )
    
    # Cargar configuración
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Crear pipeline
    pipeline = MLPipeline(config)
    
    # Ejecutar etapa específica o pipeline completo
    try:
        if args.stage == "all":
            results = pipeline.run_full_pipeline(args.input)
            pipeline.generate_final_report()
        elif args.stage == "data":
            pipeline.run_data_processing(args.input)
        elif args.stage == "features":
            pipeline.run_feature_engineering()
        elif args.stage == "models":
            pipeline.run_model_training()
        elif args.stage == "eval":
            pipeline.run_model_evaluation()
        elif args.stage == "viz":
            pipeline.run_visualization()
        
        print("\n✅ Pipeline ejecutado exitosamente!")
        print(f"📊 Resultados disponibles en: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error en el pipeline: {str(e)}")
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
