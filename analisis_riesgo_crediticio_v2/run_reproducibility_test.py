#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Reproducibilidad: Ejecuta el pipeline completo con semillas fijas
y guarda métricas de referencia para comparación entre entornos.

Este script:
1. Configura todas las semillas aleatorias
2. Ejecuta el pipeline completo
3. Guarda métricas de referencia en un archivo JSON
4. Versiona artefactos en DVC/MLflow

Uso:
    python run_reproducibility_test.py [--seed SEED] [--output-dir DIR] [--save-artifacts]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importar módulo de reproducibilidad ANTES que otros módulos
from utils.reproducibility import set_seed, DEFAULT_RANDOM_SEED

# Configurar semillas ANTES de importar otros módulos
# (El seed puede ser sobrescrito por argumentos)
set_seed(DEFAULT_RANDOM_SEED, verbose=False)

# Ahora importar componentes del pipeline
from main_pipeline import MLPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configura el sistema de logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('reproducibility_test.log')
        ]
    )


def extract_metrics_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae métricas clave de los resultados del pipeline.
    
    Args:
        results: Resultados completos del pipeline
        
    Returns:
        Diccionario con métricas extraídas
    """
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '1.0',
        'model_metrics': {},
        'data_info': {},
        'feature_info': {},
        'best_model': None
    }
    
    # Métricas de modelos
    if 'model_evaluation' in results.get('results', {}):
        for model_name, model_results in results['results']['model_evaluation'].items():
            if 'metrics' in model_results:
                metrics['model_metrics'][model_name] = {
                    'accuracy': float(model_results['metrics'].get('accuracy', 0)),
                    'precision': float(model_results['metrics'].get('precision', 0)),
                    'recall': float(model_results['metrics'].get('recall', 0)),
                    'f1': float(model_results['metrics'].get('f1', 0)),
                    'roc_auc': float(model_results['metrics'].get('roc_auc', 0)),
                    'average_precision': float(model_results['metrics'].get('average_precision', 0)),
                    'training_time': float(model_results['metrics'].get('training_time', 0))
                }
    
    # Información de datos
    if 'data_processing' in results.get('results', {}):
        data_info = results['results']['data_processing']
        metrics['data_info'] = {
            'input_shape': list(data_info.get('input_shape', [0, 0])),
            'output_shape': list(data_info.get('output_shape', [0, 0])),
            'missing_values': data_info.get('summary', {}).get('missing_values', 0),
            'duplicates': data_info.get('summary', {}).get('duplicates', 0)
        }
    
    # Información de características
    if 'feature_engineering' in results.get('results', {}):
        feature_info = results['results']['feature_engineering']
        metrics['feature_info'] = {
            'original_features': feature_info.get('original_features', 0),
            'transformed_features': feature_info.get('transformed_features', 0),
            'selected_features': feature_info.get('selected_features', 0)
        }
    
    # Mejor modelo
    if 'model_training' in results.get('results', {}):
        model_info = results['results']['model_training']
        metrics['best_model'] = model_info.get('best_model', None)
    
    return metrics

def to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj

def save_reference_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Guarda las métricas de referencia en un archivo JSON.
    
    Args:
        metrics: Diccionario con métricas
        output_path: Ruta donde guardar las métricas
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        metrics_clean = to_native(metrics)
        json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Métricas de referencia guardadas en: {output_path}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Ejecutar prueba de reproducibilidad del pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar con semilla por defecto (42)
  python run_reproducibility_test.py
  
  # Ejecutar con semilla personalizada
  python run_reproducibility_test.py --seed 123
  
  # Guardar métricas en ubicación personalizada
  python run_reproducibility_test.py --output-dir results/
  
  # Ejecutar sin guardar artefactos adicionales
  python run_reproducibility_test.py --no-save-artifacts
        """
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'Semilla aleatoria a usar (por defecto: {DEFAULT_RANDOM_SEED})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/reproducibility',
        help='Directorio donde guardar métricas de referencia'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/german_credit_modified.csv',
        help='Ruta del archivo de datos de entrada'
    )
    parser.add_argument(
        '--save-artifacts',
        action='store_true',
        default=True,
        help='Guardar artefactos del pipeline (modelos, transformadores, etc.)'
    )
    parser.add_argument(
        '--no-save-artifacts',
        dest='save_artifacts',
        action='store_false',
        help='No guardar artefactos adicionales'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Modo verbose con logging detallado'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("INICIANDO PRUEBA DE REPRODUCIBILIDAD")
    logger.info("=" * 70)
    logger.info(f"Semilla aleatoria: {args.seed}")
    logger.info(f"Archivo de entrada: {args.input}")
    logger.info(f"Directorio de salida: {args.output_dir}")
    
    # Configurar semillas
    logger.info("Configurando semillas aleatorias...")
    set_seed(args.seed, verbose=True)
    
    # Crear configuración del pipeline con semilla fija
    config = {
        'data': {
            'random_state': args.seed,
            'target_column': 'target_bad'
        },
        'features': {
            'n_features_select': 15,
            'apply_pca': False,
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'random_state': args.seed
        },
        'models': {
            'test_size': 0.25,
            'cv_folds': 5,
            'use_mlflow': True,
            'experiment_name': 'german_credit_risk_reproducibility',
            'random_state': args.seed
        },
        'prediction': {
            'decision_threshold': 0.5,
            'include_explanations': True,
            'output_format': 'csv'
        },
        'visualization': {
            'output_dir': 'reports/figures',
            'interactive': False,  # Desactivar para reproducibilidad
            'output_format': 'png'
        },
        'paths': {
            'raw_data': args.input,
            'processed_data': 'data/processed/processed_data.csv',
            'features_data': 'data/processed/features_data.csv',
            'models_dir': 'models',
            'reports_dir': 'reports'
        }
    }
    
    try:
        # Crear y ejecutar pipeline
        logger.info("Creando pipeline...")
        pipeline = MLPipeline(config)
        
        logger.info("Ejecutando pipeline completo...")
        results = pipeline.run_full_pipeline(args.input)
        
        # Extraer métricas clave
        logger.info("Extrayendo métricas de referencia...")
        reference_metrics = extract_metrics_from_results(results)
        reference_metrics['seed'] = args.seed
        reference_metrics['python_version'] = sys.version.split()[0]
        reference_metrics['pipeline_status'] = results.get('status', 'unknown')
        
        # Guardar métricas de referencia
        output_dir = Path(args.output_dir)
        metrics_file = output_dir / 'reference_metrics.json'
        save_reference_metrics(reference_metrics, metrics_file)
        
        # Guardar también con timestamp para comparación
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_file = output_dir / f'reference_metrics_{timestamp}.json'
        save_reference_metrics(reference_metrics, timestamped_file)
        
        # Mostrar resumen
        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE MÉTRICAS DE REFERENCIA")
        logger.info("=" * 70)
        
        if reference_metrics['best_model']:
            logger.info(f"Mejor modelo: {reference_metrics['best_model']}")
            if reference_metrics['best_model'] in reference_metrics['model_metrics']:
                best_metrics = reference_metrics['model_metrics'][reference_metrics['best_model']]
                logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {best_metrics['precision']:.4f}")
                logger.info(f"  Recall: {best_metrics['recall']:.4f}")
                logger.info(f"  F1: {best_metrics['f1']:.4f}")
                logger.info(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
                logger.info(f"  Average Precision: {best_metrics['average_precision']:.4f}")
        
        logger.info(f"\nMétricas guardadas en: {metrics_file}")
        logger.info(f"Timestamped file: {timestamped_file}")
        
        logger.info("\n" + "=" * 70)
        logger.info("PRUEBA DE REPRODUCIBILIDAD COMPLETADA EXITOSAMENTE")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en la prueba de reproducibilidad: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

