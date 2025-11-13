#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Comparación de Resultados: Compara métricas entre diferentes ejecuciones
del pipeline para verificar reproducibilidad entre entornos.

Este script:
1. Carga métricas de referencia y métricas actuales
2. Compara métricas de modelos
3. Genera reporte de diferencias
4. Determina si los resultados son consistentes

Uso:
    python compare_reproducibility_results.py --reference REFERENCE.json --current CURRENT.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


# Tolerancias para comparación de métricas (valores absolutos)
METRIC_TOLERANCES = {
    'accuracy': 0.01,      # 1% de tolerancia
    'precision': 0.01,     # 1% de tolerancia
    'recall': 0.01,        # 1% de tolerancia
    'f1': 0.01,            # 1% de tolerancia
    'roc_auc': 0.01,       # 1% de tolerancia
    'average_precision': 0.01,  # 1% de tolerancia
    'training_time': 10.0  # 10 segundos de tolerancia
}


def load_metrics(file_path: Path) -> Dict[str, Any]:
    """
    Carga métricas desde un archivo JSON.
    
    Args:
        file_path: Ruta del archivo JSON con métricas
        
    Returns:
        Diccionario con métricas cargadas
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    logging.info(f"Métricas cargadas desde: {file_path}")
    return metrics


def compare_model_metrics(
    ref_metrics: Dict[str, float],
    curr_metrics: Dict[str, float],
    model_name: str
) -> Dict[str, Any]:
    """
    Compara métricas de un modelo entre referencia y actual.
    
    Args:
        ref_metrics: Métricas de referencia
        curr_metrics: Métricas actuales
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con resultados de la comparación
    """
    comparison = {
        'model': model_name,
        'metrics': {},
        'differences': {},
        'within_tolerance': {},
        'is_consistent': True
    }
    
    # Comparar cada métrica
    for metric_name in METRIC_TOLERANCES.keys():
        if metric_name in ref_metrics and metric_name in curr_metrics:
            ref_value = float(ref_metrics[metric_name])
            curr_value = float(curr_metrics[metric_name])
            diff = abs(curr_value - ref_value)
            tolerance = METRIC_TOLERANCES[metric_name]
            
            within_tolerance = diff <= tolerance
            relative_diff = (diff / ref_value * 100) if ref_value != 0 else 0
            
            comparison['metrics'][metric_name] = {
                'reference': ref_value,
                'current': curr_value,
                'difference': diff,
                'relative_difference_percent': relative_diff,
                'tolerance': tolerance,
                'within_tolerance': within_tolerance
            }
            
            comparison['differences'][metric_name] = diff
            comparison['within_tolerance'][metric_name] = within_tolerance
            
            # Si alguna métrica está fuera de tolerancia, el modelo no es consistente
            if not within_tolerance:
                comparison['is_consistent'] = False
        else:
            comparison['metrics'][metric_name] = {
                'reference': None,
                'current': None,
                'status': 'missing'
            }
            comparison['is_consistent'] = False
    
    return comparison


def compare_all_models(
    ref_data: Dict[str, Any],
    curr_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Compara métricas de todos los modelos.
    
    Args:
        ref_data: Datos de referencia
        curr_data: Datos actuales
        
    Returns:
        Lista de comparaciones por modelo
    """
    comparisons = []
    
    ref_models = ref_data.get('model_metrics', {})
    curr_models = curr_data.get('model_metrics', {})
    
    # Obtener conjunto de modelos a comparar
    all_models = set(ref_models.keys()) | set(curr_models.keys())
    
    for model_name in all_models:
        if model_name in ref_models and model_name in curr_models:
            comparison = compare_model_metrics(
                ref_models[model_name],
                curr_models[model_name],
                model_name
            )
            comparisons.append(comparison)
        else:
            logging.warning(f"Modelo {model_name} no encontrado en referencia o actual")
    
    return comparisons


def compare_data_info(ref_data: Dict[str, Any], curr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara información de datos procesados.
    
    Args:
        ref_data: Datos de referencia
        curr_data: Datos actuales
        
    Returns:
        Diccionario con comparación de información de datos
    """
    comparison = {
        'data_shape': {},
        'data_quality': {},
        'is_consistent': True
    }
    
    ref_info = ref_data.get('data_info', {})
    curr_info = curr_data.get('data_info', {})
    
    # Comparar shape
    ref_shape = ref_info.get('output_shape', [0, 0])
    curr_shape = curr_info.get('output_shape', [0, 0])
    
    if ref_shape == curr_shape:
        comparison['data_shape'] = {
            'reference': ref_shape,
            'current': curr_shape,
            'match': True
        }
    else:
        comparison['data_shape'] = {
            'reference': ref_shape,
            'current': curr_shape,
            'match': False
        }
        comparison['is_consistent'] = False
    
    # Comparar calidad de datos
    for metric in ['missing_values', 'duplicates']:
        ref_val = ref_info.get(metric, 0)
        curr_val = curr_info.get(metric, 0)
        
        comparison['data_quality'][metric] = {
            'reference': ref_val,
            'current': curr_val,
            'match': ref_val == curr_val
        }
        
        if ref_val != curr_val:
            comparison['is_consistent'] = False
    
    return comparison


def compare_feature_info(ref_data: Dict[str, Any], curr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara información de características.
    
    Args:
        ref_data: Datos de referencia
        curr_data: Datos actuales
        
    Returns:
        Diccionario con comparación de información de características
    """
    comparison = {
        'feature_counts': {},
        'is_consistent': True
    }
    
    ref_info = ref_data.get('feature_info', {})
    curr_info = curr_data.get('feature_info', {})
    
    for metric in ['original_features', 'transformed_features', 'selected_features']:
        ref_val = ref_info.get(metric, 0)
        curr_val = curr_info.get(metric, 0)
        
        comparison['feature_counts'][metric] = {
            'reference': ref_val,
            'current': curr_val,
            'match': ref_val == curr_val
        }
        
        if ref_val != curr_val:
            comparison['is_consistent'] = False
    
    return comparison


def generate_comparison_report(
    comparisons: List[Dict[str, Any]],
    data_comparison: Dict[str, Any],
    feature_comparison: Dict[str, Any],
    ref_data: Dict[str, Any],
    curr_data: Dict[str, Any],
    output_path: Path
) -> str:
    """
    Genera un reporte de comparación en formato Markdown.
    
    Args:
        comparisons: Lista de comparaciones por modelo
        data_comparison: Comparación de información de datos
        feature_comparison: Comparación de información de características
        ref_data: Datos de referencia
        curr_data: Datos actuales
        output_path: Ruta donde guardar el reporte
        
    Returns:
        Ruta del archivo de reporte generado
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Comparación de Reproducibilidad\n\n")
        f.write(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Información general
        f.write("## Información General\n\n")
        f.write(f"- **Referencia (seed):** {ref_data.get('seed', 'N/A')}\n")
        f.write(f"- **Actual (seed):** {curr_data.get('seed', 'N/A')}\n")
        f.write(f"- **Referencia (timestamp):** {ref_data.get('timestamp', 'N/A')}\n")
        f.write(f"- **Actual (timestamp):** {curr_data.get('timestamp', 'N/A')}\n")
        f.write(f"- **Python (referencia):** {ref_data.get('python_version', 'N/A')}\n")
        f.write(f"- **Python (actual):** {curr_data.get('python_version', 'N/A')}\n\n")
        
        # Resumen
        f.write("## Resumen Ejecutivo\n\n")
        
        total_models = len(comparisons)
        consistent_models = sum(1 for c in comparisons if c['is_consistent'])
        
        f.write(f"- **Total de modelos:** {total_models}\n")
        f.write(f"- **Modelos consistentes:** {consistent_models}\n")
        f.write(f"- **Modelos inconsistentes:** {total_models - consistent_models}\n")
        f.write(f"- **Datos consistentes:** {'✅ Sí' if data_comparison['is_consistent'] else '❌ No'}\n")
        f.write(f"- **Características consistentes:** {'✅ Sí' if feature_comparison['is_consistent'] else '❌ No'}\n\n")
        
        overall_consistent = (
            consistent_models == total_models and
            data_comparison['is_consistent'] and
            feature_comparison['is_consistent']
        )
        
        f.write(f"### Estado General: {'✅ REPRODUCIBLE' if overall_consistent else '❌ NO REPRODUCIBLE'}\n\n")
        
        # Comparación de datos
        f.write("## Comparación de Datos\n\n")
        f.write("| Métrica | Referencia | Actual | Match |\n")
        f.write("|---------|------------|--------|-------|\n")
        
        for metric, info in data_comparison['data_quality'].items():
            match_icon = '✅' if info['match'] else '❌'
            f.write(f"| {metric} | {info['reference']} | {info['current']} | {match_icon} |\n")
        
        f.write(f"\n**Shape de datos:** Referencia={data_comparison['data_shape'].get('reference')}, "
                f"Actual={data_comparison['data_shape'].get('current')}, "
                f"Match={'✅' if data_comparison['data_shape'].get('match') else '❌'}\n\n")
        
        # Comparación de características
        f.write("## Comparación de Características\n\n")
        f.write("| Métrica | Referencia | Actual | Match |\n")
        f.write("|---------|------------|--------|-------|\n")
        
        for metric, info in feature_comparison['feature_counts'].items():
            match_icon = '✅' if info['match'] else '❌'
            f.write(f"| {metric} | {info['reference']} | {info['current']} | {match_icon} |\n")
        
        # Comparación de modelos
        f.write("\n## Comparación de Modelos\n\n")
        
        for comp in comparisons:
            model_name = comp['model']
            is_consistent = comp['is_consistent']
            status_icon = '✅' if is_consistent else '❌'
            
            f.write(f"### {model_name} {status_icon}\n\n")
            f.write("| Métrica | Referencia | Actual | Diferencia | Tolerancia | Estado |\n")
            f.write("|---------|------------|--------|------------|------------|--------|\n")
            
            for metric_name, metric_info in comp['metrics'].items():
                if metric_info.get('status') == 'missing':
                    f.write(f"| {metric_name} | - | - | - | - | ⚠️ Missing |\n")
                else:
                    ref_val = metric_info['reference']
                    curr_val = metric_info['current']
                    diff = metric_info['difference']
                    tolerance = metric_info['tolerance']
                    status = '✅' if metric_info['within_tolerance'] else '❌'
                    
                    f.write(f"| {metric_name} | {ref_val:.6f} | {curr_val:.6f} | "
                           f"{diff:.6f} | {tolerance:.6f} | {status} |\n")
            
            f.write("\n")
    
    logging.info(f"Reporte de comparación guardado en: {output_path}")
    return str(output_path)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Comparar resultados de reproducibilidad entre diferentes ejecuciones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Comparar con métricas de referencia
  python compare_reproducibility_results.py \\
      --reference reports/reproducibility/reference_metrics.json \\
      --current reports/reproducibility/reference_metrics_20241201_120000.json
  
  # Especificar tolerancias personalizadas
  python compare_reproducibility_results.py \\
      --reference reference.json \\
      --current current.json \\
      --tolerance accuracy 0.005
        """
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help='Archivo JSON con métricas de referencia'
    )
    parser.add_argument(
        '--current',
        type=str,
        required=True,
        help='Archivo JSON con métricas actuales a comparar'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/reproducibility/comparison_report.md',
        help='Ruta del archivo de reporte de comparación'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Ruta del archivo JSON con resultados de comparación (opcional)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Modo verbose con logging detallado'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("INICIANDO COMPARACIÓN DE RESULTADOS DE REPRODUCIBILIDAD")
    logger.info("=" * 70)
    
    try:
        # Cargar métricas
        ref_path = Path(args.reference)
        curr_path = Path(args.current)
        
        logger.info(f"Cargando métricas de referencia desde: {ref_path}")
        ref_data = load_metrics(ref_path)
        
        logger.info(f"Cargando métricas actuales desde: {curr_path}")
        curr_data = load_metrics(curr_path)
        
        # Comparar modelos
        logger.info("Comparando métricas de modelos...")
        comparisons = compare_all_models(ref_data, curr_data)
        
        # Comparar información de datos
        logger.info("Comparando información de datos...")
        data_comparison = compare_data_info(ref_data, curr_data)
        
        # Comparar información de características
        logger.info("Comparando información de características...")
        feature_comparison = compare_feature_info(ref_data, curr_data)
        
        # Preparar resultados completos
        results = {
            'reference_file': str(ref_path),
            'current_file': str(curr_path),
            'comparison_timestamp': datetime.now().isoformat(),
            'model_comparisons': comparisons,
            'data_comparison': data_comparison,
            'feature_comparison': feature_comparison,
            'overall_consistent': (
                all(c['is_consistent'] for c in comparisons) and
                data_comparison['is_consistent'] and
                feature_comparison['is_consistent']
            )
        }
        
        # Guardar resultados JSON si se solicita
        if args.output_json:
            json_path = Path(args.output_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Resultados JSON guardados en: {json_path}")
        
        # Generar reporte
        logger.info("Generando reporte de comparación...")
        report_path = generate_comparison_report(
            comparisons,
            data_comparison,
            feature_comparison,
            ref_data,
            curr_data,
            Path(args.output)
        )
        
        # Mostrar resumen
        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE COMPARACIÓN")
        logger.info("=" * 70)
        
        total_models = len(comparisons)
        consistent_models = sum(1 for c in comparisons if c['is_consistent'])
        
        logger.info(f"Total de modelos: {total_models}")
        logger.info(f"Modelos consistentes: {consistent_models}")
        logger.info(f"Datos consistentes: {data_comparison['is_consistent']}")
        logger.info(f"Características consistentes: {feature_comparison['is_consistent']}")
        logger.info(f"Resultado general: {'✅ REPRODUCIBLE' if results['overall_consistent'] else '❌ NO REPRODUCIBLE'}")
        logger.info(f"\nReporte completo guardado en: {report_path}")
        logger.info("=" * 70)
        
        return 0 if results['overall_consistent'] else 1
        
    except Exception as e:
        logger.error(f"Error en la comparación: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

