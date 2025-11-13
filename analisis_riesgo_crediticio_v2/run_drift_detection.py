#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script principal para ejecutar la detección de drift en el modelo de riesgo crediticio.

Este script:
1. Carga los datos de referencia (validación)
2. Carga el modelo entrenado
3. Genera datos con drift simulado
4. Evalúa el impacto en el desempeño
5. Genera reportes y visualizaciones
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar el directorio src al path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from monitoring.detect_drift import DriftDetector, DriftConfig
from data.data_processor import DataProcessor, DataConfig
from models.model_predictor import ModelPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drift_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_data_and_model():
    """Carga los datos procesados y el modelo entrenado."""
    logger.info("Cargando datos y modelo...")
    
    # Rutas - preferir features_data.csv que ya tiene las features transformadas
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
    features_data_path = PROJECT_ROOT / "data" / "processed" / "features_data.csv"
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    
    # Preferir features_data.csv si existe (ya tiene features transformadas)
    if features_data_path.exists():
        logger.info(f"Usando datos con features transformadas: {features_data_path}")
        data_path = features_data_path
        use_transformed_features = True
    elif processed_data_path.exists():
        logger.info(f"Usando datos procesados: {processed_data_path}")
        data_path = processed_data_path
        use_transformed_features = False
    else:
        raise FileNotFoundError(f"No se encontraron datos procesados en {processed_data_path.parent}")
    
    if not model_path.exists():
        # Intentar encontrar otro modelo
        models_dir = PROJECT_ROOT / "models"
        model_files = list(models_dir.glob("*.joblib"))
        if model_files:
            model_path = model_files[0]
            logger.info(f"Usando modelo alternativo: {model_path}")
        else:
            raise FileNotFoundError(f"No se encontró ningún modelo en {models_dir}")
    
    # Cargar datos
    logger.info(f"Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    
    # Verificar si tiene target_bad
    if "target_bad" not in df.columns:
        logger.warning("No se encontró columna target_bad, se generará una distribución similar")
        # Crear una distribución similar para el target
        np.random.seed(42)
        target = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        df["target_bad"] = target
    
    logger.info(f"Datos cargados: {df.shape}")
    logger.info(f"Modelo: {model_path}")
    logger.info(f"Usando features transformadas: {use_transformed_features}")
    
    return df, model_path, use_transformed_features


def prepare_data_for_model(df: pd.DataFrame, data_processor: DataProcessor):
    """
    Prepara los datos para el modelo usando el procesador.
    
    Si los datos ya están procesados (features transformadas), los devuelve tal cual.
    Si no, los procesa.
    """
    # Verificar si los datos ya están en formato de features (one-hot, etc.)
    # Si tienen muchas columnas con nombres como "laufkont_1.0", están procesados
    has_processed_features = any("_" in col and col != "target_bad" for col in df.columns)
    
    if has_processed_features:
        logger.info("Los datos parecen estar ya procesados (features transformadas)")
        return df
    else:
        logger.info("Procesando datos con DataProcessor...")
        # Limpiar y preparar
        df_clean = data_processor.clean_data(df.copy())
        X, y = data_processor.prepare_features(df_clean)
        
        # Combinar de nuevo para mantener compatibilidad
        df_processed = X.copy()
        df_processed["target_bad"] = y
        
        return df_processed


def main():
    """Función principal."""
    logger.info("=" * 60)
    logger.info("INICIANDO DETECCIÓN DE DRIFT")
    logger.info("=" * 60)
    
    try:
        # 1. Cargar datos y modelo
        df, model_path, use_transformed_features = load_data_and_model()
        
        # 2. Configurar procesador de datos (solo si no usamos features transformadas)
        data_processor = None
        if not use_transformed_features:
            data_config = DataConfig()
            data_processor = DataProcessor(config=data_config)
            # 3. Preparar datos
            df_processed = prepare_data_for_model(df, data_processor)
        else:
            # Si ya están transformadas, usar directamente
            df_processed = df.copy()
            logger.info("Usando datos con features ya transformadas")
        
        # 4. Dividir en train/test para usar test como referencia
        from sklearn.model_selection import train_test_split
        
        X = df_processed.drop(columns=["target_bad"])
        y = df_processed["target_bad"]
        
        # Usar una porción como referencia (simulando datos de validación)
        X_ref, X_temp, y_ref, y_temp = train_test_split(
            X, y, test_size=0.5, stratify=y, random_state=42
        )
        
        # Combinar de nuevo para el detector
        df_reference = X_ref.copy()
        df_reference["target_bad"] = y_ref
        
        logger.info(f"Datos de referencia: {df_reference.shape}")
        
        # 5. Configurar detector de drift
        drift_config = DriftConfig(
            accuracy_threshold=0.05,
            roc_auc_threshold=0.05,
            f1_threshold=0.05,
            mean_shift_factor=0.2,
            missing_value_rate=0.1,
            variance_change_factor=1.5,
            categorical_shift_probability=0.15
        )
        
        detector = DriftDetector(config=drift_config)
        
        # 6. Cargar datos de referencia y modelo
        # Guardar temporalmente los datos de referencia
        temp_ref_path = PROJECT_ROOT / "data" / "processed" / "temp_reference.csv"
        df_reference.to_csv(temp_ref_path, index=False)
        
        detector.load_reference_data(str(temp_ref_path), target_col="target_bad")
        detector.load_model(str(model_path))
        if data_processor is not None:
            detector.set_data_processors(data_processor)
        
        # 7. Ejecutar detección para diferentes tipos de drift
        output_dir = PROJECT_ROOT / "reports" / "drift_detection"
        
        drift_types = ["mean_shift", "missing_values", "combined"]
        
        all_results = {}
        
        for drift_type in drift_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Ejecutando detección de drift: {drift_type}")
            logger.info(f"{'='*60}\n")
            
            try:
                results = detector.run_drift_detection(
                    drift_type=drift_type,
                    save_results=True,
                    output_dir=str(output_dir / drift_type)
                )
                
                all_results[drift_type] = results
                
                # Mostrar resumen
                logger.info(f"\nResumen para {drift_type}:")
                logger.info(f"  - Drift Detectado: {results['summary']['drift_detected']}")
                logger.info(f"  - Severidad: {results['summary']['drift_severity']}")
                logger.info(f"  - Alertas: {results['summary']['alert_count']}")
                
            except Exception as e:
                logger.error(f"Error ejecutando detección para {drift_type}: {str(e)}")
                continue
        
        # 8. Generar reporte consolidado
        logger.info("\nGenerando reporte consolidado...")
        generate_consolidated_report(all_results, output_dir)
        
        # 9. Limpiar archivo temporal
        if temp_ref_path.exists():
            temp_ref_path.unlink()
        
        logger.info("\n" + "=" * 60)
        logger.info("DETECCIÓN DE DRIFT COMPLETADA")
        logger.info("=" * 60)
        logger.info(f"\nResultados guardados en: {output_dir}")
        logger.info("\nRevisa los reportes y visualizaciones generados para más detalles.")
        
    except Exception as e:
        logger.error(f"Error en la ejecución: {str(e)}", exc_info=True)
        sys.exit(1)


def generate_consolidated_report(all_results: dict, output_dir: Path):
    """Genera un reporte consolidado con todos los tipos de drift."""
    # Asegurar que el directorio existe
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "consolidated_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte Consolidado de Detección de Drift\n\n")
        f.write("Este reporte consolida los resultados de diferentes tipos de drift simulado.\n\n")
        
        f.write("## Resumen Ejecutivo\n\n")
        f.write("| Tipo de Drift | Drift Detectado | Severidad | Alertas | Degradación de Desempeño |\n")
        f.write("|---------------|-----------------|-----------|---------|---------------------------|\n")
        
        for drift_type, results in all_results.items():
            summary = results['summary']
            f.write(f"| {drift_type} | {'Sí' if summary['drift_detected'] else 'No'} | "
                   f"{summary['drift_severity']} | {summary['alert_count']} | "
                   f"{'Sí' if summary['performance_degradation'] else 'No'} |\n")
        
        f.write("\n## Detalles por Tipo de Drift\n\n")
        
        for drift_type, results in all_results.items():
            f.write(f"### {drift_type.upper()}\n\n")
            
            # Métricas
            perf_comp = results['performance_comparison']
            f.write("#### Métricas de Desempeño\n\n")
            f.write("| Métrica | Referencia | Con Drift | Cambio Relativo (%) |\n")
            f.write("|---------|------------|-----------|---------------------|\n")
            
            for metric in perf_comp['reference_metrics'].keys():
                ref_val = perf_comp['reference_metrics'][metric]
                curr_val = perf_comp['current_metrics'].get(metric, 0)
                rel_change = perf_comp['relative_changes'].get(metric, 0)
                
                f.write(f"| {metric} | {ref_val:.4f} | {curr_val:.4f} | {rel_change:+.2f}% |\n")
            
            f.write("\n")
            
            # Alertas
            if perf_comp['alerts']:
                f.write("#### Alertas\n\n")
                for alert in perf_comp['alerts']:
                    f.write(f"- **{alert['metric']}** ({alert['severity']}): {alert['message']}\n")
                f.write("\n")
        
        f.write("\n## Recomendaciones Generales\n\n")
        
        # Analizar si hay problemas críticos
        has_critical_issues = any(
            r['summary']['drift_severity'] == 'high' or r['summary']['performance_degradation']
            for r in all_results.values()
        )
        
        if has_critical_issues:
            f.write("### ⚠️ Se detectaron problemas significativos\n\n")
            f.write("**Acciones recomendadas:**\n\n")
            f.write("1. **Revisión inmediata del feature pipeline**\n")
            f.write("   - Verificar que el procesamiento de datos sigue siendo correcto\n")
            f.write("   - Validar que no hay cambios en las transformaciones aplicadas\n\n")
            
            f.write("2. **Análisis de causas del drift**\n")
            f.write("   - Investigar cambios en la distribución de datos de entrada\n")
            f.write("   - Revisar si hay cambios en el proceso de recolección de datos\n\n")
            
            f.write("3. **Considerar retraining del modelo**\n")
            f.write("   - Si la degradación es significativa (>5% en métricas clave)\n")
            f.write("   - Reentrenar con datos más recientes que reflejen la nueva distribución\n")
            f.write("   - Validar el nuevo modelo antes de desplegar\n\n")
            
            f.write("4. **Implementar monitoreo continuo**\n")
            f.write("   - Configurar alertas automáticas para drift en producción\n")
            f.write("   - Establecer umbrales de alerta basados en estos resultados\n")
            f.write("   - Programar evaluaciones periódicas del modelo\n\n")
        else:
            f.write("No se detectaron problemas críticos. El modelo mantiene un desempeño estable.\n\n")
            f.write("**Recomendaciones:**\n\n")
            f.write("- Continuar con el monitoreo regular\n")
            f.write("- Establecer umbrales de alerta basados en estos resultados\n")
            f.write("- Documentar estos valores como línea base para futuras comparaciones\n\n")
    
    logger.info(f"Reporte consolidado guardado en: {report_path}")


if __name__ == "__main__":
    main()

