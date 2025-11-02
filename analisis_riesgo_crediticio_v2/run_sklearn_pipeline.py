#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para ejecutar el pipeline automatizado de Scikit-learn.

Este script ejecuta un pipeline completo de sklearn que automatiza:
- Preprocesamiento de datos
- Ingeniería de características
- Selección de características
- Entrenamiento del modelo
- Evaluación

Uso:
    python run_sklearn_pipeline.py data/raw/german_credit_modified.csv \
        --model random_forest \
        --use-mlflow \
        --output-model models/sklearn_pipeline_rf.joblib
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from models.sklearn_pipeline import (
    create_sklearn_pipeline,
    SklearnPipelineManager
)


def setup_logging(verbose: bool = False):
    """Configura el logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sklearn_pipeline.log')
        ]
    )


def load_and_prepare_data(input_file: str, target_column: str = "target_bad"):
    """
    Carga y prepara los datos para el pipeline.
    
    Args:
        input_file: Ruta del archivo CSV
        target_column: Nombre de la columna target
        
    Returns:
        Tupla (X, y) con características y target
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cargando datos desde: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Si el archivo tiene 'kredit', convertir a 'target_bad'
    if "kredit" in df.columns and target_column not in df.columns:
        df["kredit"] = pd.to_numeric(df["kredit"], errors='coerce')
        df[target_column] = df["kredit"].map({1: 0, 0: 1}).astype("Int64")
        df = df.drop(columns=["kredit"])
    
    # Eliminar columna problemática si existe
    if "mixed_type_col" in df.columns:
        df = df.drop(columns=["mixed_type_col"])
    
    # Eliminar filas sin target
    df = df.dropna(subset=[target_column])
    
    X = df.drop(columns=[target_column])
    y = df[target_column].astype("int64")
    
    logger.info(f"Datos cargados: {X.shape[0]} filas, {X.shape[1]} características")
    logger.info(f"Distribución del target: {y.value_counts().to_dict()}")
    
    return X, y


def get_model(model_name: str, random_state: int = 42):
    """
    Obtiene una instancia del modelo especificado.
    
    Args:
        model_name: Nombre del modelo
        random_state: Semilla aleatoria
        
    Returns:
        Instancia del modelo
    """
    models = {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
        "svm": SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state
        )
    }
    
    if model_name not in models:
        raise ValueError(f"Modelo no soportado: {model_name}. Opciones: {list(models.keys())}")
    
    return models[model_name]


def get_default_config():
    """Obtiene la configuración por defecto del proyecto."""
    continuous_vars = ["hoehe", "laufzeit", "alter"]
    
    categorical_vars = [
        "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
        "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
        "bishkred", "beruf", "pers", "telef", "gastarb"
    ]
    
    valid_domains = {
        "laufkont": {1, 2, 3, 4},
        "moral": {0, 1, 2, 3, 4},
        "verw": set(range(0, 11)),
        "sparkont": {1, 2, 3, 4, 5},
        "beszeit": {1, 2, 3, 4, 5},
        "rate": {1, 2, 3, 4},
        "famges": {1, 2, 3, 4},
        "buerge": {1, 2, 3},
        "wohnzeit": {1, 2, 3, 4},
        "verm": {1, 2, 3, 4},
        "weitkred": {1, 2, 3},
        "wohn": {1, 2, 3},
        "bishkred": {1, 2, 3, 4},
        "beruf": {1, 2, 3, 4},
        "pers": {1, 2},
        "telef": {1, 2},
        "gastarb": {1, 2}
    }
    
    valid_ranges = {
        "alter": (18, 75),
        "laufzeit": (4, 72),
        "hoehe": (250, None)
    }
    
    return {
        "continuous_vars": continuous_vars,
        "categorical_vars": categorical_vars,
        "valid_domains": valid_domains,
        "valid_ranges": valid_ranges
    }


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline automatizado de Scikit-learn para análisis de riesgo crediticio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar con Logistic Regression
  python run_sklearn_pipeline.py data/raw/german_credit_modified.csv --model logistic

  # Entrenar con Random Forest y usar MLflow
  python run_sklearn_pipeline.py data/raw/german_credit_modified.csv \\
      --model random_forest --use-mlflow

  # Entrenar con configuración personalizada
  python run_sklearn_pipeline.py data/raw/german_credit_modified.csv \\
      --model gradient_boosting \\
      --n-features 20 \\
      --no-interactions \\
      --output-model models/my_model.joblib
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Archivo CSV de entrada con los datos"
    )
    parser.add_argument(
        "--target",
        default="target_bad",
        help="Nombre de la columna target (default: target_bad)"
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "gradient_boosting", "svm"],
        default="logistic",
        help="Tipo de modelo a usar (default: logistic)"
    )
    parser.add_argument(
        "--output-model",
        default="models/sklearn_pipeline.joblib",
        help="Ruta para guardar el modelo entrenado (default: models/sklearn_pipeline.joblib)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Tamaño del conjunto de prueba (default: 0.25)"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=15,
        help="Número de características a seleccionar (default: 15)"
    )
    parser.add_argument(
        "--feature-selection-method",
        choices=["mutual_info", "f_classif"],
        default="mutual_info",
        help="Método de selección de características (default: mutual_info)"
    )
    parser.add_argument(
        "--no-interactions",
        action="store_true",
        help="Desactivar creación de características de interacción"
    )
    parser.add_argument(
        "--no-ratios",
        action="store_true",
        help="Desactivar creación de características de ratio"
    )
    parser.add_argument(
        "--no-binning",
        action="store_true",
        help="Desactivar creación de características de binning"
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Usar MLflow para tracking de experimentos"
    )
    parser.add_argument(
        "--experiment-name",
        default="german_credit_risk_sklearn",
        help="Nombre del experimento MLflow (default: german_credit_risk_sklearn)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar logging detallado"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Cargar datos
        X, y = load_and_prepare_data(args.input_file, args.target)
        
        # Obtener configuración
        config = get_default_config()
        
        # Obtener modelo
        model = get_model(args.model, args.random_state)
        
        logger.info(f"Creando pipeline automatizado con modelo: {args.model}")
        
        # Crear pipeline
        pipeline = create_sklearn_pipeline(
            model=model,
            continuous_vars=config["continuous_vars"],
            categorical_vars=config["categorical_vars"],
            valid_domains=config["valid_domains"],
            valid_ranges=config["valid_ranges"],
            scale_features=True,
            feature_selection=True,
            n_features_select=args.n_features,
            feature_selection_method=args.feature_selection_method,
            apply_pca=False,
            create_interactions=not args.no_interactions,
            create_ratios=not args.no_ratios,
            create_binning=not args.no_binning
        )
        
        # Crear gestor
        manager = SklearnPipelineManager(pipeline=pipeline)
        
        # Entrenar y evaluar
        logger.info("Iniciando entrenamiento y evaluación del pipeline")
        results = manager.train_and_evaluate(
            X=X,
            y=y,
            test_size=args.test_size,
            random_state=args.random_state,
            use_mlflow=args.use_mlflow,
            experiment_name=args.experiment_name,
            run_name=f"{args.model}_pipeline"
        )
        
        # Guardar modelo
        Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
        manager.save(args.output_model)
        
        # Mostrar resultados
        print("\n" + "="*70)
        print("RESULTADOS DEL PIPELINE AUTOMATIZADO DE SCIKIT-LEARN")
        print("="*70)
        print(f"\n📊 Modelo: {results['model_type']}")
        print(f"📁 Archivo de entrada: {args.input_file}")
        print(f"💾 Modelo guardado en: {args.output_model}")
        
        print("\n📈 Métricas de Entrenamiento:")
        for metric, value in results['train_metrics'].items():
            print(f"   {metric:20s}: {value:.4f}")
        
        print("\n🧪 Métricas de Prueba:")
        for metric, value in results['test_metrics'].items():
            print(f"   {metric:20s}: {value:.4f}")
        
        # Mostrar importancia de características si está disponible
        importance = manager.get_feature_importance()
        if importance is not None and len(importance) > 0:
            print("\n🔍 Top 10 Características Más Importantes:")
            print(importance.head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("✅ Pipeline ejecutado exitosamente!")
        print("="*70 + "\n")
        
        # Guardar resumen
        summary_path = Path(args.output_model).parent / "pipeline_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("RESUMEN DEL PIPELINE AUTOMATIZADO\n")
            f.write("="*50 + "\n\n")
            f.write(f"Modelo: {results['model_type']}\n")
            f.write(f"Archivo: {args.input_file}\n\n")
            f.write("Métricas de Entrenamiento:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\nMétricas de Prueba:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Resumen guardado en: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error ejecutando pipeline: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

