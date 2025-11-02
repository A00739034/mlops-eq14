#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejemplo de uso del Pipeline Automatizado de Scikit-learn

Este script muestra cómo usar el pipeline automatizado de sklearn
para entrenar y evaluar modelos de manera sencilla.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from models.sklearn_pipeline import (
    create_sklearn_pipeline,
    SklearnPipelineManager
)


def ejemplo_basico():
    """Ejemplo básico de uso del pipeline."""
    print("="*70)
    print("EJEMPLO 1: Uso Básico del Pipeline")
    print("="*70)
    
    # Cargar datos
    data_path = Path(__file__).parent.parent / "data" / "raw" / "german_credit_modified.csv"
    if not data_path.exists():
        print(f"❌ Error: No se encontró el archivo {data_path}")
        print("Por favor, asegúrate de que el archivo de datos existe.")
        return
    
    df = pd.read_csv(data_path)
    
    # Preparar datos
    if "kredit" in df.columns:
        df["kredit"] = pd.to_numeric(df["kredit"], errors='coerce')
        df["target_bad"] = df["kredit"].map({1: 0, 0: 1}).astype("Int64")
        df = df.drop(columns=["kredit"])
    
    if "mixed_type_col" in df.columns:
        df = df.drop(columns=["mixed_type_col"])
    
    df = df.dropna(subset=["target_bad"])
    X = df.drop(columns=["target_bad"])
    y = df["target_bad"].astype("int64")
    
    print(f"\n📊 Datos cargados: {X.shape[0]} filas, {X.shape[1]} características")
    
    # Definir variables
    continuous_vars = ["hoehe", "laufzeit", "alter"]
    categorical_vars = [
        "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
        "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
        "bishkred", "beruf", "pers", "telef", "gastarb"
    ]
    
    valid_domains = {
        "laufkont": {1, 2, 3, 4}, "moral": {0, 1, 2, 3, 4},
        "verw": set(range(0, 11)), "sparkont": {1, 2, 3, 4, 5},
        "beszeit": {1, 2, 3, 4, 5}, "rate": {1, 2, 3, 4},
        "famges": {1, 2, 3, 4}, "buerge": {1, 2, 3},
        "wohnzeit": {1, 2, 3, 4}, "verm": {1, 2, 3, 4},
        "weitkred": {1, 2, 3}, "wohn": {1, 2, 3},
        "bishkred": {1, 2, 3, 4}, "beruf": {1, 2, 3, 4},
        "pers": {1, 2}, "telef": {1, 2}, "gastarb": {1, 2}
    }
    
    valid_ranges = {
        "alter": (18, 75),
        "laufzeit": (4, 72),
        "hoehe": (250, None)
    }
    
    # Crear pipeline con Logistic Regression
    print("\n🔧 Creando pipeline con Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    
    pipeline = create_sklearn_pipeline(
        model=model,
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars,
        valid_domains=valid_domains,
        valid_ranges=valid_ranges,
        scale_features=True,
        feature_selection=True,
        n_features_select=15,
        feature_selection_method='mutual_info',
        create_interactions=True,
        create_ratios=True,
        create_binning=True
    )
    
    # Crear gestor y entrenar
    print("🚀 Entrenando pipeline...")
    manager = SklearnPipelineManager(pipeline=pipeline)
    
    results = manager.train_and_evaluate(
        X=X,
        y=y,
        test_size=0.25,
        random_state=42,
        use_mlflow=False
    )
    
    # Mostrar resultados
    print("\n✅ Resultados:")
    print(f"   Modelo: {results['model_type']}")
    print("\n   Métricas de Prueba:")
    for metric, value in results['test_metrics'].items():
        print(f"   {metric:20s}: {value:.4f}")
    
    # Guardar modelo
    output_path = Path(__file__).parent.parent / "models" / "example_pipeline.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manager.save(str(output_path))
    print(f"\n💾 Modelo guardado en: {output_path}")


def ejemplo_multiple_modelos():
    """Ejemplo comparando múltiples modelos."""
    print("\n" + "="*70)
    print("EJEMPLO 2: Comparación de Múltiples Modelos")
    print("="*70)
    
    # Cargar datos (mismo proceso que antes)
    data_path = Path(__file__).parent.parent / "data" / "raw" / "german_credit_modified.csv"
    if not data_path.exists():
        print(f"❌ Error: No se encontró el archivo {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    if "kredit" in df.columns:
        df["kredit"] = pd.to_numeric(df["kredit"], errors='coerce')
        df["target_bad"] = df["kredit"].map({1: 0, 0: 1}).astype("Int64")
        df = df.drop(columns=["kredit"])
    
    if "mixed_type_col" in df.columns:
        df = df.drop(columns=["mixed_type_col"])
    
    df = df.dropna(subset=["target_bad"])
    X = df.drop(columns=["target_bad"])
    y = df["target_bad"].astype("int64")
    
    # Configuración común
    continuous_vars = ["hoehe", "laufzeit", "alter"]
    categorical_vars = [
        "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
        "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
        "bishkred", "beruf", "pers", "telef", "gastarb"
    ]
    
    valid_domains = {
        "laufkont": {1, 2, 3, 4}, "moral": {0, 1, 2, 3, 4},
        "verw": set(range(0, 11)), "sparkont": {1, 2, 3, 4, 5},
        "beszeit": {1, 2, 3, 4, 5}, "rate": {1, 2, 3, 4},
        "famges": {1, 2, 3, 4}, "buerge": {1, 2, 3},
        "wohnzeit": {1, 2, 3, 4}, "verm": {1, 2, 3, 4},
        "weitkred": {1, 2, 3}, "wohn": {1, 2, 3},
        "bishkred": {1, 2, 3, 4}, "beruf": {1, 2, 3, 4},
        "pers": {1, 2}, "telef": {1, 2}, "gastarb": {1, 2}
    }
    
    valid_ranges = {
        "alter": (18, 75),
        "laufzeit": (4, 72),
        "hoehe": (250, None)
    }
    
    # Probar múltiples modelos
    modelos = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }
    
    resultados_comparacion = {}
    
    for nombre, modelo in modelos.items():
        print(f"\n🔧 Entrenando {nombre}...")
        
        pipeline = create_sklearn_pipeline(
            model=modelo,
            continuous_vars=continuous_vars,
            categorical_vars=categorical_vars,
            valid_domains=valid_domains,
            valid_ranges=valid_ranges,
            scale_features=True,
            feature_selection=True,
            n_features_select=15,
            feature_selection_method='mutual_info'
        )
        
        manager = SklearnPipelineManager(pipeline=pipeline)
        results = manager.train_and_evaluate(
            X=X, y=y, test_size=0.25, random_state=42, use_mlflow=False
        )
        
        resultados_comparacion[nombre] = results['test_metrics']
    
    # Mostrar comparación
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    
    metricas = ['accuracy', 'f1', 'roc_auc', 'average_precision']
    
    print("\n" + " " * 20, end="")
    for metrica in metricas:
        print(f"{metrica:>15s}", end="")
    print()
    
    for nombre, metricas_modelo in resultados_comparacion.items():
        print(f"{nombre:20s}", end="")
        for metrica in metricas:
            valor = metricas_modelo.get(metrica, 0)
            print(f"{valor:15.4f}", end="")
        print()
    
    # Encontrar mejor modelo
    mejor_metrica = 'average_precision'
    mejor_valor = 0
    mejor_modelo = None
    
    for nombre, metricas_modelo in resultados_comparacion.items():
        valor = metricas_modelo.get(mejor_metrica, 0)
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_modelo = nombre
    
    print(f"\n🏆 Mejor modelo según {mejor_metrica}: {mejor_modelo} ({mejor_valor:.4f})")


def ejemplo_predicciones():
    """Ejemplo de uso del pipeline para hacer predicciones."""
    print("\n" + "="*70)
    print("EJEMPLO 3: Uso del Pipeline para Predicciones")
    print("="*70)
    
    # Este ejemplo asume que ya tienes un modelo entrenado
    model_path = Path(__file__).parent.parent / "models" / "example_pipeline.joblib"
    
    if not model_path.exists():
        print(f"⚠️  No se encontró modelo en {model_path}")
        print("   Ejecuta primero el ejemplo básico para entrenar un modelo.")
        return
    
    # Cargar modelo
    print(f"\n📂 Cargando modelo desde: {model_path}")
    manager = SklearnPipelineManager()
    manager.load(str(model_path))
    
    # Crear datos de ejemplo para predicción
    print("\n🔮 Creando datos de ejemplo...")
    datos_ejemplo = pd.DataFrame({
        'laufkont': [2],
        'moral': [4],
        'verw': [3],
        'sparkont': [1],
        'beszeit': [2],
        'rate': [4],
        'famges': [2],
        'buerge': [1],
        'wohnzeit': [2],
        'verm': [2],
        'hoehe': [1500],
        'laufzeit': [24],
        'weitkred': [3],
        'wohn': [1],
        'bishkred': [1],
        'beruf': [2],
        'pers': [2],
        'alter': [35],
        'telef': [1],
        'gastarb': [2]
    })
    
    # Hacer predicciones
    print("\n🎯 Realizando predicciones...")
    predicciones = manager.predict(datos_ejemplo)
    probabilidades = manager.predict_proba(datos_ejemplo)
    
    print(f"\n📊 Resultado:")
    print(f"   Predicción: {'Riesgo ALTO' if predicciones[0] == 1 else 'Riesgo BAJO'}")
    print(f"   Probabilidad de riesgo alto: {probabilidades[0][1]:.4f}")
    print(f"   Probabilidad de riesgo bajo: {probabilidades[0][0]:.4f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EJEMPLOS DE USO DEL PIPELINE AUTOMATIZADO DE SCIKIT-LEARN")
    print("="*70)
    
    try:
        # Ejecutar ejemplos
        ejemplo_basico()
        ejemplo_multiple_modelos()
        ejemplo_predicciones()
        
        print("\n" + "="*70)
        print("✅ Todos los ejemplos completados exitosamente!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {str(e)}")
        import traceback
        traceback.print_exc()

