# -*- coding: utf-8 -*-
"""
ModelPredictor: Clase para realizar predicciones con modelos entrenados.

Esta clase maneja la carga de modelos, procesamiento de datos de entrada
y generación de predicciones con interpretabilidad y confianza.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import joblib
import json
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix
import shap
import lime
import lime.lime_tabular


@dataclass
class PredictionConfig:
    """Configuración para las predicciones."""
    # Umbral de decisión para clasificación binaria
    decision_threshold: float = 0.5
    # Incluir explicabilidad
    include_explanations: bool = True
    # Método de explicabilidad
    explanation_method: str = "shap"  # "shap", "lime"
    # Guardar predicciones
    save_predictions: bool = True
    # Formato de salida
    output_format: str = "csv"  # "csv", "json", "parquet"
    # Incluir probabilidades
    include_probabilities: bool = True
    # Incluir explicaciones en la salida
    include_explanations_in_output: bool = False


class ModelPredictor:
    """
    Clase para realizar predicciones con modelos entrenados.
    
    Esta clase encapsula métodos para cargar modelos, procesar datos de entrada
    y generar predicciones con interpretabilidad.
    """
    
    def __init__(self, config: PredictionConfig = None):
        """
        Inicializa el predictor de modelos.
        
        Args:
            config: Configuración para las predicciones
        """
        self.config = config or PredictionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Modelos cargados
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Procesadores de datos
        self.data_processor = None
        self.feature_engineer = None
        
        # Explicadores
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Información de características
        self.feature_names = None
        self.target_classes = None
        
    def load_model(self, model_path: str, model_name: str = "model") -> None:
        """
        Carga un modelo desde archivo.
        
        Args:
            model_path: Ruta al archivo del modelo
            model_name: Nombre del modelo
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
            model = joblib.load(model_path)
            self.models[model_name] = model
            
            if model_name == "best_model" or self.best_model is None:
                self.best_model = model
                self.best_model_name = model_name
            
            self.logger.info(f"Modelo {model_name} cargado desde: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo {model_name}: {str(e)}")
            raise
    
    def load_models_from_directory(self, models_dir: str) -> None:
        """
        Carga todos los modelos desde un directorio.
        
        Args:
            models_dir: Directorio con los modelos
        """
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {models_dir}")
        
        # Cargar todos los archivos .joblib
        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            self.load_model(str(model_file), model_name)
        
        # Cargar información adicional si existe
        info_file = models_dir / "model_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                model_info = json.load(f)
                self.feature_names = model_info.get('feature_names')
                self.target_classes = model_info.get('target_classes')
        
        self.logger.info(f"Cargados {len(self.models)} modelos desde: {models_dir}")
    
    def set_data_processors(self, data_processor: Any, feature_engineer: Any) -> None:
        """
        Establece los procesadores de datos necesarios.
        
        Args:
            data_processor: Instancia de DataProcessor
            feature_engineer: Instancia de FeatureEngineer
        """
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.logger.info("Procesadores de datos establecidos")
    
    def preprocess_input_data(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """
        Preprocesa los datos de entrada para predicción.
        
        Args:
            data: Datos de entrada en diferentes formatos
            
        Returns:
            DataFrame preprocesado
        """
        self.logger.info("Preprocesando datos de entrada")
        
        # Convertir a DataFrame si es necesario
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Formato de datos no soportado")
        
        # Aplicar procesamiento de datos si está disponible
        if self.data_processor:
            df = self.data_processor.clean_data(df)
            X, _ = self.data_processor.prepare_features(df)
            df = X
        
        # Aplicar ingeniería de características si está disponible
        if self.feature_engineer:
            df = self.feature_engineer.transform(df)
        
        self.logger.info(f"Datos preprocesados: {df.shape}")
        return df
    
    def predict(self, data: Union[pd.DataFrame, Dict, List], 
                model_name: str = None, return_probabilities: bool = None) -> Dict[str, Any]:
        """
        Realiza predicciones con el modelo especificado.
        
        Args:
            data: Datos de entrada
            model_name: Nombre del modelo a usar (None para el mejor modelo)
            return_probabilities: Si incluir probabilidades
            
        Returns:
            Diccionario con predicciones y metadatos
        """
        # Seleccionar modelo
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Modelo {model_name} no encontrado")
        
        # Determinar si incluir probabilidades
        if return_probabilities is None:
            return_probabilities = self.config.include_probabilities
        
        self.logger.info(f"Realizando predicciones con modelo: {model_name}")
        
        # Preprocesar datos
        X = self.preprocess_input_data(data)
        
        # Hacer predicciones
        predictions = model.predict(X)
        
        results = {
            'model_name': model_name,
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat(),
            'data_shape': X.shape
        }
        
        # Incluir probabilidades si es posible
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            results['probabilities'] = probabilities.tolist()
            
            # Aplicar umbral de decisión personalizado
            if self.config.decision_threshold != 0.5:
                binary_predictions = (probabilities[:, 1] >= self.config.decision_threshold).astype(int)
                results['binary_predictions'] = binary_predictions.tolist()
        
        # Incluir explicaciones si está habilitado
        if self.config.include_explanations:
            explanations = self._generate_explanations(model, X)
            results['explanations'] = explanations
        
        self.logger.info(f"Predicciones completadas: {len(predictions)} muestras")
        return results
    
    def predict_batch(self, data: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """
        Realiza predicciones en lote.
        
        Args:
            data: DataFrame con datos de entrada
            model_name: Nombre del modelo a usar
            
        Returns:
            DataFrame con predicciones y metadatos
        """
        self.logger.info(f"Realizando predicciones en lote: {data.shape[0]} muestras")
        
        # Realizar predicciones
        results = self.predict(data, model_name)
        
        # Crear DataFrame de resultados
        output_df = data.copy()
        output_df['prediction'] = results['predictions']
        
        if 'probabilities' in results:
            probs = np.array(results['probabilities'])
            output_df['prob_class_0'] = probs[:, 0]
            output_df['prob_class_1'] = probs[:, 1]
        
        if 'binary_predictions' in results:
            output_df['binary_prediction'] = results['binary_predictions']
        
        # Incluir explicaciones si están disponibles
        if 'explanations' in results and self.config.include_explanations_in_output:
            explanations_df = pd.DataFrame(results['explanations'])
            output_df = pd.concat([output_df, explanations_df], axis=1)
        
        return output_df
    
    def _generate_explanations(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera explicaciones para las predicciones.
        
        Args:
            model: Modelo entrenado
            X: Características
            
        Returns:
            Diccionario con explicaciones
        """
        explanations = {}
        
        try:
            if self.config.explanation_method == "shap":
                explanations = self._generate_shap_explanations(model, X)
            elif self.config.explanation_method == "lime":
                explanations = self._generate_lime_explanations(model, X)
        except Exception as e:
            self.logger.warning(f"Error generando explicaciones: {str(e)}")
        
        return explanations
    
    def _generate_shap_explanations(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera explicaciones usando SHAP.
        
        Args:
            model: Modelo entrenado
            X: Características
            
        Returns:
            Diccionario con explicaciones SHAP
        """
        try:
            # Crear explicador SHAP
            if self.shap_explainer is None:
                # Usar un subconjunto de datos para crear el explicador
                background_data = X.sample(min(100, len(X)), random_state=42)
                self.shap_explainer = shap.Explainer(model, background_data)
            
            # Calcular valores SHAP
            shap_values = self.shap_explainer(X)
            
            explanations = {
                'shap_values': shap_values.values.tolist(),
                'base_values': shap_values.base_values.tolist(),
                'feature_names': X.columns.tolist()
            }
            
            return explanations
            
        except Exception as e:
            self.logger.warning(f"Error generando explicaciones SHAP: {str(e)}")
            return {}
    
    def _generate_lime_explanations(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera explicaciones usando LIME.
        
        Args:
            model: Modelo entrenado
            X: Características
            
        Returns:
            Diccionario con explicaciones LIME
        """
        try:
            # Crear explicador LIME
            if self.lime_explainer is None:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=X.columns,
                    class_names=['Good', 'Bad'],
                    mode='classification'
                )
            
            explanations = []
            for i in range(len(X)):
                exp = self.lime_explainer.explain_instance(
                    X.iloc[i].values,
                    model.predict_proba,
                    num_features=min(10, len(X.columns))
                )
                
                explanations.append({
                    'explanation': exp.as_list(),
                    'prediction': exp.predicted_value,
                    'confidence': exp.score
                })
            
            return {'lime_explanations': explanations}
            
        except Exception as e:
            self.logger.warning(f"Error generando explicaciones LIME: {str(e)}")
            return {}
    
    def evaluate_predictions(self, y_true: pd.Series, y_pred: np.ndarray, 
                           y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evalúa las predicciones contra valores reales.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_pred_proba: Probabilidades de predicción
            
        Returns:
            Diccionario con métricas de evaluación
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Reporte de clasificación
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        return metrics
    
    def save_predictions(self, predictions: Union[Dict, pd.DataFrame], 
                        output_path: str) -> None:
        """
        Guarda las predicciones en archivo.
        
        Args:
            predictions: Predicciones a guardar
            output_path: Ruta de salida
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(predictions, pd.DataFrame):
            if self.config.output_format == "csv":
                predictions.to_csv(output_path, index=False)
            elif self.config.output_format == "parquet":
                predictions.to_parquet(output_path, index=False)
            elif self.config.output_format == "json":
                predictions.to_json(output_path, orient='records', indent=2)
        else:
            # Guardar como JSON
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
        
        self.logger.info(f"Predicciones guardadas en: {output_path}")
    
    def get_prediction_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un resumen de las predicciones.
        
        Args:
            predictions: Diccionario con predicciones
            
        Returns:
            Diccionario con resumen estadístico
        """
        pred_array = np.array(predictions['predictions'])
        
        summary = {
            'total_predictions': len(pred_array),
            'positive_predictions': int(np.sum(pred_array)),
            'negative_predictions': int(len(pred_array) - np.sum(pred_array)),
            'positive_rate': float(np.mean(pred_array)),
            'model_used': predictions.get('model_name', 'unknown'),
            'timestamp': predictions.get('timestamp', 'unknown')
        }
        
        if 'probabilities' in predictions:
            probs = np.array(predictions['probabilities'])
            summary['avg_probability'] = float(np.mean(probs[:, 1]))
            summary['min_probability'] = float(np.min(probs[:, 1]))
            summary['max_probability'] = float(np.max(probs[:, 1]))
        
        return summary


def main():
    """Función principal para ejecutar predicciones desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Realizar predicciones con modelos entrenados")
    parser.add_argument("model_path", help="Ruta al modelo o directorio de modelos")
    parser.add_argument("input_file", help="Archivo CSV con datos de entrada")
    parser.add_argument("output_file", help="Archivo de salida para predicciones")
    parser.add_argument("--model-name", help="Nombre del modelo a usar")
    parser.add_argument("--threshold", type=float, help="Umbral de decisión", default=0.5)
    parser.add_argument("--format", choices=["csv", "json", "parquet"], 
                      help="Formato de salida", default="csv")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear configuración
    config = PredictionConfig(
        decision_threshold=args.threshold,
        output_format=args.format
    )
    
    # Crear predictor
    predictor = ModelPredictor(config)
    
    # Cargar modelo(s)
    model_path = Path(args.model_path)
    if model_path.is_file():
        predictor.load_model(str(model_path), args.model_name or "model")
    else:
        predictor.load_models_from_directory(str(model_path))
    
    # Cargar datos de entrada
    input_data = pd.read_csv(args.input_file)
    
    # Realizar predicciones
    if len(input_data) == 1:
        # Predicción individual
        results = predictor.predict(input_data.iloc[0].to_dict(), args.model_name)
    else:
        # Predicciones en lote
        results_df = predictor.predict_batch(input_data, args.model_name)
        results = results_df.to_dict('records')
    
    # Guardar predicciones
    predictor.save_predictions(results, args.output_file)
    
    # Mostrar resumen
    if isinstance(results, dict):
        summary = predictor.get_prediction_summary(results)
        print("\nResumen de predicciones:")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
