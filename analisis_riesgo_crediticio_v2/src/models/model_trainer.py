# -*- coding: utf-8 -*-
"""
ModelTrainer: Clase para entrenamiento y evaluación de modelos de machine learning.

Esta clase maneja el entrenamiento, validación cruzada, optimización de hiperparámetros
y evaluación de modelos siguiendo las mejores prácticas de MLOps.
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

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, 
    cross_val_score, validation_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


@dataclass
class ModelConfig:
    """Configuración para el entrenamiento de modelos."""
    # Modelos a entrenar
    models: Dict[str, Any] = None
    # Parámetros para GridSearch
    param_grids: Dict[str, Dict] = None
    # Configuración de validación cruzada
    cv_folds: int = 5
    # Tamaño del conjunto de prueba
    test_size: float = 0.25
    # Métricas de evaluación
    scoring_metrics: List[str] = None
    # Random state
    random_state: int = 42
    # Usar MLflow para tracking
    use_mlflow: bool = True
    # Nombre del experimento MLflow
    experiment_name: str = "german_credit_risk"
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=self.random_state
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                "GradientBoosting": GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                "SVM": SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=self.random_state
                )
            }
        
        if self.param_grids is None:
            self.param_grids = {
                "LogisticRegression": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                },
                "RandomForest": {
                    "n_estimators": [200, 300, 400],
                    "max_depth": [None, 8, 12, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "GradientBoosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                },
                "SVM": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto", 0.001, 0.01]
                }
            }
        
        if self.scoring_metrics is None:
            self.scoring_metrics = ["roc_auc", "average_precision", "f1", "accuracy"]


class ModelTrainer:
    """
    Clase para entrenamiento y evaluación de modelos de machine learning.
    
    Esta clase encapsula métodos para entrenar múltiples modelos,
    optimizar hiperparámetros y evaluar el rendimiento.
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            config: Configuración para el entrenamiento
        """
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Modelos entrenados
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Métricas de evaluación
        self.evaluation_results = {}
        self.cv_results = {}
        
        # Configurar MLflow
        if self.config.use_mlflow:
            self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Configura MLflow para tracking de experimentos."""
        try:
            mlflow.set_experiment(self.config.experiment_name)
            self.logger.info(f"MLflow configurado para experimento: {self.config.experiment_name}")
        except Exception as e:
            self.logger.warning(f"Error configurando MLflow: {str(e)}")
            self.config.use_mlflow = False
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            X: Características
            y: Target
            
        Returns:
            Tupla con (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state
        )
        
        self.logger.info(f"Conjunto de entrenamiento: {X_train.shape}")
        self.logger.info(f"Conjunto de prueba: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, model: Any, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Entrena un modelo individual y evalúa su rendimiento.
        
        Args:
            model_name: Nombre del modelo
            model: Instancia del modelo
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_test: Características de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con resultados de evaluación
        """
        self.logger.info(f"Entrenando modelo: {model_name}")
        
        # Entrenar modelo
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        
        # Validación cruzada
        cv_scores = self._cross_validate_model(model, X_train, y_train)
        
        # Guardar modelo
        self.trained_models[model_name] = model
        
        # Logging con MLflow
        if self.config.use_mlflow:
            self._log_model_mlflow(model_name, model, metrics, cv_scores, X_test, y_test)
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        self.logger.info(f"Modelo {model_name} entrenado exitosamente")
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcula métricas de evaluación.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            y_pred_proba: Probabilidades de predicción
            
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Realiza validación cruzada en un modelo.
        
        Args:
            model: Modelo a validar
            X: Características
            y: Target
            
        Returns:
            Diccionario con scores de validación cruzada
        """
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        cv_scores = {}
        for metric in self.config.scoring_metrics:
            if metric == 'average_precision':
                scoring = 'average_precision'
            elif metric == 'roc_auc':
                scoring = 'roc_auc'
            elif metric == 'f1':
                scoring = 'f1_weighted'
            elif metric == 'accuracy':
                scoring = 'accuracy'
            else:
                continue
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_scores[metric] = scores.tolist()
        
        return cv_scores
    
    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, 
                               y_train: pd.Series) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros usando GridSearchCV.
        
        Args:
            model_name: Nombre del modelo
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Diccionario con resultados de optimización
        """
        if model_name not in self.config.param_grids:
            self.logger.warning(f"No hay parámetros definidos para {model_name}")
            return {}
        
        self.logger.info(f"Optimizando hiperparámetros para {model_name}")
        
        # Crear pipeline si es necesario
        if model_name in ["LogisticRegression", "SVM"]:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', self.config.models[model_name])
            ])
            param_grid = {
                f'model__{k}': v for k, v in self.config.param_grids[model_name].items()
            }
        else:
            pipeline = self.config.models[model_name]
            param_grid = self.config.param_grids[model_name]
        
        # Configurar GridSearchCV
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1
        )
        
        # Ejecutar búsqueda
        start_time = datetime.now()
        grid_search.fit(X_train, y_train)
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Guardar mejor modelo
        self.trained_models[f"{model_name}_optimized"] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'optimization_time': optimization_time,
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Optimización completada para {model_name}")
        self.logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        self.logger.info(f"Mejor score: {grid_search.best_score_:.4f}")
        
        return results
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Entrena todos los modelos configurados.
        
        Args:
            X: Características
            y: Target
            
        Returns:
            Diccionario con resultados de todos los modelos
        """
        self.logger.info("Iniciando entrenamiento de todos los modelos")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        all_results = {}
        
        # Entrenar modelos base
        for model_name, model in self.config.models.items():
            try:
                results = self.train_single_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                all_results[model_name] = results
                
                # Optimizar hiperparámetros
                optimization_results = self.optimize_hyperparameters(
                    model_name, X_train, y_train
                )
                if optimization_results:
                    all_results[f"{model_name}_optimized"] = optimization_results
                    
            except Exception as e:
                self.logger.error(f"Error entrenando {model_name}: {str(e)}")
                continue
        
        # Seleccionar mejor modelo
        self._select_best_model(all_results)
        
        self.evaluation_results = all_results
        self.logger.info("Entrenamiento de todos los modelos completado")
        
        return all_results
    
    def _select_best_model(self, results: Dict[str, Any]) -> None:
        """
        Selecciona el mejor modelo basado en las métricas.
        
        Args:
            results: Resultados de evaluación de todos los modelos
        """
        best_score = 0
        best_model_name = None
        
        for model_name, result in results.items():
            if 'metrics' in result and 'average_precision' in result['metrics']:
                score = result['metrics']['average_precision']
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            self.best_model_name = best_model_name
            self.best_model = self.trained_models[best_model_name]
            self.logger.info(f"Mejor modelo seleccionado: {best_model_name} (score: {best_score:.4f})")
    
    def _log_model_mlflow(self, model_name: str, model: Any, metrics: Dict[str, float],
                         cv_scores: Dict[str, List[float]], X_test: pd.DataFrame, 
                         y_test: pd.Series) -> None:
        """
        Registra modelo y métricas en MLflow.
        
        Args:
            model_name: Nombre del modelo
            model: Modelo entrenado
            metrics: Métricas de evaluación
            cv_scores: Scores de validación cruzada
            X_test: Características de prueba
            y_test: Target de prueba
        """
        try:
            with mlflow.start_run(run_name=model_name):
                # Log parámetros
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                
                # Log métricas
                mlflow.log_metrics(metrics)
                
                # Log métricas de CV
                for metric, scores in cv_scores.items():
                    mlflow.log_metric(f"cv_{metric}_mean", np.mean(scores))
                    mlflow.log_metric(f"cv_{metric}_std", np.std(scores))
                
                # === Firma + ejemplo de entrada ===
                input_example = X_test.head(3)
                if hasattr(model, "predict_proba"):
                    y_hat = model.predict_proba(X_test)[:, 1]
                else:
                    # Fallback raro: si algún modelo no tiene predict_proba
                    y_hat = model.predict(X_test)
                signature = infer_signature(input_example, y_hat)

                # === Log de modelo con MLflow (usar name= en lugar de artifact_path) ===
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    input_example=input_example,
                    signature=signature
                )
                
        except Exception as e:
            self.logger.warning(f"Error logging en MLflow: {str(e)}")
    
    def save_models(self, output_dir: str) -> None:
        """
        Guarda todos los modelos entrenados.
        
        Args:
            output_dir: Directorio donde guardar los modelos
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelos individuales
        for model_name, model in self.trained_models.items():
            model_path = output_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            self.logger.info(f"Modelo {model_name} guardado en: {model_path}")
        
        # Guardar mejor modelo
        if self.best_model:
            best_model_path = output_dir / "best_model.joblib"
            joblib.dump(self.best_model, best_model_path)
            self.logger.info(f"Mejor modelo guardado en: {best_model_path}")
        
        # Guardar resultados de evaluación
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convertir numpy arrays a listas para JSON
            json_results = self._convert_for_json(self.evaluation_results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Resultados de evaluación guardados en: {results_path}")
    
    def _convert_for_json(self, obj: Any) -> Any:
        """
        Convierte objetos numpy a tipos compatibles con JSON.
        
        Args:
            obj: Objeto a convertir
            
        Returns:
            Objeto convertido
        """
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def load_models(self, input_dir: str) -> None:
        """
        Carga modelos previamente entrenados.
        
        Args:
            input_dir: Directorio donde están los modelos
        """
        input_dir = Path(input_dir)
        
        # Cargar modelos individuales
        for model_file in input_dir.glob("*.joblib"):
            if model_file.name != "best_model.joblib":
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.trained_models[model_name] = model
                self.logger.info(f"Modelo {model_name} cargado desde: {model_file}")
        
        # Cargar mejor modelo
        best_model_path = input_dir / "best_model.joblib"
        if best_model_path.exists():
            self.best_model = joblib.load(best_model_path)
            self.logger.info(f"Mejor modelo cargado desde: {best_model_path}")
        
        # Cargar resultados de evaluación
        results_path = input_dir / "evaluation_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.evaluation_results = json.load(f)
            self.logger.info(f"Resultados de evaluación cargados desde: {results_path}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Genera un resumen de todos los modelos entrenados.
        
        Returns:
            DataFrame con resumen de modelos
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0),
                    'Average Precision': metrics.get('average_precision', 0),
                    'Training Time': metrics.get('training_time', 0)
                })
        
        return pd.DataFrame(summary_data).sort_values('Average Precision', ascending=False)


def main():
    """Función principal para ejecutar el entrenamiento desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar modelos para German Credit")
    parser.add_argument("input_file", help="Archivo CSV con datos procesados")
    parser.add_argument("--target", help="Columna target", default="target_bad")
    parser.add_argument("--output-dir", help="Directorio para guardar modelos", default="models")
    parser.add_argument("--experiment-name", help="Nombre del experimento MLflow", 
                      default="german_credit_risk")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear configuración
    config = ModelConfig(experiment_name=args.experiment_name)
    
    # Crear entrenador
    trainer = ModelTrainer(config)
    
    # Cargar datos
    df = pd.read_csv(args.input_file)
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    # Entrenar modelos
    results = trainer.train_all_models(X, y)
    
    # Guardar modelos
    trainer.save_models(args.output_dir)
    
    # Mostrar resumen
    summary = trainer.get_model_summary()
    print("\nResumen de modelos:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
