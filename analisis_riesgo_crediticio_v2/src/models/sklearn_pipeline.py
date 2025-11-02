# -*- coding: utf-8 -*-
"""
Pipeline de Scikit-learn: Pipeline completo que automatiza todo el flujo
desde el preprocesamiento hasta el modelo.

Este módulo crea un Pipeline de sklearn que integra:
- Preprocesamiento de datos (limpieza, validación, imputación)
- Ingeniería de características (encoding, transformaciones)
- Selección de características
- Modelo de machine learning

Todo esto en un único objeto Pipeline que se puede entrenar y usar para predicciones.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Transformador para limpiar y validar datos.
    
    Aplica:
    - Limpieza de espacios en blanco
    - Validación de dominios categóricos
    - Validación de rangos continuos
    - Conversión a tipos numéricos
    """
    
    def __init__(self, valid_domains: Dict[str, set] = None,
                 valid_ranges: Dict[str, Tuple] = None,
                 continuous_vars: List[str] = None,
                 categorical_vars: List[str] = None):
        self.valid_domains = valid_domains or {}
        self.valid_ranges = valid_ranges or {}
        self.continuous_vars = continuous_vars or []
        self.categorical_vars = categorical_vars or []
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        """Ajusta el transformador (no requiere ajuste)."""
        return self
    
    def transform(self, X):
        """Aplica la limpieza de datos."""
        X = X.copy()
        
        # Limpiar espacios en blanco
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.strip()
                X[col] = X[col].replace('', np.nan)
        
        # Convertir a numérico
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Validar dominios categóricos
        for col, valid_values in self.valid_domains.items():
            if col in X.columns:
                mask = X[col].notna() & ~X[col].isin(valid_values)
                X.loc[mask, col] = np.nan
        
        # Validar rangos continuos
        for col, (min_val, max_val) in self.valid_ranges.items():
            if col in X.columns:
                if min_val is not None:
                    X.loc[X[col] < min_val, col] = np.nan
                if max_val is not None:
                    X.loc[X[col] > max_val, col] = np.nan
        
        return X


class DataImputer(BaseEstimator, TransformerMixin):
    """
    Transformador para imputar valores faltantes.
    
    Usa mediana para variables continuas y moda para categóricas.
    """
    
    def __init__(self, continuous_vars: List[str] = None,
                 categorical_vars: List[str] = None):
        self.continuous_vars = continuous_vars or []
        self.categorical_vars = categorical_vars or []
        self.imputers_continuous = {}
        self.imputers_categorical = {}
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        """Ajusta los imputadores."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Imputadores para variables continuas
        for col in self.continuous_vars:
            if col in X.columns:
                imputer = SimpleImputer(strategy='median')
                imputer.fit(X[[col]])
                self.imputers_continuous[col] = imputer
        
        # Imputadores para variables categóricas
        categorical_cols = [col for col in self.categorical_vars if col in X.columns]
        if categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            imputer.fit(X[categorical_cols])
            self.imputers_categorical = imputer
        
        return self
    
    def transform(self, X):
        """Aplica la imputación."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Imputar continuas
        for col, imputer in self.imputers_continuous.items():
            if col in X.columns:
                X[col] = imputer.transform(X[[col]]).ravel()
        
        # Imputar categóricas
        categorical_cols = [col for col in self.categorical_vars if col in X.columns]
        if categorical_cols and self.imputers_categorical:
            X[categorical_cols] = self.imputers_categorical.transform(X[categorical_cols])
        
        return X


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformador para ingeniería de características.
    
    Crea características derivadas como:
    - Interacciones entre variables continuas
    - Ratios entre variables
    - Binning de variables continuas
    """
    
    def __init__(self, continuous_vars: List[str] = None,
                 create_interactions: bool = True,
                 create_ratios: bool = True,
                 create_binning: bool = True):
        self.continuous_vars = continuous_vars or []
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_binning = create_binning
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        """Ajusta el transformador (no requiere ajuste)."""
        return self
    
    def transform(self, X):
        """Aplica la ingeniería de características."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        continuous_cols = [col for col in self.continuous_vars if col in X.columns]
        
        # Crear interacciones
        if self.create_interactions and len(continuous_cols) >= 2:
            for i, col1 in enumerate(continuous_cols):
                for col2 in continuous_cols[i+1:]:
                    X[f"{col1}_x_{col2}"] = X[col1] * X[col2]
        
        # Crear ratios
        if self.create_ratios and len(continuous_cols) >= 2:
            for i, col1 in enumerate(continuous_cols):
                for col2 in continuous_cols[i+1:]:
                    X[f"{col1}_div_{col2}"] = np.where(
                        X[col2] != 0,
                        X[col1] / X[col2],
                        0
                    )
        
        # Crear binning
        if self.create_binning and continuous_cols:
            for col in continuous_cols:
                try:
                    X[f"{col}_bin"] = pd.qcut(X[col], q=4, labels=False, duplicates='drop')
                except:
                    pass
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Transformador para codificar variables categóricas.
    
    Usa Label Encoding para variables con pocas categorías
    y One-Hot Encoding para variables con muchas categorías.
    """
    
    def __init__(self, categorical_vars: List[str] = None,
                 max_categories_label: int = 10):
        self.categorical_vars = categorical_vars or []
        self.max_categories_label = max_categories_label
        self.label_encoders = {}
        self.onehot_encoder = None
        self.onehot_cols = []
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        """Ajusta los encoders."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        categorical_cols = [col for col in self.categorical_vars if col in X.columns]
        
        # Label encoding para variables con pocas categorías
        label_cols = []
        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique <= self.max_categories_label:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
                label_cols.append(col)
        
        # One-hot encoding para variables con muchas categorías
        onehot_cols = [col for col in categorical_cols if col not in label_cols]
        if onehot_cols:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.onehot_encoder.fit(X[onehot_cols])
            self.onehot_cols = onehot_cols
        
        return self
    
    def transform(self, X):
        """Aplica la codificación."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Aplicar label encoding
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
        
        # Aplicar one-hot encoding
        if self.onehot_encoder and self.onehot_cols:
            available_cols = [col for col in self.onehot_cols if col in X.columns]
            if available_cols:
                oh_features = self.onehot_encoder.transform(X[available_cols])
                oh_columns = []
                for col in available_cols:
                    idx = self.onehot_cols.index(col)
                    categories = self.onehot_encoder.categories_[idx]
                    oh_columns.extend([f"{col}_{int(cat)}" for cat in categories])
                
                oh_df = pd.DataFrame(oh_features, columns=oh_columns, index=X.index)
                X = pd.concat([X, oh_df], axis=1)
                X = X.drop(columns=available_cols)
        
        return X


def create_sklearn_pipeline(
    model,
    continuous_vars: List[str],
    categorical_vars: List[str],
    valid_domains: Dict[str, set] = None,
    valid_ranges: Dict[str, Tuple] = None,
    scale_features: bool = True,
    feature_selection: bool = True,
    n_features_select: int = 15,
    feature_selection_method: str = 'mutual_info',
    apply_pca: bool = False,
    n_components_pca: int = 10,
    create_interactions: bool = True,
    create_ratios: bool = True,
    create_binning: bool = True
) -> Pipeline:
    """
    Crea un pipeline completo de sklearn que automatiza todo el flujo.
    
    Args:
        model: Modelo de sklearn a usar (LogisticRegression, RandomForest, etc.)
        continuous_vars: Lista de variables continuas
        categorical_vars: Lista de variables categóricas
        valid_domains: Diccionario con dominios válidos para variables categóricas
        valid_ranges: Diccionario con rangos válidos para variables continuas
        scale_features: Si aplicar escalado de características
        feature_selection: Si aplicar selección de características
        n_features_select: Número de características a seleccionar
        feature_selection_method: Método de selección ('mutual_info', 'f_classif')
        apply_pca: Si aplicar PCA
        n_components_pca: Número de componentes PCA
        create_interactions: Si crear características de interacción
        create_ratios: Si crear características de ratio
        create_binning: Si crear características de binning
        
    Returns:
        Pipeline de sklearn completo
    """
    steps = []
    
    # Paso 1: Limpieza de datos
    cleaner = DataCleaner(
        valid_domains=valid_domains or {},
        valid_ranges=valid_ranges or {},
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars
    )
    steps.append(('cleaner', cleaner))
    
    # Paso 2: Imputación
    imputer = DataImputer(
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars
    )
    steps.append(('imputer', imputer))
    
    # Paso 3: Ingeniería de características
    feature_eng = FeatureEngineering(
        continuous_vars=continuous_vars,
        create_interactions=create_interactions,
        create_ratios=create_ratios,
        create_binning=create_binning
    )
    steps.append(('feature_engineering', feature_eng))
    
    # Paso 4: Codificación categórica
    encoder = CategoricalEncoder(
        categorical_vars=categorical_vars,
        max_categories_label=10
    )
    steps.append(('encoder', encoder))
    
    # Paso 5: Escalado (si se requiere)
    if scale_features:
        scaler = StandardScaler()
        steps.append(('scaler', scaler))
    
    # Paso 6: Selección de características (requiere y, se añade después del fit)
    if feature_selection:
        if feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features_select)
        else:
            selector = SelectKBest(score_func=f_classif, k=n_features_select)
        steps.append(('feature_selection', selector))
    
    # Paso 7: PCA (opcional)
    if apply_pca:
        pca = PCA(n_components=n_components_pca)
        steps.append(('pca', pca))
    
    # Paso 8: Modelo
    steps.append(('model', model))
    
    # Crear pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


class SklearnPipelineManager:
    """
    Gestor del pipeline de sklearn completo.
    
    Facilita el entrenamiento, evaluación y uso del pipeline automatizado.
    """
    
    def __init__(self, pipeline: Pipeline = None):
        """
        Inicializa el gestor del pipeline.
        
        Args:
            pipeline: Pipeline de sklearn. Si es None, se debe crear después.
        """
        self.pipeline = pipeline
        self.logger = logging.getLogger(__name__)
        self.feature_names_ = None
        self.training_metrics_ = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SklearnPipelineManager':
        """
        Entrena el pipeline completo.
        
        Args:
            X: DataFrame con características
            y: Serie con target
            
        Returns:
            self
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no inicializado. Use create_pipeline() primero.")
        
        self.logger.info("Entrenando pipeline completo")
        
        # Guardar nombres de características
        self.feature_names_ = X.columns.tolist()
        
        # Entrenar pipeline
        self.pipeline.fit(X, y)
        
        self.logger.info("Pipeline entrenado exitosamente")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Hace predicciones con el pipeline.
        
        Args:
            X: DataFrame con características
            
        Returns:
            Array con predicciones
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Use fit() primero.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Obtiene probabilidades de predicción.
        
        Args:
            X: DataFrame con características
            
        Returns:
            Array con probabilidades
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Use fit() primero.")
        
        if hasattr(self.pipeline.named_steps['model'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise ValueError("El modelo no soporta predict_proba")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evalúa el pipeline en un conjunto de datos.
        
        Args:
            X: DataFrame con características
            y: Serie con target
            
        Returns:
            Diccionario con métricas
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Use fit() primero.")
        
        self.logger.info("Evaluando pipeline")
        
        y_pred = self.predict(X)
        y_pred_proba = None
        
        try:
            y_pred_proba = self.predict_proba(X)[:, 1]
        except:
            pass
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y, y_pred_proba)
        
        self.training_metrics_ = metrics
        
        return metrics
    
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        random_state: int = 42,
        use_mlflow: bool = False,
        experiment_name: str = "german_credit_risk",
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Entrena y evalúa el pipeline con división train/test.
        
        Args:
            X: DataFrame con características
            y: Serie con target
            test_size: Proporción del conjunto de prueba
            random_state: Semilla aleatoria
            use_mlflow: Si usar MLflow para tracking
            experiment_name: Nombre del experimento MLflow
            run_name: Nombre del run MLflow
            
        Returns:
            Diccionario con resultados completos
        """
        self.logger.info("Entrenando y evaluando pipeline")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Entrenar pipeline
        self.fit(X_train, y_train)
        
        # Evaluar en train
        train_metrics = self.evaluate(X_train, y_train)
        
        # Evaluar en test
        test_metrics = self.evaluate(X_test, y_test)
        
        # Logging con MLflow
        if use_mlflow:
            try:
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run(run_name=run_name or f"{type(self.pipeline.named_steps['model']).__name__}"):
                    # Log parámetros del modelo
                    if hasattr(self.pipeline.named_steps['model'], 'get_params'):
                        mlflow.log_params(self.pipeline.named_steps['model'].get_params())
                    
                    # Log métricas de train
                    for metric, value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric}", value)
                    
                    # Log métricas de test
                    for metric, value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric}", value)
                    
                    # Log modelo
                    signature = infer_signature(X_test, self.predict(X_test))
                    mlflow.sklearn.log_model(
                        self.pipeline,
                        "model",
                        signature=signature,
                        input_example=X_test.head(1)
                    )
                    
                    self.logger.info("Modelo loggeado en MLflow")
            except Exception as e:
                self.logger.warning(f"Error logging en MLflow: {str(e)}")
        
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_type': type(self.pipeline.named_steps['model']).__name__
        }
        
        self.logger.info(f"Métricas de test: {test_metrics}")
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Guarda el pipeline entrenado.
        
        Args:
            filepath: Ruta donde guardar el pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Use fit() primero.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.pipeline, filepath)
        self.logger.info(f"Pipeline guardado en: {filepath}")
    
    def load(self, filepath: str) -> 'SklearnPipelineManager':
        """
        Carga un pipeline previamente entrenado.
        
        Args:
            filepath: Ruta del pipeline guardado
            
        Returns:
            self
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline no encontrado: {filepath}")
        
        self.pipeline = joblib.load(filepath)
        self.logger.info(f"Pipeline cargado desde: {filepath}")
        
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Obtiene la importancia de características si el modelo la soporta.
        
        Returns:
            DataFrame con importancia de características o None
        """
        if self.pipeline is None:
            return None
        
        model = self.pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            # Obtener nombres de características después de transformaciones
            try:
                # Intentar obtener características seleccionadas
                if 'feature_selection' in self.pipeline.named_steps:
                    selector = self.pipeline.named_steps['feature_selection']
                    selected_features = self.feature_names_[selector.get_support()]
                else:
                    selected_features = self.feature_names_
            except:
                selected_features = range(len(model.feature_importances_))
            
            importance_df = pd.DataFrame({
                'feature': selected_features[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None


def main():
    """Función principal para demostrar el uso del pipeline."""
    import argparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    parser = argparse.ArgumentParser(description="Pipeline automatizado de sklearn")
    parser.add_argument("input_file", help="Archivo CSV de entrada")
    parser.add_argument("--target", help="Columna target", default="target_bad")
    parser.add_argument("--model", choices=["logistic", "random_forest"], 
                       default="logistic", help="Tipo de modelo")
    parser.add_argument("--output-model", help="Ruta para guardar el modelo", 
                       default="models/sklearn_pipeline.joblib")
    parser.add_argument("--test-size", type=float, default=0.25, help="Tamaño del test")
    parser.add_argument("--use-mlflow", action="store_true", help="Usar MLflow")
    parser.add_argument("--experiment-name", default="german_credit_risk", help="Nombre del experimento")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Cargar datos
    df = pd.read_csv(args.input_file)
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
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
    
    # Seleccionar modelo
    if args.model == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    
    # Crear pipeline
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
    
    # Crear gestor
    manager = SklearnPipelineManager(pipeline=pipeline)
    
    # Entrenar y evaluar
    results = manager.train_and_evaluate(
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=42,
        use_mlflow=args.use_mlflow,
        experiment_name=args.experiment_name
    )
    
    # Guardar modelo
    manager.save(args.output_model)
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS DEL PIPELINE AUTOMATIZADO")
    print("="*50)
    print(f"\nModelo: {results['model_type']}")
    print("\nMétricas de Entrenamiento:")
    for metric, value in results['train_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMétricas de Prueba:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nPipeline guardado en: {args.output_model}")
    
    # Mostrar importancia de características si está disponible
    importance = manager.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Características Más Importantes:")
        print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

