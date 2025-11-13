# -*- coding: utf-8 -*-
"""
DriftDetector: Sistema completo para detectar data drift y concept drift.

Este módulo implementa:
- Generación de datos de monitoreo con distribución alterada
- Evaluación comparativa de métricas
- Detección de drift estadístico
- Visualizaciones y reportes
- Sistema de alertas con umbrales configurables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


@dataclass
class DriftConfig:
    """Configuración para la detección de drift."""
    # Umbrales para alertas
    accuracy_threshold: float = 0.05  # Disminución relativa del 5%
    roc_auc_threshold: float = 0.05
    f1_threshold: float = 0.05
    ks_statistic_threshold: float = 0.3  # Estadístico KS para drift
    p_value_threshold: float = 0.05  # p-value para significancia estadística
    
    # Configuración de drift simulado
    mean_shift_factor: float = 0.2  # Factor de desplazamiento de medias (20%)
    missing_value_rate: float = 0.1  # 10% de valores faltantes
    variance_change_factor: float = 1.5  # Aumento de varianza (50%)
    categorical_shift_probability: float = 0.15  # Probabilidad de cambio en categóricas
    
    # Columnas continuas y categóricas
    continuous_cols: List[str] = None
    categorical_cols: List[str] = None
    
    # Random state
    random_state: int = 42
    
    def __post_init__(self):
        if self.continuous_cols is None:
            self.continuous_cols = ["hoehe", "laufzeit", "alter"]
        if self.categorical_cols is None:
            self.categorical_cols = [
                "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
                "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
                "bishkred", "beruf", "pers", "telef", "gastarb"
            ]


class DriftDetector:
    """
    Clase principal para detectar data drift y concept drift.
    
    Esta clase implementa métodos para:
    - Generar datos con distribución alterada
    - Comparar distribuciones estadísticas
    - Evaluar impacto en el desempeño del modelo
    - Generar alertas y reportes
    """
    
    def __init__(self, config: DriftConfig = None):
        """
        Inicializa el detector de drift.
        
        Args:
            config: Configuración para la detección de drift
        """
        self.config = config or DriftConfig()
        self.logger = logging.getLogger(__name__)
        
        # Datos de referencia (baseline)
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_target: Optional[pd.Series] = None
        
        # Modelo y procesadores
        self.model = None
        self.data_processor = None
        self.feature_engineer = None
        
        # Resultados
        self.drift_results: Dict[str, Any] = {}
        self.metrics_comparison: Dict[str, Any] = {}
        
    def load_reference_data(self, data_path: str, target_col: str = "target_bad") -> None:
        """
        Carga los datos de referencia (baseline).
        
        Args:
            data_path: Ruta al archivo CSV con datos de referencia
            target_col: Nombre de la columna target
        """
        try:
            self.logger.info(f"Cargando datos de referencia desde: {data_path}")
            df = pd.read_csv(data_path)
            
            if target_col not in df.columns:
                raise ValueError(f"Columna target '{target_col}' no encontrada")
            
            self.reference_data = df.drop(columns=[target_col]).copy()
            self.reference_target = df[target_col].copy()
            
            self.logger.info(f"Datos de referencia cargados: {self.reference_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos de referencia: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """
        Carga el modelo entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
            self.model = joblib.load(model_path)
            self.logger.info(f"Modelo cargado desde: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def set_data_processors(self, data_processor: Any, feature_engineer: Any = None) -> None:
        """
        Establece los procesadores de datos.
        
        Args:
            data_processor: Instancia de DataProcessor
            feature_engineer: Instancia de FeatureEngineer (opcional)
        """
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.logger.info("Procesadores de datos establecidos")
    
    def generate_drift_data(
        self,
        drift_type: str = "mean_shift",
        n_samples: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Genera datos con distribución alterada (drift simulado).
        
        Args:
            drift_type: Tipo de drift a simular:
                - "mean_shift": Desplazamiento de medias en variables continuas
                - "missing_values": Introducción de valores faltantes
                - "variance_change": Cambio en la varianza
                - "categorical_shift": Cambio en distribución categórica
                - "combined": Combinación de múltiples tipos
            n_samples: Número de muestras a generar (None = mismo que referencia)
            **kwargs: Parámetros adicionales para el tipo de drift
            
        Returns:
            DataFrame con datos que presentan drift
        """
        if self.reference_data is None:
            raise ValueError("Debe cargar datos de referencia primero")
        
        self.logger.info(f"Generando datos con drift tipo: {drift_type}")
        
        n_samples = n_samples or len(self.reference_data)
        drift_data = self.reference_data.sample(
            n=min(n_samples, len(self.reference_data)),
            replace=True,
            random_state=self.config.random_state
        ).copy()
        
        if drift_type == "mean_shift":
            drift_data = self._apply_mean_shift(drift_data, **kwargs)
        elif drift_type == "missing_values":
            drift_data = self._apply_missing_values(drift_data, **kwargs)
        elif drift_type == "variance_change":
            drift_data = self._apply_variance_change(drift_data, **kwargs)
        elif drift_type == "categorical_shift":
            drift_data = self._apply_categorical_shift(drift_data, **kwargs)
        elif drift_type == "combined":
            drift_data = self._apply_combined_drift(drift_data, **kwargs)
        else:
            raise ValueError(f"Tipo de drift no soportado: {drift_type}")
        
        self.logger.info(f"Datos con drift generados: {drift_data.shape}")
        return drift_data
    
    def _apply_mean_shift(self, data: pd.DataFrame, shift_factor: Optional[float] = None) -> pd.DataFrame:
        """Aplica desplazamiento de medias a variables continuas."""
        shift_factor = shift_factor or self.config.mean_shift_factor
        drift_data = data.copy()
        
        for col in self.config.continuous_cols:
            if col in drift_data.columns:
                # Calcular desviación estándar de la referencia
                if self.reference_data is not None and col in self.reference_data.columns:
                    ref_col = self.reference_data[col].dropna()
                    if len(ref_col) > 0:
                        std = ref_col.std()
                        if std > 0:
                            # Desplazar la media
                            drift_data[col] = drift_data[col] + (shift_factor * std)
                            self.logger.debug(f"Desplazada media de {col} en {shift_factor * std:.2f}")
        
        return drift_data
    
    def _apply_missing_values(self, data: pd.DataFrame, missing_rate: Optional[float] = None) -> pd.DataFrame:
        """Introduce valores faltantes en el dataset."""
        missing_rate = missing_rate or self.config.missing_value_rate
        drift_data = data.copy()
        
        n_missing = int(len(drift_data) * missing_rate)
        np.random.seed(self.config.random_state)
        
        for col in drift_data.columns:
            if col != "target_bad":
                n_col_missing = int(n_missing * np.random.uniform(0.5, 1.5))
                missing_indices = np.random.choice(
                    drift_data.index,
                    size=min(n_col_missing, len(drift_data)),
                    replace=False
                )
                drift_data.loc[missing_indices, col] = np.nan
        
        self.logger.debug(f"Introducidos valores faltantes: {drift_data.isnull().sum().sum()}")
        return drift_data
    
    def _apply_variance_change(self, data: pd.DataFrame, variance_factor: Optional[float] = None) -> pd.DataFrame:
        """Cambia la varianza de variables continuas."""
        variance_factor = variance_factor or self.config.variance_change_factor
        drift_data = data.copy()
        
        for col in self.config.continuous_cols:
            if col in drift_data.columns:
                mean = drift_data[col].mean()
                # Aumentar varianza
                drift_data[col] = mean + (drift_data[col] - mean) * np.sqrt(variance_factor)
                self.logger.debug(f"Cambiada varianza de {col} por factor {variance_factor}")
        
        return drift_data
    
    def _apply_categorical_shift(self, data: pd.DataFrame, shift_prob: Optional[float] = None) -> pd.DataFrame:
        """Cambia la distribución de variables categóricas."""
        shift_prob = shift_prob or self.config.categorical_shift_probability
        drift_data = data.copy()
        
        np.random.seed(self.config.random_state)
        
        for col in self.config.categorical_cols:
            if col in drift_data.columns:
                # Seleccionar una fracción de valores para cambiar
                n_to_shift = int(len(drift_data) * shift_prob)
                shift_indices = np.random.choice(
                    drift_data.index,
                    size=min(n_to_shift, len(drift_data)),
                    replace=False
                )
                
                # Cambiar a valores aleatorios dentro del dominio válido
                unique_values = drift_data[col].dropna().unique()
                if len(unique_values) > 1:
                    for idx in shift_indices:
                        # Seleccionar un valor diferente al actual
                        current_val = drift_data.loc[idx, col]
                        new_values = [v for v in unique_values if v != current_val]
                        if new_values:
                            drift_data.loc[idx, col] = np.random.choice(new_values)
        
        self.logger.debug(f"Aplicado shift categórico con probabilidad {shift_prob}")
        return drift_data
    
    def _apply_combined_drift(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica múltiples tipos de drift simultáneamente."""
        drift_data = data.copy()
        
        # Aplicar mean shift solo en columnas continuas que existan
        drift_data = self._apply_mean_shift(drift_data, shift_factor=0.15)
        
        # Aplicar missing values (menor tasa)
        drift_data = self._apply_missing_values(drift_data, missing_rate=0.05)
        
        # Aplicar categorical shift solo si hay columnas categóricas
        if any(col in drift_data.columns for col in self.config.categorical_cols):
            drift_data = self._apply_categorical_shift(drift_data, shift_prob=0.1)
        
        return drift_data
    
    def detect_statistical_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detecta drift estadístico entre dos conjuntos de datos.
        
        Args:
            reference_data: Datos de referencia
            current_data: Datos actuales a comparar
            
        Returns:
            Diccionario con resultados de detección de drift
        """
        self.logger.info("Detectando drift estadístico")
        
        drift_results = {
            "continuous_drift": {},
            "categorical_drift": {},
            "overall_drift_detected": False,
            "drift_severity": "none"
        }
        
        # Detectar drift en variables continuas (KS test)
        for col in self.config.continuous_cols:
            if col in reference_data.columns and col in current_data.columns:
                ref_col = reference_data[col].dropna()
                curr_col = current_data[col].dropna()
                
                if len(ref_col) > 0 and len(curr_col) > 0:
                    ks_stat, p_value = ks_2samp(ref_col, curr_col)
                    
                    drift_results["continuous_drift"][col] = {
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "drift_detected": ks_stat > self.config.ks_statistic_threshold and p_value < self.config.p_value_threshold
                    }
        
        # Detectar drift en variables categóricas (Chi-square test)
        for col in self.config.categorical_cols:
            if col in reference_data.columns and col in current_data.columns:
                ref_col = reference_data[col].dropna()
                curr_col = current_data[col].dropna()
                
                if len(ref_col) > 0 and len(curr_col) > 0:
                    # Crear tabla de contingencia
                    ref_counts = ref_col.value_counts()
                    curr_counts = curr_col.value_counts()
                    
                    # Combinar índices
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    
                    if len(all_categories) > 1:
                        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
                        curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
                        
                        contingency_table = np.array([ref_freq, curr_freq])
                        
                        if np.sum(contingency_table) > 0:
                            try:
                                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                                
                                drift_results["categorical_drift"][col] = {
                                    "chi2_statistic": float(chi2),
                                    "p_value": float(p_value),
                                    "drift_detected": p_value < self.config.p_value_threshold
                                }
                            except Exception as e:
                                self.logger.warning(f"Error en chi-square test para {col}: {str(e)}")
        
        # Determinar severidad general del drift
        n_continuous_drift = sum(
            1 for v in drift_results["continuous_drift"].values()
            if v.get("drift_detected", False)
        )
        n_categorical_drift = sum(
            1 for v in drift_results["categorical_drift"].values()
            if v.get("drift_detected", False)
        )
        
        total_features = len(self.config.continuous_cols) + len(self.config.categorical_cols)
        drift_ratio = (n_continuous_drift + n_categorical_drift) / total_features if total_features > 0 else 0
        
        drift_results["overall_drift_detected"] = drift_ratio > 0.2  # 20% de features con drift
        drift_results["drift_ratio"] = drift_ratio
        
        if drift_ratio > 0.5:
            drift_results["drift_severity"] = "high"
        elif drift_ratio > 0.2:
            drift_results["drift_severity"] = "medium"
        elif drift_ratio > 0:
            drift_results["drift_severity"] = "low"
        else:
            drift_results["drift_severity"] = "none"
        
        self.logger.info(f"Drift detectado: {drift_results['overall_drift_detected']}, Severidad: {drift_results['drift_severity']}")
        
        return drift_results
    
    def evaluate_model_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evalúa el desempeño del modelo en un conjunto de datos.
        
        Args:
            X: Características
            y: Target real
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con métricas de desempeño
        """
        if self.model is None:
            raise ValueError("Debe cargar un modelo primero")
        
        self.logger.info(f"Evaluando modelo en {len(X)} muestras")
        
        # Si los datos ya están transformados (tienen columnas con formato one-hot), usar directamente
        # Verificar si tiene columnas transformadas (formato con _)
        has_transformed_features = any("_" in col for col in X.columns)
        
        if has_transformed_features:
            # Datos ya transformados, usar directamente
            X_processed = X.copy()
            self.logger.debug("Usando datos con features ya transformadas")
        elif self.data_processor:
            # Intentar preprocesar
            try:
                # Si X tiene target_bad, removerlo temporalmente
                if "target_bad" in X.columns:
                    X_temp = X.drop(columns=["target_bad"]).copy()
                else:
                    X_temp = X.copy()
                
                X_clean = self.data_processor.clean_data(X_temp)
                X_processed, _ = self.data_processor.prepare_features(X_clean)
            except Exception as e:
                self.logger.warning(f"Error en preprocesamiento, usando datos tal cual: {str(e)}")
                X_processed = X.copy()
        else:
            X_processed = X.copy()
        
        # Imputar valores faltantes si existen (el modelo no puede manejar NaN)
        if X_processed.isnull().any().any():
            self.logger.info("Imputando valores faltantes antes de hacer predicciones")
            from sklearn.impute import SimpleImputer
            
            # Crear imputador con estrategia de mediana para numéricas y moda para categóricas
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns
            
            X_imputed = X_processed.copy()
            
            # Imputar numéricas con mediana
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='median')
                X_imputed[numeric_cols] = numeric_imputer.fit_transform(X_processed[numeric_cols])
            
            # Imputar categóricas con moda
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                X_imputed[categorical_cols] = categorical_imputer.fit_transform(X_processed[categorical_cols])
            
            X_processed = X_imputed
        
        # Asegurar que las columnas coincidan con lo que el modelo espera
        try:
            # Hacer predicciones
            y_pred = self.model.predict(X_processed)
            
            # Obtener probabilidades si es posible
            y_pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
                except Exception as e:
                    self.logger.warning(f"No se pudieron obtener probabilidades: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error haciendo predicciones: {str(e)}")
            # Si falla, puede ser que el modelo espere un formato diferente
            # Intentar con datos tal cual (puede ser un pipeline)
            try:
                # También imputar si hay NaN en X original
                if X.isnull().any().any():
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    X_clean = pd.DataFrame(
                        imputer.fit_transform(X.select_dtypes(include=[np.number])),
                        columns=X.select_dtypes(include=[np.number]).columns,
                        index=X.index
                    )
                    X_clean = pd.concat([X_clean, X.select_dtypes(exclude=[np.number])], axis=1)
                else:
                    X_clean = X
                
                y_pred = self.model.predict(X_clean)
                y_pred_proba = self.model.predict_proba(X_clean)[:, 1] if hasattr(self.model, 'predict_proba') else None
            except Exception as e2:
                self.logger.error(f"Error detallado: {str(e2)}")
                raise ValueError(f"No se pudo hacer predicciones. El modelo espera features en un formato específico.")
        
        # Asegurar que y tenga el mismo tamaño que y_pred
        if len(y) != len(y_pred):
            min_len = min(len(y), len(y_pred))
            y = y.iloc[:min_len] if isinstance(y, pd.Series) else y[:min_len]
            y_pred = y_pred[:min_len]
            if y_pred_proba is not None:
                y_pred_proba = y_pred_proba[:min_len]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y, y_pred_proba)
            except Exception as e:
                self.logger.warning(f"Error calculando métricas con probabilidades: {str(e)}")
        
        return metrics
    
    def compare_performance(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compara métricas de desempeño entre referencia y datos actuales.
        
        Args:
            reference_metrics: Métricas del conjunto de referencia
            current_metrics: Métricas del conjunto actual
            
        Returns:
            Diccionario con comparación y alertas
        """
        self.logger.info("Comparando desempeño del modelo")
        
        comparison = {
            "reference_metrics": reference_metrics,
            "current_metrics": current_metrics,
            "differences": {},
            "relative_changes": {},
            "alerts": []
        }
        
        # Calcular diferencias y cambios relativos
        for metric in reference_metrics.keys():
            if metric in current_metrics:
                ref_val = reference_metrics[metric]
                curr_val = current_metrics[metric]
                
                diff = curr_val - ref_val
                rel_change = (diff / ref_val) * 100 if ref_val != 0 else 0
                
                comparison["differences"][metric] = float(diff)
                comparison["relative_changes"][metric] = float(rel_change)
                
                # Generar alertas basadas en umbrales
                if metric == "accuracy" and abs(rel_change) > (self.config.accuracy_threshold * 100):
                    comparison["alerts"].append({
                        "metric": metric,
                        "severity": "high" if abs(rel_change) > (self.config.accuracy_threshold * 200) else "medium",
                        "message": f"Accuracy cambió {rel_change:.2f}% (umbral: {self.config.accuracy_threshold * 100}%)",
                        "reference": ref_val,
                        "current": curr_val
                    })
                
                elif metric == "roc_auc" and abs(rel_change) > (self.config.roc_auc_threshold * 100):
                    comparison["alerts"].append({
                        "metric": metric,
                        "severity": "high" if abs(rel_change) > (self.config.roc_auc_threshold * 200) else "medium",
                        "message": f"ROC-AUC cambió {rel_change:.2f}% (umbral: {self.config.roc_auc_threshold * 100}%)",
                        "reference": ref_val,
                        "current": curr_val
                    })
                
                elif metric == "f1" and abs(rel_change) > (self.config.f1_threshold * 100):
                    comparison["alerts"].append({
                        "metric": metric,
                        "severity": "high" if abs(rel_change) > (self.config.f1_threshold * 200) else "medium",
                        "message": f"F1 cambió {rel_change:.2f}% (umbral: {self.config.f1_threshold * 100}%)",
                        "reference": ref_val,
                        "current": curr_val
                    })
        
        comparison["alert_count"] = len(comparison["alerts"])
        comparison["has_alerts"] = len(comparison["alerts"]) > 0
        
        self.logger.info(f"Comparación completada. Alertas: {comparison['alert_count']}")
        
        return comparison
    
    def run_drift_detection(
        self,
        drift_type: str = "combined",
        save_results: bool = True,
        output_dir: str = "reports/drift_detection"
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de detección de drift.
        
        Args:
            drift_type: Tipo de drift a simular
            save_results: Si guardar resultados en archivos
            output_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con todos los resultados
        """
        self.logger.info("=== INICIANDO DETECCIÓN DE DRIFT ===")
        
        if self.reference_data is None:
            raise ValueError("Debe cargar datos de referencia primero")
        if self.model is None:
            raise ValueError("Debe cargar un modelo primero")
        
        # 1. Generar datos con drift
        drift_data = self.generate_drift_data(drift_type=drift_type)
        
        # 2. Detectar drift estadístico
        statistical_drift = self.detect_statistical_drift(
            self.reference_data,
            drift_data
        )
        
        # 3. Evaluar modelo en datos de referencia
        self.logger.info("Evaluando modelo en datos de referencia")
        reference_metrics = self.evaluate_model_performance(
            self.reference_data,
            self.reference_target
        )
        
        # 4. Evaluar modelo en datos con drift
        self.logger.info("Evaluando modelo en datos con drift")
        # Necesitamos el target para los datos con drift
        # Usaremos el mismo target de referencia (asumiendo que el drift es solo en features)
        drift_target = self.reference_target.sample(
            n=len(drift_data),
            replace=True,
            random_state=self.config.random_state
        ).reset_index(drop=True)
        
        current_metrics = self.evaluate_model_performance(
            drift_data,
            drift_target
        )
        
        # 5. Comparar desempeño
        performance_comparison = self.compare_performance(
            reference_metrics,
            current_metrics
        )
        
        # 6. Compilar resultados
        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_type": drift_type,
            "statistical_drift": statistical_drift,
            "performance_comparison": performance_comparison,
            "reference_data_shape": self.reference_data.shape,
            "drift_data_shape": drift_data.shape,
            "summary": {
                "drift_detected": statistical_drift["overall_drift_detected"],
                "drift_severity": statistical_drift["drift_severity"],
                "performance_degradation": performance_comparison["has_alerts"],
                "alert_count": performance_comparison["alert_count"]
            }
        }
        
        self.drift_results = results
        
        # 7. Guardar resultados si se solicita
        if save_results:
            self._save_results(results, output_dir)
            self._generate_visualizations(results, drift_data, output_dir)
            self._generate_report(results, output_dir)
        
        self.logger.info("=== DETECCIÓN DE DRIFT COMPLETADA ===")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Guarda los resultados en formato JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar resultados completos
        results_file = output_path / f"drift_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Resultados guardados en: {results_file}")
    
    def _generate_visualizations(
        self,
        results: Dict[str, Any],
        drift_data: pd.DataFrame,
        output_dir: str
    ) -> None:
        """Genera visualizaciones del drift detectado."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filtrar solo columnas continuas que existan en ambos datasets
        available_continuous = [
            col for col in self.config.continuous_cols 
            if col in self.reference_data.columns and col in drift_data.columns
        ]
        
        if not available_continuous:
            self.logger.warning("No hay columnas continuas disponibles para visualizar")
            return
        
        self.logger.info("Generando visualizaciones")
        
        # 1. Comparación de distribuciones (variables continuas)
        if available_continuous:
            fig, axes = plt.subplots(
                len(available_continuous),
                1,
                figsize=(12, 4 * len(available_continuous))
            )
            
            if len(available_continuous) == 1:
                axes = [axes]
            
            for idx, col in enumerate(available_continuous):
                ax = axes[idx]
                
                ref_col = self.reference_data[col].dropna()
                drift_col = drift_data[col].dropna()
                
                if len(ref_col) > 0 and len(drift_col) > 0:
                    ax.hist(ref_col, bins=30, alpha=0.5, label='Referencia', density=True)
                    ax.hist(drift_col, bins=30, alpha=0.5, label='Con Drift', density=True)
                    ax.set_xlabel(col)
                    ax.set_ylabel('Densidad')
                    ax.set_title(f'Distribución de {col}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'distributions_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Comparación de métricas
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'roc_auc' in results['performance_comparison']['reference_metrics']:
            metrics.append('roc_auc')
        
        ref_values = [results['performance_comparison']['reference_metrics'].get(m, 0) for m in metrics]
        curr_values = [results['performance_comparison']['current_metrics'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, ref_values, width, label='Referencia', alpha=0.8)
        ax.bar(x + width/2, curr_values, width, label='Con Drift', alpha=0.8)
        
        ax.set_xlabel('Métricas')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Métricas de Desempeño')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Cambios relativos en métricas
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rel_changes = results['performance_comparison']['relative_changes']
        metrics_with_changes = [m for m in metrics if m in rel_changes]
        changes = [rel_changes[m] for m in metrics_with_changes]
        
        colors = ['red' if c < 0 else 'green' for c in changes]
        ax.barh(metrics_with_changes, changes, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Cambio Relativo (%)')
        ax.set_title('Cambio Relativo en Métricas')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path / 'relative_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizaciones guardadas en: {output_path}")
    
    def _generate_report(self, results: Dict[str, Any], output_dir: str) -> None:
        """Genera un reporte en markdown."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Detección de Data Drift\n\n")
            f.write(f"**Fecha:** {results['timestamp']}\n\n")
            f.write(f"**Tipo de Drift Simulado:** {results['drift_type']}\n\n")
            
            # Resumen
            f.write("## Resumen Ejecutivo\n\n")
            summary = results['summary']
            f.write(f"- **Drift Detectado:** {'Sí' if summary['drift_detected'] else 'No'}\n")
            f.write(f"- **Severidad del Drift:** {summary['drift_severity'].upper()}\n")
            f.write(f"- **Degradación de Desempeño:** {'Sí' if summary['performance_degradation'] else 'No'}\n")
            f.write(f"- **Número de Alertas:** {summary['alert_count']}\n\n")
            
            # Drift estadístico
            f.write("## Detección de Drift Estadístico\n\n")
            stat_drift = results['statistical_drift']
            f.write(f"- **Drift General Detectado:** {'Sí' if stat_drift['overall_drift_detected'] else 'No'}\n")
            f.write(f"- **Ratio de Features con Drift:** {stat_drift['drift_ratio']:.2%}\n")
            f.write(f"- **Severidad:** {stat_drift['drift_severity'].upper()}\n\n")
            
            # Variables continuas con drift
            if stat_drift['continuous_drift']:
                f.write("### Variables Continuas con Drift\n\n")
                f.write("| Variable | KS Statistic | p-value | Drift Detectado |\n")
                f.write("|----------|--------------|---------|-----------------|\n")
                for col, drift_info in stat_drift['continuous_drift'].items():
                    f.write(f"| {col} | {drift_info['ks_statistic']:.4f} | "
                           f"{drift_info['p_value']:.4f} | "
                           f"{'Sí' if drift_info['drift_detected'] else 'No'} |\n")
                f.write("\n")
            
            # Comparación de métricas
            f.write("## Comparación de Desempeño del Modelo\n\n")
            perf_comp = results['performance_comparison']
            
            f.write("### Métricas de Referencia vs. Datos con Drift\n\n")
            f.write("| Métrica | Referencia | Con Drift | Diferencia | Cambio Relativo (%) |\n")
            f.write("|---------|------------|-----------|------------|---------------------|\n")
            
            for metric in perf_comp['reference_metrics'].keys():
                ref_val = perf_comp['reference_metrics'][metric]
                curr_val = perf_comp['current_metrics'].get(metric, 0)
                diff = perf_comp['differences'].get(metric, 0)
                rel_change = perf_comp['relative_changes'].get(metric, 0)
                
                f.write(f"| {metric} | {ref_val:.4f} | {curr_val:.4f} | "
                       f"{diff:+.4f} | {rel_change:+.2f}% |\n")
            f.write("\n")
            
            # Alertas
            if perf_comp['alerts']:
                f.write("## Alertas Generadas\n\n")
                for alert in perf_comp['alerts']:
                    f.write(f"### {alert['metric'].upper()} - Severidad: {alert['severity'].upper()}\n\n")
                    f.write(f"- **Mensaje:** {alert['message']}\n")
                    f.write(f"- **Valor Referencia:** {alert['reference']:.4f}\n")
                    f.write(f"- **Valor Actual:** {alert['current']:.4f}\n\n")
            
            # Acciones recomendadas
            f.write("## Acciones Recomendadas\n\n")
            
            if summary['drift_detected'] or summary['performance_degradation']:
                f.write("### Acciones Inmediatas:\n\n")
                f.write("1. **Revisar el Feature Pipeline:** Verificar que el procesamiento de datos sigue siendo correcto\n")
                f.write("2. **Analizar Causas del Drift:** Investigar cambios en la distribución de datos de entrada\n")
                f.write("3. **Considerar Retraining:** Si la degradación es significativa, evaluar reentrenar el modelo\n")
                f.write("4. **Monitoreo Continuo:** Implementar monitoreo automático de drift en producción\n\n")
                
                if summary['drift_severity'] == 'high':
                    f.write("### ⚠️ ALERTA CRÍTICA: Drift de Alta Severidad Detectado\n\n")
                    f.write("Se recomienda:\n")
                    f.write("- Detener el modelo en producción si es posible\n")
                    f.write("- Reentrenar inmediatamente con datos actualizados\n")
                    f.write("- Validar el modelo antes de volver a producción\n\n")
            else:
                f.write("No se detectaron problemas significativos. El modelo mantiene su desempeño.\n\n")
        
        self.logger.info(f"Reporte guardado en: {report_file}")


def main():
    """Función principal para ejecutar detección de drift desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detectar data drift en modelos de ML")
    parser.add_argument("--reference-data", required=True, help="Ruta a datos de referencia")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo entrenado")
    parser.add_argument("--drift-type", default="combined",
                       choices=["mean_shift", "missing_values", "variance_change", "categorical_shift", "combined"],
                       help="Tipo de drift a simular")
    parser.add_argument("--output-dir", default="reports/drift_detection",
                       help="Directorio para guardar resultados")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear detector
    detector = DriftDetector()
    
    # Cargar datos y modelo
    detector.load_reference_data(args.reference_data)
    detector.load_model(args.model_path)
    
    # Ejecutar detección
    results = detector.run_drift_detection(
        drift_type=args.drift_type,
        output_dir=args.output_dir
    )
    
    # Mostrar resumen
    print("\n=== RESUMEN DE DETECCIÓN DE DRIFT ===")
    print(f"Drift Detectado: {results['summary']['drift_detected']}")
    print(f"Severidad: {results['summary']['drift_severity']}")
    print(f"Alertas: {results['summary']['alert_count']}")
    print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()

