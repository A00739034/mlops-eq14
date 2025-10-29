# -*- coding: utf-8 -*-
"""
FeatureEngineer: Clase para ingeniería de características avanzada.

Esta clase maneja la creación, selección y transformación de características
para mejorar el rendimiento de los modelos de machine learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib


@dataclass
class FeatureConfig:
    """Configuración para la ingeniería de características."""
    # Variables continuas para transformaciones
    continuous_vars: List[str] = None
    # Variables categóricas para encoding
    categorical_vars: List[str] = None
    # Método de selección de características
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_classif", "random_forest"
    # Número de características a seleccionar
    n_features_select: int = 15
    # Aplicar PCA
    apply_pca: bool = False
    # Número de componentes PCA
    n_components_pca: int = 10
    # Crear características polinómicas
    create_polynomial_features: bool = False
    # Grado de características polinómicas
    polynomial_degree: int = 2
    # Crear características de interacción
    create_interaction_features: bool = False
    # Random state
    random_state: int = 42
    
    def __post_init__(self):
        if self.continuous_vars is None:
            self.continuous_vars = ["hoehe", "laufzeit", "alter"]
        
        if self.categorical_vars is None:
            self.categorical_vars = [
                "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
                "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
                "bishkred", "beruf", "pers", "telef", "gastarb"
            ]


class FeatureEngineer:
    """
    Clase para ingeniería de características avanzada.
    
    Esta clase encapsula métodos para crear, seleccionar y transformar
    características para mejorar el rendimiento de los modelos.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Inicializa el ingeniero de características.
        
        Args:
            config: Configuración para la ingeniería de características
        """
        self.config = config or FeatureConfig()
        self.logger = logging.getLogger(__name__)
        
        # Componentes de transformación
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_selector = None
        self.pca = None
        self.polynomial_features = None
        
        # Información de características
        self.feature_names_ = None
        self.selected_features_ = None
        self.feature_importance_ = None
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de interacción entre variables continuas.
        
        Args:
            df: DataFrame con características
            
        Returns:
            DataFrame con características de interacción añadidas
        """
        df_interaction = df.copy()
        continuous_cols = [col for col in self.config.continuous_vars if col in df.columns]
        
        if len(continuous_cols) >= 2:
            self.logger.info("Creando características de interacción")
            
            # Crear interacciones entre variables continuas
            for i, col1 in enumerate(continuous_cols):
                for col2 in continuous_cols[i+1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    df_interaction[interaction_name] = df[col1] * df[col2]
                    self.logger.debug(f"Creada interacción: {interaction_name}")
        
        return df_interaction
    
    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características polinómicas para variables continuas.
        
        Args:
            df: DataFrame con características
            
        Returns:
            DataFrame con características polinómicas añadidas
        """
        df_poly = df.copy()
        continuous_cols = [col for col in self.config.continuous_vars if col in df.columns]
        
        if continuous_cols and self.config.polynomial_degree > 1:
            self.logger.info(f"Creando características polinómicas de grado {self.config.polynomial_degree}")
            
            for col in continuous_cols:
                for degree in range(2, self.config.polynomial_degree + 1):
                    poly_name = f"{col}_pow_{degree}"
                    df_poly[poly_name] = df[col] ** degree
                    self.logger.debug(f"Creada característica polinómica: {poly_name}")
        
        return df_poly
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de ratio entre variables continuas.
        
        Args:
            df: DataFrame con características
            
        Returns:
            DataFrame con características de ratio añadidas
        """
        df_ratio = df.copy()
        continuous_cols = [col for col in self.config.continuous_vars if col in df.columns]
        
        if len(continuous_cols) >= 2:
            self.logger.info("Creando características de ratio")
            
            # Crear ratios entre variables continuas
            for i, col1 in enumerate(continuous_cols):
                for col2 in continuous_cols[i+1:]:
                    # Evitar división por cero
                    ratio_name = f"{col1}_div_{col2}"
                    df_ratio[ratio_name] = np.where(
                        df[col2] != 0, 
                        df[col1] / df[col2], 
                        0
                    )
                    self.logger.debug(f"Creada característica de ratio: {ratio_name}")
        
        return df_ratio
    
    def create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de binning para variables continuas.
        
        Args:
            df: DataFrame con características
            
        Returns:
            DataFrame con características de binning añadidas
        """
        df_binned = df.copy()
        continuous_cols = [col for col in self.config.continuous_vars if col in df.columns]
        
        if continuous_cols:
            self.logger.info("Creando características de binning")
            
            for col in continuous_cols:
                # Crear bins basados en cuartiles
                bin_name = f"{col}_bin"
                df_binned[bin_name] = pd.qcut(df[col], q=4, labels=False, duplicates='drop')
                self.logger.debug(f"Creada característica de binning: {bin_name}")
        
        return df_binned
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica variables categóricas usando Label Encoding y One-Hot Encoding.
        
        Args:
            df: DataFrame con características categóricas
            fit: Si True, ajusta los encoders; si False, usa encoders previamente ajustados
            
        Returns:
            DataFrame con características categóricas codificadas
        """
        df_encoded = df.copy()
        categorical_cols = [col for col in self.config.categorical_vars if col in df.columns]
        
        if categorical_cols:
            self.logger.info("Codificando características categóricas")
            
            if fit:
                # Label encoding para variables con pocas categorías
                for col in categorical_cols:
                    if df[col].nunique() <= 10:  # Solo para variables con pocas categorías
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                        self.logger.debug(f"Label encoding aplicado a: {col}")
                
                # One-hot encoding para variables con muchas categorías
                high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 10]
                if high_cardinality_cols:
                    self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    oh_features = self.onehot_encoder.fit_transform(df[high_cardinality_cols])
                    
                    # Crear nombres de columnas para one-hot encoding
                    oh_columns = []
                    for col in high_cardinality_cols:
                        categories = self.onehot_encoder.categories_[high_cardinality_cols.index(col)]
                        oh_columns.extend([f"{col}_{cat}" for cat in categories])
                    
                    # Crear DataFrame con características one-hot
                    oh_df = pd.DataFrame(oh_features, columns=oh_columns, index=df.index)
                    df_encoded = pd.concat([df_encoded, oh_df], axis=1)
                    
                    # Eliminar columnas originales de alta cardinalidad
                    df_encoded = df_encoded.drop(columns=high_cardinality_cols)
                    self.logger.debug(f"One-hot encoding aplicado a: {high_cardinality_cols}")
            else:
                # Usar encoders previamente ajustados
                for col, le in self.label_encoders.items():
                    if col in df_encoded.columns:
                        df_encoded[col] = le.transform(df[col].astype(str))
        
        return df_encoded
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """
        Selecciona las mejores características usando diferentes métodos.
        
        Args:
            X: DataFrame con características
            y: Serie con target
            fit: Si True, ajusta el selector; si False, usa selector previamente ajustado
            
        Returns:
            DataFrame con características seleccionadas
        """
        if fit:
            self.logger.info(f"Seleccionando características usando {self.config.feature_selection_method}")
            
            if self.config.feature_selection_method == "mutual_info":
                self.feature_selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.config.n_features_select, X.shape[1])
                )
            elif self.config.feature_selection_method == "f_classif":
                self.feature_selector = SelectKBest(
                    score_func=f_classif,
                    k=min(self.config.n_features_select, X.shape[1])
                )
            elif self.config.feature_selection_method == "random_forest":
                # Usar Random Forest para obtener importancia de características
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
                rf.fit(X, y)
                self.feature_importance_ = rf.feature_importances_
                
                # Seleccionar características con mayor importancia
                feature_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.feature_importance_
                }).sort_values('importance', ascending=False)
                
                self.selected_features_ = feature_importance_df.head(
                    self.config.n_features_select
                )['feature'].tolist()
                
                self.logger.info(f"Seleccionadas {len(self.selected_features_)} características usando Random Forest")
                return X[self.selected_features_]
            
            # Ajustar selector
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features_ = X.columns[self.feature_selector.get_support()].tolist()
            
            self.logger.info(f"Seleccionadas {len(self.selected_features_)} características")
            return pd.DataFrame(X_selected, columns=self.selected_features_, index=X.index)
        
        else:
            # Usar características previamente seleccionadas
            if self.selected_features_:
                return X[self.selected_features_]
            else:
                return X
    
    def apply_pca(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Aplica Análisis de Componentes Principales (PCA).
        
        Args:
            X: DataFrame con características
            fit: Si True, ajusta PCA; si False, usa PCA previamente ajustado
            
        Returns:
            DataFrame con componentes principales
        """
        if self.config.apply_pca:
            if fit:
                self.logger.info(f"Aplicando PCA con {self.config.n_components_pca} componentes")
                self.pca = PCA(
                    n_components=min(self.config.n_components_pca, X.shape[1]),
                    random_state=self.config.random_state
                )
                X_pca = self.pca.fit_transform(X)
                
                # Crear nombres de columnas para componentes PCA
                pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
                self.logger.info(f"PCA aplicado: {X.shape[1]} características -> {X_pca.shape[1]} componentes")
                
                return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
            else:
                if self.pca:
                    X_pca = self.pca.transform(X)
                    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
                    return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        return X
    
    def transform_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Aplica todas las transformaciones de características en secuencia.
        
        Args:
            df: DataFrame con características
            fit: Si True, ajusta los transformadores; si False, usa transformadores previamente ajustados
            
        Returns:
            DataFrame con características transformadas
        """
        self.logger.info("Iniciando transformación de características")
        
        # Crear características derivadas
        if self.config.create_interaction_features:
            df = self.create_interaction_features(df)
        
        if self.config.create_polynomial_features:
            df = self.create_polynomial_features(df)
        
        # Crear características adicionales
        df = self.create_ratio_features(df)
        df = self.create_binning_features(df)
        
        # Codificar características categóricas
        df = self.encode_categorical_features(df, fit=fit)
        
        # Seleccionar características
        if fit:
            # Necesitamos el target para la selección de características
            # En este caso, asumimos que el target no está en el DataFrame
            # Se debe pasar por separado en el método principal
            pass
        
        # Aplicar PCA
        df = self.apply_pca(df, fit=fit)
        
        self.feature_names_ = df.columns.tolist()
        self.logger.info(f"Transformación completada: {len(self.feature_names_)} características")
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Ajusta los transformadores y transforma los datos de entrenamiento.
        
        Args:
            X: DataFrame con características de entrenamiento
            y: Serie con target de entrenamiento
            
        Returns:
            DataFrame con características transformadas
        """
        self.logger.info("Ajustando transformadores con datos de entrenamiento")
        
        # Transformar características
        X_transformed = self.transform_features(X, fit=True)
        
        # Seleccionar características
        X_selected = self.select_features(X_transformed, y, fit=True)
        
        return X_selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma los datos usando transformadores previamente ajustados.
        
        Args:
            X: DataFrame con características a transformar
            
        Returns:
            DataFrame con características transformadas
        """
        self.logger.info("Transformando datos con transformadores ajustados")
        
        # Transformar características
        X_transformed = self.transform_features(X, fit=False)
        
        # Seleccionar características
        X_selected = self.select_features(X_transformed, None, fit=False)
        
        return X_selected
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Obtiene la importancia de las características.
        
        Returns:
            DataFrame con importancia de características o None
        """
        if self.feature_importance_ is not None and self.selected_features_ is not None:
            return pd.DataFrame({
                'feature': self.selected_features_,
                'importance': self.feature_importance_[:len(self.selected_features_)]
            }).sort_values('importance', ascending=False)
        return None
    
    def save_transformers(self, output_dir: str) -> None:
        """
        Guarda todos los transformadores ajustados.
        
        Args:
            output_dir: Directorio donde guardar los transformadores
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar encoders de etiquetas
        for col, le in self.label_encoders.items():
            joblib.dump(le, output_dir / f"label_encoder_{col}.joblib")
        
        # Guardar one-hot encoder
        if self.onehot_encoder:
            joblib.dump(self.onehot_encoder, output_dir / "onehot_encoder.joblib")
        
        # Guardar selector de características
        if self.feature_selector:
            joblib.dump(self.feature_selector, output_dir / "feature_selector.joblib")
        
        # Guardar PCA
        if self.pca:
            joblib.dump(self.pca, output_dir / "pca.joblib")
        
        # Guardar información de características
        feature_info = {
            'feature_names': self.feature_names_,
            'selected_features': self.selected_features_,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(feature_info, output_dir / "feature_info.joblib")
        
        self.logger.info(f"Transformadores guardados en: {output_dir}")
    
    def load_transformers(self, input_dir: str) -> None:
        """
        Carga transformadores previamente ajustados.
        
        Args:
            input_dir: Directorio donde están los transformadores
        """
        input_dir = Path(input_dir)
        
        # Cargar encoders de etiquetas
        for file_path in input_dir.glob("label_encoder_*.joblib"):
            col = file_path.stem.replace("label_encoder_", "")
            self.label_encoders[col] = joblib.load(file_path)
        
        # Cargar one-hot encoder
        onehot_path = input_dir / "onehot_encoder.joblib"
        if onehot_path.exists():
            self.onehot_encoder = joblib.load(onehot_path)
        
        # Cargar selector de características
        selector_path = input_dir / "feature_selector.joblib"
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)
        
        # Cargar PCA
        pca_path = input_dir / "pca.joblib"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
        
        # Cargar información de características
        info_path = input_dir / "feature_info.joblib"
        if info_path.exists():
            feature_info = joblib.load(info_path)
            self.feature_names_ = feature_info['feature_names']
            self.selected_features_ = feature_info['selected_features']
            self.feature_importance_ = feature_info['feature_importance']
        
        self.logger.info(f"Transformadores cargados desde: {input_dir}")


def main():
    """Función principal para ejecutar la ingeniería de características desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingeniería de características para German Credit")
    parser.add_argument("input_file", help="Archivo CSV de entrada")
    parser.add_argument("output_file", help="Archivo CSV de salida")
    parser.add_argument("--target", help="Columna target", default="target_bad")
    parser.add_argument("--output-dir", help="Directorio para guardar transformadores")
    parser.add_argument("--n-features", type=int, help="Número de características a seleccionar", default=15)
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear configuración
    config = FeatureConfig(n_features_select=args.n_features)
    
    # Crear ingeniero de características
    engineer = FeatureEngineer(config)
    
    # Cargar datos
    df = pd.read_csv(args.input_file)
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    # Ajustar y transformar
    X_transformed = engineer.fit_transform(X, y)
    
    # Guardar datos transformados
    output_df = X_transformed.copy()
    output_df[args.target] = y
    output_df.to_csv(args.output_file, index=False)
    
    # Guardar transformadores
    if args.output_dir:
        engineer.save_transformers(args.output_dir)
    
    print(f"Características transformadas: {X_transformed.shape}")


if __name__ == "__main__":
    main()
