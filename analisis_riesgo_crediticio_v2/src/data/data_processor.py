# -*- coding: utf-8 -*-
"""
DataProcessor: Clase para procesamiento y limpieza de datos del dataset German Credit.

Esta clase maneja la carga, validación, limpieza y transformación de datos
siguiendo las mejores prácticas de POO y MLOps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import StandardScaler


@dataclass
class DataConfig:
    """Configuración para el procesamiento de datos."""
    # Dominios válidos para variables categóricas
    valid_domains: Dict[str, set] = None
    # Variables continuas
    continuous_vars: List[str] = None
    # Rangos válidos para variables continuas
    valid_ranges: Dict[str, Tuple] = None
    # Variable objetivo
    target_column: str = "target_bad"
    # Random state para reproducibilidad
    random_state: int = 42
    
    def __post_init__(self):
        if self.valid_domains is None:
            self.valid_domains = {
                "laufkont": {1, 2, 3, 4}, "moral": {0, 1, 2, 3, 4}, 
                "verw": set(range(0, 11)), "sparkont": {1, 2, 3, 4, 5},
                "beszeit": {1, 2, 3, 4, 5}, "rate": {1, 2, 3, 4}, 
                "famges": {1, 2, 3, 4}, "buerge": {1, 2, 3},
                "wohnzeit": {1, 2, 3, 4}, "verm": {1, 2, 3, 4}, 
                "weitkred": {1, 2, 3}, "wohn": {1, 2, 3},
                "bishkred": {1, 2, 3, 4}, "beruf": {1, 2, 3, 4}, 
                "pers": {1, 2}, "telef": {1, 2}, "gastarb": {1, 2},
                "kredit": {0, 1}
            }
        
        if self.continuous_vars is None:
            self.continuous_vars = ["hoehe", "laufzeit", "alter"]
            
        if self.valid_ranges is None:
            self.valid_ranges = {
                "alter": (18, 75), 
                "laufzeit": (4, 72), 
                "hoehe": (250, None)
            }


class DataProcessor:
    """
    Clase principal para el procesamiento de datos del proyecto de riesgo crediticio.
    
    Esta clase encapsula todas las operaciones de limpieza, validación y 
    transformación de datos siguiendo principios SOLID.
    """
    
    def __init__(self, config: DataConfig = None):
        """
        Inicializa el procesador de datos.
        
        Args:
            config: Configuración para el procesamiento de datos
        """
        self.config = config or DataConfig()
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self._data_info = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo no es válido
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
                
            self.logger.info(f"Cargando datos desde: {file_path}")
            df = pd.read_csv(file_path)
            
            self.logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida la estructura y calidad de los datos.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Diccionario con información de validación
        """
        validation_info = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "issues": []
        }
        
        # Validar columnas requeridas
        required_cols = list(self.config.valid_domains.keys()) + self.config.continuous_vars
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_info["issues"].append(f"Columnas faltantes: {missing_cols}")
        
        # Validar dominios categóricos
        for col, valid_values in self.config.valid_domains.items():
            if col in df.columns:
                # Convertir a numérico si es posible
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    valid_numeric_values = numeric_values.dropna()
                    invalid_values = set(valid_numeric_values.unique()) - valid_values
                    if len(invalid_values) > 0:
                        validation_info["issues"].append(
                            f"Valores inválidos en {col}: {invalid_values}"
                        )
                except Exception as e:
                    validation_info["issues"].append(f"Error validando {col}: {str(e)}")
        
        # Validar rangos continuos
        for col, (min_val, max_val) in self.config.valid_ranges.items():
            if col in df.columns:
                try:
                    # Convertir a numérico
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    
                    if min_val is not None:
                        below_min = (numeric_col < min_val).sum()
                        if below_min > 0:
                            validation_info["issues"].append(
                                f"{col}: {below_min} valores por debajo de {min_val}"
                            )
                    if max_val is not None:
                        above_max = (numeric_col > max_val).sum()
                        if above_max > 0:
                            validation_info["issues"].append(
                                f"{col}: {above_max} valores por encima de {max_val}"
                            )
                except Exception as e:
                    validation_info["issues"].append(f"Error validando rangos en {col}: {str(e)}")
        
        self.logger.info(f"Validación completada. Issues encontrados: {len(validation_info['issues'])}")
        return validation_info
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos aplicando transformaciones necesarias.
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            DataFrame limpio
        """
        self.logger.info("Iniciando limpieza de datos")
        df_clean = df.copy()
        
        # Limpiar espacios en blanco en todas las columnas
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Reemplazar strings vacíos con NaN
                df_clean[col] = df_clean[col].replace('', np.nan)
        
        # Eliminar columnas problemáticas (como mixed_type_col)
        problematic_cols = ['mixed_type_col']
        for col in problematic_cols:
            if col in df_clean.columns:
                self.logger.info(f"Eliminando columna problemática: {col}")
                df_clean = df_clean.drop(columns=[col])
        
        # Mapear target a 'target_bad' (1 = malo, 0 = bueno)
        if "kredit" in df_clean.columns:
            # Convertir a numérico primero
            df_clean["kredit"] = pd.to_numeric(df_clean["kredit"], errors='coerce')
            df_clean[self.config.target_column] = df_clean["kredit"].map({1: 0, 0: 1}).astype("Int64")
            df_clean = df_clean.drop(columns=["kredit"])
        
        # Eliminar duplicados
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            self.logger.info(f"Eliminados {removed_duplicates} duplicados")
        
        # Imputación de valores faltantes
        df_clean = self._impute_missing_values(df_clean)
        
        # Eliminar filas con target faltante
        target_na_count = df_clean[self.config.target_column].isna().sum()
        if target_na_count > 0:
            self.logger.info(f"Eliminando {target_na_count} filas con target faltante")
            df_clean = df_clean.dropna(subset=[self.config.target_column])
        
        # Validar valores del target
        target_values = set(df_clean[self.config.target_column].unique())
        if not target_values <= {0, 1}:
            raise ValueError(f"Valores no binarios en target: {sorted(target_values)}")
        
        self.logger.info(f"Datos limpios: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")
        return df_clean
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores faltantes usando mediana para continuas y moda para categóricas.
        
        Args:
            df: DataFrame con valores faltantes
            
        Returns:
            DataFrame con valores imputados
        """
        df_imputed = df.copy()
        
        for col in df_imputed.columns:
            if col == self.config.target_column:
                continue
                
            if col in self.config.continuous_vars:
                # Variables continuas: convertir a numérico y usar mediana
                numeric_col = pd.to_numeric(df_imputed[col], errors='coerce')
                median_val = numeric_col.median()
                df_imputed[col] = numeric_col.fillna(median_val)
                self.logger.debug(f"Imputada columna continua {col} con mediana: {median_val}")
            else:
                # Variables categóricas: usar moda
                # Convertir a numérico si es posible
                try:
                    numeric_col = pd.to_numeric(df_imputed[col], errors='coerce')
                    mode_val = numeric_col.mode(dropna=True)
                    if not mode_val.empty:
                        df_imputed[col] = numeric_col.fillna(mode_val.iloc[0])
                        self.logger.debug(f"Imputada columna categórica {col} con moda: {mode_val.iloc[0]}")
                except:
                    # Si no se puede convertir a numérico, usar moda de strings
                    mode_val = df_imputed[col].mode(dropna=True)
                    if not mode_val.empty:
                        df_imputed[col] = df_imputed[col].fillna(mode_val.iloc[0])
                        self.logger.debug(f"Imputada columna categórica {col} con moda: {mode_val.iloc[0]}")
        
        return df_imputed
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara las características para el entrenamiento del modelo.
        
        Args:
            df: DataFrame limpio
            
        Returns:
            Tupla con (X, y) donde X son las características e y es el target
        """
        self.logger.info("Preparando características para entrenamiento")
        
        # Separar características y target
        X = df.drop(columns=[self.config.target_column]).copy()
        y = df[self.config.target_column].astype("int64")
        
        # Escalar variables continuas
        if self.config.continuous_vars:
            continuous_cols = [col for col in self.config.continuous_vars if col in X.columns]
            if continuous_cols:
                X_scaled = X.copy()
                X_scaled[continuous_cols] = self.scaler.fit_transform(X[continuous_cols])
                self.logger.info(f"Escaladas variables continuas: {continuous_cols}")
            else:
                X_scaled = X
        else:
            X_scaled = X
        
        self.logger.info(f"Características preparadas: {X_scaled.shape}")
        return X_scaled, y
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Guarda los datos procesados en un archivo CSV.
        
        Args:
            df: DataFrame procesado
            output_path: Ruta de salida
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Datos procesados guardados en: {output_path}")
    
    def save_scaler(self, scaler_path: str) -> None:
        """
        Guarda el scaler entrenado.
        
        Args:
            scaler_path: Ruta donde guardar el scaler
        """
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, scaler_path)
        self.logger.info(f"Scaler guardado en: {scaler_path}")
    
    def load_scaler(self, scaler_path: str) -> None:
        """
        Carga un scaler previamente entrenado.
        
        Args:
            scaler_path: Ruta del scaler a cargar
        """
        scaler_path = Path(scaler_path)
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        self.logger.info(f"Scaler cargado desde: {scaler_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un resumen estadístico de los datos.
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con estadísticas resumidas
        """
        summary = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "target_distribution": None
        }
        
        if self.config.target_column in df.columns:
            summary["target_distribution"] = df[self.config.target_column].value_counts(normalize=True).to_dict()
        
        return summary
    
    def process_pipeline(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de procesamiento de datos.
        
        Args:
            input_path: Ruta del archivo de entrada
            output_path: Ruta del archivo de salida
            
        Returns:
            Diccionario con información del procesamiento
        """
        self.logger.info("Iniciando pipeline de procesamiento de datos")
        
        # Cargar datos
        df = self.load_data(input_path)
        
        # Validar datos
        validation_info = self.validate_data(df)
        
        # Limpiar datos
        df_clean = self.clean_data(df)
        
        # Preparar características
        X, y = self.prepare_features(df_clean)
        
        # Combinar X e y para guardar
        df_processed = X.copy()
        df_processed[self.config.target_column] = y
        
        # Guardar datos procesados
        self.save_processed_data(df_processed, output_path)
        
        # Generar resumen
        summary = self.get_data_summary(df_processed)
        
        pipeline_info = {
            "input_shape": df.shape,
            "output_shape": df_processed.shape,
            "validation_info": validation_info,
            "summary": summary,
            "scaler_fitted": hasattr(self.scaler, 'mean_')
        }
        
        self.logger.info("Pipeline de procesamiento completado exitosamente")
        return pipeline_info


def main():
    """Función principal para ejecutar el procesamiento desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Procesar datos del dataset German Credit")
    parser.add_argument("input_file", help="Archivo CSV de entrada")
    parser.add_argument("output_file", help="Archivo CSV de salida")
    parser.add_argument("--scaler-path", help="Ruta para guardar el scaler")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear procesador y ejecutar pipeline
    processor = DataProcessor()
    result = processor.process_pipeline(args.input_file, args.output_file)
    
    # Guardar scaler si se especifica
    if args.scaler_path:
        processor.save_scaler(args.scaler_path)
    
    print(f"Procesamiento completado: {result['output_shape']}")


if __name__ == "__main__":
    main()
