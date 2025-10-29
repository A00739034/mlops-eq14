# -*- coding: utf-8 -*-
"""
DataVisualizer: Clase para generar visualizaciones y reportes del análisis de datos.

Esta clase maneja la creación de gráficos, reportes y visualizaciones
para el análisis exploratorio de datos y evaluación de modelos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import warnings
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configurar matplotlib para mejor calidad
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8')

# Suprimir warnings
warnings.filterwarnings('ignore')


@dataclass
class VisualizationConfig:
    """Configuración para las visualizaciones."""
    # Estilo de gráficos
    style: str = "seaborn-v0_8"
    # Tamaño de figura por defecto
    figsize: Tuple[int, int] = (12, 8)
    # DPI para guardar imágenes
    dpi: int = 300
    # Formato de salida
    output_format: str = "png"  # "png", "svg", "pdf"
    # Paleta de colores
    color_palette: str = "viridis"
    # Incluir gráficos interactivos
    interactive: bool = False
    # Directorio de salida
    output_dir: str = "reports/figures"
    # Idioma
    language: str = "es"  # "es", "en"


class DataVisualizer:
    """
    Clase para generar visualizaciones y reportes del análisis de datos.
    
    Esta clase encapsula métodos para crear gráficos exploratorios,
    visualizaciones de modelos y reportes automatizados.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Inicializa el visualizador de datos.
        
        Args:
            config: Configuración para las visualizaciones
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Configurar matplotlib
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
        # Crear directorio de salida
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Traducciones
        self.translations = {
            "es": {
                "target_distribution": "Distribución del Target",
                "missing_values": "Valores Faltantes",
                "correlation_matrix": "Matriz de Correlación",
                "feature_distribution": "Distribución de Características",
                "confusion_matrix": "Matriz de Confusión",
                "roc_curve": "Curva ROC",
                "precision_recall_curve": "Curva Precision-Recall",
                "feature_importance": "Importancia de Características",
                "model_comparison": "Comparación de Modelos"
            },
            "en": {
                "target_distribution": "Target Distribution",
                "missing_values": "Missing Values",
                "correlation_matrix": "Correlation Matrix",
                "feature_distribution": "Feature Distribution",
                "confusion_matrix": "Confusion Matrix",
                "roc_curve": "ROC Curve",
                "precision_recall_curve": "Precision-Recall Curve",
                "feature_importance": "Feature Importance",
                "model_comparison": "Model Comparison"
            }
        }
    
    def _get_text(self, key: str) -> str:
        """Obtiene texto traducido."""
        return self.translations.get(self.config.language, {}).get(key, key)
    
    def plot_target_distribution(self, df: pd.DataFrame, target_col: str = "target_bad",
                               save: bool = True) -> str:
        """
        Visualiza la distribución del target.
        
        Args:
            df: DataFrame con datos
            target_col: Nombre de la columna target
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info("Generando gráfico de distribución del target")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)
        
        # Gráfico de barras
        target_counts = df[target_col].value_counts()
        target_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title(self._get_text("target_distribution"))
        ax1.set_xlabel("Clase")
        ax1.set_ylabel("Frecuencia")
        ax1.set_xticklabels(['Bueno', 'Malo'], rotation=0)
        
        # Gráfico de torta
        target_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                          colors=['skyblue', 'salmon'])
        ax2.set_title(self._get_text("target_distribution"))
        ax2.set_ylabel("")
        
        plt.tight_layout()
        
        if save:
            filename = f"target_distribution.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_missing_values(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Visualiza valores faltantes en el dataset.
        
        Args:
            df: DataFrame con datos
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info("Generando gráfico de valores faltantes")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            self.logger.info("No hay valores faltantes en el dataset")
            return ""
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        missing_data.plot(kind='bar', ax=ax, color='coral')
        ax.set_title(self._get_text("missing_values"))
        ax.set_xlabel("Características")
        ax.set_ylabel("Número de Valores Faltantes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = f"missing_values.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Visualiza matriz de correlación.
        
        Args:
            df: DataFrame con datos numéricos
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info("Generando matriz de correlación")
        
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax, fmt='.2f')
        
        ax.set_title(self._get_text("correlation_matrix"))
        plt.tight_layout()
        
        if save:
            filename = f"correlation_matrix.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_feature_distributions(self, df: pd.DataFrame, target_col: str = "target_bad",
                                 save: bool = True) -> str:
        """
        Visualiza distribuciones de características por clase.
        
        Args:
            df: DataFrame con datos
            target_col: Nombre de la columna target
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info("Generando distribuciones de características")
        
        # Seleccionar características numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if len(numeric_cols) == 0:
            self.logger.warning("No hay características numéricas para visualizar")
            return ""
        
        # Calcular número de filas y columnas
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]
            
            # Crear histogramas por clase
            for class_val in df[target_col].unique():
                subset = df[df[target_col] == class_val]
                ax.hist(subset[col], alpha=0.7, label=f'Clase {class_val}', bins=20)
            
            ax.set_title(f'{col}')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frecuencia')
            ax.legend()
        
        # Ocultar ejes vacíos
        for i in range(len(numeric_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.suptitle(self._get_text("feature_distribution"))
        plt.tight_layout()
        
        if save:
            filename = f"feature_distributions.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Modelo", save: bool = True) -> str:
        """
        Visualiza matriz de confusión.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            model_name: Nombre del modelo
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info(f"Generando matriz de confusión para {model_name}")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{self._get_text('confusion_matrix')} - {model_name}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor Real")
        ax.set_xticklabels(['Bueno', 'Malo'])
        ax.set_yticklabels(['Bueno', 'Malo'])
        
        plt.tight_layout()
        
        if save:
            filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Modelo", save: bool = True) -> str:
        """
        Visualiza curva ROC.
        
        Args:
            y_true: Valores reales
            y_pred_proba: Probabilidades de predicción
            model_name: Nombre del modelo
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info(f"Generando curva ROC para {model_name}")
        
        from sklearn.metrics import roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.set_title(f"{self._get_text('roc_curve')} - {model_name}")
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        
        if save:
            filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Modelo", save: bool = True) -> str:
        """
        Visualiza curva Precision-Recall.
        
        Args:
            y_true: Valores reales
            y_pred_proba: Probabilidades de predicción
            model_name: Nombre del modelo
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info(f"Generando curva Precision-Recall para {model_name}")
        
        from sklearn.metrics import average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f"{self._get_text('precision_recall_curve')} - {model_name}")
        ax.legend(loc="lower left")
        
        plt.tight_layout()
        
        if save:
            filename = f"precision_recall_curve_{model_name.lower().replace(' ', '_')}.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_values: List[float],
                               model_name: str = "Modelo", save: bool = True) -> str:
        """
        Visualiza importancia de características.
        
        Args:
            feature_names: Nombres de las características
            importance_values: Valores de importancia
            model_name: Nombre del modelo
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info(f"Generando importancia de características para {model_name}")
        
        # Crear DataFrame para ordenar
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Gráfico de barras horizontales
        bars = ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importancia')
        ax.set_title(f"{self._get_text('feature_importance')} - {model_name}")
        
        # Colorear barras
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save:
            filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]],
                            save: bool = True) -> str:
        """
        Visualiza comparación de modelos.
        
        Args:
            model_results: Diccionario con resultados de modelos
            save: Si guardar la imagen
            
        Returns:
            Ruta del archivo guardado
        """
        self.logger.info("Generando comparación de modelos")
        
        # Crear DataFrame con resultados
        comparison_data = []
        for model_name, metrics in model_results.items():
            row = {'Modelo': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Seleccionar métricas principales
        main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
        available_metrics = [m for m in main_metrics if m in df_comparison.columns]
        
        if len(available_metrics) == 0:
            self.logger.warning("No hay métricas disponibles para comparar")
            return ""
        
        # Crear gráfico
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:6]):
            ax = axes[i]
            df_comparison.plot(x='Modelo', y=metric, kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Ocultar ejes vacíos
        for i in range(len(available_metrics), 6):
            axes[i].set_visible(False)
        
        plt.suptitle(self._get_text("model_comparison"))
        plt.tight_layout()
        
        if save:
            filename = f"model_comparison.{self.config.output_format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Gráfico guardado en: {filepath}")
            return str(filepath)
        else:
            plt.show()
            return ""
    
    def create_interactive_dashboard(self, df: pd.DataFrame, target_col: str = "target_bad") -> str:
        """
        Crea un dashboard interactivo con Plotly.
        
        Args:
            df: DataFrame con datos
            target_col: Nombre de la columna target
            
        Returns:
            Ruta del archivo HTML guardado
        """
        if not self.config.interactive:
            self.logger.warning("Visualizaciones interactivas deshabilitadas")
            return ""
        
        self.logger.info("Creando dashboard interactivo")
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribución del Target',
                'Correlación de Características',
                'Distribución por Edad',
                'Distribución por Monto de Crédito'
            ],
            specs=[[{"type": "pie"}, {"type": "heatmap"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Gráfico de torta del target
        target_counts = df[target_col].value_counts()
        fig.add_trace(
            go.Pie(labels=['Bueno', 'Malo'], values=target_counts.values),
            row=1, col=1
        )
        
        # Matriz de correlación
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, 
                      x=corr_matrix.columns, 
                      y=corr_matrix.columns),
            row=1, col=2
        )
        
        # Histograma de edad
        if 'alter' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['alter'], name='Edad'),
                row=2, col=1
            )
        
        # Histograma de monto
        if 'hoehe' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['hoehe'], name='Monto'),
                row=2, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Dashboard Interactivo - Análisis de Riesgo Crediticio",
            showlegend=False,
            height=800
        )
        
        # Guardar como HTML
        filename = "interactive_dashboard.html"
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        
        self.logger.info(f"Dashboard interactivo guardado en: {filepath}")
        return str(filepath)
    
    def generate_report(self, df: pd.DataFrame, model_results: Dict[str, Any] = None,
                       target_col: str = "target_bad") -> str:
        """
        Genera un reporte completo con todas las visualizaciones.
        
        Args:
            df: DataFrame con datos
            model_results: Resultados de modelos entrenados
            target_col: Nombre de la columna target
            
        Returns:
            Ruta del archivo de reporte generado
        """
        self.logger.info("Generando reporte completo")
        
        generated_files = []
        
        # Generar visualizaciones básicas
        generated_files.append(self.plot_target_distribution(df, target_col))
        generated_files.append(self.plot_missing_values(df))
        generated_files.append(self.plot_correlation_matrix(df))
        generated_files.append(self.plot_feature_distributions(df, target_col))
        
        # Generar visualizaciones de modelos si están disponibles
        if model_results:
            for model_name, results in model_results.items():
                if 'metrics' in results:
                    # Aquí podrías agregar más visualizaciones específicas del modelo
                    pass
        
        # Crear dashboard interactivo
        if self.config.interactive:
            generated_files.append(self.create_interactive_dashboard(df, target_col))
        
        # Crear archivo de índice
        index_file = self.output_dir / "index.html"
        self._create_index_html(generated_files, index_file)
        
        self.logger.info(f"Reporte completo generado en: {self.output_dir}")
        return str(self.output_dir)
    
    def _create_index_html(self, files: List[str], output_path: Path) -> None:
        """
        Crea un archivo HTML de índice con todas las visualizaciones.
        
        Args:
            files: Lista de archivos generados
            output_path: Ruta del archivo de salida
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis de Riesgo Crediticio</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .image-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .file-list {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Análisis de Riesgo Crediticio</h1>
            <p>Generado el: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="file-list">
                <h2>Archivos Generados:</h2>
                <ul>
        """
        
        for file_path in files:
            if file_path and Path(file_path).exists():
                filename = Path(file_path).name
                html_content += f'<li><a href="{filename}">{filename}</a></li>\n'
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Función principal para ejecutar visualizaciones desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar visualizaciones para German Credit")
    parser.add_argument("input_file", help="Archivo CSV con datos")
    parser.add_argument("--target", help="Columna target", default="target_bad")
    parser.add_argument("--output-dir", help="Directorio de salida", default="reports/figures")
    parser.add_argument("--interactive", action="store_true", help="Generar visualizaciones interactivas")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], help="Formato de salida", default="png")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear configuración
    config = VisualizationConfig(
        output_dir=args.output_dir,
        interactive=args.interactive,
        output_format=args.format
    )
    
    # Crear visualizador
    visualizer = DataVisualizer(config)
    
    # Cargar datos
    df = pd.read_csv(args.input_file)
    
    # Generar reporte completo
    report_path = visualizer.generate_report(df, target_col=args.target)
    
    print(f"Reporte generado en: {report_path}")


if __name__ == "__main__":
    main()
