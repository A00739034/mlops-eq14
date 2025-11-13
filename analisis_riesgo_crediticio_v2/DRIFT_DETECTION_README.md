# Sistema de Detección de Data Drift

Este módulo implementa un sistema completo para detectar data drift y concept drift en modelos de machine learning, específicamente diseñado para el proyecto de análisis de riesgo crediticio.

## Características

- ✅ **Generación de datos con drift simulado**: Múltiples tipos de drift (desplazamiento de medias, valores faltantes, cambios en varianza, cambios categóricos)
- ✅ **Detección estadística de drift**: Pruebas de Kolmogorov-Smirnov para variables continuas y Chi-square para categóricas
- ✅ **Evaluación comparativa de métricas**: Compara el desempeño del modelo entre datos de referencia y datos con drift
- ✅ **Sistema de alertas**: Umbrales configurables para generar alertas cuando se detecta degradación
- ✅ **Visualizaciones**: Gráficos comparativos de distribuciones y métricas
- ✅ **Reportes detallados**: Reportes en Markdown con recomendaciones de acción

## Estructura del Módulo

```
src/monitoring/
├── detect_drift.py          # Clase principal DriftDetector
└── ...

run_drift_detection.py       # Script principal para ejecutar detección
reports/drift_detection/     # Resultados y reportes generados
```

## Uso

### Ejecución Básica

```bash
python run_drift_detection.py
```

Este script:
1. Carga los datos de referencia desde `data/processed/processed_data.csv`
2. Carga el modelo entrenado desde `models/best_model.joblib`
3. Genera datos con diferentes tipos de drift
4. Evalúa el impacto en el desempeño
5. Genera reportes y visualizaciones en `reports/drift_detection/`

### Uso Programático

```python
from src.monitoring.detect_drift import DriftDetector, DriftConfig
from src.data.data_processor import DataProcessor

# Configurar detector
config = DriftConfig(
    accuracy_threshold=0.05,  # 5% de cambio relativo
    roc_auc_threshold=0.05,
    mean_shift_factor=0.2,    # 20% de desplazamiento
    missing_value_rate=0.1   # 10% de valores faltantes
)

detector = DriftDetector(config=config)

# Cargar datos y modelo
detector.load_reference_data("data/processed/processed_data.csv", target_col="target_bad")
detector.load_model("models/best_model.joblib")

# Configurar procesador de datos
data_processor = DataProcessor()
detector.set_data_processors(data_processor)

# Ejecutar detección
results = detector.run_drift_detection(
    drift_type="combined",  # "mean_shift", "missing_values", "combined", etc.
    save_results=True,
    output_dir="reports/drift_detection"
)

# Acceder a resultados
print(f"Drift detectado: {results['summary']['drift_detected']}")
print(f"Severidad: {results['summary']['drift_severity']}")
print(f"Alertas: {results['summary']['alert_count']}")
```

### Línea de Comandos

```bash
python -m src.monitoring.detect_drift \
    --reference-data data/processed/processed_data.csv \
    --model-path models/best_model.joblib \
    --drift-type combined \
    --output-dir reports/drift_detection
```

## Tipos de Drift Soportados

### 1. Mean Shift (Desplazamiento de Medias)
Desplaza las medias de variables continuas para simular cambios en la distribución.

```python
drift_data = detector.generate_drift_data(drift_type="mean_shift", shift_factor=0.2)
```

### 2. Missing Values (Valores Faltantes)
Introduce valores faltantes en el dataset para simular problemas de calidad de datos.

```python
drift_data = detector.generate_drift_data(drift_type="missing_values", missing_rate=0.1)
```

### 3. Variance Change (Cambio de Varianza)
Modifica la varianza de variables continuas.

```python
drift_data = detector.generate_drift_data(drift_type="variance_change", variance_factor=1.5)
```

### 4. Categorical Shift (Cambio Categórico)
Altera la distribución de variables categóricas.

```python
drift_data = detector.generate_drift_data(drift_type="categorical_shift", shift_prob=0.15)
```

### 5. Combined (Combinado)
Aplica múltiples tipos de drift simultáneamente (recomendado para pruebas realistas).

```python
drift_data = detector.generate_drift_data(drift_type="combined")
```

## Configuración de Umbrales

Los umbrales determinan cuándo se generan alertas:

```python
config = DriftConfig(
    # Umbrales de alerta (cambios relativos)
    accuracy_threshold=0.05,      # 5% de cambio en accuracy
    roc_auc_threshold=0.05,        # 5% de cambio en ROC-AUC
    f1_threshold=0.05,             # 5% de cambio en F1
    
    # Umbrales estadísticos
    ks_statistic_threshold=0.3,   # Estadístico KS para drift
    p_value_threshold=0.05,        # p-value para significancia
    
    # Parámetros de drift simulado
    mean_shift_factor=0.2,         # Factor de desplazamiento (20%)
    missing_value_rate=0.1,        # Tasa de valores faltantes (10%)
    variance_change_factor=1.5,    # Factor de cambio de varianza (50% aumento)
    categorical_shift_probability=0.15  # Probabilidad de cambio categórico (15%)
)
```

## Interpretación de Resultados

### Severidad del Drift

- **none**: No se detectó drift significativo
- **low**: Drift detectado en <20% de las características
- **medium**: Drift detectado en 20-50% de las características
- **high**: Drift detectado en >50% de las características

### Alertas de Desempeño

Las alertas se generan cuando:
- El cambio relativo en accuracy, ROC-AUC o F1 excede los umbrales configurados
- La severidad puede ser "medium" o "high" dependiendo de la magnitud del cambio

### Acciones Recomendadas

El sistema genera recomendaciones automáticas basadas en los resultados:

1. **Revisión del Feature Pipeline**: Si se detecta drift, verificar el procesamiento de datos
2. **Análisis de Causas**: Investigar cambios en la distribución de datos de entrada
3. **Considerar Retraining**: Si la degradación es significativa (>5%), evaluar reentrenar el modelo
4. **Monitoreo Continuo**: Implementar monitoreo automático en producción

## Archivos Generados

Después de ejecutar la detección, se generan los siguientes archivos en `reports/drift_detection/`:

```
reports/drift_detection/
├── {drift_type}/
│   ├── drift_results_{timestamp}.json      # Resultados completos en JSON
│   ├── drift_report_{timestamp}.md          # Reporte detallado en Markdown
│   ├── distributions_comparison.png        # Comparación de distribuciones
│   ├── metrics_comparison.png              # Comparación de métricas
│   └── relative_changes.png                # Cambios relativos
└── consolidated_report.md                   # Reporte consolidado de todos los tipos
```

## Ejemplo de Salida

```
=== RESUMEN DE DETECCIÓN DE DRIFT ===
Drift Detectado: True
Severidad: medium
Alertas: 2

Métricas:
- Accuracy: 0.6640 → 0.6120 (-7.83%)
- ROC-AUC: 0.7288 → 0.6850 (-6.02%)
- F1: 0.6773 → 0.6345 (-6.32%)

Alertas:
- ACCURACY (medium): Accuracy cambió -7.83% (umbral: 5.0%)
- ROC_AUC (medium): ROC-AUC cambió -6.02% (umbral: 5.0%)
```

## Dependencias

Asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

O instala todas las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

## Integración con MLflow

Para registrar los resultados en MLflow:

```python
import mlflow

# Después de ejecutar detección
with mlflow.start_run():
    mlflow.log_params({
        "drift_type": results['drift_type'],
        "drift_severity": results['summary']['drift_severity']
    })
    
    mlflow.log_metrics({
        "drift_ratio": results['statistical_drift']['drift_ratio'],
        "alert_count": results['summary']['alert_count']
    })
    
    mlflow.log_artifact("reports/drift_detection/")
```

## Monitoreo en Producción

Para implementar monitoreo continuo:

1. **Programar ejecuciones periódicas**: Usar cron o un scheduler
2. **Configurar alertas**: Integrar con sistemas de notificación (email, Slack, etc.)
3. **Establecer umbrales**: Basados en los resultados de estas pruebas
4. **Automatizar acciones**: Retraining automático cuando se detecta drift crítico

## Troubleshooting

### Error: "Modelo no encontrado"
- Verifica que el modelo existe en `models/best_model.joblib`
- O especifica otro modelo con `--model-path`

### Error: "Datos de referencia no encontrados"
- Asegúrate de que `data/processed/processed_data.csv` existe
- O ejecuta primero el pipeline de procesamiento de datos

### Error: "No se pudo hacer predicciones"
- Verifica que el formato de los datos coincide con lo que el modelo espera
- Revisa los logs para más detalles sobre el error específico

## Referencias

- [MLOps Best Practices](https://ml-ops.org/)
- [Data Drift Detection](https://docs.evidentlyai.com/reference/data-drift)
- [Model Monitoring](https://www.kaggle.com/code/robikscube/model-monitoring-tutorial)

