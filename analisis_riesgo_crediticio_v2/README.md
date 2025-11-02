Analisis Riesgo Crediticio V2
==============================

Analisis de riesgo crediticio del equipo 14 de la materia de MLOPS Fase 2

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar DVC con S3
python setup_dvc_s3.py

# Descargar datos desde S3 (si ya están versionados)
dvc pull
```

### Versionado de Datos con DVC

Este proyecto utiliza **DVC (Data Version Control)** para versionar datos grandes y modelos, almacenándolos en **Amazon S3**.

- **Documentación completa**: Ver [DVC_S3_SETUP.md](DVC_S3_SETUP.md)
- **Configuración rápida**: Ejecutar `python setup_dvc_s3.py`

**Comandos principales:**
- `dvc add <archivo>` - Agregar archivo a DVC
- `dvc push` - Subir datos a S3
- `dvc pull` - Descargar datos desde S3
- `dvc status` - Ver estado de los archivos

## 🤖 Pipeline Automatizado de Scikit-learn

Este proyecto incluye un **pipeline completo de Scikit-learn** que automatiza todo el flujo de machine learning, desde el preprocesamiento hasta el modelo final.

### Características del Pipeline

El pipeline automatizado integra:

1. **Preprocesamiento de datos**
   - Limpieza y validación de datos
   - Validación de dominios categóricos y rangos continuos
   - Imputación de valores faltantes (mediana para continuas, moda para categóricas)

2. **Ingeniería de características**
   - Creación de características de interacción
   - Creación de características de ratio
   - Binning de variables continuas
   - Codificación de variables categóricas (Label Encoding y One-Hot Encoding)

3. **Selección de características**
   - Selección basada en información mutua o test F
   - Reducción de dimensionalidad opcional con PCA

4. **Modelo de Machine Learning**
   - Soporte para múltiples algoritmos (Logistic Regression, Random Forest, Gradient Boosting, SVM)

### Uso Rápido

```bash
# Entrenar con Logistic Regression
python run_sklearn_pipeline.py data/raw/german_credit_modified.csv --model logistic

# Entrenar con Random Forest y usar MLflow
python run_sklearn_pipeline.py data/raw/german_credit_modified.csv \
    --model random_forest --use-mlflow

# Configuración personalizada
python run_sklearn_pipeline.py data/raw/german_credit_modified.csv \
    --model gradient_boosting \
    --n-features 20 \
    --no-interactions \
    --output-model models/my_model.joblib
```

### Opciones Disponibles

- `--model`: Tipo de modelo (`logistic`, `random_forest`, `gradient_boosting`, `svm`)
- `--n-features`: Número de características a seleccionar (default: 15)
- `--feature-selection-method`: Método de selección (`mutual_info`, `f_classif`)
- `--test-size`: Tamaño del conjunto de prueba (default: 0.25)
- `--use-mlflow`: Activar tracking con MLflow
- `--no-interactions`: Desactivar características de interacción
- `--no-ratios`: Desactivar características de ratio
- `--no-binning`: Desactivar características de binning

### Uso Programático

```python
from src.models.sklearn_pipeline import create_sklearn_pipeline, SklearnPipelineManager
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Cargar datos
df = pd.read_csv("data/raw/german_credit_modified.csv")
X = df.drop(columns=["target_bad"])
y = df["target_bad"]

# Crear pipeline
pipeline = create_sklearn_pipeline(
    model=RandomForestClassifier(n_estimators=300, random_state=42),
    continuous_vars=["hoehe", "laufzeit", "alter"],
    categorical_vars=["laufkont", "moral", "verw", ...],
    scale_features=True,
    feature_selection=True,
    n_features_select=15
)

# Entrenar y evaluar
manager = SklearnPipelineManager(pipeline=pipeline)
results = manager.train_and_evaluate(X, y, use_mlflow=True)

# Guardar modelo
manager.save("models/my_pipeline.joblib")

# Cargar y usar para predicciones
manager.load("models/my_pipeline.joblib")
predictions = manager.predict(X_new)
```

### Ventajas del Pipeline Automatizado

✅ **Todo en un solo objeto**: Preprocesamiento, transformaciones y modelo unificados  
✅ **Fácil de usar**: Una sola llamada `fit()` y `predict()`  
✅ **Reproducible**: Todos los pasos están versionados y guardados  
✅ **MLflow compatible**: Tracking automático de experimentos  
✅ **Listo para producción**: Fácil de desplegar y usar en servicios

### Archivos del Pipeline

- `src/models/sklearn_pipeline.py`: Implementación del pipeline automatizado
- `run_sklearn_pipeline.py`: Script para ejecutar el pipeline desde línea de comandos

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
