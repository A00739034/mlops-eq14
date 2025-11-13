# Guía de Reproducibilidad - Análisis de Riesgo Crediticio

Esta guía explica cómo demostrar que el mismo modelo/binario produce resultados consistentes en un entorno diferente al de entrenamiento.

## Índice

1. [Introducción](#introducción)
2. [Requisitos Previos](#requisitos-previos)
3. [Configuración del Entorno](#configuración-del-entorno)
4. [Ejecución del Pipeline con Reproducibilidad](#ejecución-del-pipeline-con-reproducibilidad)
5. [Comparación de Resultados](#comparación-de-resultados)
6. [Versionado de Artefactos](#versionado-de-artefactos)
7. [Verificación en Entorno Limpio](#verificación-en-entorno-limpio)
8. [Troubleshooting](#troubleshooting)

---

## Introducción

La reproducibilidad es un requisito fundamental en MLOps para:
- ✅ Auditar y depurar modelos
- ✅ Cumplir estándares de calidad y gobernanza
- ✅ Garantizar consistencia entre entornos
- ✅ Facilitar el despliegue y mantenimiento

Este proyecto implementa:
- **Semillas aleatorias fijas** en todas las operaciones aleatorias
- **Versiones exactas** de todas las dependencias
- **Configuración centralizada** de reproducibilidad
- **Scripts automatizados** para pruebas y comparación

---

## Requisitos Previos

### Software Necesario

**Opción A: Docker (Recomendado - Solo requiere Docker)**
- **Docker** versión 20.10 o superior
- **Docker Compose** versión 1.29 o superior (opcional pero recomendado)
- **Git** (para clonar el repositorio)

**Opción B: Entorno Local**
- **Python 3.11** (versión exacta recomendada para máxima reproducibilidad)
- **Git** (para clonar el repositorio)
- **DVC** (para versionado de datos)
- **MLflow** (para tracking de experimentos, opcional)

### Credenciales AWS (si se usa S3 para DVC)

```bash
# Configurar credenciales AWS (si aplica)
aws configure
```

---

## Configuración del Entorno

### Opción 1: Entorno Virtual (Recomendado)

```bash
# 1. Crear entorno virtual
python3.11 -m venv venv_reproducibilidad

# 2. Activar entorno virtual
# En Linux/Mac:
source venv_reproducibilidad/bin/activate
# En Windows:
venv_reproducibilidad\Scripts\activate

# 3. Actualizar pip
pip install --upgrade pip setuptools wheel

# 4. Instalar dependencias (versiones exactas)
pip install -r requirements.txt

# 5. Verificar instalación
python -c "import pandas, numpy, sklearn; print('✓ Dependencias instaladas correctamente')"
```

### Opción 2: Docker (Máxima Reproducibilidad) - RECOMENDADO

**Ventajas:**
- ✅ Solo requiere Docker instalado (no Python ni dependencias)
- ✅ Reproducibilidad 100% garantizada
- ✅ Funciona en cualquier sistema operativo
- ✅ Entorno completamente aislado

**Pasos rápidos:**

```bash
# 1. Construir imagen Docker
./docker_build.sh
# O manualmente: docker build -t riesgo-crediticio:latest .

# 2. Ejecutar pipeline con Docker
./docker_run_reproducibility.sh --seed 42 --verbose

# 3. Ver resultados
ls -la reports/reproducibility/docker_run/
```

**Documentación completa:** Ver `DOCKER_README.md` para instrucciones detalladas y ejemplos.

**Comandos Docker directos:**

```bash
# Construir imagen
docker build -t riesgo-crediticio:reproducible .

# Ejecutar pipeline
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  riesgo-crediticio:reproducible \
  python run_reproducibility_test.py --seed 42

# Ejecutar interactivo (para debugging)
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  riesgo-crediticio:reproducible bash
```

### Opción 3: Conda

```bash
# 1. Crear entorno conda
conda create -n riesgo-crediticio python=3.11 -y
conda activate riesgo-crediticio

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución del Pipeline con Reproducibilidad

### Paso 1: Configurar Semillas

Las semillas se configuran automáticamente al ejecutar el script de reproducibilidad. Por defecto, se usa la semilla `42`.

### Paso 2: Obtener Datos Versionados (DVC)

```bash
# Si los datos están versionados con DVC en S3
dvc pull

# Si los datos están en local, asegurarse de tener:
# - data/raw/german_credit_modified.csv
```

### Paso 3: Ejecutar Pipeline con Reproducibilidad

```bash
# Ejecutar pipeline completo con semillas fijas
python run_reproducibility_test.py

# Con opciones personalizadas:
python run_reproducibility_test.py \
    --seed 42 \
    --input data/raw/german_credit_modified.csv \
    --output-dir reports/reproducibility \
    --save-artifacts \
    --verbose
```

**Parámetros disponibles:**

- `--seed`: Semilla aleatoria (por defecto: 42)
- `--input`: Ruta del archivo de datos de entrada
- `--output-dir`: Directorio donde guardar métricas de referencia
- `--save-artifacts`: Guardar artefactos del pipeline (modelos, transformadores, etc.)
- `--no-save-artifacts`: No guardar artefactos adicionales
- `--verbose`: Modo verbose con logging detallado

### Paso 4: Verificar Métricas de Referencia

Después de ejecutar el pipeline, se generarán:

- `reports/reproducibility/reference_metrics.json`: Métricas de referencia
- `reports/reproducibility/reference_metrics_TIMESTAMP.json`: Métricas con timestamp
- `models/`: Modelos entrenados y transformadores
- `reports/pipeline_summary.json`: Resumen del pipeline

**Estructura de métricas de referencia:**

```json
{
  "timestamp": "2024-12-01T12:00:00",
  "seed": 42,
  "python_version": "3.11.0",
  "model_metrics": {
    "RandomForest": {
      "accuracy": 0.7500,
      "precision": 0.7523,
      "recall": 0.7500,
      "f1": 0.7511,
      "roc_auc": 0.8234,
      "average_precision": 0.7456,
      "training_time": 2.34
    },
    ...
  },
  "data_info": {...},
  "feature_info": {...},
  "best_model": "RandomForest_optimized"
}
```

---

## Comparación de Resultados

### Comparar Resultados entre Ejecuciones

```bash
# Comparar métricas de referencia con nuevas métricas
python compare_reproducibility_results.py \
    --reference reports/reproducibility/reference_metrics.json \
    --current reports/reproducibility/reference_metrics_20241201_120000.json \
    --output reports/reproducibility/comparison_report.md \
    --output-json reports/reproducibility/comparison_results.json \
    --verbose
```

**Parámetros:**

- `--reference`: Archivo JSON con métricas de referencia
- `--current`: Archivo JSON con métricas actuales a comparar
- `--output`: Ruta del archivo de reporte Markdown
- `--output-json`: Ruta del archivo JSON con resultados de comparación (opcional)
- `--verbose`: Modo verbose

### Interpretar Resultados de Comparación

El script de comparación genera:

1. **Reporte Markdown** (`comparison_report.md`):
   - Resumen ejecutivo
   - Comparación detallada por modelo
   - Estado de consistencia

2. **Resultados JSON** (opcional):
   - Datos estructurados de la comparación
   - Diferencias calculadas
   - Estado de tolerancia

**Tolerancias por defecto:**

| Métrica | Tolerancia |
|---------|-----------|
| accuracy | ±0.01 (1%) |
| precision | ±0.01 (1%) |
| recall | ±0.01 (1%) |
| f1 | ±0.01 (1%) |
| roc_auc | ±0.01 (1%) |
| average_precision | ±0.01 (1%) |
| training_time | ±10 segundos |

**Ejemplo de salida:**

```
RESUMEN DE COMPARACIÓN
======================================================================
Total de modelos: 4
Modelos consistentes: 4
Datos consistentes: True
Características consistentes: True
Resultado general: ✅ REPRODUCIBLE
```

---

## Versionado de Artefactos

### Versionar con DVC

```bash
# 1. Añadir artefactos a DVC
dvc add models/best_model.joblib
dvc add models/transformers/
dvc add data/processed/features_data.csv

# 2. Commit cambios
git add models/best_model.joblib.dvc models/transformers.dvc data/processed/features_data.csv.dvc
git commit -m "Versionar modelo y artefactos de reproducibilidad"

# 3. Push a S3 (si está configurado)
dvc push
```

### Versionar con MLflow

Los modelos se versionan automáticamente en MLflow durante el entrenamiento:

```bash
# Iniciar servidor MLflow (opcional, para visualización)
mlflow ui

# Acceder a: http://localhost:5000
```

**Parámetros registrados en MLflow:**

- Semilla aleatoria (`random_state`)
- Hiperparámetros del modelo
- Métricas de evaluación
- Artefactos (modelo, transformadores)
- Firma del modelo

---

## Verificación en Entorno Limpio

### Pasos Completos para Verificación

#### En Máquina Local Diferente o VM

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd analisis_riesgo_crediticio_v2

# 2. Crear entorno virtual limpio
python3.11 -m venv venv_limpio
source venv_limpio/bin/activate  # Linux/Mac
# venv_limpio\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Obtener datos versionados
dvc pull  # Si los datos están en DVC

# 5. Ejecutar pipeline con la misma semilla
python run_reproducibility_test.py \
    --seed 42 \
    --input data/raw/german_credit_modified.csv \
    --output-dir reports/reproducibility/new_run

# 6. Comparar con métricas de referencia
python compare_reproducibility_results.py \
    --reference reports/reproducibility/reference_metrics.json \
    --current reports/reproducibility/new_run/reference_metrics.json \
    --output reports/reproducibility/comparison_report.md

# 7. Verificar que el resultado sea "REPRODUCIBLE"
```

#### En Contenedor Docker (Recomendado)

```bash
# 1. Construir imagen
./docker_build.sh
# O: docker build -t riesgo-crediticio:test .

# 2. Ejecutar pipeline en contenedor
./docker_run_reproducibility.sh --seed 42 --verbose

# 3. Comparar resultados automáticamente
./docker_run_reproducibility.sh --seed 42 --compare

# O manualmente:
docker run --rm \
  -v $(pwd)/reports:/app/reports \
  riesgo-crediticio:test \
  python compare_reproducibility_results.py \
    --reference reports/reproducibility/reference_metrics.json \
    --current reports/reproducibility/docker_run/reference_metrics.json \
    --output reports/reproducibility/comparison_report_docker.md
```

**Ventajas de Docker:**
- ✅ No requiere Python ni dependencias instaladas localmente
- ✅ Entorno idéntico garantizado
- ✅ Funciona en cualquier sistema con Docker

**Ver documentación completa:** `DOCKER_README.md`

### Checklist de Verificación

- [ ] Python versión correcta (3.11)
- [ ] Todas las dependencias instaladas con versiones exactas
- [ ] Semilla aleatoria configurada a 42
- [ ] Datos de entrada idénticos
- [ ] Pipeline ejecutado sin errores
- [ ] Métricas generadas correctamente
- [ ] Comparación muestra "REPRODUCIBLE"
- [ ] Modelos guardados y accesibles
- [ ] Artefactos versionados en DVC/MLflow

---

## Troubleshooting

### Problemas Comunes

#### 1. Métricas no coinciden exactamente

**Causa:** Diferencias menores en operaciones de punto flotante entre sistemas.

**Solución:**
- Verificar que las tolerancias sean apropiadas
- Asegurarse de usar las mismas versiones de dependencias
- Verificar que el sistema operativo y arquitectura sean compatibles

#### 2. Error al cargar datos con DVC

**Causa:** Credenciales AWS no configuradas o datos no disponibles.

**Solución:**
```bash
# Configurar credenciales AWS
aws configure

# Verificar configuración DVC
dvc remote list

# Forzar descarga
dvc pull --force
```

#### 3. Modelos no se generan

**Causa:** Error en el pipeline de entrenamiento.

**Solución:**
```bash
# Ejecutar con modo verbose
python run_reproducibility_test.py --verbose

# Verificar logs
tail -f reproducibility_test.log
```

#### 4. Diferentes versiones de Python

**Causa:** Versiones de Python diferentes entre entornos.

**Solución:**
- Usar exactamente Python 3.11
- Verificar versión: `python --version`
- Usar Docker para garantizar entorno idéntico

#### 5. Semillas no se aplican correctamente

**Causa:** Orden de importación incorrecto.

**Solución:**
- Asegurarse de importar `reproducibility` antes que otros módulos
- Verificar que `set_seed()` se llame antes de crear modelos

---

## Estructura de Archivos de Reproducibilidad

```
analisis_riesgo_crediticio_v2/
├── src/
│   └── utils/
│       └── reproducibility.py          # Módulo de configuración de semillas
├── run_reproducibility_test.py         # Script de ejecución con reproducibilidad
├── compare_reproducibility_results.py  # Script de comparación de métricas
├── requirements.txt                    # Dependencias con versiones exactas
├── REPRODUCIBILIDAD.md                 # Esta guía
├── reports/
│   └── reproducibility/
│       ├── reference_metrics.json      # Métricas de referencia
│       ├── reference_metrics_*.json    # Métricas con timestamp
│       ├── comparison_report.md        # Reporte de comparación
│       └── comparison_results.json     # Resultados de comparación JSON
└── reproducibility_test.log            # Log de ejecución
```

---

## Referencias

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn Random State](https://scikit-learn.org/stable/glossary.html#term-random_state)
- [NumPy Random Seed](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html)

---

## Contacto y Soporte

Para problemas o preguntas sobre reproducibilidad:
1. Revisar logs en `reproducibility_test.log`
2. Verificar configuración en `src/utils/reproducibility.py`
3. Consultar esta documentación

---

**Última actualización:** 2024-12-01

