# Pasos para Demostrar Reproducibilidad

Este documento proporciona los pasos concretos para demostrar que el mismo binario/modelo produce resultados consistentes en un entorno diferente al de entrenamiento.

## Resumen Ejecutivo

✅ **Implementado:**
- Módulo centralizado de configuración de semillas (`src/utils/reproducibility.py`)
- Versiones exactas en `requirements.txt`
- Script de reproducibilidad (`run_reproducibility_test.py`)
- Script de comparación (`compare_reproducibility_results.py`)
- Documentación completa (`REPRODUCIBILIDAD.md`)
- Pipeline actualizado para usar semillas centralizadas

---

## Pasos Rápidos (Entorno Actual)

### 1. Generar Métricas de Referencia

```bash
# Ejecutar pipeline con semillas fijas y guardar métricas
python run_reproducibility_test.py --seed 42 --verbose
```

Esto generará:
- `reports/reproducibility/reference_metrics.json`: Métricas de referencia
- `models/`: Modelos entrenados
- `reports/pipeline_summary.json`: Resumen del pipeline

### 2. Versionar Artefactos (DVC)

```bash
# Añadir modelos y transformadores a DVC
dvc add models/best_model.joblib
dvc add models/transformers/

# Commit y push
git add models/*.dvc
git commit -m "Versionar modelo y artefactos para reproducibilidad"
dvc push  # Si está configurado S3
```

### 3. Verificación Rápida (Mismo Entorno)

```bash
# Ejecutar verificación rápida (ejecuta pipeline 2 veces y compara)
./quick_reproducibility_check.sh
```

---

## Pasos Completos (Entorno Limpio/Diferente)

### Paso 1: Preparar Entorno Limpio

#### Opción A: Máquina/VM Nueva

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd analisis_riesgo_crediticio_v2

# 2. Crear entorno virtual
python3.11 -m venv venv_limpio
source venv_limpio/bin/activate  # Linux/Mac
# venv_limpio\Scripts\activate  # Windows

# 3. Actualizar pip
pip install --upgrade pip setuptools wheel

# 4. Instalar dependencias EXACTAS
pip install -r requirements.txt

# 5. Verificar versiones instaladas
pip list | grep -E "(pandas|numpy|scikit-learn|mlflow)"
```

#### Opción B: Contenedor Docker

```bash
# 1. Construir imagen
docker build -t riesgo-crediticio:reproducible .

# 2. Ejecutar contenedor
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  riesgo-crediticio:reproducible bash
```

### Paso 2: Obtener Datos y Artefactos

```bash
# 1. Obtener datos versionados (si están en DVC)
dvc pull

# 2. Verificar que existen los datos
ls -la data/raw/german_credit_modified.csv

# 3. Verificar artefactos (si están en DVC)
ls -la models/best_model.joblib
ls -la models/transformers/
```

### Paso 3: Ejecutar Pipeline con Semillas Fijas

```bash
# Ejecutar pipeline completo con semillas configuradas
python run_reproducibility_test.py \
    --seed 42 \
    --input data/raw/german_credit_modified.csv \
    --output-dir reports/reproducibility/new_run \
    --save-artifacts \
    --verbose
```

**Salida esperada:**
- ✅ Pipeline ejecutado exitosamente
- ✅ Métricas guardadas en `reports/reproducibility/new_run/reference_metrics.json`
- ✅ Modelos guardados en `models/`

### Paso 4: Comparar con Métricas de Referencia

```bash
# Comparar métricas actuales con referencia
python compare_reproducibility_results.py \
    --reference reports/reproducibility/reference_metrics.json \
    --current reports/reproducibility/new_run/reference_metrics.json \
    --output reports/reproducibility/comparison_report.md \
    --output-json reports/reproducibility/comparison_results.json \
    --verbose
```

**Resultado esperado:**
```
RESUMEN DE COMPARACIÓN
======================================================================
Total de modelos: 4
Modelos consistentes: 4
Datos consistentes: True
Características consistentes: True
Resultado general: ✅ REPRODUCIBLE
```

### Paso 5: Verificar Artefactos Versionados

```bash
# Si se usó DVC para versionar
dvc list models/
dvc show models/best_model.joblib.dvc

# Verificar que los modelos son idénticos (hash)
md5sum models/best_model.joblib
# Comparar con el hash de referencia
```

---

## Comparación de Métricas Esperadas

### Métricas del Mejor Modelo (ejemplo)

| Métrica | Valor Esperado | Tolerancia |
|---------|---------------|-----------|
| Accuracy | ~0.75 | ±0.01 |
| Precision | ~0.75 | ±0.01 |
| Recall | ~0.75 | ±0.01 |
| F1 | ~0.75 | ±0.01 |
| ROC-AUC | ~0.82 | ±0.01 |
| Average Precision | ~0.75 | ±0.01 |

### Verificación de Datos

- **Shape de datos:** Debe ser idéntico entre ejecuciones
- **Valores faltantes:** Debe ser idéntico
- **Duplicados:** Debe ser idéntico
- **Características:** Número de características debe ser idéntico

---

## Checklist de Verificación

- [ ] Python versión 3.11 instalada
- [ ] Todas las dependencias instaladas con versiones exactas de `requirements.txt`
- [ ] Semilla aleatoria configurada a 42
- [ ] Datos de entrada idénticos (`data/raw/german_credit_modified.csv`)
- [ ] Pipeline ejecutado sin errores
- [ ] Métricas generadas correctamente
- [ ] Comparación muestra "✅ REPRODUCIBLE"
- [ ] Modelos guardados y accesibles
- [ ] Artefactos versionados en DVC/MLflow (opcional)

---

## Solución de Problemas

### Problema: Métricas no coinciden

**Solución:**
1. Verificar versión de Python: `python --version` (debe ser 3.11)
2. Verificar versiones de dependencias: `pip list`
3. Verificar semilla: debe ser 42 en ambas ejecuciones
4. Verificar datos de entrada: deben ser idénticos

### Problema: Error al cargar datos con DVC

**Solución:**
```bash
# Configurar credenciales AWS (si aplica)
aws configure

# Forzar descarga
dvc pull --force
```

### Problema: Modelos no se generan

**Solución:**
```bash
# Ejecutar con modo verbose para ver errores
python run_reproducibility_test.py --verbose

# Verificar logs
tail -f reproducibility_test.log
```

---

## Estructura de Archivos de Salida

```
analisis_riesgo_crediticio_v2/
├── reports/
│   └── reproducibility/
│       ├── reference_metrics.json           # Métricas de referencia
│       ├── new_run/
│       │   └── reference_metrics.json      # Métricas de nueva ejecución
│       ├── comparison_report.md            # Reporte de comparación
│       └── comparison_results.json         # Resultados JSON
├── models/
│   ├── best_model.joblib                   # Mejor modelo
│   ├── *.joblib                            # Otros modelos
│   └── transformers/                       # Transformadores
└── reproducibility_test.log                # Log de ejecución
```

---

## Comandos de Referencia Rápida

```bash
# Generar métricas de referencia
python run_reproducibility_test.py --seed 42

# Comparar resultados
python compare_reproducibility_results.py \
    --reference reports/reproducibility/reference_metrics.json \
    --current reports/reproducibility/new_run/reference_metrics.json

# Verificación rápida (automatizada)
./quick_reproducibility_check.sh

# Ver métricas de referencia
cat reports/reproducibility/reference_metrics.json | jq

# Ver reporte de comparación
cat reports/reproducibility/comparison_report.md
```

---

## Referencias

- **Documentación completa:** Ver `REPRODUCIBILIDAD.md`
- **Módulo de reproducibilidad:** `src/utils/reproducibility.py`
- **Script de reproducibilidad:** `run_reproducibility_test.py`
- **Script de comparación:** `compare_reproducibility_results.py`

---

**Última actualización:** 2024-12-01

