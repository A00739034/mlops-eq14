# 🔧 Solución: No ves experimentos en MLflow UI

## Problema común:
MLflow UI puede estar ejecutándose desde un directorio diferente al que contiene tus experimentos.

## ✅ Solución Paso a Paso:

### 1. **Verifica dónde están tus experimentos**

Los experimentos deberían estar en:
```
analisis_riesgo_crediticio_v2/mlruns/
```

### 2. **Detén MLflow UI actual** (si está corriendo)

Busca el proceso:
```bash
ps aux | grep mlflow
```

O simplemente presiona `Ctrl+C` en la terminal donde está corriendo.

### 3. **Inicia MLflow UI desde el directorio correcto**

**IMPORTANTE**: Debes estar EN EL DIRECTORIO DEL PROYECTO:

```bash
cd /Users/manuelambriz/Documents/MaestriaIa/MLops/MLOps/mlops-eq14/analisis_riesgo_crediticio_v2
mlflow ui --backend-store-uri file://$(pwd)/mlruns --host 0.0.0.0 --port 5000
```

O más simple, desde el directorio del proyecto:
```bash
cd analisis_riesgo_crediticio_v2
mlflow ui
```

### 4. **Verifica que los experimentos existan**

Antes de iniciar MLflow UI, verifica que hay experimentos:

```bash
ls -la mlruns/
```

Deberías ver algo como:
```
mlruns/
├── 0/                    # Experimento por defecto
└── 671460200784342881/   # Tu experimento german_credit_risk
```

### 5. **Si no ves experimentos, reejecuta el pipeline**

Si el directorio `mlruns` está vacío o solo tiene el experimento 0, necesitas reejecutar:

```bash
cd analisis_riesgo_crediticio_v2
python3 upload_to_mlflow.py --action upload --environment local
```

### 6. **Verificación alternativa - Usar el script de diagnóstico**

Ejecuta el script de verificación:

```bash
cd analisis_riesgo_crediticio_v2
python3 check_mlflow.py --show-instructions
```

## 🎯 **Comando Completo Recomendado:**

```bash
# 1. Ve al directorio del proyecto
cd /Users/manuelambriz/Documents/MaestriaIa/MLops/MLOps/mlops-eq14/analisis_riesgo_crediticio_v2

# 2. Verifica que existen experimentos
ls -la mlruns/

# 3. Si no hay experimentos, ejecuta el pipeline
python3 upload_to_mlflow.py --action upload --environment local

# 4. Inicia MLflow UI desde este directorio
mlflow ui --host 0.0.0.0 --port 5000

# 5. Abre tu navegador en: http://localhost:5000
```

## 📝 **Nota Importante:**

MLflow guarda los experimentos en el directorio `mlruns/` **relativo al directorio desde donde inicias mlflow ui**.

Si inicias MLflow desde `/Users/manuelambriz/Documents/MaestriaIa/MLops/MLOps/mlops-eq14/`, buscará experimentos en:
```
/Users/manuelambriz/Documents/MaestriaIa/MLops/MLOps/mlops-eq14/mlruns/
```

Pero tus experimentos están en:
```
/Users/manuelambriz/Documents/MaestriaIa/MLops/MLOps/mlops-eq14/analisis_riesgo_crediticio_v2/mlruns/
```

Por eso es **CRÍTICO** iniciar MLflow UI desde el directorio `analisis_riesgo_crediticio_v2`.
