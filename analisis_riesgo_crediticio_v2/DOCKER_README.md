# Guía de Docker - Pipeline de Reproducibilidad


## Requisitos Previos

### Software Necesario

- **Docker** versión 20.10 o superior
- **Docker Compose** versión 1.29 o superior (opcional pero recomendado)

### Verificar Instalación

```bash
# Verificar Docker
docker --version
# Debe mostrar: Docker version 20.10.x o superior

# Verificar Docker Compose
docker-compose --version
# Debe mostrar: docker-compose version 1.29.x o superior
```

---

##  Inicio Rápido

### Opción 1: Usar Scripts de Ayuda (Recomendado)

```bash
# 1. Construir imagen Docker
./docker_build.sh

# 2. Ejecutar pipeline de reproducibilidad
./docker_run_reproducibility.sh --seed 42 --verbose

# 3. Ver resultados
ls -la reports/reproducibility/
```

### Opción 2: Usar Docker Compose

```bash
# 1. Construir y ejecutar pipeline
docker-compose up pipeline

# 2. Ver logs
docker-compose logs -f pipeline

# 3. Limpiar
docker-compose down
```

### Opción 3: Comandos Docker Directos

```bash
# 1. Construir imagen
docker build -t riesgo-crediticio:latest .

# 2. Ejecutar pipeline
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  riesgo-crediticio:latest \
  python run_reproducibility_test.py --seed 42 --verbose
```

---

## 📖 Instrucciones Detalladas

### Paso 1: Construir la Imagen Docker

```bash
# Usar script de ayuda
./docker_build.sh

# O manualmente
docker build -t riesgo-crediticio:latest .

# Con tag específico
docker build -t riesgo-crediticio:v1.0 .
```

**Tiempo estimado:** 5-10 minutos (primera vez)

**¿Qué incluye la imagen?**
- Python 3.11
- Todas las dependencias de `requirements.txt` (versiones exactas)
- Código del pipeline completo
- Scripts de reproducibilidad

### Paso 2: Preparar Datos

Asegúrate de tener los datos en el directorio local:

```bash
# Verificar que existe el archivo de datos
ls -la data/raw/german_credit_modified.csv

# Si usas DVC, los datos se montarán desde tu máquina local
```

### Paso 3: Ejecutar Pipeline

#### Usando Script de Ayuda

```bash
# Ejecución básica
./docker_run_reproducibility.sh --seed 42

# Con opciones personalizadas
./docker_run_reproducibility.sh \
    --seed 42 \
    --data-dir ./data \
    --models-dir ./models \
    --reports-dir ./reports \
    --verbose

# Ejecutar y comparar automáticamente
./docker_run_reproducibility.sh --seed 42 --compare
```

#### Usando Docker Compose

```bash
# Ejecutar pipeline
docker-compose up pipeline

# Ejecutar en segundo plano
docker-compose up -d pipeline

# Ver logs
docker-compose logs -f pipeline

# Ejecutar con perfil de comparación
docker-compose --profile compare up compare
```

#### Usando Docker Directo

```bash
docker run --rm \
  --name riesgo_crediticio_pipeline \
  -e PYTHONHASHSEED=42 \
  -e RANDOM_SEED=42 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/logs:/app/logs \
  riesgo-crediticio:latest \
  python run_reproducibility_test.py --seed 42 --verbose
```

### Paso 4: Ver Resultados

Los resultados se guardan en los directorios montados:

```bash
# Métricas de referencia
cat reports/reproducibility/docker_run/reference_metrics.json

# Modelos entrenados
ls -la models/

# Logs
tail -f logs/reproducibility_test.log
```

---

## 🔄 Comparar Resultados

### Opción 1: Automática (con script)

```bash
./docker_run_reproducibility.sh --seed 42 --compare
```

### Opción 2: Manual con Docker

```bash
docker run --rm \
  -v $(pwd)/reports:/app/reports \
  riesgo-crediticio:latest \
  python compare_reproducibility_results.py \
    --reference reports/reproducibility/reference_metrics.json \
    --current reports/reproducibility/docker_run/reference_metrics.json \
    --output reports/reproducibility/comparison_report_docker.md \
    --verbose
```

### Opción 3: Con Docker Compose

```bash
docker-compose --profile compare up compare
```

---

##  Estructura de Volúmenes

Los volúmenes Docker montan directorios locales para:

| Volumen | Tipo | Propósito |
|---------|------|-----------|
| `./data` | Solo lectura (ro) | Datos de entrada |
| `./models` | Lectura/escritura | Modelos entrenados |
| `./reports` | Lectura/escritura | Reportes y métricas |
| `./logs` | Lectura/escritura | Logs de ejecución |

**Nota:** Los datos se montan como **solo lectura** para garantizar reproducibilidad.

---

## 🛠️ Scripts Disponibles

### `docker_build.sh`
Construye la imagen Docker.

```bash
./docker_build.sh [IMAGE_NAME] [TAG]
# Ejemplo: ./docker_build.sh riesgo-crediticio v1.0
```

### `docker_run_reproducibility.sh`
Ejecuta el pipeline de reproducibilidad.

**Opciones:**
- `-s, --seed SEED`: Semilla aleatoria (default: 42)
- `-i, --image IMAGE`: Nombre de imagen (default: riesgo-crediticio)
- `-d, --data-dir DIR`: Directorio de datos
- `-m, --models-dir DIR`: Directorio de modelos
- `-r, --reports-dir DIR`: Directorio de reports
- `-b, --build`: Construir imagen antes de ejecutar
- `-c, --compare`: Comparar resultados después
- `-v, --verbose`: Modo verbose
- `-h, --help`: Mostrar ayuda

---

## 📊 Ejemplos de Uso

### Ejemplo 1: Ejecución Completa

```bash
# 1. Construir imagen
./docker_build.sh

# 2. Ejecutar pipeline
./docker_run_reproducibility.sh --seed 42 --verbose

# 3. Verificar resultados
cat reports/reproducibility/docker_run/reference_metrics.json | jq
```

### Ejemplo 2: Comparación entre Ejecuciones

```bash
# Primera ejecución (guarda como referencia)
./docker_run_reproducibility.sh --seed 42
# Guardar manualmente: cp reports/reproducibility/docker_run/reference_metrics.json reports/reproducibility/reference_metrics.json

# Segunda ejecución (compara automáticamente)
./docker_run_reproducibilidad.sh --seed 42 --compare
```

### Ejemplo 3: Múltiples Semillas

```bash
# Ejecutar con diferentes semillas
for seed in 42 123 456; do
    echo "Ejecutando con semilla: $seed"
    ./docker_run_reproducibility.sh --seed $seed \
        --reports-dir "./reports/seed_$seed"
done
```

### Ejemplo 4: Ejecutar en Entorno Limpio (Nueva Máquina)

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd analisis_riesgo_crediticio_v2

# 2. Construir imagen (solo requiere Docker)
./docker_build.sh

# 3. Copiar datos (o descargar con DVC)
# Si usas DVC localmente, los datos se montan automáticamente
# Si los datos están en S3, necesitas configurar credenciales

# 4. Ejecutar pipeline
./docker_run_reproducibility.sh --seed 42 --verbose

# 5. Comparar con referencia
./docker_run_reproducibility.sh --seed 42 --compare
```

---

## 🔍 Verificar Reproducibilidad

### Checklist Docker

- [ ] Docker instalado y funcionando
- [ ] Imagen construida exitosamente
- [ ] Datos disponibles en `./data/raw/`
- [ ] Pipeline ejecutado sin errores
- [ ] Métricas generadas en `./reports/reproducibility/`
- [ ] Modelos guardados en `./models/`
- [ ] Comparación muestra "✅ REPRODUCIBLE"

### Verificación Rápida

```bash
# Ejecutar verificación completa
./quick_reproducibility_check.sh

# O con Docker
docker run --rm \
  -v $(pwd):/app \
  riesgo-crediticio:latest \
  bash -c "./quick_reproducibility_check.sh"
```

---

## 🐛 Troubleshooting

### Problema: Imagen no se construye

**Solución:**
```bash
# Limpiar cache de Docker
docker builder prune

# Construir sin cache
docker build --no-cache -t riesgo-crediticio:latest .
```

### Problema: Error de permisos

**Solución:**
```bash
# En Linux, ajustar permisos de directorios
sudo chown -R $USER:$USER data/ models/ reports/ logs/

# O ejecutar con permisos correctos
docker run --rm -u $(id -u):$(id -g) ...
```

### Problema: Datos no se encuentran

**Solución:**
```bash
# Verificar que los datos existen localmente
ls -la data/raw/german_credit_modified.csv

# Verificar montaje de volúmenes
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  riesgo-crediticio:latest \
  ls -la /app/data/raw/
```

### Problema: Contenedor se detiene inmediatamente

**Solución:**
```bash
# Ejecutar en modo interactivo para ver errores
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  riesgo-crediticio:latest \
  bash

# Dentro del contenedor:
python run_reproducibility_test.py --seed 42 --verbose
```

### Problema: Memoria insuficiente

**Solución:**
```bash
# Verificar memoria disponible
docker info | grep Memory

# Ajustar límites de memoria en Docker Desktop
# O usar menos recursos:
docker run --rm --memory="2g" ...
```

---

## 📦 Optimización de Imagen

### Tamaño de Imagen

La imagen incluye todas las dependencias necesarias. Para reducir tamaño:

```bash
# Ver tamaño actual
docker images | grep riesgo-crediticio

# Usar multi-stage build (avanzado)
# Ver Dockerfile.optimized para ejemplo
```

### Cache de Build

Docker usa cache de capas. El Dockerfile está optimizado para maximizar cache:

1. Copiar `requirements.txt` primero (cambia menos frecuentemente)
2. Instalar dependencias
3. Copiar código al final (cambia más frecuentemente)


---




