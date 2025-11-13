# Guía: Levantar el Proyecto con Docker Desktop

Esta guía explica cómo ejecutar el pipeline de reproducibilidad usando **Docker Desktop** (interfaz gráfica).

## 📋 Requisitos Previos

1. **Docker Desktop instalado** (para Windows o Mac)
   - Descargar desde: https://www.docker.com/products/docker-desktop/
   - Verificar instalación: Docker Desktop debe estar ejecutándose

2. **Verificar que Docker Desktop está funcionando:**
   ```bash
   docker --version
   docker-compose --version
   ```

---

## 🚀 Método 1: Usando Docker Desktop (Interfaz Gráfica)

### Paso 1: Abrir Docker Desktop

1. Inicia **Docker Desktop** desde las aplicaciones
2. Espera a que el ícono de Docker en la barra de tareas muestre "Docker Desktop is running"

### Paso 2: Abrir Terminal Integrada

1. En Docker Desktop, ve a **Settings/Preferences**
2. Asegúrate de que la terminal esté configurada (Bash, PowerShell, o tu terminal preferida)
3. O usa tu terminal favorita (Terminal en Mac, PowerShell/CMD en Windows)

### Paso 3: Navegar al Proyecto

```bash
cd ruta/a/tu/proyecto/analisis_riesgo_crediticio_v2
```

### Paso 4: Construir la Imagen

**Opción A: Usando el Script (Más Fácil)**
```bash
./docker_build.sh
```

**Opción B: Usando Docker Desktop**
1. Abre Docker Desktop
2. Ve a la pestaña **"Images"**
3. Click en **"Build"**
4. Selecciona el directorio del proyecto: `analisis_riesgo_crediticio_v2`
5. Nombre de imagen: `riesgo-crediticio:latest`
6. Dockerfile: `Dockerfile` (debe estar en el directorio)
7. Click en **"Build"**

**Opción C: Comando Docker**
```bash
docker build -t riesgo-crediticio:latest .
```

### Paso 5: Verificar que la Imagen se Construyó

1. En Docker Desktop, ve a la pestaña **"Images"**
2. Debes ver `riesgo-crediticio:latest` en la lista
3. Verifica el tamaño y fecha de creación

### Paso 6: Ejecutar el Pipeline

**Opción A: Usando el Script (Más Fácil)**
```bash
./docker_run_reproducibility.sh --seed 42 --verbose
```

**Opción B: Usando Docker Desktop**
1. En Docker Desktop, ve a la pestaña **"Images"**
2. Encuentra `riesgo-crediticio:latest`
3. Click en el botón **"Run"** (▶️)
4. Configura:
   - **Container name**: `riesgo_crediticio_pipeline`
   - **Ports**: No necesario para pipeline (solo para API)
   - **Volumes**: Click en **"Optional settings"** → **"Volumes"**
     - Agregar bind mount:
       - Host path: `./data` → Container path: `/app/data:ro`
       - Host path: `./models` → Container path: `/app/models`
       - Host path: `./reports` → Container path: `/app/reports`
   - **Environment variables**: Click en **"Optional settings"** → **"Environment variables"**
     - Agregar: `PYTHONHASHSEED=42`
     - Agregar: `RANDOM_SEED=42`
   - **Command**: Click en **"Optional settings"** → **"Command"**
     - Ingresar: `python run_reproducibility_test.py --seed 42 --verbose`
5. Click en **"Run"**

**Opción C: Usando Docker Compose (Recomendado)**
```bash
docker-compose up pipeline
```

### Paso 7: Ver los Logs

**En Docker Desktop:**
1. Ve a la pestaña **"Containers"**
2. Encuentra `riesgo_crediticio_pipeline`
3. Click en el contenedor para ver detalles
4. Click en la pestaña **"Logs"** para ver la salida en tiempo real

**En Terminal:**
```bash
docker-compose logs -f pipeline
```

---

## 🎯 Método 2: Usando Docker Compose desde Docker Desktop

### Paso 1: Abrir Terminal

Usa la terminal integrada de Docker Desktop o tu terminal favorita.

### Paso 2: Navegar al Proyecto

```bash
cd ruta/a/tu/proyecto/analisis_riesgo_crediticio_v2
```

### Paso 3: Construir y Ejecutar con Docker Compose

```bash
# Construir y ejecutar en un solo comando
docker-compose up --build pipeline

# O separado:
# 1. Construir
docker-compose build pipeline

# 2. Ejecutar
docker-compose up pipeline
```

### Paso 4: Ver Resultados

Los resultados se guardan en tu máquina local en:
- `./reports/reproducibility/` - Métricas y reportes
- `./models/` - Modelos entrenados
- `./logs/` - Logs de ejecución

---

## 📊 Método 3: Usando los Scripts de Ayuda (Más Fácil)

### Paso 1: Abrir Terminal

Abre tu terminal (Docker Desktop puede estar ejecutándose en segundo plano).

### Paso 2: Navegar al Proyecto

```bash
cd ruta/a/tu/proyecto/analisis_riesgo_crediticio_v2
```

### Paso 3: Ejecutar Scripts

```bash
# 1. Construir imagen (primera vez o cuando cambies código)
./docker_build.sh

# 2. Ejecutar pipeline
./docker_run_reproducibility.sh --seed 42 --verbose

# 3. Ver resultados
ls -la reports/reproducibility/docker_run/
```

### Paso 4: Ver Resultados en Docker Desktop

1. Abre Docker Desktop
2. Ve a **"Containers"** para ver el contenedor ejecutándose
3. Ve a **"Images"** para ver la imagen
4. Ve a **"Volumes"** para ver volúmenes montados (si aplica)

---

## 🔍 Verificar que Todo Funciona

### 1. Verificar Docker Desktop está Ejecutándose

- El ícono de Docker en la barra de tareas debe estar verde/activo
- Docker Desktop debe mostrar "Docker Desktop is running"

### 2. Verificar Imagen Construida

```bash
docker images | grep riesgo-crediticio
```

Deberías ver algo como:
```
riesgo-crediticio   latest   abc123def456   2 minutes ago   2.5GB
```

### 3. Verificar Contenedor en Ejecución

```bash
docker ps
```

Si está ejecutándose, deberías ver `riesgo_crediticio_pipeline` en la lista.

### 4. Ver Logs

```bash
docker logs riesgo_crediticio_pipeline
```

O en Docker Desktop:
- Ve a **"Containers"** → Click en contenedor → **"Logs"**

### 5. Ver Resultados

```bash
# Ver métricas generadas
cat reports/reproducibility/docker_run/reference_metrics.json

# Ver modelos generados
ls -la models/

# Ver logs
tail -f logs/reproducibility_test.log
```

---

## 🐛 Troubleshooting en Docker Desktop

### Problema: Docker Desktop no inicia

**Solución:**
1. Reiniciar Docker Desktop
2. Verificar que Hyper-V/VirtualBox está habilitado (Windows)
3. Verificar que tienes suficientes recursos (RAM mínimo 4GB)

### Problema: No puedo construir la imagen

**Solución:**
1. Verificar que Docker Desktop está ejecutándose
2. Verificar que estás en el directorio correcto
3. Verificar que `Dockerfile` existe:
   ```bash
   ls -la Dockerfile
   ```

### Problema: Volúmenes no funcionan

**En Windows/Mac:**
1. Docker Desktop debe tener permisos para acceder a las carpetas
2. Ve a Docker Desktop → **Settings** → **Resources** → **File Sharing**
3. Asegúrate de que la ruta del proyecto esté compartida

**En Linux:**
- Los volúmenes funcionan directamente sin configuración adicional

### Problema: Contenedor se detiene inmediatamente

**Solución:**
1. Ver logs para ver el error:
   ```bash
   docker logs riesgo_crediticio_pipeline
   ```
2. Verificar que los datos existen:
   ```bash
   ls -la data/raw/german_credit_modified.csv
   ```
3. Ejecutar interactivamente para debuggear:
   ```bash
   docker run -it --rm \
     -v $(pwd)/data:/app/data:ro \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/reports:/app/reports \
     riesgo-crediticio:latest bash
   ```

---

## 💡 Tips para Docker Desktop

### 1. Ver Uso de Recursos

- Docker Desktop muestra CPU, memoria y disco en la pestaña **"Dashboard"**
- Útil para verificar que tienes suficientes recursos

### 2. Limpiar Recursos

```bash
# Limpiar contenedores detenidos
docker container prune

# Limpiar imágenes no usadas
docker image prune

# Limpiar todo (cuidado!)
docker system prune -a
```

O en Docker Desktop:
- **"Settings"** → **"Resources"** → **"Advanced"** → **"Clean / Purge data"**

### 3. Ver Logs en Tiempo Real

```bash
docker-compose logs -f pipeline
```

O en Docker Desktop:
- **"Containers"** → Click en contenedor → **"Logs"** (se actualiza automáticamente)

### 4. Ejecutar Comandos Dentro del Contenedor

**Desde Terminal:**
```bash
docker exec -it riesgo_crediticio_pipeline bash
```

**Desde Docker Desktop:**
1. Ve a **"Containers"**
2. Click en el contenedor
3. Click en **"Exec"** (terminal)
4. Ejecuta comandos dentro del contenedor

---

## 📝 Resumen de Comandos Rápidos

```bash
# 1. Construir imagen
./docker_build.sh
# O: docker build -t riesgo-crediticio:latest .

# 2. Ejecutar pipeline
./docker_run_reproducibility.sh --seed 42 --verbose
# O: docker-compose up pipeline

# 3. Ver logs
docker-compose logs -f pipeline
# O: docker logs riesgo_crediticio_pipeline

# 4. Ver resultados
ls -la reports/reproducibility/docker_run/

# 5. Comparar resultados
./docker_run_reproducibility.sh --seed 42 --compare

# 6. Limpiar
docker-compose down
docker system prune
```

---

## 🎯 Flujo de Trabajo Recomendado

1. **Iniciar Docker Desktop** (si no está ejecutándose)

2. **Construir imagen** (primera vez o cuando cambies código):
   ```bash
   ./docker_build.sh
   ```

3. **Ejecutar pipeline**:
   ```bash
   ./docker_run_reproducibility.sh --seed 42 --verbose
   ```

4. **Monitorear ejecución**:
   - Ver logs en terminal o Docker Desktop
   - Verificar que el contenedor está ejecutándose

5. **Ver resultados**:
   - Revisar métricas en `reports/reproducibility/`
   - Revisar modelos en `models/`

6. **Comparar resultados** (opcional):
   ```bash
   ./docker_run_reproducibility.sh --seed 42 --compare
   ```

7. **Limpiar** (cuando termines):
   ```bash
   docker-compose down
   ```

---

## 📚 Referencias Adicionales

- **Docker Desktop Documentation**: https://docs.docker.com/desktop/
- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **Guía de Reproducibilidad**: Ver `REPRODUCIBILIDAD.md`
- **Guía Docker Completa**: Ver `DOCKER_README.md`
- **Pasos Rápidos**: Ver `PASOS_REPRODUCIBILIDAD.md`

---

**Última actualización:** 2024-12-01

