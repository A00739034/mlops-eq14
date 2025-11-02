# Configuración de DVC con S3

Este documento explica cómo se ha configurado DVC (Data Version Control) para usar S3 como almacenamiento remoto.

## 📋 Requisitos Previos

- Python 3.7+
- Credenciales de AWS configuradas
- Bucket de S3 creado: `mlops-eq-14`
- Variables de entorno configuradas en `.env`

## 🚀 Instalación y Configuración

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará `dvc[s3]` que incluye el soporte para S3.

### 2. Configurar Variables de Entorno

Asegúrate de tener un archivo `.env` en la raíz del proyecto con:

```env
# AWS Credenciales
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_REGION=us-east-2

# S3 Bucket
S3_BUCKET=mlops-eq-14
S3_BUCKET_NAME=mlops-eq-14
```

**⚠️ IMPORTANTE**: El archivo `.env` está en `.gitignore` y no debe ser commiteado al repositorio.

### 3. Configurar DVC

Ejecuta uno de los siguientes scripts de configuración:

**Opción A: Script Python (Recomendado)**
```bash
python setup_dvc_s3.py
```

**Opción B: Script Bash**
```bash
bash setup_dvc_s3.sh
```

Estos scripts:
- Verificarán/instalarán DVC
- Inicializarán DVC en el proyecto
- Configurarán el remoto S3: `s3://mlops-eq-14/dvc`
- Configurarán las credenciales de AWS

## 📦 Uso de DVC

### Agregar Archivos/Carpetas a DVC

```bash
# Agregar archivo individual
dvc add data/raw/german_credit_modified.csv

# Agregar carpeta completa
dvc add data/processed/
dvc add models/
```

Esto creará archivos `.dvc` que contienen metadatos sobre los archivos versionados.

### Subir Datos a S3

```bash
# Subir todos los datos tracked por DVC a S3
dvc push
```

### Descargar Datos desde S3

```bash
# Descargar todos los datos tracked desde S3
dvc pull
```

### Comandos Útiles

```bash
# Ver estado de los archivos
dvc status

# Ver información del remoto configurado
dvc remote list
dvc remote default

# Verificar conexión con S3
dvc push --verbose
```

## 🔄 Flujo de Trabajo con Git + DVC

1. **Agregar datos a DVC:**
   ```bash
   dvc add data/raw/german_credit_modified.csv
   ```

2. **Commitear archivos .dvc (metadatos) a Git:**
   ```bash
   git add data/raw/german_credit_modified.csv.dvc .gitignore
   git commit -m "Add dataset to DVC"
   ```

3. **Subir datos a S3:**
   ```bash
   dvc push
   ```

4. **Push de código a Git:**
   ```bash
   git push
   ```

**⚠️ Nota**: Los archivos de datos grandes NO se suben a Git, solo los metadatos (archivos `.dvc`). Los datos reales se almacenan en S3.

## 📁 Archivos a Versionar con DVC

Recomendamos versionar:

- `data/raw/` - Datos crudos originales
- `data/processed/` - Datos procesados listos para modelado
- `models/` - Modelos entrenados y transformers

**Ejemplo de setup completo:**
```bash
# Versionar datos
dvc add data/raw/german_credit_modified.csv
dvc add data/processed/
dvc add models/

# Commitear metadatos
git add *.dvc .gitignore
git commit -m "Add data and models to DVC"

# Subir a S3
dvc push
```

## 🔧 Configuración Manual (Opcional)

Si prefieres configurar manualmente:

```bash
# Inicializar DVC
dvc init --no-scm

# Agregar remoto S3
dvc remote add -d storage s3://mlops-eq-14/dvc

# Configurar credenciales
dvc remote modify storage access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify storage secret_access_key $AWS_SECRET_ACCESS_KEY
dvc remote modify storage region us-east-2
```

## 🐛 Solución de Problemas

### Error: "Access Denied"
- Verifica que las credenciales de AWS sean correctas
- Verifica que el bucket `mlops-eq-14` exista y tengas permisos
- Verifica los permisos IAM del usuario

### Error: "Bucket does not exist"
- Crea el bucket en S3 primero
- Verifica que el nombre del bucket sea correcto

### Error: "Region mismatch"
- Verifica que la región en `.env` coincida con la región del bucket

## 📚 Referencias

- [Documentación oficial de DVC](https://dvc.org/doc)
- [DVC con S3](https://dvc.org/doc/command-reference/remote/add#amazon-s3)
- [Guía de mejores prácticas DVC](https://dvc.org/doc/user-guide/best-practices)

## 🔐 Seguridad

- **NUNCA** commitees el archivo `.env` con credenciales reales
- Usa variables de entorno o AWS IAM roles en producción
- Considera usar AWS Secrets Manager para credenciales en producción
- El archivo `.env` está en `.gitignore` para evitar commits accidentales
