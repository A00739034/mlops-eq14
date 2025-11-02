#!/bin/bash
# Script para configurar DVC con S3 como almacenamiento remoto

set -e

echo "🚀 Configurando DVC con S3..."

# Verificar que DVC esté instalado
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC no está instalado. Instalando..."
    pip install 'dvc[s3]>=3.55.0'
fi

# Cargar variables de entorno desde .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  Archivo .env no encontrado. Por favor, créalo basándote en .env.example"
    exit 1
fi

# Verificar variables de entorno necesarias
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$S3_BUCKET_NAME" ]; then
    echo "❌ Faltan variables de entorno necesarias en .env"
    echo "   Se requieren: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME"
    exit 1
fi

# Inicializar DVC si no está inicializado
if [ ! -d .dvc ]; then
    echo "📦 Inicializando DVC..."
    dvc init --no-scm
fi

# Configurar el remoto S3
echo "🔧 Configurando remoto S3..."
S3_URL="s3://${S3_BUCKET_NAME}/dvc"

# Remover remoto anterior si existe
dvc remote remove default 2>/dev/null || true

# Agregar remoto S3
dvc remote add -d storage "$S3_URL"

# Configurar credenciales para S3
dvc remote modify storage access_key_id "$AWS_ACCESS_KEY_ID"
dvc remote modify storage secret_access_key "$AWS_SECRET_ACCESS_KEY"
dvc remote modify storage region "$AWS_REGION"

echo "✅ DVC configurado con S3:"
echo "   Remote URL: $S3_URL"
echo "   Region: ${AWS_REGION:-us-east-2}"
echo ""
echo "📝 Próximos pasos:"
echo "   1. Agregar archivos a DVC: dvc add data/raw/german_credit_modified.csv"
echo "   2. Hacer commit de los archivos .dvc: git add *.dvc .gitignore"
echo "   3. Subir a S3: dvc push"
