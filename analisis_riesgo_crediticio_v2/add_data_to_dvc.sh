#!/bin/bash
# Script helper para agregar archivos comunes del proyecto a DVC

set -e

echo "📦 Agregando archivos a DVC..."

# Verificar que DVC esté configurado
if [ ! -d .dvc ]; then
    echo "❌ DVC no está inicializado. Ejecuta primero: python setup_dvc_s3.py"
    exit 1
fi

# Agregar datos raw
if [ -f "data/raw/german_credit_modified.csv" ]; then
    echo "📊 Agregando datos raw..."
    dvc add data/raw/german_credit_modified.csv
else
    echo "⚠️  Archivo data/raw/german_credit_modified.csv no encontrado"
fi

# Agregar datos procesados (si existen archivos)
if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "📊 Agregando datos procesados..."
    # Agregar archivos individuales en processed
    for file in data/processed/*.csv; do
        if [ -f "$file" ]; then
            dvc add "$file"
        fi
    done
else
    echo "⚠️  Carpeta data/processed vacía o no existe"
fi

# Agregar modelos
if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
    echo "🤖 Agregando modelos..."
    # Agregar carpeta de modelos completa
    dvc add models/
else
    echo "⚠️  Carpeta models vacía o no existe"
fi

echo ""
echo "✅ Archivos agregados a DVC"
echo ""
echo "📝 Próximos pasos:"
echo "   1. Revisar los cambios: git status"
echo "   2. Agregar archivos .dvc a Git:"
echo "      git add *.dvc .gitignore"
echo "   3. Commit: git commit -m 'Add data to DVC'"
echo "   4. Subir a S3: dvc push"
