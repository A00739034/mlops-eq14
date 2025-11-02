#!/bin/bash
# Script para iniciar MLflow UI desde el directorio correcto

# Cambiar al directorio del proyecto
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📂 Directorio del proyecto: $SCRIPT_DIR"
echo ""

# Verificar que existe mlruns
if [ ! -d "mlruns" ]; then
    echo "❌ No se encontró el directorio mlruns"
    echo "💡 Ejecuta primero: python3 upload_to_mlflow.py --action upload --environment local"
    exit 1
fi

# Contar experimentos
EXPERIMENTS=$(ls -d mlruns/*/ 2>/dev/null | grep -v "^mlruns/0$" | wc -l | tr -d ' ')
echo "📊 Experimentos encontrados: $EXPERIMENTS"
echo ""

# Listar experimentos
echo "📋 Experimentos disponibles:"
for dir in mlruns/*/; do
    if [ -f "$dir/meta.yaml" ]; then
        EXP_NAME=$(grep "name:" "$dir/meta.yaml" | cut -d' ' -f2)
        EXP_ID=$(basename "$dir")
        if [ "$EXP_ID" != "0" ]; then
            echo "   - $EXP_NAME (ID: $EXP_ID)"
        fi
    fi
done
echo ""

# Iniciar MLflow UI
echo "🚀 Iniciando MLflow UI..."
echo "🌐 Abre tu navegador en: http://localhost:5000"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

mlflow ui --backend-store-uri "file://$SCRIPT_DIR/mlruns" --host 0.0.0.0 --port 5000
