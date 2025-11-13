#!/bin/bash
# Script rápido para verificar reproducibilidad del pipeline
# Este script ejecuta el pipeline dos veces con la misma semilla y compara resultados

set -e  # Salir si hay errores

echo "=========================================="
echo "Verificación Rápida de Reproducibilidad"
echo "=========================================="
echo ""

# Configuración
SEED=42
OUTPUT_DIR="reports/reproducibility/quick_check"
REFERENCE_FILE="${OUTPUT_DIR}/run1_metrics.json"
CURRENT_FILE="${OUTPUT_DIR}/run2_metrics.json"

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

echo "Paso 1: Ejecutar pipeline (primera ejecución)..."
python run_reproducibility_test.py \
    --seed "$SEED" \
    --output-dir "${OUTPUT_DIR}/run1" \
    --save-artifacts \
    --verbose 2>&1 | tee "${OUTPUT_DIR}/run1.log"

# Mover métricas a ubicación de referencia
cp "${OUTPUT_DIR}/run1/reference_metrics.json" "$REFERENCE_FILE" 2>/dev/null || \
cp "${OUTPUT_DIR}/run1"/*reference_metrics*.json "$REFERENCE_FILE" 2>/dev/null || \
echo "⚠️  No se encontraron métricas de referencia"

echo ""
echo "Paso 2: Ejecutar pipeline (segunda ejecución)..."
python run_reproducibility_test.py \
    --seed "$SEED" \
    --output-dir "${OUTPUT_DIR}/run2" \
    --save-artifacts \
    --verbose 2>&1 | tee "${OUTPUT_DIR}/run2.log"

# Mover métricas a ubicación actual
cp "${OUTPUT_DIR}/run2/reference_metrics.json" "$CURRENT_FILE" 2>/dev/null || \
cp "${OUTPUT_DIR}/run2"/*reference_metrics*.json "$CURRENT_FILE" 2>/dev/null || \
echo "⚠️  No se encontraron métricas actuales"

echo ""
echo "Paso 3: Comparar resultados..."

if [ -f "$REFERENCE_FILE" ] && [ -f "$CURRENT_FILE" ]; then
    python compare_reproducibility_results.py \
        --reference "$REFERENCE_FILE" \
        --current "$CURRENT_FILE" \
        --output "${OUTPUT_DIR}/comparison_report.md" \
        --output-json "${OUTPUT_DIR}/comparison_results.json" \
        --verbose
    
    # Verificar resultado
    if grep -q "✅ REPRODUCIBLE" "${OUTPUT_DIR}/comparison_report.md"; then
        echo ""
        echo "✅ ¡VERIFICACIÓN EXITOSA! Los resultados son reproducibles."
        exit 0
    else
        echo ""
        echo "❌ ERROR: Los resultados NO son reproducibles."
        exit 1
    fi
else
    echo "⚠️  No se pudieron encontrar archivos de métricas para comparar."
    echo "   Archivos buscados:"
    echo "   - Referencia: $REFERENCE_FILE"
    echo "   - Actual: $CURRENT_FILE"
    exit 1
fi

