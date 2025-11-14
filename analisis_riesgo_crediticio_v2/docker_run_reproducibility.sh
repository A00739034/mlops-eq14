#!/bin/bash
# Script para ejecutar el pipeline de reproducibilidad usando Docker
# Este script simplifica la ejecución del pipeline en contenedor Docker

set -e  # Salir si hay errores

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuración por defecto
SEED=42
IMAGE_NAME="riesgo-crediticio"
CONTAINER_NAME="riesgo_crediticio_pipeline"
DATA_DIR="./data"
MODELS_DIR="./models"
REPORTS_DIR="./reports"
LOGS_DIR="./logs"

# Función de ayuda
show_help() {
    cat << EOF
Uso: $0 [OPCIONES]

Script para ejecutar el pipeline de reproducibilidad usando Docker.

OPCIONES:
    -s, --seed SEED          Semilla aleatoria (por defecto: 42)
    -i, --image IMAGE        Nombre de la imagen Docker (por defecto: riesgo-crediticio)
    -d, --data-dir DIR       Directorio de datos (por defecto: ./data)
    -m, --models-dir DIR     Directorio de modelos (por defecto: ./models)
    -r, --reports-dir DIR    Directorio de reports (por defecto: ./reports)
    -b, --build              Construir imagen antes de ejecutar
    -c, --compare            Ejecutar comparación de resultados después
    -v, --verbose            Modo verbose
    -h, --help               Mostrar esta ayuda

EJEMPLOS:
    # Construir imagen y ejecutar pipeline
    $0 --build --seed 42

    # Solo ejecutar pipeline (asumiendo que la imagen ya existe)
    $0 --seed 42

    # Ejecutar pipeline y comparar resultados
    $0 --seed 42 --compare

    # Ejecutar con directorios personalizados
    $0 --data-dir ./my_data --models-dir ./my_models
EOF
}

# Parsear argumentos
BUILD=false
COMPARE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -m|--models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -r|--reports-dir)
            REPORTS_DIR="$2"
            shift 2
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -c|--compare)
            COMPARE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline de Reproducibilidad con Docker${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Crear directorios si no existen
mkdir -p "$DATA_DIR"/{raw,processed,interim}
mkdir -p "$MODELS_DIR"/transformers
mkdir -p "$REPORTS_DIR"/{figures,reproducibility}
mkdir -p "$LOGS_DIR"

# Construir imagen si se solicita
if [ "$BUILD" = true ]; then
    echo -e "${YELLOW}Construyendo imagen Docker...${NC}"
    docker build -t "$IMAGE_NAME:latest" .
    echo -e "${GREEN}✓ Imagen construida exitosamente${NC}"
    echo ""
fi

# Verificar que la imagen existe
if ! docker images | grep -q "^$IMAGE_NAME"; then
    echo -e "${YELLOW}⚠ Imagen no encontrada. Construyendo...${NC}"
    docker build -t "$IMAGE_NAME:latest" .
    echo ""
fi

# Verificar si DVC está configurado y preparar descarga de datos
DVC_ENV_ARGS=""
DVC_VOLUME_ARGS=""
DVC_PULL_CMD=""

if [ -d ".dvc" ]; then
    echo -e "${YELLOW}DVC detectado. Preparando descarga de datos desde S3...${NC}"
    
    # Montar directorio .dvc y archivos .dvc para que el contenedor tenga acceso a la configuración
    DVC_VOLUME_ARGS="-v $(pwd)/.dvc:/app/.dvc"
    
    # Montar archivos .dvc individuales si existen
    for dvc_file in $(find . -maxdepth 1 -name "*.dvc" -type f 2>/dev/null); do
        DVC_VOLUME_ARGS="$DVC_VOLUME_ARGS -v $(pwd)/$(basename $dvc_file):/app/$(basename $dvc_file)"
    done
    
    # Montar .dvcignore si existe
    if [ -f ".dvcignore" ]; then
        DVC_VOLUME_ARGS="$DVC_VOLUME_ARGS -v $(pwd)/.dvcignore:/app/.dvcignore"
    fi
    
    # Pasar variables de entorno de AWS si existen
    if [ -f ".env" ]; then
        echo -e "${YELLOW}  Cargando variables de entorno desde .env...${NC}"

        while IFS='=' read -r key value; do
            # Saltar líneas vacías o comentadas
            if [[ -z "$key" || "$key" =~ ^# ]]; then
                continue
            fi

            # Solo cargar variables relacionadas con AWS o S3
            if [[ "$key" =~ ^AWS_ || "$key" =~ ^S3_ ]]; then
                # Remover posibles caracteres CRLF
                clean_value=$(echo "$value" | tr -d '\r')
                export "$key=$clean_value"
            fi
        done < .env
    fi
    
    # Pasar variables de entorno de AWS al contenedor
    if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
        DVC_ENV_ARGS="$DVC_ENV_ARGS -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
    fi
    if [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
        DVC_ENV_ARGS="$DVC_ENV_ARGS -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
    fi
    if [ ! -z "$AWS_DEFAULT_REGION" ]; then
        DVC_ENV_ARGS="$DVC_ENV_ARGS -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION"
    fi
    if [ ! -z "$AWS_REGION" ]; then
        DVC_ENV_ARGS="$DVC_ENV_ARGS -e AWS_REGION=$AWS_REGION"
    fi
    
    # Comando para descargar datos con DVC
    DVC_PULL_CMD="cd /app && echo 'Descargando datos desde DVC...' && dvc pull && echo '✓ Datos descargados exitosamente' && "
    echo -e "${GREEN}  ✓ DVC configurado. Los datos se descargarán automáticamente.${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ DVC no detectado. Asegúrate de que los datos estén en $DATA_DIR/raw/${NC}"
    echo ""
fi

# Ejecutar pipeline
echo -e "${BLUE}Ejecutando pipeline con semilla: $SEED${NC}"
echo ""

VERBOSE_FLAG=""
if [ "$VERBOSE" = true ]; then
    VERBOSE_FLAG="--verbose"
fi

docker run --rm \
    --name "$CONTAINER_NAME" \
    -e PYTHONHASHSEED="$SEED" \
    -e RANDOM_SEED="$SEED" \
    $DVC_ENV_ARGS \
    -v "$(pwd)/$DATA_DIR:/app/data" \
    -v "$(pwd)/$MODELS_DIR:/app/models" \
    -v "$(pwd)/$REPORTS_DIR:/app/reports" \
    -v "$(pwd)/$LOGS_DIR:/app/logs" \
    $DVC_VOLUME_ARGS \
    "$IMAGE_NAME:latest" \
    bash -c "$DVC_PULL_CMD python run_reproducibility_test.py --seed $SEED --output-dir reports/reproducibility/docker_run $VERBOSE_FLAG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Pipeline ejecutado exitosamente${NC}"
    echo ""
    echo "Resultados disponibles en:"
    echo "  - Métricas: $REPORTS_DIR/reproducibility/docker_run/"
    echo "  - Modelos: $MODELS_DIR/"
    echo "  - Logs: $LOGS_DIR/"
    
    # Ejecutar comparación si se solicita
    if [ "$COMPARE" = true ]; then
        echo ""
        echo -e "${BLUE}Ejecutando comparación de resultados...${NC}"
        
        # Verificar que existen archivos de referencia
        if [ -f "$REPORTS_DIR/reproducibility/reference_metrics.json" ]; then
            docker run --rm \
                --name "${CONTAINER_NAME}_compare" \
                -v "$(pwd)/$REPORTS_DIR:/app/reports" \
                "$IMAGE_NAME:latest" \
                python compare_reproducibility_results.py \
                    --reference reports/reproducibility/reference_metrics.json \
                    --current reports/reproducibility/docker_run/reference_metrics.json \
                    --output reports/reproducibility/comparison_report_docker.md \
                    --verbose
            
            echo -e "${GREEN}✓ Comparación completada${NC}"
            echo "Reporte disponible en: $REPORTS_DIR/reproducibility/comparison_report_docker.md"
        else
            echo -e "${YELLOW}⚠ No se encontró archivo de referencia para comparar${NC}"
            echo "  Primera ejecución guardada como referencia"
        fi
    fi
else
    echo ""
    echo -e "${YELLOW}⚠ Pipeline terminó con código de salida: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi

