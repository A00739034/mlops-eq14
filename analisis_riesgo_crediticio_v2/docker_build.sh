#!/bin/bash
# Script para construir la imagen Docker del pipeline

set -e

# Colores
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

IMAGE_NAME="${1:-riesgo-crediticio}"
TAG="${2:-latest}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Construyendo Imagen Docker${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Imagen: $IMAGE_NAME:$TAG"
echo ""

# Construir imagen
docker build -t "$IMAGE_NAME:$TAG" .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Imagen construida exitosamente${NC}"
    echo ""
    echo "Para ejecutar el pipeline:"
    echo "  ./docker_run_reproducibility.sh --seed 42"
    echo ""
    echo "O usando docker directamente:"
    echo "  docker run --rm -v \$(pwd)/data:/app/data -v \$(pwd)/models:/app/models -v \$(pwd)/reports:/app/reports $IMAGE_NAME:$TAG"
else
    echo ""
    echo -e "${YELLOW}⚠ Error construyendo la imagen${NC}"
    exit 1
fi

