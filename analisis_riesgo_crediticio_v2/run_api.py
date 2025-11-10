"""
Script para iniciar la API de análisis de riesgo crediticio
"""

import uvicorn
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

def main():
    """
    Iniciar el servidor de la API
    """
    print("=" * 60)
    print("🚀 Iniciando API de Análisis de Riesgo Crediticio")
    print("=" * 60)
    print(f"📁 Directorio: {ROOT_DIR}")
    print(f"📊 Modelo: {ROOT_DIR / 'models' / 'best_model.joblib'}")
    print(f"🌐 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    # Verificar que el modelo existe
    model_path = ROOT_DIR / "models" / "best_model.joblib"
    if not model_path.exists():
        print(f"❌ ERROR: Modelo no encontrado en {model_path}")
        print("   Por favor, entrena el modelo primero o verifica la ruta.")
        sys.exit(1)
    
    print("✅ Modelo encontrado")
    print("\n💡 Presiona CTRL+C para detener el servidor\n")
    
    # Iniciar servidor
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
