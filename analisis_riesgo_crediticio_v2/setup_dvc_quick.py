
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script rápido para configurar DVC con S3 usando las credenciales de aws_config.py
"""

import subprocess
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.aws_config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

def run_cmd(cmd, check=True):
    """Ejecuta un comando."""
    print(f"🔧 Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def main():
    """Función principal."""
    print("🚀 Configurando DVC con S3...\n")
    
    # Verificar que DVC esté instalado
    try:
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        print(f"✅ DVC instalado: {result.stdout.strip()}\n")
    except FileNotFoundError:
        print("❌ DVC no está instalado. Instalando...")
        if not run_cmd([sys.executable, "-m", "pip", "install", "dvc[s3]>=3.55.0"]):
            return False
        print()
    
    # Inicializar DVC si no está inicializado
    dvc_dir = Path(".dvc")
    if not dvc_dir.exists():
        print("📦 Inicializando DVC...")
        if not run_cmd(["dvc", "init", "--no-scm"]):
            return False
        print("✅ DVC inicializado\n")
    else:
        print("✅ DVC ya está inicializado\n")
    
    # Configurar remoto S3
    s3_url = f"s3://{S3_BUCKET_NAME}/dvc"
    print(f"🔧 Configurando remoto S3: {s3_url}")
    
    # Remover remoto anterior si existe
    result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
    if "storage" in result.stdout:
        print("   Removiendo remoto 'storage' anterior...")
        subprocess.run(["dvc", "remote", "remove", "storage"], capture_output=True)
    
    # Agregar remoto S3
    if not run_cmd(["dvc", "remote", "add", "-d", "storage", s3_url]):
        return False
    
    # Configurar credenciales
    if not run_cmd(["dvc", "remote", "modify", "storage", "access_key_id", AWS_ACCESS_KEY_ID]):
        return False
    if not run_cmd(["dvc", "remote", "modify", "storage", "secret_access_key", AWS_SECRET_ACCESS_KEY]):
        return False
    if not run_cmd(["dvc", "remote", "modify", "storage", "region", AWS_REGION]):
        return False
    
    print("\n✅ DVC configurado exitosamente con S3!")
    print(f"\n📊 Configuración:")
    print(f"   Remote URL: {s3_url}")
    print(f"   Region: {AWS_REGION}")
    
    # Verificar configuración
    print("\n🔍 Verificando configuración...")
    result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
    print(result.stdout)
    
    print("\n📝 Próximos pasos para agregar y subir archivos:")
    print("   1. Agregar archivos a DVC:")
    print("      dvc add data/raw/german_credit_modified.csv")
    print("      dvc add data/processed/")
    print("      dvc add models/")
    print("\n   2. Commitear metadatos:")
    print("      git add *.dvc .gitignore")
    print("      git commit -m 'Add data to DVC'")
    print("\n   3. Subir a S3:")
    print("      dvc push")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

