#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para configurar DVC con S3 como almacenamiento remoto.
Este script inicializa DVC y configura el remoto S3 usando las credenciales de AWS.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv


def run_command(cmd, check=True):
    """Ejecuta un comando en la terminal."""
    print(f"🔧 Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    return result


def check_dvc_installed():
    """Verifica si DVC está instalado."""
    try:
        run_command(["dvc", "--version"], check=False)
        return True
    except FileNotFoundError:
        return False


def main():
    """Función principal."""
    print("🚀 Configurando DVC con S3...\n")
    
    # Cargar variables de entorno
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Variables de entorno cargadas desde .env\n")
    else:
        print("⚠️  Archivo .env no encontrado. Usando variables de entorno del sistema.\n")
    
    # Verificar e instalar DVC
    if not check_dvc_installed():
        print("📦 Instalando DVC con soporte S3...")
        run_command([sys.executable, "-m", "pip", "install", "dvc[s3]>=3.55.0"])
        print("✅ DVC instalado\n")
    else:
        print("✅ DVC ya está instalado\n")
    
    # Obtener variables de entorno
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-2")
    s3_bucket = os.getenv("S3_BUCKET_NAME") or os.getenv("S3_BUCKET")
    
    # Validar variables
    if not aws_access_key or not aws_secret_key:
        print("❌ Error: AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY son requeridos")
        sys.exit(1)
    
    if not s3_bucket:
        print("❌ Error: S3_BUCKET_NAME o S3_BUCKET es requerido")
        sys.exit(1)
    
    # Inicializar DVC si no está inicializado
    dvc_dir = Path(".dvc")
    if not dvc_dir.exists():
        print("📦 Inicializando DVC...")
        run_command(["dvc", "init", "--no-scm"])
        print("✅ DVC inicializado\n")
    else:
        print("✅ DVC ya está inicializado\n")
    
    # Configurar remoto S3
    s3_url = f"s3://{s3_bucket}/dvc"
    print(f"🔧 Configurando remoto S3: {s3_url}")
    
    # Remover remoto anterior si existe
    result = run_command(["dvc", "remote", "list"], check=False)
    if "storage" in result.stdout:
        print("   Removiendo remoto 'storage' anterior...")
        run_command(["dvc", "remote", "remove", "storage"], check=False)
    
    # Agregar remoto S3
    run_command(["dvc", "remote", "add", "-d", "storage", s3_url])
    
    # Configurar credenciales
    run_command(["dvc", "remote", "modify", "storage", "access_key_id", aws_access_key])
    run_command(["dvc", "remote", "modify", "storage", "secret_access_key", aws_secret_key])
    run_command(["dvc", "remote", "modify", "storage", "region", aws_region])
    
    print("\n✅ DVC configurado exitosamente con S3!")
    print(f"\n📊 Configuración:")
    print(f"   Remote URL: {s3_url}")
    print(f"   Region: {aws_region}")
    
    print("\n📝 Próximos pasos:")
    print("   1. Agregar datos a DVC:")
    print("      dvc add data/raw/german_credit_modified.csv")
    print("      dvc add data/processed/")
    print("      dvc add models/")
    print("\n   2. Hacer commit de los archivos .dvc:")
    print("      git add *.dvc .gitignore dvc.yaml dvc.lock")
    print("      git commit -m 'Add DVC configuration and tracked files'")
    print("\n   3. Subir datos a S3:")
    print("      dvc push")
    print("\n   4. Para descargar datos desde S3:")
    print("      dvc pull")


if __name__ == "__main__":
    main()
