# -*- coding: utf-8 -*-
"""
Script para verificar y corregir la configuración de MLflow.

Este script ayuda a diagnosticar por qué no se ven los experimentos en MLflow UI.
"""

import os
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

def check_mlflow_setup():
    """Verifica la configuración actual de MLflow."""
    print("🔍 Verificando configuración de MLflow...\n")
    
    # 1. Verificar tracking URI
    tracking_uri = mlflow.get_tracking_uri()
    print(f"📍 Tracking URI actual: {tracking_uri}")
    
    # 2. Verificar si existe el directorio mlruns
    if tracking_uri.startswith('file:'):
        mlruns_path = tracking_uri.replace('file:', '')
        mlruns_path = Path(mlruns_path).resolve()
        
        if mlruns_path.exists():
            print(f"✅ Directorio mlruns encontrado: {mlruns_path}")
            
            # Contar experimentos
            experiments_count = 0
            for item in mlruns_path.iterdir():
                if item.is_dir() and item.name.isdigit():
                    experiments_count += 1
                    print(f"   - Experimento encontrado: {item.name}")
            
            print(f"\n📊 Total de experimentos encontrados: {experiments_count}")
        else:
            print(f"❌ Directorio mlruns NO encontrado en: {mlruns_path}")
    else:
        print(f"⚠️  Usando servidor remoto: {tracking_uri}")
    
    # 3. Intentar listar experimentos con el cliente
    print("\n🔍 Intentando listar experimentos con MLflow Client...")
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        
        print(f"✅ Se encontraron {len(experiments)} experimentos:")
        for exp in experiments:
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
            
            # Contar ejecuciones
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            print(f"     Ejecuciones: {len(runs)}")
            
            if runs:
                best_run = runs[0]
                metrics = best_run.data.metrics
                if 'roc_auc' in metrics:
                    print(f"     Mejor ROC-AUC: {metrics['roc_auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Error listando experimentos: {str(e)}")
    
    # 4. Verificar si hay experimentos en otros lugares
    print("\n🔍 Buscando otros directorios mlruns...")
    current_dir = Path.cwd()
    for mlruns_dir in current_dir.rglob('mlruns'):
        if mlruns_dir.is_dir() and str(mlruns_dir) != str(mlruns_path):
            print(f"⚠️  Encontrado otro directorio mlruns en: {mlruns_dir}")
    
    return tracking_uri, mlruns_path if 'mlruns_path' in locals() else None


def fix_mlflow_config(project_dir: str = None):
    """Corrige la configuración de MLflow apuntando al directorio correcto."""
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
    
    mlruns_dir = project_dir / "mlruns"
    
    if not mlruns_dir.exists():
        print(f"❌ No se encontró mlruns en: {mlruns_dir}")
        return False
    
    # Configurar MLflow para usar este directorio
    tracking_uri = f"file://{mlruns_dir.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"✅ MLflow configurado para usar: {tracking_uri}")
    
    # Verificar experimentos
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        
        if experiments:
            print(f"\n📊 Experimentos disponibles:")
            for exp in experiments:
                print(f"   - {exp.name} (ID: {exp.experiment_id})")
        else:
            print("\n⚠️  No se encontraron experimentos. Puede que necesites ejecutar el pipeline nuevamente.")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    return True


def show_mlflow_ui_instructions():
    """Muestra instrucciones para iniciar MLflow UI correctamente."""
    project_dir = Path.cwd()
    mlruns_dir = project_dir / "mlruns"
    
    print("\n" + "="*60)
    print("📋 INSTRUCCIONES PARA INICIAR MLFLOW UI")
    print("="*60)
    print(f"\n1. Asegúrate de estar en el directorio del proyecto:")
    print(f"   cd {project_dir}")
    
    print(f"\n2. Inicia MLflow UI apuntando al directorio correcto:")
    print(f"   mlflow ui --backend-store-uri file://{mlruns_dir.resolve()}")
    
    print(f"\n   O simplemente:")
    print(f"   mlflow ui")
    print(f"   (si ya estás en el directorio del proyecto)")
    
    print(f"\n3. Abre tu navegador en:")
    print(f"   http://localhost:5000")
    
    print(f"\n4. Si necesitas usar un puerto diferente:")
    print(f"   mlflow ui --port 5001")
    
    print("\n" + "="*60)


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verificar y corregir configuración de MLflow")
    parser.add_argument("--fix", action="store_true", help="Corregir configuración automáticamente")
    parser.add_argument("--project-dir", help="Directorio del proyecto", default=None)
    parser.add_argument("--show-instructions", action="store_true", help="Mostrar instrucciones para MLflow UI")
    
    args = parser.parse_args()
    
    if args.show_instructions:
        show_mlflow_ui_instructions()
        return
    
    # Verificar configuración actual
    tracking_uri, mlruns_path = check_mlflow_setup()
    
    if args.fix:
        print("\n🔧 Corrigiendo configuración...")
        fix_mlflow_config(args.project_dir)
    
    print("\n💡 TIP: Usa --show-instructions para ver cómo iniciar MLflow UI correctamente")


if __name__ == "__main__":
    main()
