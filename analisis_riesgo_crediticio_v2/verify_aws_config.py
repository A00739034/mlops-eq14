#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar la configuración de AWS y conexión con S3.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.aws_config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET_NAME,
    s3_client
)

def verify_aws_config():
    """Verifica la configuración de AWS."""
    print("=" * 60)
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN AWS")
    print("=" * 60)
    
    # Verificar credenciales
    print("\n📋 Credenciales:")
    print(f"   AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY_ID[:15]}..." if AWS_ACCESS_KEY_ID else "   ❌ AWS_ACCESS_KEY_ID: NO CONFIGURADO")
    print(f"   AWS_SECRET_ACCESS_KEY: {'*' * 10}..." if AWS_SECRET_ACCESS_KEY else "   ❌ AWS_SECRET_ACCESS_KEY: NO CONFIGURADO")
    print(f"   AWS_REGION: {AWS_REGION}")
    print(f"   S3_BUCKET_NAME: {S3_BUCKET_NAME}")
    
    # Verificar que las credenciales estén configuradas
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("\n❌ ERROR: Credenciales de AWS no configuradas")
        return False
    
    if not S3_BUCKET_NAME:
        print("\n❌ ERROR: Nombre del bucket S3 no configurado")
        return False
    
    print("\n✅ Credenciales básicas configuradas correctamente")
    
    # Verificar conexión con S3
    print("\n🌐 Verificando conexión con S3...")
    try:
        # Intentar listar buckets
        response = s3_client.list_buckets()
        print("   ✅ Conexión con AWS S3 exitosa")
        
        # Verificar si el bucket existe
        bucket_names = [b['Name'] for b in response.get('Buckets', [])]
        
        if S3_BUCKET_NAME in bucket_names:
            print(f"   ✅ Bucket '{S3_BUCKET_NAME}' existe en tu cuenta")
            
            # Verificar acceso al bucket
            try:
                s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
                print(f"   ✅ Tienes acceso al bucket '{S3_BUCKET_NAME}'")
                
                # Intentar listar objetos en el bucket
                try:
                    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=5)
                    obj_count = response.get('KeyCount', 0)
                    print(f"   📦 Objetos en el bucket: {obj_count} (mostrando primeros 5)")
                    
                    if obj_count > 0:
                        print("   Archivos encontrados:")
                        for obj in response.get('Contents', [])[:5]:
                            print(f"      - {obj['Key']} ({obj['Size']} bytes)")
                    else:
                        print("   ℹ️  El bucket está vacío (esto es normal si es nuevo)")
                        
                except Exception as e:
                    print(f"   ⚠️  No se pudieron listar objetos: {e}")
                    
            except s3_client.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '403':
                    print(f"   ❌ Sin acceso al bucket '{S3_BUCKET_NAME}' (permisos insuficientes)")
                    print("      Verifica los permisos IAM de tu usuario")
                    return False
                else:
                    print(f"   ❌ Error al acceder al bucket: {e}")
                    return False
                    
        else:
            print(f"   ⚠️  El bucket '{S3_BUCKET_NAME}' NO existe en tu cuenta")
            print(f"   💡 Puedes crearlo con:")
            print(f"      aws s3 mb s3://{S3_BUCKET_NAME} --region {AWS_REGION}")
            print(f"   O desde Python:")
            print(f"      s3_client.create_bucket(Bucket='{S3_BUCKET_NAME}', CreateBucketConfiguration={{'LocationConstraint': '{AWS_REGION}'}})")
            
            # Preguntar si quiere crearlo
            create = input(f"\n   ¿Deseas crear el bucket '{S3_BUCKET_NAME}' ahora? (s/n): ").lower()
            if create == 's':
                try:
                    if AWS_REGION == 'us-east-1':
                        # us-east-1 no requiere LocationConstraint
                        s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
                    else:
                        s3_client.create_bucket(
                            Bucket=S3_BUCKET_NAME,
                            CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                        )
                    print(f"   ✅ Bucket '{S3_BUCKET_NAME}' creado exitosamente")
                except Exception as e:
                    print(f"   ❌ Error al crear el bucket: {e}")
                    return False
            else:
                return False
                
    except s3_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidAccessKeyId':
            print(f"   ❌ AWS_ACCESS_KEY_ID inválido")
            return False
        elif error_code == 'SignatureDoesNotMatch':
            print(f"   ❌ AWS_SECRET_ACCESS_KEY inválido")
            return False
        else:
            print(f"   ❌ Error de autenticación: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
        print("      Verifica tu conexión a internet y las credenciales")
        return False
    
    # Verificar región
    print(f"\n🌍 Región configurada: {AWS_REGION}")
    
    print("\n" + "=" * 60)
    print("✅ CONFIGURACIÓN AWS VERIFICADA CORRECTAMENTE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = verify_aws_config()
    sys.exit(0 if success else 1)

