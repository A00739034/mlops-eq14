"""
Utilidades para interactuar con Amazon S3
Funciones para cargar, descargar y gestionar archivos en S3 para el proyecto de análisis de riesgo crediticio
"""

import os
import logging
import pickle
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from io import StringIO, BytesIO
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Utils:
    """Clase para gestionar operaciones con Amazon S3"""
    
    def __init__(self, bucket_name: str = None):
        """
        Inicializar cliente S3
        
        Args:
            bucket_name: Nombre del bucket de S3 por defecto
        """
        try:
            # Importar configuración de AWS
            from ..config.aws_config import s3_client
            self.s3_client = s3_client
            self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
            
            # Verificar conexión
            self.s3_client.list_buckets()
            logger.info("Conexión a S3 establecida correctamente")
            
        except NoCredentialsError:
            logger.error("Credenciales de AWS no encontradas")
            raise
        except Exception as e:
            logger.error(f"Error al conectar con S3: {e}")
            raise
    
    def upload_file(self, local_file_path: str, s3_key: str, bucket_name: str = None) -> bool:
        """
        Subir un archivo local a S3
        
        Args:
            local_file_path: Ruta del archivo local
            s3_key: Clave (path) del archivo en S3
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            True si la subida fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            self.s3_client.upload_file(local_file_path, bucket, s3_key)
            logger.info(f"Archivo subido exitosamente: {local_file_path} -> s3://{bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error al subir archivo: {e}")
            return False
    
    def download_file(self, s3_key: str, local_file_path: str, bucket_name: str = None) -> bool:
        """
        Descargar un archivo de S3 a local
        
        Args:
            s3_key: Clave del archivo en S3
            local_file_path: Ruta donde guardar el archivo local
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            True si la descarga fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            self.s3_client.download_file(bucket, s3_key, local_file_path)
            logger.info(f"Archivo descargado exitosamente: s3://{bucket}/{s3_key} -> {local_file_path}")
            return True
        except ClientError as e:
            logger.error(f"Error al descargar archivo: {e}")
            return False
    
    def delete_file(self, s3_key: str, bucket_name: str = None) -> bool:
        """
        Eliminar un archivo de S3
        
        Args:
            s3_key: Clave del archivo en S3
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Archivo eliminado exitosamente: s3://{bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error al eliminar archivo: {e}")
            return False
    
    def list_files(self, prefix: str = "", bucket_name: str = None) -> List[str]:
        """
        Listar archivos en S3 con un prefijo específico
        
        Args:
            prefix: Prefijo para filtrar archivos
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            Lista de claves de archivos
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            files = [obj['Key'] for obj in response.get('Contents', [])]
            logger.info(f"Encontrados {len(files)} archivos con prefijo '{prefix}'")
            return files
        except ClientError as e:
            logger.error(f"Error al listar archivos: {e}")
            return []
    
    def check_file_exists(self, s3_key: str, bucket_name: str = None) -> bool:
        """
        Verificar si un archivo existe en S3
        
        Args:
            s3_key: Clave del archivo en S3
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            True si el archivo existe, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, s3_key: str, format: str = 'csv', 
                        bucket_name: str = None, **kwargs) -> bool:
        """
        Subir un DataFrame a S3
        
        Args:
            df: DataFrame de pandas
            s3_key: Clave del archivo en S3
            format: Formato del archivo ('csv', 'parquet', 'json')
            bucket_name: Nombre del bucket (opcional)
            **kwargs: Argumentos adicionales para el método de escritura
        
        Returns:
            True si la subida fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            if format.lower() == 'csv':
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, **kwargs)
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=csv_buffer.getvalue()
                )
            elif format.lower() == 'parquet':
                parquet_buffer = BytesIO()
                df.to_parquet(parquet_buffer, index=False, **kwargs)
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=parquet_buffer.getvalue()
                )
            elif format.lower() == 'json':
                json_buffer = StringIO()
                df.to_json(json_buffer, orient='records', **kwargs)
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=json_buffer.getvalue()
                )
            else:
                logger.error(f"Formato no soportado: {format}")
                return False
            
            logger.info(f"DataFrame subido exitosamente: s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error al subir DataFrame: {e}")
            return False
    
    def download_dataframe(self, s3_key: str, format: str = 'csv', 
                          bucket_name: str = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Descargar un DataFrame desde S3
        
        Args:
            s3_key: Clave del archivo en S3
            format: Formato del archivo ('csv', 'parquet', 'json')
            bucket_name: Nombre del bucket (opcional)
            **kwargs: Argumentos adicionales para el método de lectura
        
        Returns:
            DataFrame de pandas o None si hay error
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            
            if format.lower() == 'csv':
                df = pd.read_csv(BytesIO(response['Body'].read()), **kwargs)
            elif format.lower() == 'parquet':
                df = pd.read_parquet(BytesIO(response['Body'].read()), **kwargs)
            elif format.lower() == 'json':
                df = pd.read_json(BytesIO(response['Body'].read()), **kwargs)
            else:
                logger.error(f"Formato no soportado: {format}")
                return None
            
            logger.info(f"DataFrame descargado exitosamente: s3://{bucket}/{s3_key}")
            return df
        except Exception as e:
            logger.error(f"Error al descargar DataFrame: {e}")
            return None
    
    def upload_model(self, model: Any, s3_key: str, bucket_name: str = None, 
                    use_joblib: bool = True) -> bool:
        """
        Subir un modelo entrenado a S3
        
        Args:
            model: Modelo a subir
            s3_key: Clave del archivo en S3
            bucket_name: Nombre del bucket (opcional)
            use_joblib: Si usar joblib o pickle
        
        Returns:
            True si la subida fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            buffer = BytesIO()
            
            if use_joblib:
                joblib.dump(model, buffer)
            else:
                pickle.dump(model, buffer)
            
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            
            logger.info(f"Modelo subido exitosamente: s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error al subir modelo: {e}")
            return False
    
    def download_model(self, s3_key: str, bucket_name: str = None, 
                      use_joblib: bool = True) -> Optional[Any]:
        """
        Descargar un modelo desde S3
        
        Args:
            s3_key: Clave del archivo en S3
            bucket_name: Nombre del bucket (opcional)
            use_joblib: Si usar joblib o pickle
        
        Returns:
            Modelo cargado o None si hay error
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            
            if use_joblib:
                model = joblib.load(BytesIO(response['Body'].read()))
            else:
                model = pickle.load(BytesIO(response['Body'].read()))
            
            logger.info(f"Modelo descargado exitosamente: s3://{bucket}/{s3_key}")
            return model
        except Exception as e:
            logger.error(f"Error al descargar modelo: {e}")
            return None
    
    def create_folder(self, folder_path: str, bucket_name: str = None) -> bool:
        """
        Crear una "carpeta" en S3 (realmente un objeto vacío con "/" al final)
        
        Args:
            folder_path: Ruta de la carpeta (debe terminar con "/")
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            True si la creación fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        try:
            self.s3_client.put_object(Bucket=bucket, Key=folder_path)
            logger.info(f"Carpeta creada exitosamente: s3://{bucket}/{folder_path}")
            return True
        except ClientError as e:
            logger.error(f"Error al crear carpeta: {e}")
            return False
    
    def delete_folder(self, folder_path: str, bucket_name: str = None) -> bool:
        """
        Eliminar una carpeta completa en S3
        
        Args:
            folder_path: Ruta de la carpeta
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        bucket = bucket_name or self.bucket_name
        
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        try:
            # Listar todos los objetos en la carpeta
            files = self.list_files(prefix=folder_path, bucket_name=bucket)
            
            if not files:
                logger.info(f"No se encontraron archivos en la carpeta: {folder_path}")
                return True
            
            # Eliminar todos los objetos
            delete_keys = [{'Key': file} for file in files]
            self.s3_client.delete_objects(
                Bucket=bucket,
                Delete={'Objects': delete_keys}
            )
            
            logger.info(f"Carpeta eliminada exitosamente: s3://{bucket}/{folder_path}")
            return True
        except ClientError as e:
            logger.error(f"Error al eliminar carpeta: {e}")
            return False
    
    def get_file_metadata(self, s3_key: str, bucket_name: str = None) -> Optional[Dict]:
        """
        Obtener metadatos de un archivo en S3
        
        Args:
            s3_key: Clave del archivo en S3
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            Diccionario con metadatos o None si hay error
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            metadata = {
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'etag': response.get('ETag'),
                'metadata': response.get('Metadata', {})
            }
            return metadata
        except ClientError as e:
            logger.error(f"Error al obtener metadatos: {e}")
            return None
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600, 
                              bucket_name: str = None) -> Optional[str]:
        """
        Generar URL firmada para acceso temporal a un archivo
        
        Args:
            s3_key: Clave del archivo en S3
            expiration: Tiempo de expiración en segundos (default: 1 hora)
            bucket_name: Nombre del bucket (opcional)
        
        Returns:
            URL firmada o None si hay error
        """
        bucket = bucket_name or self.bucket_name
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            logger.info(f"URL firmada generada para: s3://{bucket}/{s3_key}")
            return url
        except ClientError as e:
            logger.error(f"Error al generar URL firmada: {e}")
            return None
    
    # Métodos específicos para MLOps
    
    def upload_training_data(self, df: pd.DataFrame, experiment_name: str, 
                           version: str = None) -> bool:
        """
        Subir datos de entrenamiento con versionado
        
        Args:
            df: DataFrame con datos de entrenamiento
            experiment_name: Nombre del experimento
            version: Versión específica (si no se proporciona, usa timestamp)
        
        Returns:
            True si la subida fue exitosa, False en caso contrario
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        s3_key = f"data/training/{experiment_name}/v{version}/training_data.csv"
        return self.upload_dataframe(df, s3_key, format='csv')
    
    def upload_model_artifacts(self, model: Any, metrics: Dict, experiment_name: str, 
                              version: str = None) -> bool:
        """
        Subir artefactos del modelo (modelo + métricas)
        
        Args:
            model: Modelo entrenado
            metrics: Diccionario con métricas del modelo
            experiment_name: Nombre del experimento
            version: Versión específica
        
        Returns:
            True si ambas subidas fueron exitosas, False en caso contrario
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Subir modelo
        model_key = f"models/{experiment_name}/v{version}/model.joblib"
        model_success = self.upload_model(model, model_key)
        
        # Subir métricas
        metrics_df = pd.DataFrame([metrics])
        metrics_key = f"models/{experiment_name}/v{version}/metrics.csv"
        metrics_success = self.upload_dataframe(metrics_df, metrics_key, format='csv')
        
        return model_success and metrics_success
    
    def download_latest_model(self, experiment_name: str) -> Optional[Any]:
        """
        Descargar la versión más reciente del modelo
        
        Args:
            experiment_name: Nombre del experimento
        
        Returns:
            Modelo cargado o None si hay error
        """
        # Listar todas las versiones del modelo
        prefix = f"models/{experiment_name}/"
        files = self.list_files(prefix=prefix)
        
        # Filtrar archivos de modelo
        model_files = [f for f in files if f.endswith('/model.joblib')]
        
        if not model_files:
            logger.warning(f"No se encontraron modelos para el experimento: {experiment_name}")
            return None
        
        # Obtener la versión más reciente (asumiendo formato de timestamp)
        model_files.sort(reverse=True)
        latest_model_key = model_files[0]
        
        return self.download_model(latest_model_key)
    
    def list_experiments(self) -> List[str]:
        """
        Listar todos los experimentos disponibles
        
        Returns:
            Lista de nombres de experimentos
        """
        files = self.list_files(prefix="models/")
        experiments = set()
        
        for file in files:
            parts = file.split('/')
            if len(parts) >= 2:
                experiments.add(parts[1])
        
        return sorted(list(experiments))
    
    def get_experiment_versions(self, experiment_name: str) -> List[str]:
        """
        Obtener todas las versiones de un experimento
        
        Args:
            experiment_name: Nombre del experimento
        
        Returns:
            Lista de versiones disponibles
        """
        prefix = f"models/{experiment_name}/"
        files = self.list_files(prefix=prefix)
        versions = set()
        
        for file in files:
            parts = file.split('/')
            if len(parts) >= 3:
                versions.add(parts[2])
        
        return sorted(list(versions), reverse=True)


# Funciones de conveniencia para uso directo
def get_s3_utils(bucket_name: str = None) -> S3Utils:
    """
    Obtener instancia de S3Utils configurada
    
    Args:
        bucket_name: Nombre del bucket (opcional)
    
    Returns:
        Instancia de S3Utils
    """
    return S3Utils(bucket_name=bucket_name)


def quick_upload(local_file: str, s3_key: str, bucket_name: str = None) -> bool:
    """
    Función rápida para subir un archivo
    
    Args:
        local_file: Ruta del archivo local
        s3_key: Clave del archivo en S3
        bucket_name: Nombre del bucket (opcional)
    
    Returns:
        True si la subida fue exitosa, False en caso contrario
    """
    s3_utils = get_s3_utils(bucket_name)
    return s3_utils.upload_file(local_file, s3_key, bucket_name)


def quick_download(s3_key: str, local_file: str, bucket_name: str = None) -> bool:
    """
    Función rápida para descargar un archivo
    
    Args:
        s3_key: Clave del archivo en S3
        local_file: Ruta donde guardar el archivo local
        bucket_name: Nombre del bucket (opcional)
    
    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    s3_utils = get_s3_utils(bucket_name)
    return s3_utils.download_file(s3_key, local_file, bucket_name)
