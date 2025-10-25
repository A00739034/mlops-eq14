"""
Configuración del proyecto de análisis de riesgo crediticio
"""

from .aws_config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_PROFILE,
    S3_BUCKET_NAME,
    s3_client
)

__all__ = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY', 
    'AWS_REGION',
    'AWS_PROFILE',
    'S3_BUCKET_NAME',
    's3_client'
]