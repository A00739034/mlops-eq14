import os
import boto3
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener credenciales desde variables de entorno
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')
AWS_PROFILE = os.environ.get('AWS_PROFILE', 'default')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'mlops-eq-14')

# Configurar cliente S3
# Si las credenciales no están en variables de entorno, boto3 usará las credenciales por defecto del sistema
s3_config = {
    'region_name': AWS_REGION
}

# Solo agregar credenciales si están disponibles en variables de entorno
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_config['aws_access_key_id'] = AWS_ACCESS_KEY_ID
    s3_config['aws_secret_access_key'] = AWS_SECRET_ACCESS_KEY

s3_client = boto3.client('s3', **s3_config)
