import boto3
import logging

logger = logging.getLogger(__name__)

def check_bedrock_availability():
    try:
        bedrock_client = boto3.client('bedrock')
        bedrock_client.list_foundation_models()
        logger.info("Bedrock service is available")
    except Exception as e:
        logger.error(f"Bedrock service is not available: {str(e)}")
        raise
