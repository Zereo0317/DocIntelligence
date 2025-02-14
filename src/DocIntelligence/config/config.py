import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Application configuration class to manage environment variables and constants."""

    # Load environment variables
    load_dotenv()

    # Directory Paths
    ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_DIR = ROOT_DIR / "output"

    # PDF Processing
    PDF_IMAGE_DPI = 300  # Resolution for PDF to image conversion

    # YOLO Model Configuration
    YOLO_MODEL_NAME = "juliozhao/DocLayout-YOLO-DocStructBench"
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IMAGE_SIZE = 1024

    # Google Cloud Configuration
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
    GCP_LOCATION = os.getenv('GCP_LOCATION', 'us-central1')
    GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
    GCP_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    GCP_BIGQUERY_INSERT_BATCH = os.getenv('GCP_BIGQUERY_INSERT_BATCH')
    GCP_DATASET_ID = os.getenv('GCP_DATASET_ID')
    GCP_CONNECTION_ID = os.getenv('GCP_CONNECTION_ID')

    @classmethod
    def validate(cls):
        """Ensure required environment variables are set."""
        required_vars = ["GCP_PROJECT_ID", "GCP_BUCKET_NAME"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required Google Cloud environment variables: {', '.join(missing_vars)}")

# Validate configuration on module import
Config.validate()