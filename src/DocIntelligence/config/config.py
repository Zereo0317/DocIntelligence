import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = ROOT_DIR / "output"
DOCUMENTS_DIR = ROOT_DIR / "Documents"

PDF_IMAGE_DPI = 300  # Resolution for PDF to image conversion

YOLO_MODEL_NAME = "juliozhao/DocLayout-YOLO-DocStructBench"
YOLO_CONF_THRESHOLD = 0.25
YOLO_IMAGE_SIZE = 1024

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
GCP_LOCATION = os.getenv('GCP_LOCATION', 'us-central1')
GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
GCP_ANTHROPIC_ENDPOINT_LOCATION = os.getenv('GCP_ANTHROPIC_ENDPOINT_LOCATION', 'us-east5')
GCP_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
GCP_BIGQUERY_INSERT_BATCH = os.getenv('GCP_BIGQUERY_INSERT_BATCH')
GCP_DATASET_ID = os.getenv('GCP_DATASET_ID')
GCP_CONNECTION_ID = os.getenv('GCP_CONNECTION_ID')


if not all([GCP_PROJECT_ID, GCP_BUCKET_NAME]):
    raise ValueError("Missing required Google Cloud environment variables. Please check .env file.")


# System constants
MAX_RECURSION = 3           # 降低最大遞迴次數
ACCEPTABLE_RECURSION = 2    # 降低可接受遞迴次數
EARLY_STOP_RECURSION = 1    # 提早停止遞迴次數

# Document quality thresholds
MIN_RELEVANT_DOCS = 1       # 降低最小相關文檔要求
EXCELLENT_DOC_SCORE = 0.8   # 維持優秀文檔分數
GOOD_DOC_SCORE = 0.6       # 維持良好文檔分數

# Generation quality thresholds
EXCELLENT_GEN_SCORE = 0.8   # 維持優秀生成分數
GOOD_GEN_SCORE = 0.6       # 維持良好生成分數
ACCEPTABLE_GEN_SCORE = 0.4  # 維持可接受生成分數


# Document retrived from BigQuery
NUM_DOCUMENTS_RETRIEVED = 25  # 降低從BigQuery檢索的文檔數量