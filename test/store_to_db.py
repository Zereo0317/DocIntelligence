import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

from DocIntelligence import DocIntelligence
# from DocIntelligence.config import Config

## Check the environment variables
# Config.print_all_env_vars()

engine = DocIntelligence(use_gpu_yolo=False)

# The results is None
results = engine.process_documents(
    input_dir="/Users/charless/Desktop/DocIntelligence/test/Documents/",
    output_dir="/Users/charless/Desktop/DocIntelligence/test/Output/",
    store_to_db=True,
    cloud_storage=True,
)