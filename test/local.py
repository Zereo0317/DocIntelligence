import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

from DocIntelligence import DocIntelligence
# from DocIntelligence.config import Config


engine = DocIntelligence(use_gpu_yolo=False)

## Check the environment variables
# Config.print_all_env_vars()

# # Returns three lists of outputs: elements, documents, and embeddings
elements, documents, embeddings = engine.process_documents(
    input_dir="/Users/charless/Desktop/DocIntelligence/test/Documents/",
    output_dir="/Users/charless/Desktop/DocIntelligence/test/Output/",
)

print("This is the element outputs:")
print(elements)

print("This is the document outputs:")
print(documents)

print("This is the embeddings outputs:")
print(embeddings)