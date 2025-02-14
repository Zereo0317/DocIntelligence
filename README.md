# DocIntelligence

A Python package that extracts, analyzes and indexes content from PDF documents. It automatically detects page layouts, extracts text/tables/images, and generates searchable embeddings - making documents ready for RAG applications.

## What It Does

DocIntelligence breaks down PDF documents into structured, searchable components:

- Detects and extracts text, tables, images, formulas, and captions
- Preserves document structure (chapters, sections, hierarchies) 
- Generates vector embeddings for similarity search
- Stores processed content locally or in cloud (GCP)

## How to Use

1. Clone this Repo:
```bash
git clone https://github.com/Zereo0317/DocIntelligence.git

cd ./DocIntelligence
```

2. Install this package:
```bash
pip install -e .
```

3. Set environment variables for Google Cloud Platform:
```bash
# Google Cloud Platform (required for OCR)
GCP_PROJECT_ID=""              # Project ID
GCP_LOCATION=""               # Service location

# Cloud Storage (Optional for storing visual & tabular context)
GCP_BUCKET_NAME=""            # Cloud Storage Bucket for storing PDFs

# BigQuery (Optional for embedding generation & storing)
GCP_DATASET_ID=""            # BigQuery Dataset
GCP_CONNECTION_ID=""         # BigQuery Model Connection
GOOGLE_APPLICATION_CREDENTIALS=""   # Path to the JSON file of the service account
GCP_BIGQUERY_INSERT_BATCH=500      # Number of rows to insert in a single batch
```

4. Process documents:
```python
from DocIntelligence import DocIntelligence

# Initialize engine
engine = DocIntelligence(use_gpu_yolo=False)

# Local processing - returns extracted elements, documents and embeddings
elements, documents, embeddings = engine.process_documents(
    input_dir="./Documents/",
    output_dir="./Output/"
)

# Cloud processing - stores results in GCP and returns None
result = engine.process_documents(
    input_dir="./Documents/",
    output_dir="./Output/",
    store_to_db=True,
    cloud_storage=True
)
```

* When `store_to_db=False`, extracted contents (including pictures and tables) will be stored in the output directory (defaults to "./output/"). 
* When `cloud_storage=False`, elements will not be stored in BigQuery.

## Return Values Structure

When `store_to_db=False`, the function returns three lists of dictionaries:

### elements
List of detected elements from the document:
```python
{
    'doc_id': str,              # Document ID (Required)
    'page_num': int,            # Page number (Required)
    'element_type': str,        # Element type (Text, Table, Picture, etc.)
    'element_id': str,          # Unique identifier
    'storage_path': str,        # Storage path for the element
    'embedding_id': str,        # Embedding identifier
    'content': str,             # Element content
    'metadata': dict,           # Element metadata in JSON format
    'mapped_to_element_id': str,# Related element ID if applicable
    'store_in_bigquery': bool,  # Whether to store in BigQuery
    'section': str,             # Section information
    'title': str               # Title information
}
```

### documents
List of document metadata:
```python
{
    'doc_id': str,          # Document ID (Required)
    'title': str,           # Document title
    'total_pages': int,     # Total number of pages
    'storage_path': str     # Storage path
}
```

### embeddings
List of vector embeddings:
```python
{
    'embedding_id': str,           # Embedding identifier
    'vector': "" | List[float]     # Empty string when store_to_db=False, 
                                  # List of floats when store_to_db=True
    'original_text': str,          # Original text content
    'content_type': str,           # Content type
    'doc_id': str,                 # Document ID
    'page_num': int,               # Page number
    'element_id': str,             # Element identifier
    'mapped_to_element_id': str,   # Related element ID
    'coordinates': dict            # Position coordinates in JSON
}
```

## Output Directory Structure

```
Output/
└── {document_name}/
    ├── images/         # Extracted page images
    ├── labeled/        # Detected elements
    │   ├── text/
    │   ├── table/
    │   ├── formula/
    │   └── picture/
    └── ocr/           # Processed content
        ├── text/
        ├── table/
        └── caption/
```

## Requirements

- Python ≥ 3.10
- Google Cloud Vision API (required for OCR)
- Optional GCP services for cloud storage and search:
  - Cloud Storage 
  - BigQuery
  - Service account with appropriate permissions