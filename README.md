# DocIntelligence

[![Python Version](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`DocIntelligence` is a powerful document processing engine that transforms PDFs into structured, searchable knowledge bases. It combines advanced layout analysis with intelligent content extraction to make your documents truly AI-ready for RAG (Retrieval-Augmented Generation) applications.

## Background

In today's AI-driven world, RAG (Retrieval-Augmented Generation) has become essential for creating AI applications that can access and reason about specific document collections. While many excellent open-source PDF processors rely solely on OCR (Optical Character Recognition), they often struggle with:

- Complex document layouts
- Mixed content types (text, tables, images, formulas)
- Maintaining document structure and hierarchy
- Generating high-quality embeddings for semantic search

DocIntelligence addresses these challenges by combining advanced layout analysis with intelligent content extraction, making your documents truly AI-ready.

## Key Features

| Feature | Description | Benefit |
|---------|-------------|----------|
| 🔍 Smart Layout Analysis | Automatically detects and preserves document structure | Maintains context and relationships between elements |
| 📊 Multi-Modal Extraction | Handles text, tables, images, and formulas | Complete document understanding |
| 🧠 Semantic Embeddings | Generates vector embeddings for all content | Enables powerful similarity search |
| ☁️ Flexible Storage | Store locally or in Google Cloud (BigQuery/Storage) | Scales with your needs |
| 🔗 Knowledge Base Ready | Structured output perfect for RAG applications | Build smarter AI applications |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zereo0317/DocIntelligence.git
cd DocIntelligence
```

2. Install dependencies:
```bash
pip install -e .  # Install DocIntelligence in development mode
```

## Google Cloud Setup

`DocIntelligence` uses Google Cloud Platform (GCP) for OCR and optionally for enhanced storage features. You can set up your environment using tools like [`python-dotenv`](https://pypi.org/project/python-dotenv/) or [`direnv`](https://direnv.net/).

1. Set up GCP Authentication:
   - Create a [Google Cloud Project](https://console.cloud.google.com/)
   - Enable the [Cloud Vision API](https://console.cloud.google.com/apis/library/vision.googleapis.com)
   - Create a [Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts) with these roles:
     - `Cloud Vision API User`
     - `Storage Object Viewer` (if using Cloud Storage)
     - `BigQuery Data Editor` (if using BigQuery)
   - Download the JSON key file

2. Create a `.env` file in your project root:
```bash
# Required: Google Cloud Vision API for OCR
GCP_PROJECT_ID=""              # Your GCP Project ID
GCP_LOCATION=""               # Service location (defaults to us-central1)

# Optional: Cloud Storage for visual & tabular content
GCP_BUCKET_NAME=""            # Cloud Storage bucket name
GOOGLE_APPLICATION_CREDENTIALS=""   # Path to service account JSON file

# Optional: BigQuery for embeddings & search
GCP_DATASET_ID=""            # BigQuery dataset name
GCP_CONNECTION_ID=""         # BigQuery connection ID
GCP_BIGQUERY_INSERT_BATCH=500      # Batch size for insertions
```

3. Load environment variables:
```python
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file
```

## Usage

DocIntelligence offers two main processing modes:

### Local Processing
Perfect for smaller projects or testing. Stores results locally:

```python
from DocIntelligence import DocIntelligence

engine = DocIntelligence()

# Process documents and get results directly
elements, documents, embeddings = engine.process_documents(
    input_dir="./Documents/",
    output_dir="./Output/",
    store_to_db=False,    # Store locally
    cloud_storage=False   # Don't use GCP
)
```

### Cloud Processing
Ideal for large-scale applications. Stores results in GCP:

```python
engine = DocIntelligence()

# Process and store in GCP
engine.process_documents(
    input_dir="./Documents/",
    output_dir="./Output/",
    store_to_db=True,     # Store in BigQuery
    cloud_storage=True    # Use Cloud Storage
)
```

### Understanding Storage Options

- `store_to_db=False`: Results are stored in the local output directory
  - Extracted text stored as JSON
  - Images and tables saved as files
  - Embeddings returned as Python objects
  
- `store_to_db=True`: Results are stored in BigQuery
  - Structured data stored in tables
  - Enables fast querying and search
  - Perfect for production deployments

- `cloud_storage=False`: Visual elements stored locally
- `cloud_storage=True`: Visual elements stored in Google Cloud Storage
  - Better for sharing and scalability
  - Enables cloud-based processing

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

## Output Structure

```
Output/
└── {document_name}/
    ├── images/         # Page images
    ├── labeled/        # Extracted elements
    │   ├── text/      # Text blocks
    │   ├── table/     # Table images
    │   ├── formula/   # Mathematical formulas
    │   └── picture/   # Figures and diagrams
    └── ocr/           # Processed content
        ├── text/      # OCR results
        ├── table/     # Table data
        └── caption/   # Image captions
```

## Contributing

We welcome contributions! Here's how you can help:

1. Check our [Issues](https://github.com/Zereo0317/DocIntelligence/issues) for tasks
2. Fork the repository and create a new branch
3. Submit a Pull Request with your changes
4. Join our community discussions

## Future Roadmap

We're actively developing new features to make `DocIntelligence` even more powerful:

1. **Knowledge Graph Integration**
   - Automatically build knowledge graphs from documents
   - Discover relationships between concepts
   - Enable graph-based querying

2. **AgentBasis Integration**
   - Seamless connection to our upcoming AgentBasis framework
   - Build intelligent agents that can reason over your documents
   - Create powerful RAG applications with minimal code

## Requirements

- Python ≥ 3.10
- Google Cloud Vision API (for OCR)
- Optional: Google Cloud Storage and BigQuery (for cloud features)

## License

`DocIntelligence` is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.
