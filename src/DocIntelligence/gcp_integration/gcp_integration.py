# src/gcp_integration.py

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from google.cloud import storage, bigquery, vision
import json
import datetime
import base64
from urllib.parse import quote
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GCPIntegration:
    def __init__(self, project_id: str, bucket_name: str, credentials_path: str = None, credentials_json: str = None):
        """
        Initialize GCP services integration
        """
        try:
            self.project_id = project_id
            self.bucket_name = bucket_name

            # 載入服務帳戶憑證
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
            elif credentials_json:
                import json
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info
                )
            else:
                # 嘗試從環境變量加載憑證
                credentials = None  # 將使用預設憑證

            # 使用服務帳戶憑證初始化客戶端
            self.storage_client = storage.Client(
                project=project_id,
                credentials=credentials
            )
            self.bigquery_client = bigquery.Client(
                project=project_id,
                credentials=credentials
            )
            self.vision_client = vision.ImageAnnotatorClient(
                credentials=credentials
            )

            # try to init bucket
            self.bucket = self.storage_client.bucket(bucket_name)
            if not self.bucket.exists():
                self.bucket = self.storage_client.create_bucket(bucket_name)

            logger.info("GCP services initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing GCP services: {str(e)}")
            raise

    def download_and_encode_image(self, gcs_uri: str) -> Optional[str]:
        """Download image from GCS and encode to base64"""
        try:
            # Extract blob name from GCS URI
            if not gcs_uri.startswith('gs://'):
                raise ValueError(f"Invalid GCS URI: {gcs_uri}")
            
            bucket_name = gcs_uri.split('/')[2]
            blob_name = '/'.join(gcs_uri.split('/')[3:])
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_path = temp_file.name
            
            # Download blob
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(temp_path)
            
            # Read and encode
            with open(temp_path, 'rb') as image_file:
                encoded = base64.standard_b64encode(image_file.read()).decode('utf-8')
            
            # Clean up
            import os
            os.unlink(temp_path)
            
            return encoded
            
        except Exception as e:
            logger.error(f"Error downloading/encoding image: {str(e)}")
            return None

    def generate_gcs_url(self, blob_name: str, public: bool = False) -> str:
        try:
            # 生成 GCS URL 時也對 blob_name 進行 quote，以避免空白或特殊字元問題
            encoded_blob_name = quote(blob_name)
            blob = self.bucket.blob(blob_name)
            if public:
                return f"https://storage.googleapis.com/{self.bucket_name}/{encoded_blob_name}"
            else:
                # 簽名URL不需再次quote, blob自行處理
                return blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(hours=1),
                    method="GET"
                )
        except Exception as e:
            logger.error(f"Error generating GCS URL: {str(e)}")
            raise

    def upload_to_storage(self, file_path: Path, destination_blob_name: str) -> str:
        """上傳文件到 Cloud Storage"""
        try:
            # 確保檔案存在
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # 確保目標路徑有效
            destination_blob_name = quote(destination_blob_name)
            
            # 檢查 bucket 是否存在
            if not self.bucket.exists():
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
                
            # 上傳檔案
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(str(file_path))
            
            # 驗證上傳
            if not blob.exists():
                raise RuntimeError(f"Upload failed: {destination_blob_name}")
                
            logger.info(f"File uploaded to GCS: {destination_blob_name}")
            return f"gs://{self.bucket_name}/{destination_blob_name}"
            
        except Exception as e:
            logger.error(f"Error uploading file to Cloud Storage: {str(e)}")
            logger.error(f"File path: {file_path}")
            logger.error(f"Destination: {destination_blob_name}")
            raise


    def store_document_metadata(self, doc_id: str, title: str, total_pages: int, storage_path: str):
        try:
            query = f"""
            INSERT INTO `{self.project_id}.rag_system.documents`
            (doc_id, title, total_pages, processed_date, storage_path)
            VALUES
            (@doc_id, @title, @total_pages, CURRENT_TIMESTAMP(), @storage_path)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id),
                    bigquery.ScalarQueryParameter("title", "STRING", title),
                    bigquery.ScalarQueryParameter("total_pages", "INTEGER", total_pages),
                    bigquery.ScalarQueryParameter("storage_path", "STRING", storage_path)
                ]
            )
            query_job = self.bigquery_client.query(query, job_config=job_config)
            query_job.result()
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            raise

    def store_page_metadata(self, doc_id: str, page_num: int, storage_path: str):
        try:
            query = f"""
            INSERT INTO `{self.project_id}.rag_system.pages`
            (doc_id, page_num, storage_path)
            VALUES
            (@doc_id, @page_num, @storage_path)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id),
                    bigquery.ScalarQueryParameter("page_num", "INTEGER", page_num),
                    bigquery.ScalarQueryParameter("storage_path", "STRING", storage_path)
                ]
            )
            query_job = self.bigquery_client.query(query, job_config=job_config)
            query_job.result()
        except Exception as e:
            logger.error(f"Error storing page metadata: {str(e)}")
            raise

    def store_element_metadata(self, doc_id: str, page_num: int, element_type: str,
                             element_id: str, storage_path: str, embedding_id: str,
                             content: Optional[str], metadata: Dict[str, Any]):
        try:
            # Ensure coordinates are JSON serialized
            if 'coordinates' in metadata and metadata['coordinates'] is not None:
                metadata['coordinates'] = json.dumps(metadata['coordinates'])

            mapped_to_element_id = metadata.get('mapped_to')
            store_in_bigquery = metadata.get('store_in_bigquery', True)
            doc_title = metadata.get('doc_title', '')
            section = metadata.get('section', '')

            query = f"""
            INSERT INTO `{self.project_id}.rag_system.elements`
            (doc_id, page_num, element_type, element_id, storage_path, embedding_id, 
             content, mapped_to_element_id, store_in_bigquery, metadata, title, section)
            VALUES
            (@doc_id, @page_num, @element_type, @element_id, @storage_path, @embedding_id,
             @content, @mapped_to_element_id, @store_in_bigquery, @metadata, @title, @section)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id),
                    bigquery.ScalarQueryParameter("page_num", "INTEGER", page_num),
                    bigquery.ScalarQueryParameter("element_type", "STRING", element_type),
                    bigquery.ScalarQueryParameter("element_id", "STRING", element_id),
                    bigquery.ScalarQueryParameter("storage_path", "STRING", storage_path),
                    bigquery.ScalarQueryParameter("embedding_id", "STRING", embedding_id),
                    bigquery.ScalarQueryParameter("content", "STRING", content),
                    bigquery.ScalarQueryParameter("mapped_to_element_id", "STRING", mapped_to_element_id),
                    bigquery.ScalarQueryParameter("store_in_bigquery", "BOOL", store_in_bigquery),
                    bigquery.ScalarQueryParameter("metadata", "JSON", json.dumps(metadata)),
                    bigquery.ScalarQueryParameter("title", "STRING", doc_title),
                    bigquery.ScalarQueryParameter("section", "STRING", section)
                ]
            )
            query_job = self.bigquery_client.query(query, job_config=job_config)
            query_job.result()
        except Exception as e:
            logger.error(f"Error storing element metadata: {str(e)}")
            raise

    def setup_bigquery_tables(self):
        try:
            dataset_id = f"{self.project_id}.rag_system"
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            dataset = self.bigquery_client.create_dataset(dataset, exists_ok=True)

            element_metadata_schema = [
                bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("page_num", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("element_type", "STRING"),
                bigquery.SchemaField("element_id", "STRING"),
                bigquery.SchemaField("storage_path", "STRING"),
                bigquery.SchemaField("embedding_id", "STRING"),
                bigquery.SchemaField("content", "STRING"),
                bigquery.SchemaField("mapped_to_element_id", "STRING"),
                bigquery.SchemaField("store_in_bigquery", "BOOLEAN"),
                bigquery.SchemaField("metadata", "JSON"),
                bigquery.SchemaField("title", "STRING"),
                bigquery.SchemaField("section", "STRING"),
                bigquery.SchemaField(
                    "created_at", 
                    "TIMESTAMP", 
                    default_value_expression="CURRENT_TIMESTAMP()"
                )
            ]

            tables = {
                "documents": [
                    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("title", "STRING"),
                    bigquery.SchemaField("total_pages", "INTEGER"),
                    bigquery.SchemaField("processed_date", "TIMESTAMP"),
                    bigquery.SchemaField("storage_path", "STRING")
                ],
                "pages": [
                    bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("page_num", "INTEGER", mode="REQUIRED"),
                    bigquery.SchemaField("chapter", "STRING"),
                    bigquery.SchemaField("section", "STRING"),
                    bigquery.SchemaField("storage_path", "STRING")
                ],
                "elements": element_metadata_schema
            }

            for table_name, schema in tables.items():
                table_id = f"{dataset_id}.{table_name}"
                table = bigquery.Table(table_id, schema=schema)
                table = self.bigquery_client.create_table(table, exists_ok=True)
                logger.info(f"Created table {table_id}")

        except Exception as e:
            logger.error(f"Error setting up BigQuery tables: {str(e)}")
            raise

    def create_vector_index(self):
        try:
            query = f"""
            CREATE OR REPLACE VECTOR INDEX text_embeddings_index
            ON `{self.project_id}.rag_system.elements`(embedding_id)
            OPTIONS(
                index_type='IVF',
                distance_type='COSINE'
            );
            """
            query_job = self.bigquery_client.query(query)
            query_job.result()
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            raise

    def generate_resource_urls(self, doc_id: str, page_num: int, element_id: str) -> Dict[str, str]:
        """
        生成資源的訪問 URLs
        
        Args:
            doc_id: 文檔 ID
            page_num: 頁碼
            element_id: 元素 ID
            
        Returns:
            包含各種資源 URL 的字典
        """
        try:
            # 將所有路徑使用相對安全的名稱格式
            safe_doc_id = quote(doc_id)
            element_path = f"documents/{safe_doc_id}/elements/{element_id}"
            document_path = f"documents/{safe_doc_id}/{safe_doc_id}.pdf"
            page_path = f"documents/{safe_doc_id}/pages/page_{page_num}.png"
            
            # 生成帶有一小時過期時間的簽名 URLs
            exp_time = datetime.timedelta(hours=1)
            
            urls = {
                'element': f"https://console.cloud.google.com/bigquery?project={self.project_id}&p={self.project_id}&d=rag_system&t=elements&page=table"
            }
            
            try:
                element_blob = self.bucket.blob(element_path)
                if element_blob.exists():
                    urls['content'] = element_blob.generate_signed_url(
                        expiration=exp_time,
                        version="v4",
                        method="GET"
                    )
                else:
                    urls['content'] = ''
            except Exception as e:
                logger.warning(f"Error generating content URL: {str(e)}")
                urls['content'] = ''
                
            try:
                document_blob = self.bucket.blob(document_path)
                if document_blob.exists():
                    urls['document'] = document_blob.generate_signed_url(
                        expiration=exp_time,
                        version="v4",
                        method="GET"
                    )
                else:
                    urls['document'] = ''
            except Exception as e:
                logger.warning(f"Error generating document URL: {str(e)}")
                urls['document'] = ''
                
            try:
                page_blob = self.bucket.blob(page_path)
                if page_blob.exists():
                    urls['page'] = page_blob.generate_signed_url(
                        expiration=exp_time,
                        version="v4",
                        method="GET"
                    )
                else:
                    urls['page'] = ''
            except Exception as e:
                logger.warning(f"Error generating page URL: {str(e)}")
                urls['page'] = ''
                
            return urls
            
        except Exception as e:
            logger.error(f"Error generating resource URLs: {str(e)}")
            return {
                'content': '',
                'document': '',
                'page': '',
                'element': ''
            }