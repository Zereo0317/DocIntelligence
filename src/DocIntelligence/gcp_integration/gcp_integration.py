# src/gcp_integration.py

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import functools
from google.cloud import storage, bigquery, vision
import json
import datetime
import base64
from urllib.parse import quote
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

from DocIntelligence.config import Config

def require_bucket(func):
    """Decorator to ensure self.bucket is not None before calling the function"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.bucket is None:
            raise ValueError("GCP_BUCKET is not provided in the environment, please check the .env file")
        return func(self, *args, **kwargs)
    return wrapper


class GCPIntegration:
    def __init__(self):
        """
        Initialize GCP services integration
        """
        try:
            self.project_id = Config.GCP_PROJECT_ID
            self.bucket_name = Config.GCP_BUCKET_NAME

            # 載入服務帳戶憑證
            if Config.GCP_APPLICATION_CREDENTIALS:
                credentials = service_account.Credentials.from_service_account_file(
                    Config.GCP_APPLICATION_CREDENTIALS
                )
            else:
                # 嘗試從環境變量加載憑證
                credentials = None  # 將使用預設憑證

            # 使用服務帳戶憑證初始化客戶端
            self.storage_client = storage.Client(
                project=self.project_id,
                credentials=credentials
            )
            self.bigquery_client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
            self.vision_client = vision.ImageAnnotatorClient(
                credentials=credentials
            )

            # try to init bucket
            if self.bucket_name:
                self.bucket = self.storage_client.bucket(self.bucket_name)
                if not self.bucket.exists():
                    self.bucket = self.storage_client.create_bucket(self.bucket_name)
            else:
                self.bucket = None

            logger.info("GCP services initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing GCP services: {str(e)}")
            raise

    @require_bucket
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