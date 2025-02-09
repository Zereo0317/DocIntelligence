import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.cloud import aiplatform
from google.cloud import bigquery
import numpy as np
import uuid
import json

from src.config import GCP_BIGQUERY_INSERT_BATCH, GCP_BUCKET_NAME, GCP_DATASET_ID, GCP_CONNECTION_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the embedding processor
        
        Args:
            project_id (str): Google Cloud Project ID
            location (str): GCP region for Vertex AI
        """
        try:
            self.project_id = project_id
            self.location = location
            self.dataset_id = GCP_DATASET_ID  # 資料集名稱

            self.connection_id = GCP_CONNECTION_ID
            self.connection_string = f"{project_id}.{location}.{self.connection_id}"

            # Initialize Vertex AI
            aiplatform.init(
                project=project_id,
                location=location
            )
            
            # Initialize BigQuery client
            self.bq_client = bigquery.Client(project=project_id)
            
            logger.info("Embedding processor initialized successfully")
            self.embedding_buffer = []

        except Exception as e:
            logger.error(f"Error initializing embedding processor: {str(e)}")
            raise

    def _handle_text_embedding(self, content: str, model_name: str) -> List[Any]:
        """
        處理文字內容的 embedding 生成
        """
        query = f"""
        SELECT * FROM ML.GENERATE_EMBEDDING(
            MODEL `{model_name}`,
            (SELECT @content AS content)
        );
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("content", "STRING", content)
            ]
        )
        
        query_job = self.bq_client.query(query, job_config=job_config)
        return list(query_job.result())
        

    def _handle_image_embedding(self, content: str, model_name: str) -> List[Any]:
        """
        處理圖片內容的 embedding 生成
        
        Args:
            content: 圖片的 GCS URI
            model_name: 模型的完整名稱
            
        Returns:
            List[Any]: 包含生成的 embedding 結果的列表
        """
        if not isinstance(content, str) or not content.startswith('gs://'):
            raise ValueError(f"Image content must be a valid GCS URI")
        
        # 創建臨時表格
        temp_table_id = f"{self.project_id}.{self.dataset_id}.temp_uri_table_{str(uuid.uuid4()).replace('-', '_')}"
    
        # 創建表格並插入數據
        create_table_query = f"""
        CREATE OR REPLACE EXTERNAL TABLE `{temp_table_id}`
        WITH CONNECTION `{self.project_id}.US.{self.connection_id}`
        OPTIONS (
            object_metadata='SIMPLE',
            uris = ['{content}']
        );
        """
        
        # 生成 embedding
        embedding_query = f"""
        SELECT * FROM ML.GENERATE_EMBEDDING(
            MODEL `{model_name}`,
            TABLE `{temp_table_id}`
        );
        """
            
        try:
            logger.info(f"Creating external table for image embedding at {temp_table_id} with URI: {content}")
            self.bq_client.query(create_table_query).result()

            logger.info(f"Generating image embedding using model {model_name}")
            results = list(self.bq_client.query(embedding_query).result())

            if not results:
                raise ValueError(f"No embedding generated for image: {content}")

            logger.info(f"Successfully generated {len(results)} embedding result(s) for the image.")
            return results

        except Exception as e:
            logger.error(f"Error during image embedding generation for content: {content}, temp table: {temp_table_id}")
            logger.error("Error details:", exc_info=True)
            raise e

        finally:
            try:
                drop_query = f"DROP TABLE IF EXISTS `{temp_table_id}`"
                self.bq_client.query(drop_query).result()
                logger.info(f"Temporary table {temp_table_id} dropped successfully.")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary table {temp_table_id}: {str(cleanup_error)}")



    def generate_embeddings(self, content: Union[str, Path], content_type: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        為給定內容生成 embeddings。  
        若為文字類型且內容過長，會先進行分塊，對每個塊產生 embedding 並加入緩存 (embedding_buffer)。  
        若為影像類型，則直接產生單一 embedding 並加入緩存。  
        所有 embedding 都不會立即寫入 BigQuery，而是等到 batch_insert_embeddings() 被呼叫時一次性寫入。

        Args:
            content (Union[str, Path]): 要產生 embedding 的內容 (文字或圖片 GCS URI)。
            content_type (str): 內容的類型，如 'Text', 'Table', 'Picture' 等。
            metadata (Optional[Dict[str, Any]]): 此 embedding 關聯的來源資訊 (如 doc_id, page_num, element_id...)。

        Returns:
            List[Dict[str, Any]] or Dict[str, Any] or None:
            - 若分塊後的文字內容，會返回一個 embedding dict list。
            - 若為單一 embedding (短文本或圖片)，回傳單一 embedding dict。
            - 若 store_in_bigquery 為 False，則返回 None。
        """
        try:
            logger.info(f"Generating embeddings for content_type={content_type}")
            
            if metadata and not metadata.get('store_in_bigquery', True):
                logger.info(f"Skipping embedding generation for {content_type}")
                return None

            model_name = f"{self.project_id}.{self.dataset_id}.multimodal_embedding_model"
            
            # 對於文本內容，處理可能的分塊
            if content_type in ['Text', 'Caption', 'Title', 'Footnote', 'Table', "Formula"]:
                
                # 如果內容太長，進行分塊
                if isinstance(content, str) and len(content) > 900:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=900,
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_text(content)
                    
                    # 為每個 chunk 生成 embedding
                    embeddings = []
                    for i, chunk in enumerate(chunks):
                        # 如果有 metadata，為每個 chunk 創建新的 metadata
                        chunk_metadata = metadata.copy() if metadata else {}
                        if chunk_metadata:
                            chunk_metadata['chunk_index'] = i
                        
                        result = self._handle_text_embedding(chunk, model_name)
                        if not result:
                            continue
                            
                        # 正常來說 result[0].ml_generate_embedding_result 應包含 embedding vector
                        embedding_vector = result[0].ml_generate_embedding_result
                        embedding_id = str(uuid.uuid4())

                        # 將 embedding 資料存入 buffer 而非立即寫入 BigQuery
                        self.embedding_buffer.append({
                            "embedding_id": embedding_id,
                            "vector": embedding_vector,
                            "content_type": content_type,
                            "doc_id": chunk_metadata.get('doc_id'),
                            "page_num": chunk_metadata.get('page_num'),
                            "element_id": chunk_metadata.get('element_id'),
                            "mapped_to_element_id": chunk_metadata.get('mapped_to'),
                            "coordinates": chunk_metadata.get('coordinates')
                        })

                        embeddings.append({
                            'embedding_id': embedding_id,
                            'vector': embedding_vector,
                            'chunk': chunk
                        })

                    return embeddings
                    
                # 如果內容夠短，直接生成單個 embedding
                results = self._handle_text_embedding(str(content), model_name)
                
            else:
                # 處理圖片內容
                results = self._handle_image_embedding(str(content), model_name)
            
            if not results:
                raise ValueError("No embedding generated")
                
            embedding_vector = results[0].ml_generate_embedding_result
            embedding_id = str(uuid.uuid4())
            
            self.embedding_buffer.append({
                "embedding_id": embedding_id,
                "vector": embedding_vector,
                "content_type": content_type,
                "doc_id": metadata.get('doc_id') if metadata else None,
                "page_num": metadata.get('page_num') if metadata else None,
                "element_id": metadata.get('element_id') if metadata else None,
                "mapped_to_element_id": metadata.get('mapped_to') if metadata else None,
                "coordinates": metadata.get('coordinates') if metadata else None
            })

            return {
                'embedding_id': embedding_id,
                'vector': embedding_vector
            }
                
        except Exception as e:
            logger.error("Error generating embeddings:")
            logger.error(f"Content type: {content_type}")
            logger.error(f"Content: {content}")
            logger.error(f"Metadata: {metadata}")
            logger.error("Error details:", exc_info=True)
            raise


    # def create_multimodal_model(self) -> str:
    #     """
    #     Create or get multimodal embedding model
        
    #     Returns:
    #         str: Model resource name
    #     """
    #     try:
    #         model_id = f"{self.project_id}.{self.dataset_id}.multimodal_embedding_model"
            
    #         query = f"""
    #         CREATE OR REPLACE MODEL `{model_id}`
    #         REMOTE WITH CONNECTION `{self.location}.plasma_line_remote_model`
    #         OPTIONS (endpoint = 'multimodalembedding@001');
    #         """
            
    #         query_job = self.bq_client.query(query)
    #         query_job.result()
            
    #         return model_id
            
    #     except Exception as e:
    #         logger.error(f"Error creating multimodal model: {str(e)}")
    #         raise


    # 較彈性，先上傳到 Cloud Storage，再從 Storage 上傳到 BigQuery
    def batch_insert_embeddings_via_load_job(self):
        """
        使用 Load Job 將 self.embedding_buffer 中的所有 embedding 一次性寫入 BigQuery。
        要點：
        - 單次 load job，避免多次 load job 操作導致超過每日 1000 次上限。
        - max_bad_records = 0，確保無 error bit，如有錯誤立即整體失敗。
        - load_job.result() 會阻塞至完成，確保我們可確定 load job 結果。
        """

        if not self.embedding_buffer:
            logger.info("No embeddings to insert, buffer is empty.")
            return

        # 建立 embeddings 表 (若不存在)
        table_id = f"{self.project_id}.rag_system.embeddings"
        schema = [
            bigquery.SchemaField("embedding_id", "STRING"),
            bigquery.SchemaField("vector", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("content_type", "STRING"),
            bigquery.SchemaField("doc_id", "STRING"),
            bigquery.SchemaField("page_num", "INTEGER"),
            bigquery.SchemaField("element_id", "STRING"),
            bigquery.SchemaField("mapped_to_element_id", "STRING"),
            bigquery.SchemaField("coordinates", "JSON"),
            bigquery.SchemaField(
                "created_at", 
                "TIMESTAMP", 
                default_value_expression="CURRENT_TIMESTAMP()"
            )
        ]

        table = bigquery.Table(table_id, schema=schema)
        self.bq_client.create_table(table, exists_ok=True)

        # 將 embedding_buffer 資料轉為 NDJSON
        ndjson_lines = []
        for emb in self.embedding_buffer:
            record = {
                "embedding_id": emb["embedding_id"],
                "vector": emb["vector"],
                "content_type": emb["content_type"],
                "doc_id": emb["doc_id"],
                "page_num": emb["page_num"],
                "element_id": emb["element_id"],
                "mapped_to_element_id": emb["mapped_to_element_id"],
                "coordinates": emb["coordinates"],
            }
            ndjson_lines.append(json.dumps(record))

        ndjson_data = "\n".join(ndjson_lines)

        # 上傳至 GCS
        from google.cloud import storage
        import uuid
        blob_name = f"temp_embeddings_{uuid.uuid4()}.json"
        bucket_name = GCP_BUCKET_NAME
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # 上傳 NDJSON 至 GCS
        blob.upload_from_string(ndjson_data, content_type="application/json")
        gcs_uri = f"gs://{bucket_name}/{blob_name}"

        # 建立 Load Job Config
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        )

        # 執行 load job
        load_job = self.bq_client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
        logger.info("Starting load job, waiting for it to complete...")
        load_job.result()  # 等待 load job 完成

        if load_job.errors:
            # 如有 errors，代表 load job 整體失敗，不會有 partial success
            # 這裡可以紀錄或 raise exception
            logger.error(f"Load job failed with errors: {load_job.errors}")
            # 選擇是否要 raise，以便外層捕捉
            raise RuntimeError("Load job failed.")
        else:
            logger.info("Load job completed successfully. All embeddings inserted.")

        # 清空 buffer
        self.embedding_buffer.clear()
        # 若需要，刪除 GCS 暫存檔案
        # blob.delete()
        # logger.info("Temporary file removed from GCS.")


    # # 上傳拆成總共 500 次插入行為 (還是 < 1000 次)
    # def batch_insert_embeddings(self):
    #     """
    #     一次性將 self.embedding_buffer 中的所有 embedding 寫入 BigQuery。
    #     藉由一次性 INSERT 多筆資料的方式，大幅減少對 table 的操作次數。
    #     """
    #     if not self.embedding_buffer:
    #         logger.info("No embeddings to insert, buffer is empty.")
    #         return

    #     # 建立 embeddings 表 (若不存在)
    #     create_table_query = f"""
    #     CREATE TABLE IF NOT EXISTS `{self.project_id}.rag_system.embeddings`
    #     (
    #         embedding_id STRING,
    #         vector ARRAY<FLOAT64>,
    #         content_type STRING,
    #         doc_id STRING,
    #         page_num INTEGER,
    #         element_id STRING,
    #         mapped_to_element_id STRING,
    #         coordinates JSON,
    #         created_at TIMESTAMP
    #     );
    #     """
    #     self.bq_client.query(create_table_query).result()

    #     # 將 buffer 中的資料轉換成可插入的參數
    #     rows = []
    #     for emb in self.embedding_buffer:
    #         rows.append((
    #             emb["embedding_id"],
    #             emb["vector"],
    #             emb["content_type"],
    #             emb["doc_id"],
    #             emb["page_num"],
    #             emb["element_id"],
    #             emb["mapped_to_element_id"],
    #             json.dumps(emb["coordinates"]) if emb["coordinates"] else None
    #         ))

    #     total_rows = len(rows)
    #     logger.info(f"Total rows to insert: {total_rows}")
    #     batch_size = int(GCP_BIGQUERY_INSERT_BATCH)

    #     # 分批處理 rows
    #     for batch_start in range(0, total_rows, batch_size):
    #         batch_rows = rows[batch_start: batch_start + batch_size]

    #         parameters = []
    #         row_placeholders = []
    #         for i, row in enumerate(batch_rows):
    #             idx = batch_start + i
    #             parameters.extend([
    #                 bigquery.ScalarQueryParameter(f"embedding_id_{idx}", "STRING", row[0]),
    #                 bigquery.ArrayQueryParameter(f"vector_{idx}", "FLOAT64", row[1]),
    #                 bigquery.ScalarQueryParameter(f"content_type_{idx}", "STRING", row[2]),
    #                 bigquery.ScalarQueryParameter(f"doc_id_{idx}", "STRING", row[3]),
    #                 bigquery.ScalarQueryParameter(f"page_num_{idx}", "INTEGER", row[4]),
    #                 bigquery.ScalarQueryParameter(f"element_id_{idx}", "STRING", row[5]),
    #                 bigquery.ScalarQueryParameter(f"mapped_to_element_id_{idx}", "STRING", row[6]),
    #                 bigquery.ScalarQueryParameter(f"coordinates_{idx}", "JSON", row[7])
    #             ])
    #             row_placeholders.append(
    #                 f"(@embedding_id_{idx}, @vector_{idx}, @content_type_{idx}, @doc_id_{idx}, @page_num_{idx}, "
    #                 f"@element_id_{idx}, @mapped_to_element_id_{idx}, @coordinates_{idx}, CURRENT_TIMESTAMP())"
    #             )

    #         insert_query = f"""
    #         INSERT INTO `{self.project_id}.rag_system.embeddings`
    #         (embedding_id, vector, content_type, doc_id, page_num, element_id, mapped_to_element_id, coordinates, created_at)
    #         VALUES {", ".join(row_placeholders)}
    #         """

    #         job_config = bigquery.QueryJobConfig(query_parameters=parameters)
    #         logger.info(f"Inserting batch from row {batch_start} to {batch_start + len(batch_rows) - 1}, size: {len(batch_rows)}")
    #         self.bq_client.query(insert_query, job_config=job_config).result()

    #     # 清空 buffer
    #     self.embedding_buffer.clear()
    #     logger.info("All batch insert of embeddings completed.")



    def _store_embedding(self, embedding_id: str, vector: List[float], 
                        content_type: str, source_info: Optional[Dict[str, Any]] = None):
        """
        Store embedding vector in BigQuery with enhanced source tracking
        
        Args:
            embedding_id: Unique identifier for the embedding
            vector: Embedding vector
            content_type: Type of content ('text', 'image', 'caption')
            source_info: Source information including page, coordinates, etc.
        """
        try:
            # Create embeddings table if it doesn't exist
            query = f"""
            CREATE TABLE IF NOT EXISTS `{self.project_id}.rag_system.embeddings`
            (
                embedding_id STRING,
                vector ARRAY<FLOAT64>,
                content_type STRING,
                doc_id STRING,
                page_num INTEGER,
                element_id STRING,
                mapped_to_element_id STRING,
                coordinates JSON,
                created_at TIMESTAMP
            );
            """
            
            self.bq_client.query(query).result()
            
            # Extract source information
            doc_id = source_info.get('doc_id') if source_info else None
            page_num = source_info.get('page_num') if source_info else None
            element_id = source_info.get('element_id') if source_info else None
            mapped_to = source_info.get('mapped_to') if source_info else None
            coordinates = source_info.get('coordinates') if source_info else None
            
            # Insert embedding with source information
            query = f"""
            INSERT INTO `{self.project_id}.rag_system.embeddings`
            (embedding_id, vector, content_type, doc_id, page_num, element_id, 
             mapped_to_element_id, coordinates, created_at)
            VALUES
            (@embedding_id, @vector, @content_type, @doc_id, @page_num, @element_id,
             @mapped_to_element_id, @coordinates, CURRENT_TIMESTAMP())
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("embedding_id", "STRING", embedding_id),
                    bigquery.ArrayQueryParameter("vector", "FLOAT64", vector),
                    bigquery.ScalarQueryParameter("content_type", "STRING", content_type),
                    bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id),
                    bigquery.ScalarQueryParameter("page_num", "INTEGER", page_num),
                    bigquery.ScalarQueryParameter("element_id", "STRING", element_id),
                    bigquery.ScalarQueryParameter("mapped_to_element_id", "STRING", mapped_to),
                    bigquery.ScalarQueryParameter("coordinates", "JSON", 
                                                json.dumps(coordinates) if coordinates else None)
                ]
            )
            
            self.bq_client.query(query, job_config=job_config).result()
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise

    def search_similar(self, query_embedding: List[float], top_k: int = 10, similarity_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        搜索相似的文檔並返回標準格式的結果
        
        Args:
            query_embedding: 查詢向量
            top_k: 返回結果數量
            similarity_threshold: 相似度閾值（0-1之間），低於此值的結果將被過濾
                
        Returns:
            List[Dict[str, Any]]: 標準格式的文檔列表，每個文檔包含 content 和 metadata
        """
        try:
            query_embedding_str = ",".join(map(str, query_embedding))

            # 修改查詢以包含相似度計算和過濾
            query = f"""
            WITH 
            query_embedding AS (
                SELECT ARRAY<FLOAT64>[{query_embedding_str}] AS vector
            ),
            vector_similarity AS (
                SELECT 
                    e.embedding_id,
                    e.content_type,
                    e.doc_id,
                    e.page_num,
                    e.element_id,
                    e.mapped_to_element_id,
                    TO_JSON_STRING(e.coordinates) as coordinates,  -- Convert to JSON string
                    elem.content,
                    elem.metadata as additional_metadata,
                    elem.title,
                    elem.section,
                    -- 計算餘弦相似度
                    (
                        SELECT 
                            SUM(e_v * q_v) / (
                                SQRT(SUM(POW(e_v, 2))) * SQRT(SUM(POW(q_v, 2)))
                            )
                        FROM UNNEST(e.vector) AS e_v WITH OFFSET e_idx
                        JOIN UNNEST(q.vector) AS q_v WITH OFFSET q_idx
                        ON e_idx = q_idx
                    ) AS similarity_score
                FROM 
                    `{self.project_id}.rag_system.embeddings` e
                JOIN 
                    `{self.project_id}.rag_system.elements` elem
                ON 
                    e.element_id = elem.element_id
                CROSS JOIN 
                    query_embedding q
                WHERE
                    elem.store_in_bigquery = TRUE
                    AND elem.element_type != 'Title'        -- Avoid `Title` elements
                    AND NOT (e.content_type = "Caption" AND e.mapped_to_element_id IS NULL)  -- 跳過 Caption 且 mapped_to_element_id 為 NULL
            )
            SELECT *
            FROM vector_similarity
            WHERE similarity_score >= {similarity_threshold}
            ORDER BY similarity_score DESC
            LIMIT {top_k}
            """

            results = []
            query_job = self.bq_client.query(query)

            for row in query_job:
                # Parse coordinates back from JSON string if it exists
                coordinates = json.loads(row.coordinates) if row.coordinates else None

                # 構建標準格式的文檔對象
                standardized_metadata = {
                    'doc_id': row.doc_id,
                    'page_num': row.page_num,
                    'element_id': row.element_id,
                    'element_type': row.content_type,
                    'mapped_to': row.mapped_to_element_id,
                    'coordinates': coordinates,
                    'title': row.title if hasattr(row, 'title') else None,
                    'section': row.section if hasattr(row, 'section') else None,
                    'similarity_score': float(row.similarity_score)
                }
                
                # 合併額外的元數據
                if row.additional_metadata:
                    try:
                        extra_metadata = (
                            json.loads(row.additional_metadata) 
                            if isinstance(row.additional_metadata, str) 
                            else row.additional_metadata
                        )
                        # 確保 title 和 section 不被覆蓋
                        title = standardized_metadata['title']
                        section = standardized_metadata['section']
                        standardized_metadata.update(extra_metadata)
                        if title is not None:
                            standardized_metadata['title'] = title
                        if section is not None:
                            standardized_metadata['section'] = section
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not parse additional metadata for document {row.doc_id}"
                        )


                # DEBUG MESSAGE
                logger.info("Queried content:")
                logger.info(row.content)

                logger.info("Standardized metadata:")
                logger.info(standardized_metadata)



                # 構建最終的文檔對象
                document = {
                    'content': row.content,
                    'metadata': standardized_metadata
                }

                # 如果有映射的視覺元素，添加相關信息
                if row.mapped_to_element_id:
                    visual_info = self._get_mapped_visual_info(row.mapped_to_element_id)
                    if visual_info:
                        document['mapped_element'] = visual_info

                results.append(document)

            logger.info(f"Found {len(results)} similar documents with similarity threshold {similarity_threshold}")
            
            # 記錄詳細的相似度信息
            if results:
                similarity_scores = [doc['metadata']['similarity_score'] for doc in results]
                logger.info(f"Similarity scores range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")
                
            return results

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            raise

    def _get_mapped_visual_info(self, element_id: str) -> Optional[Dict[str, Any]]:
        """獲取映射的視覺元素信息"""
        try:
            query = f"""
            SELECT content, metadata
            FROM `{self.project_id}.rag_system.elements`
            WHERE element_id = @element_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("element_id", "STRING", element_id)
                ]
            )
            
            results = list(self.bq_client.query(query, job_config=job_config))
            if results:
                row = results[0]
                return {
                    'content': row.content,
                    'metadata': json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                }
            return None
            
        except Exception as e:
            logger.warning(f"Error getting mapped visual info: {str(e)}")
            return None
