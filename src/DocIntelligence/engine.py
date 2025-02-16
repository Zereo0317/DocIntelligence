import argparse
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Union
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from DocIntelligence.pdf_process import PDFProcessor, OCRProcessor, LayoutAnalyzer
from DocIntelligence.gcp_integration import GCPIntegration
from DocIntelligence.embeddings import EmbeddingProcessor
from DocIntelligence.config import Config


class DocIntelligence:
    def __init__(self, use_gpu_yolo: bool = False):
        """Initialize the RAG system components"""
        try:
            self.start_time = time.time()
            self.processed_documents_count = 0
            self.failed_stage = None
            self.unprocessed_documents = []

            self.elements = []
            self.documents = []
            self.embeddings = []

            load_dotenv()
            self.pdf_processor = PDFProcessor()
            self.layout_analyzer = LayoutAnalyzer(use_gpu=use_gpu_yolo)
            self.ocr_processor = OCRProcessor()
            self.gcp_integration = GCPIntegration()
            logger.info("DocIntelligence system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DocIntelligence system: {str(e)}")
            raise

    def process_documents(self, input_dir: str, output_dir: str = None, 
                          chunk_size: int = 900, overlap: int = 100,
                          store_to_db: bool = False, cloud_storage: bool = False):
        """
        Process all PDF documents in the input directory.

        Args:
            input_dir (str): Path to the directory containing PDF documents.
            output_dir (str, optional): Directory for output files. Defaults to "Config.ROOTDIR / output" .
            chunk_size (int): Maximum chunk size for text splitting.
            overlap (int): Overlap between chunks.
            store_to_db (bool): Whether to store results to DB.
            cloud_storage (bool): Whether to store files to cloud storage.
        """
        # 使用傳入的 input_dir 與 output_dir；若 output_dir 為 None 則使用 Config.OUTPUT_DIR
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve() if output_dir else Config.ROOT_DIR / "output"
        
        logger.info(f"Input directory: {input_path}")
        logger.info(f"Output directory: {output_path}")
        
        pdf_files = list(input_path.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pdf = {
                executor.submit(self.process_single_document, pdf_path, output_path, chunk_size, overlap, store_to_db, cloud_storage): pdf_path
                for pdf_path in pdf_files
            }
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    # Update counts and accumulate results if not storing to DB
                    if not store_to_db and result:
                        self.elements.extend(result.get('elements', []))
                        self.documents.append(result.get('document'))
                        self.embeddings.extend(result.get('embeddings', []))
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {str(e)}")
                    self.unprocessed_documents.append(str(pdf_path))
                    continue
        logger.info("Document processing completed")
        
        if not store_to_db:
            return self.elements, self.documents, self.embeddings
        else:
            return None

    def process_single_document(self, pdf_path: Path, output_dir: Path, chunk_size: int = 900, overlap: int = 100,
                                store_to_db: bool = False, cloud_storage: bool = False) -> Union[Dict[str, Any], None]:
        """
        Process a single PDF document:
        - Convert PDF pages to images.
        - Perform layout analysis on the images.
        - Run OCR on the analyzed elements.
        - Generate embeddings for each element (e.g., Text, Table) using the provided chunk_size and overlap.
        - If store_to_db is True and cloud_storage is True, upload the original PDF and associated assets to GCP;
            otherwise, use local file paths.
        - If store_to_db is False, accumulate results locally (elements, document metadata, and embeddings)
            without calling any GCP store functions.

        Args:
            pdf_path (Path): Path to the PDF file.
            output_dir (Path): Base directory to store output files.
            chunk_size (int): Maximum chunk size for text splitting.
            overlap (int): Overlap between chunks.
            store_to_db (bool): Whether to generate and upload embeddings to the database.
            cloud_storage (bool): Whether to store files to GCP cloud storage.

        Returns:
            dict or None: If store_to_db is False, returns a dict containing:
                        - 'elements': Processed OCR elements.
                        - 'document': Document metadata (doc_id, title, total_pages, storage_path).
                        - 'embeddings': List of embedding results.
                        Otherwise, returns None.
        """
        try:
            logger.info(f"Processing document: {pdf_path}")
            pdf_name = pdf_path.stem
            pdf_output_dir = output_dir / pdf_name
            pdf_images_dir = pdf_output_dir / "images"
            pdf_labeled_dir = pdf_output_dir / "labeled"
            pdf_ocr_dir = pdf_output_dir / "ocr"

            # Create necessary output directories
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_images_dir.mkdir(exist_ok=True)
            pdf_labeled_dir.mkdir(exist_ok=True)
            pdf_ocr_dir.mkdir(exist_ok=True)

            # Convert PDF pages to images
            images = self.pdf_processor.convert_pdf_to_images(pdf_path, pdf_images_dir)
            logger.info(f"Converted PDF to {len(images)} images")

            # Perform layout analysis on the images (debug mode enabled)
            labeled_elements = self.layout_analyzer.analyze_layouts(images, pdf_labeled_dir, debug=True)
            logger.info("Layout analysis completed")

            # Run OCR processing using multithreading
            ordered_results = [None] * len(labeled_elements)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_pageidx = {}
                for page_idx, page_data in enumerate(labeled_elements, start=1):
                    future = executor.submit(self.ocr_processor.process_elements, [page_data], pdf_ocr_dir)
                    future_to_pageidx[future] = page_idx

                for future in concurrent.futures.as_completed(future_to_pageidx):
                    page_idx = future_to_pageidx[future]
                    try:
                        result = future.result()  # result is a list; one dict per page
                        ordered_results[page_idx - 1] = result
                    except Exception as e:
                        logger.error(f"OCR error on page {page_idx}: {str(e)}")

            # Merge OCR results from all pages into a single list
            processed_elements = []
            for res in ordered_results:
                if res is not None:
                    processed_elements.extend(res)
            logger.info("OCR processing completed")

            # ----------------------------------------------------------------
            # (A) Determine PDF storage path
            # ----------------------------------------------------------------
            # If store_to_db is True and cloud_storage is True, then upload the PDF to GCP.
            # Otherwise (store_to_db is False or cloud_storage is False), use local path.
            if store_to_db and cloud_storage:
                pdf_storage_path = self.gcp_integration.upload_to_storage(
                    pdf_path,
                    f"documents/{pdf_name}/{pdf_path.name}"
                )
            else:
                pdf_storage_path = str(pdf_path)

            # ----------------------------------------------------------------
            # (B) Create Title Map and update metadata
            # ----------------------------------------------------------------
            final_doc_title = None
            title_map = {}
            for page_dict in processed_elements:
                if 'Title' in page_dict:
                    for item in page_dict['Title']:
                        t_id = item['metadata'].get('element_id')
                        t_text = item.get('content', '')
                        if t_id and t_text:
                            title_map[t_id] = t_text

            for page_dict in processed_elements:
                for etype, items in page_dict.items():
                    for item in items:
                        meta = item.get('metadata', {})
                        if 'doc_title' in meta and meta['doc_title'] in title_map:
                            meta['doc_title'] = title_map[meta['doc_title']]
                        if 'section' in meta and meta['section'] in title_map:
                            meta['section'] = title_map[meta['section']]

            for t in title_map.values():
                if t.strip():
                    final_doc_title = t.strip()
                    break

            if not final_doc_title:
                final_doc_title = pdf_name

            logger.info("Title map created, metadata updated...")

            # ----------------------------------------------------------------
            # (C) Chunk & Embedding Generation
            # ----------------------------------------------------------------
            local_embedding_processor = EmbeddingProcessor()

            local_elements = []  # for accumulating embeddings when store_to_db is False
            # Iterate over each page's processed elements
            for page_num, page_dict in enumerate(processed_elements, start=1):
                for element_type, items in page_dict.items():
                    for idx, item in enumerate(items):
                        metadata = item.get('metadata', {})
                        # Add final document title if not already present
                        if final_doc_title and not metadata.get('doc_title'):
                            metadata['doc_title'] = final_doc_title

                        # Update metadata with doc_id, page_num, and chunk_index
                        metadata.update({
                            'doc_id': pdf_name,
                            'page_num': page_num,
                            'chunk_index': idx
                        })

                        # Determine content for embedding:
                        # For Picture and Table types, if 'region_path' exists, decide based on store_to_db:
                        #   - If store_to_db is True and cloud_storage is True, upload the image via GCP.
                        #   - Otherwise, use the local file path.
                        if element_type in ['Picture', 'Table']:
                            if 'region_path' in metadata:
                                if store_to_db and cloud_storage:
                                    image_path = Path(metadata['region_path'])
                                    image_storage_path = self.gcp_integration.upload_to_storage(
                                        image_path,
                                        f"documents/{pdf_name}/images/{element_type}_{page_num}_{idx}.png"
                                    )
                                else:
                                    image_storage_path = str(Path(metadata['region_path']))
                                metadata['storage_path'] = image_storage_path
                                content = image_storage_path
                            else:
                                continue
                        else:
                            content = item.get('content')
                            if not content:
                                continue

                        logger.info("Chunking and embedding...")
                        logger.info(f"Metadata: {metadata}")
                        # Chunk the content according to element type
                        chunked_items = self.process_element_content(element_type, content, metadata, chunk_size, overlap)

                        for chunk_item in chunked_items:
                            content_for_embedding = chunk_item['content']
                            c_type = chunk_item['content_type']
                            meta = chunk_item['metadata']

                            # Only process embedding for those elements whose metadata indicates to store in BigQuery.
                            if store_to_db and not meta.get('store_in_bigquery', True):
                                continue

                            # Call embedding processor: if store_to_db is False, generate_embeddings will not produce a real vector.
                            embeddings = local_embedding_processor.generate_embeddings(
                                content=content_for_embedding,
                                content_type=c_type,
                                metadata=meta,
                                chunk_size=chunk_size,
                                chunk_overlap=overlap,
                                store_to_db=store_to_db
                            )
                            if store_to_db:
                                # When storing to DB, call GCPIntegration to store element metadata.
                                for embedding in embeddings:
                                    try:
                                        self.gcp_integration.store_element_metadata(
                                            doc_id=meta['doc_id'],
                                            page_num=meta['page_num'],
                                            element_type=c_type,
                                            element_id=meta['element_id'],
                                            storage_path=meta.get('storage_path', ''),
                                            embedding_id=embedding['embedding_id'] if embedding else None,
                                            content=embedding.get('original_text', content_for_embedding),
                                            metadata=meta
                                        )
                                    except Exception as e:
                                        logger.error(f"Error processing {c_type} element {meta['element_id']}: {str(e)}")
                                        continue
                            else:
                                # If not storing to DB, accumulate embedding info locally.
                                mapped_to_element_id = metadata.get('mapped_to')
                                store_in_bigquery = metadata.get('store_in_bigquery', True)
                                doc_title = metadata.get('doc_title', '')
                                section = metadata.get('section', '')

                                for embedding in embeddings:
                                    local_elements.append({
                                        'doc_id': meta.get('doc_id'),
                                        'page_num': meta.get('page_num'),
                                        'element_type': c_type,
                                        'storage_path': meta.get('storage_path', ''),
                                        'embedding_id': embedding.get('embedding_id', None),
                                        'content': embedding.get('original_text', content_for_embedding),
                                        'mapped_to_element_id': mapped_to_element_id,
                                        'store_in_bigquery': store_in_bigquery,
                                        'metadata': meta,
                                        'title': doc_title,
                                        'section': section
                                    })

            logger.info("Embedding generation completed")

            # ----------------------------------------------------------------
            # (D) Document Metadata Storage
            # ----------------------------------------------------------------
            if store_to_db:
                # When storing to DB, call the GCP function to store document metadata.
                self.gcp_integration.store_document_metadata(
                    doc_id=pdf_name,
                    title=final_doc_title,
                    total_pages=len(images),
                    storage_path=pdf_storage_path
                )
            else:
                document_metadata = {
                    'doc_id': pdf_name,
                    'title': final_doc_title,
                    'total_pages': len(images),
                    'storage_path': pdf_storage_path
                }

            logger.info(f"Successfully processed document: {pdf_path}")

            # Return local results if not storing to DB
            if not store_to_db:
                return {
                    'elements': local_elements,
                    'document': document_metadata,
                    'embeddings': local_embedding_processor.embedding_buffer
                }
            else:
                local_embedding_processor.batch_insert_embeddings_via_load_job()
                return None

        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            raise



    def chunk_text_level2(self, text: str, max_length: int = 900, overlap: int = 100) -> List[str]:
        """
        使用LangChain的 RecursiveCharacterTextSplitter
        chunk_size=900, chunk_overlap=100
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=overlap
        )
        chunks = splitter.split_text(text)
        return chunks

    def chunk_table_level3(self, markdown_table: str, max_length=900) -> List[str]:
        if len(markdown_table) <= max_length:
            return [markdown_table]
        lines = markdown_table.split('\n')
        chunks = []
        current_chunk = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            # 如果單行就超過max_length，需先將此行再細分
            if line_len > max_length:
                # 先把current_chunk flush出去(如果有內容)
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                # 將 line 再切成多個 ≤900 chars 的小塊
                small_chunks = self.chunk_line_by_length(line, max_length)
                for sc in small_chunks:
                    # 每個小塊本身都不超過max_length，所以獨立成chunk直接加入
                    chunks.append(sc)
                continue

            # 原本的行合併邏輯
            if current_len + line_len > max_length:
                # flush current_chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_len = line_len
            else:
                current_chunk.append(line)
                current_len += line_len
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        return chunks

    def chunk_line_by_length(self, line: str, max_length: int = 900) -> List[str]:
        chunks = []
        start = 0
        while start < len(line):
            end = min(start + max_length, len(line))
            chunk = line[start:end]
            chunks.append(chunk)
            start = end
        return chunks


    def process_element_content(self, element_type: str, content: str, metadata: dict, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        根據元素類型與內容，回傳已 chunk 好的內容清單與對應 metadata。
        這裡除了 Picture 以外，都會在前面加上 prefix (doc_id, doc_title, section)。
        若是 Table，會把 prefix + caption / markdown 一起 chunk。
        """
        logger.info(f"Processing element type: {element_type}")

        chunks_info = []

        # ========== 幫助函式：用來組 prefix ==========
        def build_prefix(metadata: dict) -> str:
            logger.info(f"Building prefix for doc_title: {metadata.get('doc_title', '')}, section: {metadata.get('section', '')}")

            doc_title = metadata.get('doc_title', '').replace("\n", " ")
            section_title = metadata.get('section', '').replace("\n", " ")

            # 02/05 Update，用於標註 Title, Section 段落
            prefix_list = []
            if doc_title:
                prefix_list.append(f"Title: [{doc_title}]")
            if section_title:
                prefix_list.append(f"Section: [{section_title}]")

            print(f"prefix_list: {prefix_list}")

            if prefix_list:
                return "\n".join(prefix_list) + "\n"

            return ""

        # 圖片類型 => 不加 prefix
        if element_type == 'Picture':
            # 若有 caption => 將 caption 做 chunk
            if 'caption' in metadata and metadata['caption']:
                caption_text = metadata['caption']
                if len(caption_text) > chunk_size:
                    text_chunks = self.chunk_text_level2(caption_text, chunk_size, overlap)
                else:
                    text_chunks = [caption_text]
                for i, c in enumerate(text_chunks):
                    new_metadata = metadata.copy()
                    new_metadata['element_id'] = (
                        f"{metadata['doc_id']}_p{metadata['page_num']}_PicCap_"
                        f"{metadata.get('chunk_index',0)}_{i}"
                    )
                    chunks_info.append({
                        'content': c,
                        'content_type': 'Caption',
                        'metadata': new_metadata
                    })
            else:
                # 沒有 caption => 直接存圖片 (gs:// URI)
                chunks_info.append({
                    'content': content,
                    'content_type': 'Picture',
                    'metadata': metadata
                })
            return chunks_info

        # 其餘類型 (包含 Text, Caption, Footnote, Formula, Title, Table) => 要加 prefix
        prefix = build_prefix(metadata)
        prefix_len = len(prefix)
        if prefix_len >= chunk_size:
            # prefix 已經過長，截斷一部分，避免沒空間給正文
            prefix = prefix[:(chunk_size // 2)] + "...\n"
            prefix_len = len(prefix)
            chunk_size = chunk_size - prefix_len
            if chunk_size < overlap:
                chunk_size = overlap
        else:
            chunk_size = chunk_size - prefix_len

        # ========== 如果是 Table，先檢查 caption，否則用整個 markdown 內容 ==========
        if element_type == 'Table':
            if 'caption' in metadata and metadata['caption']:
                caption_text = metadata['caption']
                # prefix + caption_text
                final_text = prefix + caption_text
                if len(final_text) > chunk_size:
                    text_chunks = self.chunk_text_level2(final_text, chunk_size, overlap)
                else:
                    text_chunks = [final_text]

                for i, c in enumerate(text_chunks):
                    new_metadata = metadata.copy()
                    new_metadata['element_id'] = (
                        f"{metadata['doc_id']}_p{metadata['page_num']}_TableCap_"
                        f"{metadata.get('chunk_index',0)}_{i}"
                    )
                    chunks_info.append({
                        'content': c,
                        'content_type': 'Caption',  # 這裡沿用你原本的做法，以 'Caption' 標示 table caption
                        'metadata': new_metadata
                    })

            else:
                # 沒有 caption => prefix + table markdown
                final_text = prefix + content
                if len(final_text) > chunk_size:
                    table_chunks = self.chunk_table_level3(final_text, chunk_size)
                else:
                    table_chunks = [final_text]

                for i, c in enumerate(table_chunks):
                    new_metadata = metadata.copy()
                    new_metadata['element_id'] = (
                        f"{metadata['doc_id']}_p{metadata['page_num']}_Table_"
                        f"{metadata.get('chunk_index',0)}_{i}"
                    )
                    chunks_info.append({
                        'content': c,
                        'content_type': 'Table',
                        'metadata': new_metadata
                    })

            return chunks_info

        # ========== 其餘文字類型 (Text, Caption, Footnote, Formula, Title) ==========
        # prefix + content
        final_text = prefix + content
        if len(final_text) > chunk_size:
            text_chunks = self.chunk_text_level2(final_text, chunk_size, overlap)
        else:
            text_chunks = [final_text]

        for i, c in enumerate(text_chunks):
            new_metadata = metadata.copy()
            new_metadata['element_id'] = (
                f"{metadata['doc_id']}_p{metadata['page_num']}_{element_type}_"
                f"{metadata.get('chunk_index',0)}_{i}"
            )
            chunks_info.append({
                'content': c,
                'content_type': element_type,
                'metadata': new_metadata
            })

        return chunks_info


    def final_logging(self):
        end_time = time.time()
        elapsed = end_time - self.start_time
        logger.info("========== Final Run Report ==========")
        logger.info(f"Processed {self.processed_documents_count} documents in total.")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")

        if self.failed_stage:
            logger.error(f"Execution failed at stage: {self.failed_stage}")
            if self.unprocessed_documents:
                logger.error("Unprocessed documents:")
                for doc in self.unprocessed_documents:
                    logger.error(f"- {doc}")
        else:
            if self.unprocessed_documents:
                logger.info("Some documents were not processed successfully:")
                for doc in self.unprocessed_documents:
                    logger.info(f"- {doc}")
            else:
                logger.info("Execution completed successfully.")



def main():
    try:
        parser = argparse.ArgumentParser(description="Run the RAG system for document processing and querying")
        parser.add_argument(
            "--input_dir",
            type=str,
            required=True,
            help="Path to the directory containing PDF documents"
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            help="Path to the directory for output files (optional)"
        )
        args = parser.parse_args()

        doc_intelligence = DocIntelligence(use_gpu_yolo=True)

        if not args.input_dir:
            parser.error("--input-dir is required")
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.is_dir():
            raise ValueError(f"Invalid input directory: {input_dir}")

        logger.info("Starting document processing...")
        doc_intelligence.process_documents()
        logger.info("Document processing completed")


    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        doc_intelligence.failed_stage = doc_intelligence.failed_stage or "main"
    finally:
        doc_intelligence.final_logging()

if __name__ == "__main__":
    main()