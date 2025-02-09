import argparse
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Union
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures

from src.pdf_processor import PDFProcessor
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_processor import OCRProcessor
from src.gcp_integration import GCPIntegration
from src.embedding_processor import EmbeddingProcessor
from src.rag.graph import SelfRAGGraph
from src.config import (
    DOCUMENTS_DIR, 
    OUTPUT_DIR, 
    GCP_PROJECT_ID, 
    GCP_BUCKET_NAME,
    GCP_LOCATION,
    GCP_ANTHROPIC_ENDPOINT_LOCATION,
    GCP_APPLICATION_CREDENTIALS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, use_gpu_yolo: bool = False):
        """Initialize the RAG system components"""
        try:
            self.start_time = time.time()
            self.processed_documents_count = 0
            self.failed_stage = None
            self.unprocessed_documents = []

            load_dotenv()
            self.pdf_processor = PDFProcessor()
            self.layout_analyzer = LayoutAnalyzer(use_gpu=use_gpu_yolo)
            self.ocr_processor = OCRProcessor()
            self.gcp_integration = GCPIntegration(GCP_PROJECT_ID, GCP_BUCKET_NAME, GCP_APPLICATION_CREDENTIALS)
            self.embedding_processor = EmbeddingProcessor(GCP_PROJECT_ID, GCP_LOCATION)
            self.rag_graph = SelfRAGGraph(GCP_PROJECT_ID, GCP_ANTHROPIC_ENDPOINT_LOCATION, gcp_integration=self.gcp_integration, embedding_processor=self.embedding_processor)
            logger.info("RAG system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise

    def chunk_text_level2(self, text: str, max_length=900) -> List[str]:
        """
        使用LangChain的 RecursiveCharacterTextSplitter
        chunk_size=900, chunk_overlap=100
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=100
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


    def process_element_content(self, element_type: str, content: str, metadata: dict) -> List[Dict[str, Any]]:
        """
        根據元素類型與內容，回傳已 chunk 好的內容清單與對應 metadata。
        這裡除了 Picture 以外，都會在前面加上 prefix (doc_id, doc_title, section)。
        若是 Table，會把 prefix + caption / markdown 一起 chunk。
        """
        chunks_info = []

        # ========== 幫助函式：用來組 prefix ==========
        def build_prefix(metadata: dict) -> str:
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
                if len(caption_text) > 900:
                    text_chunks = self.chunk_text_level2(caption_text, 900)
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
        base_chunk_size = 900
        prefix_len = len(prefix)
        if prefix_len >= base_chunk_size:
            # prefix 已經過長，截斷一部分，避免沒空間給正文
            prefix = prefix[:(base_chunk_size // 2)] + "...\n"
            prefix_len = len(prefix)
            chunk_size = base_chunk_size - prefix_len
            if chunk_size < 100:
                chunk_size = 100
        else:
            chunk_size = base_chunk_size - prefix_len

        # ========== 如果是 Table，先檢查 caption，否則用整個 markdown 內容 ==========
        if element_type == 'Table':
            if 'caption' in metadata and metadata['caption']:
                caption_text = metadata['caption']
                # prefix + caption_text
                final_text = prefix + caption_text
                if len(final_text) > base_chunk_size:
                    text_chunks = self.chunk_text_level2(final_text, chunk_size)
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
                if len(final_text) > base_chunk_size:
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
        if len(final_text) > base_chunk_size:
            text_chunks = self.chunk_text_level2(final_text, chunk_size)
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


    def process_documents(self):
        """Process all PDF documents in the Documents directory in parallel using ThreadPoolExecutor."""
        pdf_files = list(DOCUMENTS_DIR.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # 使用 ThreadPoolExecutor 進行並行化
        # 若OCR或上傳GCS是I/O bound(最有可能)，使用ThreadPoolExecutor可加速I/O相關任務
        # 若需要重度CPU計算，可考慮ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pdf = {executor.submit(self.process_single_document, pdf_path): pdf_path for pdf_path in pdf_files}

            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    future.result()
                    self.processed_documents_count += 1
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {str(e)}")
                    # 將處理失敗的文件加入 unprocessed_documents
                    self.unprocessed_documents.append(str(pdf_path))
                    continue

        logger.info("Document processing completed")

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


    def process_single_document(self, pdf_path: Path):
        try:
            logger.info(f"Processing document: {pdf_path}")
            pdf_name = pdf_path.stem
            pdf_output_dir = OUTPUT_DIR / pdf_name
            pdf_images_dir = pdf_output_dir / "images"
            pdf_labeled_dir = pdf_output_dir / "labeled"
            pdf_ocr_dir = pdf_output_dir / "ocr"

            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_images_dir.mkdir(exist_ok=True)
            pdf_labeled_dir.mkdir(exist_ok=True)
            pdf_ocr_dir.mkdir(exist_ok=True)

            images = self.pdf_processor.convert_pdf_to_images(pdf_path, pdf_images_dir)
            logger.info(f"Converted PDF to {len(images)} images")

            labeled_elements = self.layout_analyzer.analyze_layouts(images, pdf_labeled_dir, debug=True)    # Debug mode
            logger.info("Layout analysis completed")

            # labeled_elements 的長度通常 == PDF頁數
            # 我們要確保收集 OCR 後的結果，也能按原本 PDF 頁序放好
            ordered_results = [None] * len(labeled_elements)

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_to_pageidx = {}
                for page_idx, page_data in enumerate(labeled_elements, start=1):
                    # 提交 OCR 處理時，把 page_idx 帶進去，避免用 enumerate 結束順序
                    future = executor.submit(
                        self.ocr_processor.process_elements,
                        [page_data],
                        pdf_ocr_dir
                    )
                    future_to_pageidx[future] = page_idx

                for future in concurrent.futures.as_completed(future_to_pageidx):
                    page_idx = future_to_pageidx[future]
                    try:
                        result = future.result()  # result是一個list, 通常包含1個page dict
                        # 依 page_idx 放回ordered_results
                        # 注意 page_idx 是從1起算，但list index從0起算，所以減1
                        ordered_results[page_idx - 1] = result
                    except Exception as e:
                        logger.error(f"OCR error on page {page_idx}: {str(e)}")

            processed_elements = []
            for res in ordered_results:
                if res is not None:
                    processed_elements.extend(res)

            logger.info("OCR processing completed")

            # 上傳 PDF 檔到 GCS (僅一次)
            pdf_gcs_path = self.gcp_integration.upload_to_storage(
                pdf_path,
                f"documents/{pdf_name}/{pdf_path.name}"
            )

            #------------------------------------------
            # (1) 建立 Title Map
            #    title_map[ title_element_id ] = ocr_text
            #------------------------------------------
            title_map = {}
            for page_dict in processed_elements:
                # page_dict = {'Text': [...], 'Title': [...], 'Table': [...], ...}
                # 取出 Title 裡的 content
                if 'Title' in page_dict:
                    for item in page_dict['Title']:
                        t_id = item['metadata'].get('element_id')
                        t_ocr = item.get('content', '')
                        if t_id and t_ocr:
                            title_map[t_id] = t_ocr

            #------------------------------------------
            # (2) 依照 layout_analyzer，元素可能有:
            #    'doc_title' 或 'section'
            #    => 如果值是某個 title_element_id => 用 title_map 替換
            #------------------------------------------
            for page_dict in processed_elements:
                for etype, items in page_dict.items():
                    for item in items:
                        # print("====================================")
                        meta = item.get('metadata', {})
                        # print(f"Before: metadata -> {meta}")

                        # doc_title
                        if 'doc_title' in meta and meta['doc_title'] in title_map:
                            meta['doc_title'] = title_map[meta['doc_title']]
                        # section
                        if 'section' in meta and meta['section'] in title_map:
                            meta['section'] = title_map[meta['section']]

                        # print(f"After: metadata -> {meta}")

            #------------------------------------------
            # (3) 在本地先決定文件最終要顯示的 title
            #     若 layout_analyzer 找到 "doc_title", 以最後/最早? 為最終
            #     這裡簡易: 若有多個 doc_title, 就取第一個不為空的
            #------------------------------------------
            final_doc_title_ocr = None
            for t_ocr in title_map.values():
                if t_ocr.strip():
                    final_doc_title_ocr = t_ocr.strip()
                    break

            # #------------------------------------------
            # # (4) 建立 embedding 前，先存 page metadata 到 local (以便 embedding 時參考)
            # #     這裡只做 local list，避免重複呼叫 BigQuery
            # #------------------------------------------
            # local_page_metadata = []
            # total_pages = len(ordered_results)
            # for p_i in range(total_pages):
            #     page_num = p_i + 1
            #     local_page_metadata.append({
            #         'doc_id': pdf_name,
            #         'page_num': page_num,
            #         'storage_path': f"documents/{pdf_name}/pages/page_{page_num}"
            #     })

            #------------------------------------------
            # (5) chunk & embedding
            #------------------------------------------
            for page_num, page_dict in enumerate(processed_elements, start=1):
                for element_type, items in page_dict.items():
                    for idx, item in enumerate(items):
                        metadata = item.get('metadata', {})
                        # 加入最終 doc_title (若先前找到)
                        if final_doc_title_ocr and 'doc_title' not in metadata:
                            metadata['doc_title'] = final_doc_title_ocr

                        # 這裡更新 doc_id/page_num/chunk_index
                        metadata.update({
                            'doc_id': pdf_name,
                            'page_num': page_num,
                            'chunk_index': idx
                        })

                        content = None
                        if element_type in ['Picture', 'Table']:
                            if element_type == 'Picture':
                                if 'region_path' in metadata:
                                    image_gcs_path = self.gcp_integration.upload_to_storage(
                                        Path(metadata['region_path']),
                                        f"documents/{pdf_name}/images/{element_type}_{page_num}_{idx}.png"
                                    )
                                    metadata['storage_path'] = image_gcs_path
                                    content = image_gcs_path
                                else:
                                    continue
                            else:  # Table
                                content = item.get('content')
                                if not content:
                                    continue
                        else:
                            content = item.get('content')
                            if not content:
                                continue

                        # chunking + embedding
                        print("Chunking and embedding...")
                        print(f"metadata: {metadata}")
                        chunked_items = self.process_element_content(element_type, content, metadata)

                        for chunk_item in chunked_items:
                            content_for_embedding = chunk_item['content']
                            c_type = chunk_item['content_type']
                            meta = chunk_item['metadata']

                            if not meta.get('store_in_bigquery', True):
                                continue

                            try:
                                embedding = self.embedding_processor.generate_embeddings(
                                    content=content_for_embedding,
                                    content_type=c_type,
                                    metadata=meta
                                )
                                self.gcp_integration.store_element_metadata(
                                    doc_id=meta['doc_id'],
                                    page_num=meta['page_num'],
                                    element_type=c_type,
                                    element_id=meta['element_id'],
                                    storage_path=meta.get('storage_path', ''),
                                    embedding_id=embedding['embedding_id'] if embedding else None,
                                    content=content_for_embedding,
                                    metadata=meta
                                )
                            except Exception as e:
                                logger.error(f"Error processing {c_type} element {meta['element_id']}: {str(e)}")
                                continue

            #------------------------------------------
            # (6) 最後更新 documents metadata 到 BigQuery
            #------------------------------------------
            if not final_doc_title_ocr:
                final_doc_title_ocr = pdf_name  # 如果沒找到任何 title，就用原始 pdf 檔名

            # 寫入 documents 資料表
            self.gcp_integration.store_document_metadata(
                doc_id=pdf_name,
                title=final_doc_title_ocr,
                total_pages=len(images),
                storage_path=pdf_gcs_path
            )
            # # 寫入 pages 資料表
            # for p_meta in local_page_metadata:
            #     self.gcp_integration.store_page_metadata(
            #         doc_id=p_meta['doc_id'],
            #         page_num=p_meta['page_num'],
            #         storage_path=p_meta['storage_path']
            #     )

            logger.info(f"Successfully processed document: {pdf_path}")

        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        Returns a dictionary containing the response details
        """
        try:
            logger.info(f"Processing query: {question}")
            result = self.rag_graph.run(question)
            
            # 先準備 supporting_documents
            supporting_docs = []
            documents = result.get('documents', [])
            grades = result.get('grades', {}).get('documents', [])
            
            # 用傳統的 for 循環來構建 supporting_documents
            for doc, grade in zip(documents, grades):
                doc_info = {
                    'doc_id': doc.get('doc_id'),
                    'page_num': doc.get('page_num'),
                    'content': doc.get('content'),
                    'relevance_score': grade.score
                }
                supporting_docs.append(doc_info)
            
            # 取得 confidence score
            grades_info = result.get('grades', {})
            generation_grade = grades_info.get('generation')
            confidence_score = generation_grade.score if generation_grade else 0.0
            
            # 構建最終的 response
            response = {
                'question': question,
                'answer': result.get('generation', ''),
                'supporting_documents': supporting_docs,
                'confidence_score': confidence_score
            }
            
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise


def main():
    try:
        parser = argparse.ArgumentParser(description="Run the RAG system for document processing and querying")
        parser.add_argument(
            "--mode",
            type=str,
            required=True,
            choices=['embedding', 'query'],
            help="Operation mode: 'embedding' for document processing or 'query' for asking questions"
        )
        parser.add_argument(
            "--input_dir",
            type=str,
            help="Path to the directory containing PDF documents (required for embedding mode)"
        )
        parser.add_argument(
            "--instruction",
            type=str,
            help="Question to ask the system (required for query mode)"
        )
        args = parser.parse_args()

        rag_system = RAGSystem(use_gpu_yolo=True)

        if args.mode == 'embedding':
            if not args.input_dir:
                parser.error("--input-dir is required when mode is 'embedding'")
            input_dir = Path(args.input_dir).resolve()
            if not input_dir.is_dir():
                raise ValueError(f"Invalid input directory: {input_dir}")

            global DOCUMENTS_DIR
            DOCUMENTS_DIR = input_dir
            logger.info(f"DOCUMENTS_DIR set to: {DOCUMENTS_DIR}")

            logger.info("Starting document processing...")
            rag_system.process_documents()
            logger.info("Document processing completed")

            try:
                rag_system.embedding_processor.batch_insert_embeddings_via_load_job()
                logger.info("All embeddings have been batch inserted to BigQuery.")
            except Exception as e:
                rag_system.failed_stage = "batch_insert_embeddings_via_load_job"
                logger.error(f"Error during batch insert embeddings: {str(e)}")

        else:
            if not args.instruction:
                parser.error("--instruction is required when mode is 'query'")
            response = rag_system.query(args.instruction)

            print("\nQuestion:", response["question"])
            print("\nAnswer:", response["answer"])
            print("\nConfidence Score:", response["confidence_score"])
            print("\nSupporting Documents:")
            for doc in response["supporting_documents"]:
                print(f"\n- Document {doc['doc_id']}, Page {doc['page_num']}")
                print(f"  Relevance Score: {doc['relevance_score']:.2f}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        rag_system.failed_stage = rag_system.failed_stage or "main"
    finally:
        rag_system.final_logging()

if __name__ == "__main__":
    main()