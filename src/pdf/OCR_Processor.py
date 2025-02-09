import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from google.cloud import vision
from google.cloud import storage

from src.config import GCP_PROJECT_ID, GCP_BUCKET_NAME, GCP_LOCATION, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """Initialize Google Cloud Vision client"""
        try:
            self.vision_client = vision.ImageAnnotatorClient()
            self.storage_client = storage.Client(project=GCP_PROJECT_ID)
            self.bucket = self.storage_client.bucket(GCP_BUCKET_NAME)
            logger.info("Google Cloud Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Vision client: {str(e)}")
            raise

    def _upload_to_gcs(self, image_path: Path) -> Optional[str]:
        """Upload image to Google Cloud Storage"""
        try:
            blob_name = f"temp/{image_path.name}"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(str(image_path))
            return f"gs://{GCP_BUCKET_NAME}/{blob_name}"
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return None

    def _delete_from_gcs(self, gcs_uri: str):
        """Delete temporary image from Google Cloud Storage"""
        try:
            blob_name = gcs_uri.split(f"gs://{GCP_BUCKET_NAME}/")[1]
            blob = self.bucket.blob(blob_name)
            blob.delete()
        except Exception as e:
            logger.error(f"Error deleting from GCS: {str(e)}")

    def _process_text(self, response) -> Dict[str, Any]:
        if not response.text_annotations:
            return {'text': '', 'confidence': 0.0}

        full_text = response.text_annotations[0].description
        confidences = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            confidences.append(symbol.confidence)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            'text': full_text,
            'confidence': avg_confidence
        }

    def _process_table(self, response) -> Dict[str, Any]:
        if not response.full_text_annotation:
            return {'markdown': '', 'confidence': 0.0}

        elements = []
        confidences = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    text = ''
                    para_confidences = []
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            text += symbol.text
                            para_confidences.append(symbol.confidence)
                        text += ' '
                    vertices = paragraph.bounding_box.vertices
                    elements.append({
                        'text': text.strip(),
                        'x': min(v.x for v in vertices),
                        'y': min(v.y for v in vertices)
                    })
                    confidences.extend(para_confidences)

        elements.sort(key=lambda e: (e['y'], e['x']))
        markdown = self._elements_to_markdown_table(elements)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            'markdown': markdown,
            'confidence': avg_confidence
        }

    def _elements_to_markdown_table(self, elements) -> str:
        if not elements:
            return ""
        row_tolerance = 10
        rows = []
        current_row = []
        current_y = elements[0]['y']

        for element in elements:
            if abs(element['y'] - current_y) > row_tolerance:
                if current_row:
                    rows.append(current_row)
                current_row = [element]
                current_y = element['y']
            else:
                current_row.append(element)
        if current_row:
            rows.append(current_row)

        markdown = []
        if rows:
            header = " | ".join(cell['text'] for cell in rows[0])
            markdown.append(f"| {header} |")
            separator = "|" + "|".join("---" for _ in rows[0]) + "|"
            markdown.append(separator)
        for row in rows[1:]:
            row_text = " | ".join(cell['text'] for cell in row)
            markdown.append(f"| {row_text} |")

        return "\n".join(markdown)

    def process_image(self, image_path: Path, element_type: str, pdf_ocr_dir: Path) -> Dict[str, Any]:
        """
        Process an image with Google Cloud Vision OCR

        If element_type == 'Table', return markdown table
        else: return text
        """
        try:
            logger.info(f"Processing {element_type} from {image_path}")
            gcs_uri = self._upload_to_gcs(image_path)
            if not gcs_uri:
                raise Exception("Failed to upload image to GCS")

            image = vision.Image()
            image.source.image_uri = gcs_uri

            if element_type == 'Table':
                response = self.vision_client.document_text_detection(image=image)
                result = self._process_table(response)
            else:
                # For Text, Title, Picture, Caption, etc. use text_detection
                response = self.vision_client.text_detection(image=image)
                result = self._process_text(response)

            # Save OCR result
            ocr_type_dir = pdf_ocr_dir / element_type.lower()
            ocr_type_dir.mkdir(exist_ok=True, parents=True)

            ocr_file_name = f"{image_path.stem}_ocr.md"
            output_path = ocr_type_dir / ocr_file_name

            with open(output_path, 'w', encoding='utf-8') as f:
                if element_type == 'Table':
                    f.write("```markdown\n")
                    f.write(result['markdown'])
                    f.write("\n```")
                else:
                    f.write(result['text'])

            self._delete_from_gcs(gcs_uri)
            return result

        except Exception as e:
            logger.error(f"Error processing image with OCR: {str(e)}")
            return {}

    def process_elements(self, labeled_elements: List[Dict[str, List[Dict[str, Any]]]], pdf_ocr_dir: Path) -> List[Dict[str, Any]]:
        """
        Process elements from layout analysis, preserving caption mapping information.
        
        Returns a list of pages, each page is a dict with keys:
        'Text', 'Table', 'Picture', 'Caption', etc.
        Each element in these lists contains 'content' and 'metadata'.
        """
        processed_pages = []
        textual_labels = ['Text','Caption','Footnote','Formula','Title']
        for page_dict in labeled_elements:
            page_result = {
                'Text': [],
                'Table': [],
                'Picture': [],
                'Caption': [],
                'Footnote': [],
                'Formula': [],
                'Title': []
            }

            # CHANGED: 根據前面layout_analyzer的store_in_bigquery / mapped_to_element_id資訊決定OCR結果儲存
            for etype, items in page_dict.items():
                # 跳過非元素屬性
                if etype.startswith('_'):
                    continue

                if etype == 'Picture':
                    # 圖像不做OCR，但要保留region_path與metadata
                    for item in items:
                        page_result['Picture'].append({
                            'content': '[Image content not extracted]',
                            'metadata': {
                                'coordinates': item['coordinates'],
                                'element_id': item.get('element_id'),
                                'store_in_bigquery': item.get('store_in_bigquery', True),
                                'mapped_to': item.get('mapped_to'),
                                'region_path': item.get('region_path'),  # 保留 region_path
                                'doc_title': item.get('doc_title'),
                                'section': item.get('section')
                            }
                        })
                elif etype == 'Table':
                    # Table 使用 table OCR
                    for item in items:
                        ocr_result = self.process_image(Path(item['region_path']), 'Table', pdf_ocr_dir)
                        page_result['Table'].append({
                            'content': ocr_result.get('markdown', ''),
                            'metadata': {
                                'confidence': ocr_result.get('confidence', 0.0),
                                'coordinates': item['coordinates'],
                                'element_id': item.get('element_id'),
                                'caption': item.get('caption'),
                                'store_in_bigquery': item.get('store_in_bigquery', True),
                                'mapped_to': item.get('mapped_to'),
                                'region_path': item.get('region_path'),  # 保留 region_path
                                'doc_title': item.get('doc_title'),
                                'section': item.get('section')
                            }
                        })
                elif etype in textual_labels:
                    # 文字類型則進行 OCR
                    for item in items:
                        ocr_result = self.process_image(Path(item['region_path']), etype, pdf_ocr_dir)
                        page_result[etype].append({
                            'content': ocr_result.get('text', ''),
                            'metadata': {
                                'confidence': ocr_result.get('confidence', 0.0),
                                'coordinates': item['coordinates'],
                                'element_id': item.get('element_id'),
                                'mapped_to': item.get('mapped_to'),
                                'store_in_bigquery': item.get('store_in_bigquery', True),
                                'region_path': item.get('region_path'),  # 保留 region_path
                                'doc_title': item.get('doc_title'),
                                'section': item.get('section')
                            }
                        })
                else:
                    # 非預期類別（如有的話）
                    for item in items:
                        page_result[etype].append({
                            'content': '[No OCR for this element type]',
                            'metadata': {
                                'coordinates': item['coordinates'],
                                'element_id': item.get('element_id'),
                                'store_in_bigquery': item.get('store_in_bigquery', True),
                                'mapped_to': item.get('mapped_to'),
                                'region_path': item.get('region_path'),  # 保留 region_path
                                'doc_title': item.get('doc_title'),
                                'section': item.get('section')
                            }
                        })

            processed_pages.append(page_result)
        return processed_pages