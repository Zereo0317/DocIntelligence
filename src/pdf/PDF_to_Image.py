import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List
from src.config import PDF_IMAGE_DPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    @staticmethod
    def convert_pdf_to_images(pdf_path: Path, pdf_images_dir: Path) -> List[Path]:
        """
        Convert a PDF file to images, one per page.
        
        Args:
            pdf_path (Path): Path to the PDF file
            pdf_images_dir (Path): Directory to save the page images
            
        Returns:
            List[Path]: List of paths to the generated images
        """
        image_paths = []
        try:
            pdf_images_dir.mkdir(parents=True, exist_ok=True)
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=PDF_IMAGE_DPI)
                    image_path = pdf_images_dir / f"page_{page_num + 1}.png"
                    pix.save(str(image_path))
                    image_paths.append(image_path)
                    logger.info(f"Converted page {page_num + 1} of {pdf_path.name}")
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1} of {pdf_path.name}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()

        return image_paths
