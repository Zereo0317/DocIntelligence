# src/layout_analyzer.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

# Machine learning and clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.config import YOLO_CONF_THRESHOLD, YOLO_IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    def __init__(self, use_gpu: bool = False):
        """Initialize the YOLO model for document layout analysis."""
        try:
            logger.info("Loading YOLO model...")
            try:
                logger.info("Attempting to load model using from_pretrained...")
                self.device = "cuda:1" if use_gpu else "cpu"
                self.model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench", device=self.device)
            except Exception as e1:
                logger.warning(f"from_pretrained failed: {str(e1)}")
                logger.info("Attempting alternative loading method from hf_hub_download...")
                filepath = hf_hub_download(
                    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
                )
                self.model = YOLOv10(filepath)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise

    def _extract_region(self, image: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
        """Extract a sub-region from the image based on the bounding box."""
        return image[
            bbox['y1']:bbox['y2'],
            bbox['x1']:bbox['x2']
        ]

    def _find_caption_for_visual(self, visual_bbox: Dict[str, int],
                                 captions: List[Dict[str, Any]],
                                 all_elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find a suitable caption for a visual element (e.g., picture/table/formula).
        We try: below, beside, above – whichever is physically closest and not blocked.
        """

        def has_overlap(box1: Dict[str, int], box2: Dict[str, int]) -> bool:
            """Check if two boxes overlap."""
            return not (box1['x2'] < box2['x1'] or
                        box1['x1'] > box2['x2'] or
                        box1['y2'] < box2['y1'] or
                        box1['y1'] > box2['y2'])

        def is_path_clear(visual: Dict[str, int],
                          caption_box: Dict[str, int],
                          elements: List[Dict[str, Any]]) -> bool:
            """
            Check if the direct path between the visual and a candidate caption is not obstructed.
            """
            region = {
                'x1': min(visual['x1'], caption_box['x1']),
                'x2': max(visual['x2'], caption_box['x2']),
                'y1': min(visual['y2'], caption_box['y1']),
                'y2': max(visual['y2'], caption_box['y1'])
            }
            for elem in elements:
                if elem['coordinates'] in [visual, caption_box]:
                    continue
                if has_overlap(region, elem['coordinates']):
                    return False
            return True

        # Below
        below_captions = [cap for cap in captions if cap['coordinates']['y1'] > visual_bbox['y2']]
        if below_captions:
            below_captions.sort(key=lambda x: (
                abs(x['coordinates']['y1'] - visual_bbox['y2']),
                abs(((x['coordinates']['x1'] + x['coordinates']['x2']) / 2.0 -
                     (visual_bbox['x1'] + visual_bbox['x2']) / 2.0))
            ))
            for c in below_captions:
                if is_path_clear(visual_bbox, c['coordinates'], all_elements):
                    return c

        # Beside
        side_captions = [cap for cap in captions if (
            cap['coordinates']['y1'] >= visual_bbox['y1'] and
            cap['coordinates']['y2'] <= visual_bbox['y2']
        )]
        if side_captions:
            side_captions.sort(key=lambda x:
                abs(((x['coordinates']['x1'] + x['coordinates']['x2']) / 2.0 -
                     (visual_bbox['x1'] + visual_bbox['x2']) / 2.0))
            )
            for c in side_captions:
                if is_path_clear(visual_bbox, c['coordinates'], all_elements):
                    return c

        # Above
        above_captions = [cap for cap in captions if cap['coordinates']['y2'] < visual_bbox['y1']]
        if above_captions:
            above_captions.sort(key=lambda x: (
                abs(visual_bbox['y1'] - x['coordinates']['y2']),
                abs(((x['coordinates']['x1'] + x['coordinates']['x2']) / 2.0 -
                     (visual_bbox['x1'] + visual_bbox['x2']) / 2.0))
            ))
            for c in above_captions:
                if is_path_clear(visual_bbox, c['coordinates'], all_elements):
                    return c

        return None

    @staticmethod
    def _detect_columns_kmeans(elements: List[Dict[str, Any]]) -> Optional[float]:
        """
        Use ML-based clustering to detect whether this page is multi-column.
        1) KMeans(n=2). If silhouette is good, we pick that.
        2) Otherwise, fallback to DBSCAN, then GMM.
        3) Return boundary (avg of cluster centers) if 2 columns are found; else None => single column.
        """
        base_types = ['Text', 'Title']
        base_elems = [e for e in elements if e['type'] in base_types]
        if not base_elems:
            return None

        # Prepare data for clustering (center x)
        X = []
        for e in base_elems:
            x1 = e['coordinates']['x1']
            x2 = e['coordinates']['x2']
            cx = (x1 + x2) / 2.0
            X.append([cx])
        X = np.array(X, dtype=np.float32)

        if len(X) < 4:
            # Not enough data to reliably form two columns
            return None

        def boundary_from_centers(centers: np.ndarray) -> float:
            """Compute boundary line from two cluster centers."""
            c1, c2 = centers[0, 0], centers[1, 0]
            return float((c1 + c2) / 2.0)

        # 1) KMeans
        km = KMeans(n_clusters=2, random_state=42)
        labels_km = km.fit_predict(X)
        km_score = silhouette_score(X, labels_km)
        cluster_counts = [np.sum(labels_km == i) for i in [0, 1]]
        if km_score >= 0.2 and all(c >= 2 for c in cluster_counts):
            boundary = boundary_from_centers(km.cluster_centers_)
            logger.info(f"KMeans => 2 columns, silhouette={km_score:.2f}, boundary={boundary:.2f}")
            return boundary

        # 2) DBSCAN fallback
        logger.info("Falling back to DBSCAN for columns...")
        x_min, x_max = float(np.min(X)), float(np.max(X))
        x_range = x_max - x_min
        if x_range < 10:
            return None  # probably single-column if range is small

        eps_guess = x_range / 10.0
        dbscan = DBSCAN(eps=eps_guess, min_samples=2)
        labels_db = dbscan.fit_predict(X)

        unique_clusters = set(labels_db) - {-1}
        if len(unique_clusters) == 2:
            filtered_mask = labels_db != -1
            db_score = silhouette_score(X[filtered_mask], labels_db[filtered_mask])
            if db_score >= 0.2:
                # compute centers
                c_list = []
                for uc in unique_clusters:
                    pts = X[labels_db == uc]
                    c_list.append(np.mean(pts[:, 0]))
                boundary = float(np.mean(c_list))
                logger.info(f"DBSCAN => 2 columns, silhouette={db_score:.2f}, boundary={boundary:.2f}")
                return boundary

        # 3) GMM fallback
        logger.info("Falling back to GMM for columns...")
        if len(X) < 6:
            return None
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm_labels = gmm.fit_predict(X)
        gmm_score = silhouette_score(X, gmm_labels)
        g0 = np.sum(gmm_labels == 0)
        g1 = np.sum(gmm_labels == 1)

        if gmm_score >= 0.2 and g0 >= 2 and g1 >= 2:
            means = gmm.means_.flatten()
            boundary = float((means[0] + means[1]) / 2.0)
            logger.info(f"GMM => 2 columns, silhouette={gmm_score:.2f}, boundary={boundary:.2f}")
            return boundary

        logger.info("All clustering indicates single-column or insufficient separation.")
        return None

    def _find_page_margins(self, elements: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Estimate page margins using DBSCAN on all x1, x2 to remove outliers, then pick min and max.
        """
        if not elements:
            return {'left': 0, 'right': 0, 'width': 0}

        all_x1 = [float(e['coordinates']['x1']) for e in elements]
        all_x2 = [float(e['coordinates']['x2']) for e in elements]
        combined_x = all_x1 + all_x2

        if len(combined_x) < 6:
            # fallback if not enough data
            left_m = min(all_x1)
            right_m = max(all_x2)
            return {'left': left_m, 'right': right_m, 'width': right_m - left_m}

        x_min_val, x_max_val = min(combined_x), max(combined_x)
        x_range = x_max_val - x_min_val
        if x_range <= 0:
            return {'left': x_min_val, 'right': x_max_val, 'width': 0}

        # DBSCAN to remove outliers
        X_coord = np.array(combined_x).reshape(-1, 1)
        eps_guess = x_range / 5.0
        clusterer = DBSCAN(eps=eps_guess, min_samples=3)
        labels = clusterer.fit_predict(X_coord)

        label_counts = {}
        for lb in labels:
            if lb == -1:
                continue
            label_counts[lb] = label_counts.get(lb, 0) + 1
        if not label_counts:
            # fallback
            left_m = min(all_x1)
            right_m = max(all_x2)
            return {'left': left_m, 'right': right_m, 'width': right_m - left_m}

        # pick largest cluster
        main_label = max(label_counts, key=label_counts.get)
        inliers = X_coord[labels == main_label].flatten()
        left_m = float(np.min(inliers))
        right_m = float(np.max(inliers))
        return {'left': left_m, 'right': right_m, 'width': right_m - left_m}

    def _cluster_widths_for_full_width(self, elements: List[Dict[str, Any]]) -> set:
        """
        Determine which elements are "full width" via clustering on widths:
        1) Gather widths of all bounding boxes.
        2) Use KMeans(n=2) or fallback to DBSCAN to separate wide vs. narrow.
        3) Return a set of element_ids that belong to the 'widest' cluster.
        """
        if not elements:
            return set()

        # Collect widths
        data = []
        for e in elements:
            w = float(e['coordinates']['x2'] - e['coordinates']['x1'])
            data.append((w, e['element_id']))
        widths = np.array([[d[0]] for d in data], dtype=np.float32)

        if len(widths) < 2:
            return set()  # no clustering possible

        # 1) KMeans
        km = KMeans(n_clusters=2, random_state=42)
        labels_km = km.fit_predict(widths)
        if len(widths) > 2:
            score_km = silhouette_score(widths, labels_km)
        else:
            score_km = 1.0  # trivial if only 2 items

        if score_km >= 0.2:
            # pick cluster with larger mean as "full width"
            c0_mean = float(np.mean(widths[labels_km == 0]))
            c1_mean = float(np.mean(widths[labels_km == 1]))
            if c0_mean > c1_mean:
                fw_label = 0
            else:
                fw_label = 1
            fw_ids = [data[i][1] for i in range(len(data)) if labels_km[i] == fw_label]
            return set(fw_ids)

        # 2) DBSCAN fallback
        w_min, w_max = float(np.min(widths)), float(np.max(widths))
        w_range = w_max - w_min
        if w_range < 1:
            return set()
        eps_guess = w_range / 4.0
        db = DBSCAN(eps=eps_guess, min_samples=2)
        labels_db = db.fit_predict(widths)
        valid_clusters = set(labels_db) - {-1}
        if not valid_clusters:
            return set()

        # pick cluster with largest mean width
        best_label = None
        best_mean = -999999
        for cl in valid_clusters:
            cluster_w = widths[labels_db == cl]
            mean_w = float(np.mean(cluster_w))
            if mean_w > best_mean:
                best_mean = mean_w
                best_label = cl

        fw_ids = [data[i][1] for i in range(len(data)) if labels_db[i] == best_label]
        return set(fw_ids)

    def _sort_and_link_elements(self,
                                all_elements: List[Dict[str, Any]],
                                debug: bool = False,
                                doc_title: Optional[str] = None,
                                prev_page_section: Optional[str] = None
                                ) -> (List[Dict[str, Any]], bool):
        """
        Sort elements in reading order, detect columns, maintain sections, etc.
        """
        if not all_elements:
            return (all_elements, False)

        # 1) Page margins
        margins = self._find_page_margins(all_elements)

        # 2) Identify which elements are "full width" via width clustering
        fw_ids = self._cluster_widths_for_full_width(all_elements)

        # 3) Detect multi-column layout for this page
        col_boundary = self._detect_columns_kmeans(all_elements)
        is_multi_col = col_boundary is not None

        if debug:
            logger.info("==== Layout Analysis (Single Page) ====")
            logger.info(f"page_margins: left={margins['left']:.2f}, right={margins['right']:.2f}, width={margins['width']:.2f}")
            logger.info(f"multi_col={is_multi_col}, boundary={col_boundary}, full_width_count={len(fw_ids)}")

        # 4) Basic doc_title / section logic
        current_section = prev_page_section

        # 5) If single column, just sort vertically
        if not is_multi_col:
            all_elements.sort(key=lambda x: x['coordinates']['y1'])
            for elem in all_elements:
                if elem['type'] == 'Title' and elem['element_id'] != doc_title:
                    current_section = elem['element_id']
                elem['doc_title'] = doc_title
                elem['section'] = current_section
            return (all_elements, False)

        # 6) Multi-column => separate into full-width vs. left vs. right
        full_width_elems = []
        left_col = []
        right_col = []

        for elem in all_elements:
            elem_id = elem['element_id']
            if elem_id in fw_ids:
                full_width_elems.append(elem)
            else:
                x1, x2 = elem['coordinates']['x1'], elem['coordinates']['x2']
                cx = (x1 + x2) / 2.0
                if col_boundary and cx < col_boundary:
                    left_col.append(elem)
                else:
                    right_col.append(elem)

        # sort by y in each group
        for grp in [full_width_elems, left_col, right_col]:
            grp.sort(key=lambda x: x['coordinates']['y1'])

        final_elems = []

        # find the top_y among left/right
        top_y = float('inf')
        if left_col:
            top_y = min(top_y, left_col[0]['coordinates']['y1'])
        if right_col:
            top_y = min(top_y, right_col[0]['coordinates']['y1'])

        # 6.1 top full-width
        top_full = [e for e in full_width_elems if e['coordinates']['y2'] <= top_y]
        for elem in top_full:
            if elem['type'] == 'Title' and elem['element_id'] != doc_title:
                current_section = elem['element_id']
            elem['doc_title'] = doc_title
            elem['section'] = current_section
            final_elems.append(elem)

        # 6.2 left col
        for elem in left_col:
            if elem['type'] == 'Title' and elem['element_id'] != doc_title:
                current_section = elem['element_id']
            elem['doc_title'] = doc_title
            elem['section'] = current_section
            final_elems.append(elem)

        # 6.3 right col
        for elem in right_col:
            if elem['type'] == 'Title' and elem['element_id'] != doc_title:
                current_section = elem['element_id']
            elem['doc_title'] = doc_title
            elem['section'] = current_section
            final_elems.append(elem)

        # 6.4 remaining full-width
        remaining_fw = [e for e in full_width_elems if e not in top_full]
        for elem in remaining_fw:
            if elem['type'] == 'Title' and elem['element_id'] != doc_title:
                current_section = elem['element_id']
            elem['doc_title'] = doc_title
            elem['section'] = current_section
            final_elems.append(elem)

        if debug:
            logger.info("==== Final Reading Order (Single Page) ====")
            for fe in final_elems:
                logger.info(f"{fe['element_id']} => type={fe['type']}, y1={fe['coordinates']['y1']}")

        return (final_elems, True)

    def analyze_image(self,
                      image_path: Path,
                      page_num: int,
                      pdf_labeled_dir: Path,
                      debug: bool = False,
                      doc_title: Optional[str] = None,
                      prev_page_section: Optional[str] = None
                      ) -> Dict[str, Any]:
        """
        Analyze a single page image; supports Title/Section tracking, multi-column, etc.
        """
        logger.info(f"\nAnalyzing image {image_path}, Page {page_num}...")
        try:
            # Map YOLO classes to existing bounding-box categories
            class_name_map = {
                'title': 'Title',
                'plain text': 'Text',
                'figure': 'Picture',
                'figure_caption': 'Caption',
                'isolate_formula': 'Formula',
                'formula_caption': 'Caption',
                'table': 'Table',
                'table_caption': 'Caption',
                'table_footnote': 'Footnote'
            }

            # Prepare results structure
            results = {
                'Caption': [], 'Footnote': [], 'Formula': [], 'Picture': [],
                'Table': [], 'Text': [], 'Title': []
            }

            # Load the page image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return results

            # YOLO inference
            det_res = self.model.predict(
                image,
                imgsz=YOLO_IMAGE_SIZE,
                conf=YOLO_CONF_THRESHOLD,
                device=self.device
            )

            # Build a list of all elements
            all_elements = []
            if det_res and len(det_res) > 0:
                for detection in det_res[0].boxes:
                    class_id = int(detection.cls)
                    yolo_class_name = det_res[0].names[class_id]
                    confidence = float(detection.conf)
                    bbox = detection.xyxy[0].tolist()

                    coordinates = {
                        'x1': int(bbox[0]),
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]),
                        'y2': int(bbox[3])
                    }

                    mapped_class_name = class_name_map.get(yolo_class_name.lower())
                    if mapped_class_name and mapped_class_name in results:
                        element_dir = pdf_labeled_dir / mapped_class_name.lower()
                        element_dir.mkdir(parents=True, exist_ok=True)
                        elem_count = len(results[mapped_class_name])

                        # Extract region
                        region = self._extract_region(image, coordinates)
                        region_path = element_dir / f"page_{page_num}_{elem_count}.png"
                        cv2.imwrite(str(region_path), region)

                        element = {
                            'coordinates': coordinates,
                            'confidence': confidence,
                            'region_path': str(region_path),
                            'page_num': page_num,
                            'element_id': f"p{page_num}_{mapped_class_name}_{elem_count}",
                            'type': mapped_class_name
                        }
                        results[mapped_class_name].append(element)
                        all_elements.append(element)

            # Sort/link elements => reading order, multi-col
            ordered_elems, is_double_column = self._sort_and_link_elements(
                all_elements,
                debug=debug,
                doc_title=doc_title,
                prev_page_section=prev_page_section
            )

            # Re-distribute sorted elements
            for k in results.keys():
                if not k.startswith('_'):
                    results[k] = []
            for elem in ordered_elems:
                results[elem['type']].append(elem)

            # Associate captions => pictures/tables/formulas
            pictures = [e for e in ordered_elems if e['type'] == 'Picture']
            tables = [e for e in ordered_elems if e['type'] == 'Table']
            formulas = [e for e in ordered_elems if e['type'] == 'Formula']
            captions = [e for e in ordered_elems if e['type'] == 'Caption']
            footnotes = [e for e in ordered_elems if e['type'] == 'Footnote']

            for pic in pictures:
                c = self._find_caption_for_visual(pic['coordinates'], captions, ordered_elems)
                if c:
                    c['mapped_to'] = pic['element_id']
                    pic['caption'] = c['element_id']

            for tbl in tables:
                c = self._find_caption_for_visual(tbl['coordinates'], captions, ordered_elems)
                if c:
                    c['mapped_to'] = tbl['element_id']
                    tbl['caption'] = c['element_id']
                # Link footnote just below table if any
                fset = [fn for fn in footnotes if fn['coordinates']['y1'] > tbl['coordinates']['y2']]
                if fset:
                    fset.sort(key=lambda x: x['coordinates']['y1'] - tbl['coordinates']['y2'])
                    fset[0]['mapped_to'] = tbl['element_id']

            for fm in formulas:
                c = self._find_caption_for_visual(fm['coordinates'], captions, ordered_elems)
                if c:
                    c['mapped_to'] = fm['element_id']
                    fm['caption'] = c['element_id']

            # Save layout info
            results['_layout_info'] = 'double' if is_double_column else 'single'
            results['_ordered_elements'] = ordered_elems

        except Exception as e:
            logger.warning(f"analyze_image failed: {str(e)}")
            raise

        return results

    def analyze_layouts(self,
                        image_paths: List[Path],
                        pdf_labeled_dir: Path,
                        debug: bool = False
                        ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze all pages while maintaining consistent sections across pages.
        Then, cluster Title bounding boxes across *all pages* to identify
        multi-page chapter/subchapter structure.
        """
        page_results = []
        doc_title = None
        current_section = None

        for page_num, img_path in enumerate(image_paths, start=1):
            logger.info(f"\nProcessing Page {page_num}...")
            page_result = self.analyze_image(
                img_path,
                page_num,
                pdf_labeled_dir,
                debug=debug,
                doc_title=doc_title,
                prev_page_section=current_section
            )
            # Update doc_title/section if discovered
            if '_ordered_elements' in page_result and page_result['_ordered_elements']:
                elems = page_result['_ordered_elements']
                # If we haven't set doc_title yet, pick the first Title in reading order
                if not doc_title:
                    titles = [e for e in elems if e['type'] == 'Title']
                    if titles:
                        doc_title = titles[0]['element_id']
                current_section = elems[-1].get('section')

            page_results.append(page_result)

        # After analyzing all pages, we have page-wise results with Title bounding boxes.
        # We'll do a cross-page structure analysis to mark heading levels (e.g., chapters, subchapters).
        # 1) Gather all Title elements from all pages
        all_titles = []
        for pr in page_results:
            if '_ordered_elements' in pr:
                for elem in pr['_ordered_elements']:
                    if elem['type'] == 'Title':
                        all_titles.append(elem)

        # 2) Cluster these Title bounding boxes to identify levels
        if len(all_titles) > 1:
            self._cluster_chapter_subchapter_titles(all_titles, debug=debug)

        # Remove the _layout_info / _ordered_elements if not needed
        for pr in page_results:
            if '_layout_info' in pr:
                del pr['_layout_info']
            if '_ordered_elements' in pr:
                del pr['_ordered_elements']

        return page_results

    def _cluster_chapter_subchapter_titles(self, title_elements: List[Dict[str, Any]], debug: bool = False):
        """
        Identify multi-level headings across pages. For example:
        - Larger bounding boxes (area) => Chapter
        - Medium => Subchapter
        - Smaller => Section

        We'll do KMeans(n=3) or fallback to DBSCAN if silhouette is low.
        Then assign "heading_level" = 1,2,3 based on cluster.
        """
        # Build feature array (area, maybe width)
        feats = []
        for t in title_elements:
            coords = t['coordinates']
            w = float(coords['x2'] - coords['x1'])
            h = float(coords['y2'] - coords['y1'])
            area = w * h
            feats.append([area, w])

        feats_np = np.array(feats, dtype=np.float32)
        if len(feats_np) < 3:
            # If we have only 1-2 title boxes total, not enough for multi-level
            # => all become level=1
            for t in title_elements:
                t['heading_level'] = 1
            return

        # Try KMeans(n=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels_km = kmeans.fit_predict(feats_np)

        if len(feats_np) > 3:
            score_km = silhouette_score(feats_np, labels_km)
        else:
            score_km = 1.0  # trivial if only 3 items

        if debug:
            logger.info(f"Cross-page Title Clustering => KMeans(3) silhouette={score_km:.2f}")

        if score_km < 0.2:
            # fallback to DBSCAN
            if debug:
                logger.info("Silhouette too low => fallback to DBSCAN for Title clustering.")
            area_min, area_max = float(np.min(feats_np[:, 0])), float(np.max(feats_np[:, 0]))
            arange = area_max - area_min
            if arange <= 1.0:
                # all same => level=1
                for t in title_elements:
                    t['heading_level'] = 1
                return

            eps_guess = arange / 4.0
            db = DBSCAN(eps=eps_guess, min_samples=2)
            db_labels = db.fit_predict(feats_np)
            valid_labels = set(db_labels) - {-1}
            # We can define up to 3 levels max
            # Sort each cluster by mean area descending => level 1 = largest, level 2 = medium, level 3 = small
            cluster_areas = {}
            for lbl in valid_labels:
                indices = np.where(db_labels == lbl)[0]
                mean_area = float(np.mean(feats_np[indices, 0]))
                cluster_areas[lbl] = mean_area
            sorted_clusters = sorted(valid_labels, key=lambda c: cluster_areas[c], reverse=True)

            # Assign heading_level by sorted order
            level_map = {}
            current_level = 1
            for sc in sorted_clusters:
                level_map[sc] = current_level
                current_level += 1
                if current_level > 3:
                    current_level = 3  # cap at 3

            # apply
            for i, t in enumerate(title_elements):
                lbl = db_labels[i]
                if lbl in level_map:
                    t['heading_level'] = level_map[lbl]
                else:
                    # noise => smallest
                    t['heading_level'] = 3
            return
        else:
            # KMeans was good => we have 3 clusters
            # compute mean area in each cluster => largest => heading_level=1, etc.
            cluster_areas = {}
            for cl in [0, 1, 2]:
                idxs = np.where(labels_km == cl)[0]
                mean_area = float(np.mean(feats_np[idxs, 0]))
                cluster_areas[cl] = mean_area
            sorted_clusters = sorted(cluster_areas.keys(), key=lambda c: cluster_areas[c], reverse=True)

            # top cluster => level1, next => level2, last => level3
            level_map = {}
            level_map[sorted_clusters[0]] = 1
            if len(sorted_clusters) > 1:
                level_map[sorted_clusters[1]] = 2
            if len(sorted_clusters) > 2:
                level_map[sorted_clusters[2]] = 3

            for i, t in enumerate(title_elements):
                lbl = labels_km[i]
                if lbl in level_map:
                    t['heading_level'] = level_map[lbl]
                else:
                    t['heading_level'] = 3  # fallback
            return

    def analyze_all_pages(self,
                          image_paths: List[Path],
                          pdf_labeled_dir: Path,
                          debug: bool = False
                          ) -> List[Dict[str, Any]]:
        """
        High-level method to process all pages in reading order.
        1) analyze_layouts => produce page_results with bounding boxes, reading order, etc.
        2) each page may have different columns or no columns
        3) after cross-page Title clustering, we can enhance final structure
        """
        all_page_results = self.analyze_layouts(image_paths, pdf_labeled_dir, debug=debug)
        # Return page-wise results (where Title elements might have 'heading_level' assigned).
        return all_page_results


if __name__ == "__main__":
    from pathlib import Path
    # Example usage with three pages (possibly with different columns)
    image_paths = [
        Path('/path/to/acl_page_1.png'),
        Path('/path/to/acl_page_2.png'),
        Path('/path/to/resume_page.png')
    ]
    output_dir = Path('./labeled_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = LayoutAnalyzer(use_gpu=True)
    all_results = analyzer.analyze_all_pages(image_paths, output_dir, debug=True)

    print("=== Analysis Complete ===")
    for i, page_dict in enumerate(all_results, start=1):
        print(f"\n--- Page {i} ---")
        for cat, elems in page_dict.items():
            if cat.startswith('_'):
                continue
            print(f"{cat}: {len(elems)} elements")
            for e in elems:
                level_info = e.get('heading_level', 'N/A')
                print(f"   ID={e['element_id']}, coords={e['coordinates']}, page={e['page_num']}, heading_level={level_info}")
