
import streamlit as st
import os
import re
import cv2
import fitz  # PyMuPDF
import pytesseract
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
import tempfile
import zipfile
import io

class PDFExtractor:
    def __init__(self):
        # Configuration (same as original)
        self.config = {
            'dpi': 400,
            'min_area_ratio': 0.02,
            'max_area_ratio': 0.96,
            'min_width_px': 200,
            'min_height_px': 220,
            'inset_px': 6,
            'stitch': {
                'y_tol': 60,
                'h_tol': 120,
                'x_tol': 60,
                'w_tol': 120,
            },
            'caption_regex': r"^\s*(?:Figure|Fig\.?|Panel|Table)\s*[\dA-Za-z\-\.]*",
            'ocr_lang': 'eng',
            'rotate_on_demand': False,
            'debug_mode': False,
            'max_caption_search_pages_ahead': 1,
        }
        self.setup_tesseract()
        
    def setup_tesseract(self):
        """Try to find Tesseract executable"""
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            'tesseract'  # If in PATH
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path) or path == 'tesseract':
                    pytesseract.pytesseract.tesseract_cmd = path
                    # Test if it works
                    test_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
                    pytesseract.image_to_string(test_img)
                    return True
            except:
                continue
        return False
            
    def process_single_pdf(self, pdf_path: str, out_dir: str):
        """Process a single PDF file (adapted from original code)"""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        os.makedirs(out_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise Exception(f"Error opening PDF: {e}")

        detections_by_page = []
        total_pages = len(doc)

        # Progress tracking for Streamlit
        if hasattr(self, 'progress_callback'):
            self.progress_callback(f"Analyzing {total_pages} pages...")
        
        for pno, page in enumerate(doc):
            img = self.render_page_to_bgr(page, self.config['dpi'])
            boxes, _ = self.detect_boxes_on_image(
                img,
                min_area_ratio=self.config['min_area_ratio'],
                max_area_ratio=self.config['max_area_ratio'],
                min_w=self.config['min_width_px'],
                min_h=self.config['min_height_px'],
                inset_px=self.config['inset_px'],
                debug_overlay=self.config['debug_mode'],
            )
            for b in boxes:
                b['page'] = pno
            detections_by_page.append(boxes)
            if hasattr(self, 'progress_callback'):
                self.progress_callback(f"  - Page {pno+1}: {len(boxes)} region(s)")

        doc.close()

        self.classify_boxes_with_ocr(detections_by_page, self.config['ocr_lang'])
        figures = self.stitch_split_figures(detections_by_page)
        self.save_results(figures, detections_by_page, out_dir)

    # Original algorithm methods (adapted for the class)
    def render_page_to_bgr(self, page: fitz.Page, dpi: int) -> np.ndarray:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def detect_boxes_on_image(self, img: np.ndarray, min_area_ratio: float, max_area_ratio: float, 
                            min_w: int, min_h: int, inset_px: int, debug_overlay: bool = False
                            ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        H, W = img.shape[:2]
        page_area = W * H

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Dict] = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < 80:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            if w < min_w or h < min_h:
                continue
            area = w * h
            area_ratio = area / page_area
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue
            if (w / (h + 1e-6) > 12) or (h / (w + 1e-6) > 12):
                continue

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)

            def edge_present(slice_arr: np.ndarray) -> bool:
                if slice_arr.size == 0:
                    return False
                return (np.mean(slice_arr) > 20)

            edge_thickness = 8
            top_slice = mask[y:y+edge_thickness, x:x+w] if y+edge_thickness < H else mask[y:H, x:x+w]
            bottom_slice = mask[max(0, y+h-edge_thickness):y+h, x:x+w]
            left_slice = mask[y:y+h, x:x+edge_thickness] if x+edge_thickness < W else mask[y:y+h, x:W]
            right_slice = mask[y:y+h, max(0, x+w-edge_thickness):x+w]

            top_edge = edge_present(top_slice)
            bottom_edge = edge_present(bottom_slice)
            left_edge = edge_present(left_slice)
            right_edge = edge_present(right_slice)

            open_sides = []
            if not top_edge: open_sides.append("top")
            if not bottom_edge: open_sides.append("bottom")
            if not left_edge: open_sides.append("left")
            if not right_edge: open_sides.append("right")

            x1 = max(0, x + inset_px)
            y1 = max(0, y + inset_px)
            x2 = min(W, x + w - inset_px)
            y2 = min(H, y + h - inset_px)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2].copy()

            box = {
                'coords': (x, y, w, h),
                'image': crop,
                'open_sides': open_sides,
                'area_ratio': float(area_ratio),
            }

            boxes.append(box)

        boxes.sort(key=lambda b: (b['coords'][1], b['coords'][0]))
        return boxes, None

    def ocr_text(self, image: np.ndarray, lang: str) -> str:
        try:
            txt = pytesseract.image_to_string(image, lang=lang)
        except Exception:
            txt = ""
        return (txt or "").strip()

    def classify_boxes_with_ocr(self, detections_by_page: List[List[Dict]], lang: str) -> None:
        caption_re = re.compile(self.config['caption_regex'], re.IGNORECASE)
        
        jobs = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            for p_idx, page_boxes in enumerate(detections_by_page):
                for b_idx, box in enumerate(page_boxes):
                    jobs.append(((p_idx, b_idx), ex.submit(self.ocr_text, box['image'], lang)))

            for (p_idx, b_idx), fut in jobs:
                text = fut.result() or ""
                box = detections_by_page[p_idx][b_idx]
                if caption_re.match(text):
                    box['type'] = 'caption'
                    box['text'] = text
                else:
                    box['type'] = 'figure'
                    box['text'] = text

    def stitch_split_figures(self, detections_by_page: List[List[Dict]]) -> List[Dict]:
        # Mark boxes with IDs and stitch flags
        for p_idx, page_boxes in enumerate(detections_by_page):
            for b_idx, box in enumerate(page_boxes):
                box['id'] = f"p{p_idx+1}_b{b_idx+1}"
                box['used_for_stitch'] = False

        figures: List[Dict] = []

        for p_idx, page_boxes in enumerate(detections_by_page):
            for b_idx, box in enumerate(page_boxes):
                if box.get('type') == 'caption':
                    continue
                if box['used_for_stitch']:
                    continue

                cur_img = box['image']
                cur_coords = box['coords']
                pages = [p_idx]
                bbox_refs = [(p_idx, b_idx)]
                box['used_for_stitch'] = True

                np_idx = p_idx + 1
                candidate = None
                if np_idx < len(detections_by_page):
                    for nb_idx, nb in enumerate(detections_by_page[np_idx]):
                        if nb.get('type') == 'caption' or nb['used_for_stitch']:
                            continue
                        x, y, w, h = cur_coords
                        nx, ny, nw, nh = nb['coords']

                        if abs(x - nx) < 50 and abs((x+w) - (nx+nw)) < 50:
                            candidate = (np_idx, nb_idx, nb, 'vertical')
                            break
                        if abs(y - ny) < 50 and abs((y+h) - (ny+nh)) < 50:
                            candidate = (np_idx, nb_idx, nb, 'horizontal')
                            break

                if candidate:
                    np_idx, nb_idx, nb, stitch_type = candidate
                    nb['used_for_stitch'] = True
                    pages.append(np_idx)
                    bbox_refs.append((np_idx, nb_idx))

                    if stitch_type == 'vertical':
                        w_max = max(cur_img.shape[1], nb['image'].shape[1])

                        def pad_to_width(img, target_w):
                            pad_w = target_w - img.shape[1]
                            if pad_w <= 0:
                                return img
                            return np.pad(img, ((0,0),(0,pad_w),(0,0)), 
                                          mode="constant", constant_values=255)

                        cur_img = pad_to_width(cur_img, w_max)
                        nb_img = pad_to_width(nb['image'], w_max)
                        cur_img = np.vstack([cur_img, nb_img])

                        x1 = min(cur_coords[0], nb['coords'][0])
                        y1 = min(cur_coords[1], nb['coords'][1])
                        x2 = max(cur_coords[0]+cur_coords[2], nb['coords'][0]+nb['coords'][2])
                        y2 = max(cur_coords[1]+cur_coords[3], nb['coords'][1]+nb['coords'][3])
                        cur_coords = (x1, y1, x2-x1, y2-y1)

                    else:  # horizontal
                        h_max = max(cur_img.shape[0], nb['image'].shape[0])

                        def pad_to_height(img, target_h):
                            pad_h = target_h - img.shape[0]
                            if pad_h <= 0:
                                return img
                            return np.pad(img, ((0,pad_h),(0,0),(0,0)), 
                                          mode="constant", constant_values=255)

                        cur_img = pad_to_height(cur_img, h_max)
                        nb_img = pad_to_height(nb['image'], h_max)
                        cur_img = np.hstack([cur_img, nb_img])

                        x1 = min(cur_coords[0], nb['coords'][0])
                        y1 = min(cur_coords[1], nb['coords'][1])
                        x2 = max(cur_coords[0]+cur_coords[2], nb['coords'][0]+nb['coords'][2])
                        y2 = max(cur_coords[1]+cur_coords[3], nb['coords'][1]+nb['coords'][3])
                        cur_coords = (x1, y1, x2-x1, y2-y1)

                figures.append({
                    'id': f"f{len(figures)+1:03d}",
                    'pages': pages,
                    'image': cur_img,
                    'bbox_refs': bbox_refs,
                    'base_page': pages[0],
                    'coords_hint': cur_coords,
                })

        return figures

    def pick_best_caption_for_figure(self, fig: Dict, detections_by_page: List[List[Dict]], 
                                   used_caption_ids: set) -> Optional[Tuple[int, int, Dict]]:
        base_p = fig['base_page']
        x, y, w, h = fig['coords_hint']

        max_ahead = self.config['max_caption_search_pages_ahead']
        candidates = []
        for p in range(base_p, min(base_p + 1 + max_ahead, len(detections_by_page))):
            for b_idx, box in enumerate(detections_by_page[p]):
                if box.get('type') != 'caption':
                    continue
                if box.get('caption_used_id'):
                    continue
                bx, by, bw, bh = box['coords']
                same_page = (p == base_p)
                after_figure = (not same_page) or (by >= y)
                if not after_figure:
                    continue
                vdist = abs((by) - (y + h)) if same_page else 0
                wdiff = abs(bw - w)
                score = vdist + 0.5 * wdiff
                candidates.append((score, p, b_idx, box))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        for _, p, b_idx, box in candidates:
            box_id = (p, b_idx)
            if box_id not in used_caption_ids:
                return (p, b_idx, box)
        return None

    def rotate_if_needed(self, img: np.ndarray) -> np.ndarray:
        if not self.config['rotate_on_demand']:
            return img
        h, w = img.shape[:2]
        if h > w * 1.2:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def save_results(self, figures: List[Dict], detections_by_page: List[List[Dict]], out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        used_captions = set()
        saved = 0

        for fig in figures:
            cap = self.pick_best_caption_for_figure(fig, detections_by_page, used_captions)
            if cap is not None:
                p, b_idx, cap_box = cap
                used_captions.add((p, b_idx))
                fig_img = fig['image']
                cap_img = cap_box['image']
                if cap_img.shape[1] != fig_img.shape[1]:
                    new_h = int(cap_img.shape[0] * (fig_img.shape[1] / cap_img.shape[1]))
                    cap_img = cv2.resize(cap_img, (fig_img.shape[1], new_h))
                stitched = cv2.vconcat([fig_img, cap_img])
                stitched = self.rotate_if_needed(stitched)
                fname = f"figure_with_caption_{fig['id']}.png"
                cv2.imwrite(os.path.join(out_dir, fname), stitched)
                saved += 1
            else:
                fig_img = self.rotate_if_needed(fig['image'])
                fname = f"figure_{fig['id']}.png"
                cv2.imwrite(os.path.join(out_dir, fname), fig_img)
                saved += 1

        cap_count = 0
        for p_idx, page_boxes in enumerate(detections_by_page):
            for b_idx, box in enumerate(page_boxes):
                if box.get('type') == 'caption' and (p_idx, b_idx) not in used_captions:
                    cap_count += 1
                    cv2.imwrite(os.path.join(out_dir, f"standalone_caption_{cap_count:03d}.png"), box['image'])

        if hasattr(self, 'progress_callback'):
            self.progress_callback(f"Saved {saved} figure image(s) (+ any standalone captions) to: {out_dir}")

def main():
    st.set_page_config(
        page_title="PDF Figure Extractor",
        page_icon="📄",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
        }
        .error-box {
            padding: 1rem;
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            background-color: #d1ecf1;
            border-left: 5px solid #17a2b8;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-title">📄 PDF Figure Extractor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize extractor in session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = PDFExtractor()
        tesseract_found = st.session_state.extractor.setup_tesseract()
        if not tesseract_found:
            st.info("ℹ️ **Tesseract OCR not detected.** "
                   "Caption detection will be limited. "
                   "For local development, install Tesseract from: "
                   "https://github.com/UB-Mannheim/tesseract/wiki")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        dpi = st.slider(
            "Image Quality (DPI)",
            min_value=150,
            max_value=600,
            value=400,
            step=50,
            help="Higher DPI means better quality but slower processing"
        )
        
        rotate_images = st.checkbox(
            "Auto-rotate tall images",
            value=False,
            help="Automatically rotate images that are taller than they are wide"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool extracts figures and captions from PDF files using:
        - **Computer Vision** for figure detection
        - **OCR** for caption recognition
        - **Smart Stitching** for multi-page figures
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="section-title">1️⃣ Upload PDF Files</h3>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to extract figures from"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} PDF file(s) selected")
            for i, file in enumerate(uploaded_files, 1):
                st.text(f"  {i}. {file.name}")
        else:
            # Show welcome message when no files uploaded
            st.info("""
            👋 **Welcome!** Upload your PDF files to get started.
            
            This tool will:
            - 🔍 Detect figures, charts, and diagrams
            - 📝 Extract and match captions
            - 🔄 Stitch multi-page figures
            - 💾 Package everything for easy download
            """)
    
    with col2:
        st.markdown('<h3 class="section-title">2️⃣ Process</h3>', unsafe_allow_html=True)
        
        process_button = st.button(
            "🚀 Extract Figures",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True
        )
    
    # Processing section
    if process_button and uploaded_files:
        st.markdown("---")
        st.markdown('<h3 class="section-title">📊 Processing Status</h3>', unsafe_allow_html=True)
        
        # Update config
        st.session_state.extractor.config['dpi'] = dpi
        st.session_state.extractor.config['rotate_on_demand'] = rotate_images
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def log_callback(message):
            pass  # Silent processing
        
        st.session_state.extractor.progress_callback = log_callback
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            total_files = len(uploaded_files)
            all_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = i / total_files
                progress_bar.progress(progress)
                status_text.markdown(f"**Processing:** {uploaded_file.name} ({i+1}/{total_files})")
                
                # Save uploaded file temporarily
                temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create output directory for this PDF
                pdf_name = os.path.splitext(uploaded_file.name)[0]
                out_dir = os.path.join(temp_dir, pdf_name)
                
                try:
                    st.session_state.extractor.process_single_pdf(temp_pdf_path, out_dir)
                    
                    # Collect results
                    if os.path.exists(out_dir):
                        for filename in os.listdir(out_dir):
                            if filename.endswith('.png'):
                                filepath = os.path.join(out_dir, filename)
                                all_results.append((pdf_name, filename, filepath))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.markdown("**✅ Processing completed!**")
            
            # Display results
            if all_results:
                st.markdown("---")
                st.markdown('<h3 class="section-title">🎉 Extraction Results</h3>', unsafe_allow_html=True)
                st.success(f"Successfully extracted {len(all_results)} figure(s) from {total_files} PDF(s)")
                
                # Group by PDF
                results_by_pdf = {}
                for pdf_name, filename, filepath in all_results:
                    if pdf_name not in results_by_pdf:
                        results_by_pdf[pdf_name] = []
                    results_by_pdf[pdf_name].append((filename, filepath))
                
                # Display results by PDF with auto-expanded previews
                for pdf_name, files in results_by_pdf.items():
                    st.markdown(f"### 📄 {pdf_name} ({len(files)} figures)")
                    
                    # Display images in columns
                    cols = st.columns(3)
                    for idx, (filename, filepath) in enumerate(files):
                        with cols[idx % 3]:
                            st.image(filepath, caption=filename, use_container_width=True)
                
                # Create download button for all results (placed after previews)
                st.markdown("---")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for pdf_name, filename, filepath in all_results:
                        arcname = f"{pdf_name}/{filename}"
                        zip_file.write(filepath, arcname)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download All Figures (ZIP)",
                    data=zip_buffer,
                    file_name="extracted_figures.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.warning("No figures were extracted. The PDFs may not contain detectable figures.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>Made with ❤️ using Streamlit | 
            <a href='https://github.com' target='_blank'>GitHub</a> | 
            Need help? Check the processing log for details</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()