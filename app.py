# app.py
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import camelot
import json
import tempfile
import os
import re
import base64
from io import BytesIO
from statistics import mean, pstdev

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3})?[\s\-.(]*(\d{2,4})[\s\-.)]*(\d{3,4})[\s\-]*(\d{3,4})")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
CIN_RE = re.compile(r"\bCIN\b.*", flags=re.IGNORECASE)

def image_bytes_to_base64(img_bytes, mime="image/png"):
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def detect_headings(spans):
    """
    Heuristic detection of sections/subsections using font sizes in spans.
    spans: list of (text, size, flags, font)
    Returns thresholds (section_threshold, subsection_threshold)
    """
    sizes = [s for (_, s, _, _) in spans if s > 0]
    if not sizes:
        return (16, 12)
    avg = mean(sizes)
    sd = pstdev(sizes) if len(sizes) > 1 else 0
    # Section threshold: avg + 1*sd or at least 14
    section_t = max(14, avg + sd)
    subsection_t = max(11, avg)
    return (section_t, subsection_t)

def classify_footer_and_signature(lines):
    """
    Given list of lines (strings) attempt to classify footer, signature, or normal.
    Returns (type, combined_text) where type in {"footer","signature","paragraph"}.
    """
    combined = "\n".join(lines).strip()
    # Look for signature clues
    if any(x in combined.lower() for x in ["yours sincerely", "yours faithfully", "for "]) or re.search(r"\b(dean|director|manager|ceo|coo)\b", combined.lower()):
        return "signature", combined
    if EMAIL_RE.search(combined) or URL_RE.search(combined) or PHONE_RE.search(combined) or CIN_RE.search(combined):
        return "footer", combined
    return "paragraph", combined

def extract_images_from_page(page, embed_images):
    """
    Extract images from a PyMuPDF page.
    Returns list of dicts: {"type":"chart","description":...,"image_b64":...}
    """
    imgs = []
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n - pix.alpha >= 4:  # e.g., CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")

            img_entry = {
                "type": "chart",
                "description": f"Image {img_index} on page {page.number + 1}",
            }
            if embed_images:
                img_entry["image_b64"] = image_bytes_to_base64(img_bytes, mime="image/png")
            imgs.append(img_entry)

            pix = None  # free memory
        except Exception as e:
            print(f"⚠️ Could not extract image {img_index} on page {page.number+1}: {e}")
            continue
    return imgs


def ocr_image_bytes(img_b64):
    """
    If OCR available, decode base64 and run OCR to extract text.
    Returns OCR text or None.
    """
    if not OCR_AVAILABLE:
        return None
    header, data = img_b64.split(",", 1)
    img_bytes = base64.b64decode(data)
    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    text = pytesseract.image_to_string(im)
    return text.strip()

def extract_pdf_content(pdf_path, embed_images=False, do_ocr_images=False):
    """
    Main extraction pipeline:
    - Uses PyMuPDF for text with spans/size metadata (section/subsection detection)
    - Uses Camelot for tables
    - Detects images and optionally embeds them
    - Classifies signature/footer blocks
    """
    result = {"pages": []}
    doc = fitz.open(pdf_path)
    # Pre-open pdfplumber for alternate text extraction if needed
    plumber_doc = pdfplumber.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        page_entry = {"page_number": page_number, "content": []}

        # --- Collect spans for heuristics ---
        # each span: (text, size, flags, font)
        spans = []
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    font = span.get("font", "")
                    if text:
                        spans.append((text, size, flags, font))

        section_t, subsection_t = detect_headings(spans)

        # --- Walk blocks and create paragraphs or headings ---
        current_section = None
        current_subsection = None
        # We'll group by block for better paragraph sense
        for block in blocks:
            if "lines" not in block:
                continue
            block_lines = []
            # For each line, decide if it's heading/subheading/paragraph
            for line in block["lines"]:
                # join spans of the line preserving style info
                line_spans = line.get("spans", [])
                if not line_spans:
                    continue
                # Determine the largest font size in the line
                sizes = [s.get("size", 0) for s in line_spans if s.get("text", "").strip()]
                if not sizes:
                    continue
                max_size = max(sizes)
                text_line = " ".join(s.get("text", "").strip() for s in line_spans).strip()
                if not text_line:
                    continue

                # Heading heuristics
                if max_size >= section_t and (text_line.isupper() or len(text_line.split()) <= 6):
                    # Section heading
                    current_section = text_line
                    current_subsection = None
                    page_entry["content"].append({
                        "type": "section",
                        "section": current_section,
                        "sub_section": None,
                        "text": None
                    })
                elif max_size >= subsection_t and (len(text_line.split()) <= 8):
                    current_subsection = text_line
                    page_entry["content"].append({
                        "type": "sub_section",
                        "section": current_section,
                        "sub_section": current_subsection,
                        "text": None
                    })
                else:
                    block_lines.append(text_line)

            if block_lines:
                # Try to classify block (footer/signature) heuristics
                btype, combined = classify_footer_and_signature(block_lines)
                if btype == "signature":
                    page_entry["content"].append({
                        "type": "signature",
                        "section": current_section,
                        "sub_section": current_subsection,
                        "text": combined
                    })
                elif btype == "footer":
                    page_entry["content"].append({
                        "type": "footer",
                        "section": current_section,
                        "sub_section": current_subsection,
                        "text": combined
                    })
                else:
                    # regular paragraph
                    page_entry["content"].append({
                        "type": "paragraph",
                        "section": current_section,
                        "sub_section": current_subsection,
                        "text": combined
                    })

        # --- Camelot tables for this page ---
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_number))
            for idx, table in enumerate(tables, start=1):
                table_data = table.df.values.tolist()
                page_entry["content"].append({
                    "type": "table",
                    "section": current_section,
                    "sub_section": current_subsection,
                    "description": f"Table {idx} on page {page_number}",
                    "table_data": table_data
                })
        except Exception:
            # camelot may raise when no tables or not supported; ignore
            pass

        # --- Images / Charts detection ---
        images = extract_images_from_page(page, embed_images)
        # If OCR on images requested, attempt to extract text
        if do_ocr_images and OCR_AVAILABLE:
            for img in images:
                if "image_b64" in img:
                    ocr_text = ocr_image_bytes(img["image_b64"])
                    if ocr_text:
                        img["ocr_text"] = ocr_text
        # Append images as chart entries
        for img in images:
            page_entry["content"].append(img)

        # If pdfplumber can find elements (fallback), add any missing text blocks (optional)
        # (Skipping to avoid duplication — pdfplumber often duplicates fitz results.)

        result["pages"].append(page_entry)

    plumber_doc.close()
    doc.close()
    return result

# ---------------- Streamlit App UI ----------------
st.set_page_config(page_title="PDF → Structured JSON (Robust)", layout="wide")
st.title("PDF Parsing and Structured JSON Extraction")

st.markdown(
    """
Upload a PDF and the app will:
- detect sections/subsections by font-size heuristics,
- extract paragraphs and group them,
- extract tables (Camelot),
- detect images/charts and optionally embed them (base64),
- identify signature/footer/contact blocks,
- optionally OCR text inside images (Tesseract required).
"""
)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    embed_images = st.checkbox("Embed images (base64) into JSON", value=False)
with col2:
    do_ocr_images = st.checkbox("Run OCR on images (pytesseract)", value=False)
with col3:
    pretty = st.checkbox("Pretty-print JSON preview", value=True)

if do_ocr_images and not OCR_AVAILABLE:
    st.warning("pytesseract or PIL not available in environment — OCR disabled. Install pytesseract and Tesseract engine.")

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info(f"Saved uploaded PDF to `{tmp_path}`")

    if st.button("Extract → JSON"):
        try:
            with st.spinner("Extracting..."):
                json_data = extract_pdf_content(tmp_path, embed_images=embed_images, do_ocr_images=do_ocr_images)

            st.success("Extraction complete ✅")

            # JSON preview
            if pretty:
                st.json(json_data)
            else:
                st.code(json.dumps(json_data, ensure_ascii=False))

            # Offer download of JSON
            json_bytes = json.dumps(json_data, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button("⬇️ Download JSON", data=json_bytes, file_name="extracted.json", mime="application/json")

            # If images embedded, show thumbnails (first page few)
            if embed_images:
                shown = 0
                st.write("Extracted Images (embedded):")
                for p in json_data["pages"]:
                    for content in p["content"]:
                        if content.get("type") == "chart" and content.get("image_b64"):
                            st.image(content["image_b64"], width=300)
                            shown += 1
                            if shown >= 6:
                                break
                    if shown >= 6:
                        break

        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.exception(e)

    # Cleanup temp file if desired (keep for debugging)
    # os.remove(tmp_path)
else:
    st.info("Upload a PDF to begin.")

st.markdown("---")
st.markdown("**Notes / Requirements**:")
st.markdown(
    """
- **Camelot** requires Ghostscript and a compatible environment (works best with Linux).
- **pytesseract** requires the Tesseract engine installed on your system.
- Embedding images as base64 increases JSON size considerably; disable embedding if you only need metadata.
- The heuristics (font-size thresholds, regexes) are conservative — you may need to tweak thresholds for certain document families.
"""
)
