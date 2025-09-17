## PDF → Structured JSON Extractor

Convert unstructured PDF files into well-structured JSON while preserving the hierarchical organization of the document.
This project goes beyond simple text extraction by identifying sections, subsections, paragraphs, tables, images, signatures, and footers.
APP

## APPLICATION LIVE LINK :  https://huggingface.co/spaces/lithish2602/PDF-Structured-JSON-Extractor 

## Features

 Hierarchical Parsing – Detects sections & subsections using font-size heuristics

 Paragraph Grouping – Preserves context within sections

 Table Extraction – Extracts tables into structured row–column arrays via Camelot
 
 Image & Chart Detection – Optionally embed images as base64 in JSON

 Footer & Signature Classification – Identifies emails, URLs, phone numbers, CIN, and signature blocks

 OCR Support (optional) – Extracts text from scanned PDFs/images using Tesseract OCR

 Streamlit UI – Upload PDFs, preview structured JSON, and download results

## Tech Stack

Python 3.9+

Streamlit
 – Web app interface

PyMuPDF (fitz)
 – Text, font size, and image extraction

pdfplumber
 – Alternate text extraction

Camelot
 – Table extraction

pytesseract
 + Pillow
 – OCR for scanned images

Regex – Detecting structured metadata (emails, URLs, phone numbers, CIN)


## Installation

Clone this repository:

git clone https://github.com/lithish262004/PDF-Structured-JSON-Extractor.git

cd <repo-folder>

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


## Usage

Run the Streamlit app:

streamlit run app.py


Open the link in your browser (default: http://localhost:8501).
