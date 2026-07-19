"""
Expense Analyzer - Web App
---------------------------
Upload a Cash Book (PDF or Excel) in the browser and see a total spent
per category. This is a Streamlit port of the original CLI script so it
can be deployed and used from a phone or any browser.

Core extraction logic (PDF text/OCR extraction, Excel parsing, column
detection) is unchanged from the original script. What changed:
  - input()/file-path based I/O -> st.file_uploader (works with in-memory
    uploaded files, no need to save to disk first)
  - print() output -> st.dataframe / st.metric / st.bar_chart
  - the tesseract_cmd path is now auto-detected (works on Linux servers,
    where Streamlit Community Cloud runs) instead of being hardcoded to
    the Windows Tesseract install path
"""

import os
import re
import shutil

import pandas as pd
import pdfplumber
import pytesseract
from PIL import ImageOps
import streamlit as st

# --- Point pytesseract to the Tesseract executable --------------------------
# On Streamlit Community Cloud (Linux) tesseract is installed via
# packages.txt and available on PATH. Fall back to the Windows default
# path only if nothing is found on PATH and we're actually on Windows,
# so this still works if you run the script locally on your own PC.
_tess_path = shutil.which("tesseract")
if _tess_path:
    pytesseract.pytesseract.tesseract_cmd = _tess_path
elif os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- OCR settings -------------------------------------------------------
OCR_DPI = 600
OCR_PAD_PX = 5
OCR_CONFIG = "--psm 13"


def parse_amount(value):
    if value is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(value).strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_cell(value):
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()
    text = re.sub(r'^[\|\-\u2014_~"\']+\s*', "", text)
    text = re.sub(r'\s*[\|\-\u2014_~"\']+$', "", text)
    return text.strip()


def extract_text_by_position(page, bbox):
    """Extract real embedded text whose character centers fall inside bbox."""
    x0, top, x1, bottom = bbox
    chars = [
        c for c in page.chars
        if x0 <= (c["x0"] + c["x1"]) / 2 <= x1
        and top <= (c["top"] + c["bottom"]) / 2 <= bottom
    ]
    if not chars:
        return ""
    chars.sort(key=lambda c: (round(c["top"], 1), c["x0"]))
    pieces, prev = [], None
    for c in chars:
        if prev is not None and (
            abs(c["top"] - prev["top"]) > 3 or c["x0"] - prev["x1"] > 2
        ):
            pieces.append(" ")
        pieces.append(c["text"])
        prev = c
    return normalize_cell("".join(pieces))


def ocr_cell(page_image, bbox, scale):
    """Rasterize just this cell and read it visually with Tesseract."""
    left, top, right, bottom = [coord * scale for coord in bbox]
    crop = page_image.original.crop(
        (left - OCR_PAD_PX, top - OCR_PAD_PX, right + OCR_PAD_PX, bottom + OCR_PAD_PX)
    ).convert("L")
    crop = ImageOps.autocontrast(crop)
    text = pytesseract.image_to_string(crop, config=OCR_CONFIG)
    return normalize_cell(text)


def get_cell_text(page, page_image, bbox, scale):
    if bbox is None:
        return ""
    text = extract_text_by_position(page, bbox)
    if text:
        return text
    return ocr_cell(page_image, bbox, scale)


def detect_columns(header_texts):
    note_idx, cash_out_idx = 2, 4

    note_keywords = ["note", "particular", "description", "detail", "remark", "category"]
    out_keywords = ["cash out", "out", "payment", "withdrawal"]

    for i, cell in enumerate(header_texts):
        low = cell.lower()
        if any(kw in low for kw in note_keywords):
            note_idx = i
        if any(kw in low for kw in out_keywords):
            cash_out_idx = i

    return note_idx, cash_out_idx


def extract_data_from_pdf(pdf_file):
    """pdf_file: a path OR a file-like object (e.g. an uploaded file)."""
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            if not tables:
                continue

            page_image = page.to_image(resolution=OCR_DPI)
            scale = OCR_DPI / 72

            for table in tables:
                rows = table.rows
                if len(rows) < 2:
                    continue

                header_texts = [
                    get_cell_text(page, page_image, cell, scale)
                    for cell in rows[0].cells
                ]
                note_idx, cash_out_idx = detect_columns(header_texts)
                max_idx = max(note_idx, cash_out_idx)
                if max_idx >= len(rows[0].cells):
                    continue

                for row in rows[1:]:
                    cells = row.cells
                    if len(cells) <= max_idx or cells[cash_out_idx] is None:
                        continue

                    cash_out_val = parse_amount(
                        get_cell_text(page, page_image, cells[cash_out_idx], scale)
                    )
                    if cash_out_val is None or cash_out_val == 0:
                        continue

                    note = get_cell_text(page, page_image, cells[note_idx], scale)
                    data.append({"Category": note or "Uncategorized", "Amount": cash_out_val})

    return pd.DataFrame(data)


def extract_data_from_excel(excel_file):
    """excel_file: a path OR a file-like object (e.g. an uploaded file)."""
    df = pd.read_excel(excel_file)
    df.columns = [str(col).strip().lower() for col in df.columns]

    note_col = next((col for col in df.columns if "note" in col), None)
    cash_out_col = next((col for col in df.columns if "out" in col), None)

    if not note_col or not cash_out_col:
        raise ValueError("Could not find 'Notes' or 'Cash Out' columns.")

    df = df.dropna(subset=[note_col, cash_out_col])
    df = df[~df[note_col].astype(str).str.contains("Total|Balance", case=False)]
    df["Amount"] = df[cash_out_col].astype(str).str.replace(r"[^\d.]", "", regex=True)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    return pd.DataFrame({"Category": df[note_col].astype(str).str.strip(), "Amount": df["Amount"]})


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Expense Analyzer", page_icon="💰", layout="centered")

st.title("💰 Expense Analyzer")
st.write(
    "Upload your Cash Book (PDF or Excel) and get a total spent per category. "
    "Works the same on your phone as on desktop."
)

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "xlsx", "xls"])

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    with st.spinner("Reading and analyzing your file... (PDFs with OCR can take a minute)"):
        try:
            if ext == ".pdf":
                df = extract_data_from_pdf(uploaded_file)
            elif ext in (".xlsx", ".xls"):
                df = extract_data_from_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a PDF or Excel file.")
                df = pd.DataFrame()
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            df = pd.DataFrame()

    if not df.empty:
        summary = df.groupby("Category")["Amount"].sum().reset_index()
        summary = summary.sort_values(by="Amount", ascending=False)
        total_spent = summary["Amount"].sum()

        st.subheader("Expense Breakdown")
        st.dataframe(
            summary.rename(columns={"Category": "Category / Note", "Amount": "Amount Spent"}),
            use_container_width=True,
            hide_index=True,
        )
        st.metric("Total Spent", f"{total_spent:,.2f}")
        st.bar_chart(summary.set_index("Category")["Amount"])
    else:
        st.warning("No transaction data found in the file.")
else:
    st.info("👆 Upload a Cash Book file to get started.")
