"""
Expense Analyzer
----------------
Reads a Cash Book (PDF or Excel) and prints a total spent per category.

FIX (2026-07-19): The PDF path previously returned wrong / ungrouped totals
because the source PDF renders its category text ("ICICI Credit Card",
"SBI Card", "PazCare", etc.) in a decorative font that was flattened to
vector outlines instead of embedded as real text. pdfplumber therefore read
those cells as blank, and the script's fallback labeled each row
"Entry <n>" -- so the same category (e.g. two ICICI Credit Card charges)
was never grouped together.

The fix below tries normal text extraction first (fast + exact), and only
falls back to OCR (rendering that specific cell as an image and reading it
visually) when a cell truly has no embedded text. This keeps things fast for
normal PDFs and correctly recovers category names from decorative-font ones.

Extra dependencies vs. the original script:
    pip install pytesseract --break-system-packages
    (the tesseract-ocr binary must also be installed on the system, e.g.
     `sudo apt install tesseract-ocr` on Ubuntu/Debian)
"""

import os
import re
import pandas as pd
import pdfplumber
import pytesseract
from PIL import ImageOps

# Point pytesseract to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- OCR settings -----------------------------------------------------------
OCR_DPI = 600          # higher DPI = much better OCR accuracy on small cells
OCR_PAD_PX = 5         # small padding so glyphs aren't clipped at cell edges
OCR_CONFIG = "--psm 13"  # treat each cropped cell as one raw line of text


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
    # strip stray OCR noise characters (|, —, -, etc.) from the ends only
    text = re.sub(r'^[\|\-\u2014_~"\']+\s*', "", text)
    text = re.sub(r'\s*[\|\-\u2014_~"\']+$', "", text)
    return text.strip()


def extract_text_by_position(page, bbox):
    """Extract real embedded text whose character centers fall inside bbox.

    More reliable than pdfplumber's crop()/within_bbox(), which can either
    bleed in text from neighboring rows or drop characters that slightly
    overhang a cell's border.
    """
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
    """Rasterize just this cell and read it visually with Tesseract.

    Used only when a cell has no real embedded text (e.g. a decorative
    font that was flattened into vector artwork rather than real glyphs).
    """
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
    note_idx, cash_out_idx = 2, 4  # sensible defaults if headers can't be read

    note_keywords = ["note", "particular", "description", "detail", "remark", "category"]
    out_keywords = ["cash out", "out", "payment", "withdrawal"]

    for i, cell in enumerate(header_texts):
        low = cell.lower()
        if any(kw in low for kw in note_keywords):
            note_idx = i
        if any(kw in low for kw in out_keywords):
            cash_out_idx = i

    return note_idx, cash_out_idx


def extract_data_from_pdf(pdf_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            if not tables:
                continue

            # Render the page once per page (not once per cell) for speed.
            page_image = page.to_image(resolution=OCR_DPI)
            scale = OCR_DPI / 72

            for table in tables:
                rows = table.rows
                if len(rows) < 2:
                    continue  # need at least a header + one data row

                header_texts = [
                    get_cell_text(page, page_image, cell, scale)
                    for cell in rows[0].cells
                ]
                note_idx, cash_out_idx = detect_columns(header_texts)
                max_idx = max(note_idx, cash_out_idx)
                if max_idx >= len(rows[0].cells):
                    continue  # this table doesn't have the columns we need

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


def extract_data_from_excel(excel_path):
    df = pd.read_excel(excel_path)
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


def process_file():
    print("-" * 50)
    file_path = input("Enter the name of your file (e.g., Cash Book 19-Jul-2026.pdf): ").strip()

    file_path = file_path.strip("\"'")

    if not os.path.exists(file_path):
        for ext in (".pdf", ".xlsx", ".xls"):
            candidate = file_path if file_path.lower().endswith(ext) else file_path + ext
            if os.path.exists(candidate):
                file_path = candidate
                break

    if not os.path.exists(file_path):
        print(f"\nError: Could not find '{file_path}'. Make sure it is in the same folder as this script.")
        return

    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            df = extract_data_from_pdf(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = extract_data_from_excel(file_path)
        else:
            print("\nError: Unsupported file format. Please use PDF or Excel.")
            return

        if df.empty:
            print("\nNo transaction data found in the file.")
            return

        summary = df.groupby("Category")["Amount"].sum().reset_index()
        total_spent = summary["Amount"].sum()
        summary = summary.sort_values(by="Amount", ascending=False)

        print("\n" + "=" * 40)
        print("          EXPENSE BREAKDOWN          ")
        print("=" * 40)
        print(f"{'Category / Note':<25} | {'Amount Spent':<10}")
        print("-" * 40)

        for _, row in summary.iterrows():
            print(f"{row['Category']:<25} | {row['Amount']:<10.2f}")

        print("-" * 40)
        print(f"{'TOTAL':<25} | {total_spent:<10.2f}")
        print("=" * 40)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    process_file()