"""
extract_text.py

Step 1 of the book search pipeline: pulls raw text out of PDF files.

Handles two kinds of PDFs:
  - Native-text PDFs: text is extracted directly (fast, free, exact).
  - Scanned PDFs: pages are images with no text layer, so we fall back
    to OCR (Tesseract) to "read" the page image.

To keep OCR cheap, a fully-scanned book is NOT OCR'd page by page.
Instead we OCR a small representative sample (cover, table of
contents / intro pages, and a spread of pages through the body) —
enough for the embedding model to understand what the book is about,
without burning time OCRing every single page.

Exposes one function other files can import:
    extract_text_from_pdf(pdf_path) -> str

Can also be run directly from the command line — see main() at the
bottom for usage.
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import pymupdf          # reads PDF files, renders pages to images
import pytesseract       # Python wrapper around the Tesseract OCR engine
from PIL import Image    # converts a rendered page into an image OCR can read


# pytesseract calls the Tesseract program as an external process — it
# needs to know where that .exe lives. If Tesseract is on your system
# PATH this line isn't needed, but on Windows it usually isn't unless
# you added it manually during install, so point at it explicitly here.
# Adjust the path if you installed it somewhere else.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------------------------------------------------------------------
# Config — tweak these numbers as you learn more about your real books
# ---------------------------------------------------------------------------

OCR_LANGUAGE = "ara"          # Tesseract language pack to use
OCR_DPI = 300                 # higher = better OCR accuracy, slower to render
SCANNED_BOOK_THRESHOLD = 0.7  # if >=70% of pages are effectively empty, treat
                               # the whole book as scanned and switch to sampled OCR
SAMPLE_PAGE_COUNT = 15        # how many pages to OCR for a fully-scanned book
FRONT_MATTER_PAGES = 8        # of those, how many are "grab the first N pages"
                               # (cover/title/TOC/intro) vs. spread through the book
MIN_SUBSTANTIAL_CHARS = 40    # pages with fewer real characters than this are
                               # treated as "effectively empty" even if they
                               # technically have some text (e.g. a burned-in
                               # page number or watermark on an otherwise
                               # scanned page) — this is what catches books
                               # like "الحرب على الكسل" that have no literally
                               # empty pages but also no real extractable content


def get_sample_page_indices(total_pages: int, sample_size: int = SAMPLE_PAGE_COUNT) -> set[int]:
    """
    Decide which page numbers (0-indexed) to OCR for a fully-scanned book.

    Strategy: take the first few pages (cover, title page, table of
    contents, introduction usually live here and carry a lot of topical
    signal), then fill the rest of the sample with pages spread evenly
    through the remainder of the book for broader coverage.

    Args:
        total_pages: how many pages the book has
        sample_size: max number of pages to OCR

    Returns:
        A set of page indices to OCR.
    """
    if total_pages <= sample_size:
        # Short book — cheaper to just OCR all of it than to build a sample
        return set(range(total_pages))

    front_count = min(FRONT_MATTER_PAGES, sample_size)
    front_pages = set(range(front_count))

    remaining_slots = sample_size - len(front_pages)
    if remaining_slots > 0:
        # Evenly space the remaining sample across the rest of the book
        step = (total_pages - front_count) / remaining_slots
        spaced_pages = {
            front_count + int(i * step) for i in range(remaining_slots)
        }
    else:
        spaced_pages = set()

    return front_pages | spaced_pages


def ocr_page(page, dpi: int = OCR_DPI, lang: str = OCR_LANGUAGE) -> str:
    """
    Render a single PDF page to an image and run Tesseract OCR on it.

    This is the expensive operation in the whole pipeline — rendering
    at high DPI and running OCR takes real time per page, which is
    exactly why we only do this for a sample of pages, not every page.
    """
    # Render the page as a pixel image at the given resolution
    pix = page.get_pixmap(dpi=dpi)

    # Convert PyMuPDF's raw pixmap into a PIL Image, which is what
    # pytesseract expects as input
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Run the actual OCR. lang="ara" tells Tesseract to use the Arabic
    # trained model rather than English.
    text = pytesseract.image_to_string(img, lang=lang)

    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and return its text as a single string, using OCR
    automatically wherever native text extraction comes up empty.

    Args:
        pdf_path: path to the .pdf file on disk

    Returns:
        Extracted (and/or OCR'd) text, pages joined by a separator.
        For fully-scanned books, only a sample of pages is included
        (see get_sample_page_indices) — this is intentional, not a bug.

    Raises:
        FileNotFoundError: if pdf_path doesn't exist
        RuntimeError: if the PDF can't be opened at all (corrupted,
            password-protected, etc.)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"No PDF found at: {pdf_path}")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF '{pdf_path.name}': {e}")

    total_pages = doc.page_count
    pages_text = [""] * total_pages
    effectively_empty_indices = []

    # --- Pass 1: try normal (free, fast) text extraction on every page ---
    for i, page in enumerate(doc):
        text = page.get_text()
        pages_text[i] = text
        # Treat a page as needing OCR if it has no text at all, OR if it
        # has only a trivial amount (a page number, a watermark) — real
        # body content is virtually always well over MIN_SUBSTANTIAL_CHARS
        # characters, so a page under that isn't giving the embedding
        # model anything useful even though extraction "succeeded".
        if len(text.strip()) < MIN_SUBSTANTIAL_CHARS:
            effectively_empty_indices.append(i)

    effective_empty_ratio = len(effectively_empty_indices) / total_pages if total_pages else 0

    # --- Pass 2: decide how to handle the effectively-empty pages ---
    if effective_empty_ratio >= SCANNED_BOOK_THRESHOLD:
        # Most/all of the book has no usable text layer -> this is a
        # scanned book (even if a few pages have stray thin text like
        # page numbers). OCR a representative sample instead of every page.
        sample_indices = get_sample_page_indices(total_pages)
        print(
            f"  {pdf_path.name}: detected as scanned "
            f"({effective_empty_ratio:.0%} of pages effectively empty) -> "
            f"OCR sampling {len(sample_indices)}/{total_pages} pages",
            file=sys.stderr,
        )
        for i in sample_indices:
            pages_text[i] = ocr_page(doc[i])
        # Pages outside the sample stay as-is (thin/empty) and get
        # filtered out below — we deliberately don't OCR the whole book.

    elif effectively_empty_indices:
        # Only a handful of pages are effectively empty in an otherwise
        # native-text book. The volume is small, so it's cheap to just
        # OCR those specific pages directly.
        print(
            f"  {pdf_path.name}: OCR fallback for {len(effectively_empty_indices)} "
            f"page(s) out of {total_pages}",
            file=sys.stderr,
        )
        for i in effectively_empty_indices:
            pages_text[i] = ocr_page(doc[i])

    doc.close()

    # Drop any pages that are still empty (untouched pages in a sampled
    # scanned book) and join what's left.
    full_text = "\n\f\n".join(t for t in pages_text if t.strip())

    # Some Arabic PDFs (especially older typeset ones) store text using
    # "presentation form" glyph codepoints instead of standard Arabic
    # letters — this extracts as disconnected, garbled-looking characters
    # even though the underlying content is correct. NFKC normalization
    # converts these back to standard letters. This is safe to run on
    # any text (already-correct text passes through unchanged), so we
    # apply it unconditionally rather than trying to detect which PDFs
    # need it.
    full_text = unicodedata.normalize("NFKC", full_text)

    return full_text


def extract_folder(input_dir: str, output_dir: str) -> None:
    """
    Batch version: extract text from every PDF in input_dir and save
    each one as a matching .txt file in output_dir. Skips PDFs that
    already have a corresponding .txt file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    for pdf_path in pdf_files:
        output_path = output_dir / (pdf_path.stem + ".txt")

        if output_path.exists():
            print(f"Skipping {pdf_path.name} (already extracted)")
            continue

        print(f"Extracting {pdf_path.name}...")

        try:
            text = extract_text_from_pdf(pdf_path)
        except RuntimeError as e:
            print(f"  [error] {e}", file=sys.stderr)
            continue

        if not text.strip():
            # Even OCR found nothing usable — flag it loudly so it
            # doesn't silently disappear from the collection.
            print(f"  [warning] no usable text extracted from {pdf_path.name}", file=sys.stderr)

        output_path.write_text(text, encoding="utf-8")
        print(f"  -> saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from one PDF or a folder of PDFs, with OCR fallback for scanned pages."
    )
    parser.add_argument("--pdf", help="Path to a single PDF file")
    parser.add_argument("--folder", help="Path to a folder of PDF files")
    parser.add_argument(
        "--out",
        default="data/extracted_text",
        help="Output folder for extracted .txt files (used with --folder)",
    )
    args = parser.parse_args()

    if args.pdf:
        text = extract_text_from_pdf(args.pdf)
        print(text)
    elif args.folder:
        extract_folder(args.folder, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
