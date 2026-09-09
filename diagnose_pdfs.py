"""
diagnose_pdfs.py

Quick diagnostic tool — run this BEFORE the real extract_text.py pass
on a new batch of books. It does NOT run any OCR (so it's fast, even
on hundreds of pages) and just reports, per PDF:

    - total page count
    - how many pages have zero extractable text
    - thin and corrupt/non-Arabic native text pages
    - the empty-page and unusable-page ratios
    - which branch extract_text.py would take, and how many pages it
      would actually OCR

This exists to answer "why is it processing so many pages?" without
guessing — run it and look at the numbers directly.

Usage:
    python diagnose_pdfs.py --folder "path/to/your/pdfs"
"""

import argparse
import unicodedata
from pathlib import Path
from project_config import project_path

import regex
import pymupdf

# Keep these in sync with extract_text.py's config
SCANNED_BOOK_THRESHOLD = 0.7
SAMPLE_PAGE_COUNT = 30
FRONT_MATTER_PAGES = 8
MIN_SUBSTANTIAL_CHARS = 40


# Keep this validator identical to extract_text.py without importing OCR setup.
# Match genuine Arabic letters, excluding digits, marks and punctuation.
ARABIC_LETTER = regex.compile(r"[\p{Script=Arabic}&&\p{Letter}]", regex.VERSION1)
MIN_ARABIC_LETTER_RATIO = 0.30


def has_usable_arabic_text(text: str) -> bool:
    """Reject thin text and broken/non-Arabic native font extraction."""
    text = unicodedata.normalize("NFKC", text).strip()
    if len(text) < MIN_SUBSTANTIAL_CHARS:
        return False
    alphabetic_count = sum(char.isalpha() for char in text)
    if not alphabetic_count:
        return False
    arabic_count = len(ARABIC_LETTER.findall(text))
    return arabic_count / alphabetic_count >= MIN_ARABIC_LETTER_RATIO


def get_sample_page_indices(total_pages: int, sample_size: int = SAMPLE_PAGE_COUNT) -> set[int]:
    # Same logic as extract_text.py — duplicated here so this script
    # has no dependency on it and can be run standalone.
    if total_pages <= sample_size:
        return set(range(total_pages))
    front_count = min(FRONT_MATTER_PAGES, sample_size)
    front_pages = set(range(front_count))
    remaining_slots = sample_size - len(front_pages)
    if remaining_slots > 0:
        step = (total_pages - front_count) / remaining_slots
        spaced_pages = {front_count + int(i * step) for i in range(remaining_slots)}
    else:
        spaced_pages = set()
    return front_pages | spaced_pages


def diagnose_pdf(pdf_path: Path) -> dict:
    pdf_path = project_path(pdf_path)
    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count

    empty_count = 0
    # Track how much non-empty text actually exists on "non-empty" pages —
    # this catches the case where a page technically has SOME text (like a
    # burned-in page number or watermark) but is still effectively a scanned
    # image page. A real content page usually has hundreds of characters;
    # a bare page number is a handful.
    thin_text_count = 0  # pages with some text, but suspiciously little

    corrupt_text_count = 0
    unusable_indices = set()
    for i, page in enumerate(doc):
        text = unicodedata.normalize("NFKC", page.get_text()).strip()
        if not has_usable_arabic_text(text):
            unusable_indices.add(i)
        if not text:
            empty_count += 1
        elif len(text) < MIN_SUBSTANTIAL_CHARS:
            # Non-empty but tiny — likely just a header/footer/page number,
            # not real body content. Worth knowing about separately.
            thin_text_count += 1
        elif i in unusable_indices:
            corrupt_text_count += 1

    doc.close()

    empty_ratio = empty_count / total_pages if total_pages else 0
    # Retain the legacy result key, now including corrupt/non-Arabic text.
    effective_empty_count = len(unusable_indices)
    effective_empty_ratio = effective_empty_count / total_pages if total_pages else 0

    if effective_empty_ratio >= SCANNED_BOOK_THRESHOLD:
        sample = get_sample_page_indices(total_pages).intersection(unusable_indices)
        branch = f"SAMPLED OCR — would OCR {len(sample)}/{total_pages} pages"
    elif effective_empty_count > 0:
        branch = f"PER-PAGE FALLBACK — would OCR {effective_empty_count}/{total_pages} pages individually"
    else:
        branch = "NATIVE TEXT ONLY — no OCR"

    return {
        "file": pdf_path.name,
        "total_pages": total_pages,
        "empty_pages": empty_count,
        "thin_text_pages": thin_text_count,
        "corrupt_text_pages": corrupt_text_count,
        "unusable_ratio": effective_empty_ratio,
        "empty_ratio": empty_ratio,
        "effective_empty_ratio": effective_empty_ratio,
        "branch": branch,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose PDFs before running OCR extraction.")
    parser.add_argument("--folder", required=True, help="Folder of PDFs to inspect")
    args = parser.parse_args()

    folder = project_path(args.folder)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {folder}")
        return

    print(f"{'File':<35} {'Pages':>6} {'Empty':>6} {'Thin':>5} {'Corrupt':>7} {'Empty%':>7} {'Unusable%':>10}  Branch")
    print("-" * 110)

    for pdf_path in pdf_files:
        try:
            r = diagnose_pdf(pdf_path)
        except Exception as e:
            print(f"{pdf_path.name:<35} [error opening file: {e}]")
            continue

        print(
            f"{r['file']:<35} {r['total_pages']:>6} {r['empty_pages']:>6} "
            f"{r['thin_text_pages']:>5} {r['corrupt_text_pages']:>7} {r['empty_ratio']:>6.0%} "
            f"{r['effective_empty_ratio']:>9.0%}  {r['branch']}"
        )


if __name__ == "__main__":
    main()