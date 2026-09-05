"""
clean_text.py

Step 2 of the book search pipeline: turns raw extracted text (from
extract_text.py) into normalized text ready for embedding.

Built against real noise patterns found in actual extracted books:

  - Native-text PDFs wrap every visual line as its own text line,
    fragmenting sentences into hundreds of short lines (and
    occasionally splitting a single word across two lines).
  - Diacritic marks (tashkeel) sometimes extract with a stray space
    before them, e.g. "محمدا ً" instead of "محمداً".
  - Publisher watermarks (e.g. a website URL) repeated on every page.
  - Stray page numbers and footnote-marker lines sitting alone.
  - OCR misreading decorative covers/borders/logos as gibberish
    Arabic-looking text — real words with no real meaning.
  - Sampled OCR books jump between non-adjacent pages, so the result
    is a set of independent chunks, not one flowing narrative — this
    function treats it that way rather than trying to force a single
    continuous story.

Exposes:
    clean_arabic_text(raw_text) -> str

Can also be run directly on a folder of extracted .txt files.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Arabic diacritic (tashkeel) codepoints — fatha, damma, kasra, sukun,
# shadda, tanween, superscript alef, quranic annotation marks, etc.
ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u0652\u0670\u0653-\u065F\u06D6-\u06ED]"
)

TATWEEL = "\u0640"  # ـ elongation character, purely decorative

# A line that's only digits (Western or Arabic-Indic), punctuation,
# and whitespace — catches stray page numbers and footnote markers
# like "٢" or "(١)" sitting alone on a line.
STANDALONE_NUMBER_LINE = re.compile(r"^[\s0-9٠-٩\(\)\.\-–—]+$")

# Bare URLs / watermark lines (e.g. "www.ibtesama.com" repeated on
# every page of a scanned book)
URL_LINE = re.compile(r"^\s*(www\.|https?://)\S+\s*$", re.IGNORECASE)

# Sentence-terminal punctuation — a wrapped line ending in one of these
# is treated as a real paragraph/sentence boundary, not just a
# line-wrap, when rejoining fragmented lines.
SENTENCE_END = re.compile(r"[.!?؟۔]\s*[\"»)]?\s*$")

# A small set of very common Arabic function words. Real sentences of
# any length almost always contain several of these. Decorative-cover
# OCR gibberish (e.g. "معمعم يمو موعويوه") is built from letter
# clusters that are NOT real words, so it scores ~0 matches here even
# though it uses ordinary Arabic letters. This is what separates
# "garbled OCR of real text" (keep — still has real words mixed in)
# from "OCR of a decorative design" (discard — no real words at all).
ARABIC_STOPWORDS = {
    "و", "في", "من", "على", "الى", "ان", "لا", "ما", "هذا", "هذه",
    "ذلك", "تلك", "التي", "الذي", "كان", "كانت", "قال", "قالت", "الله",
    "بن", "عن", "مع", "كل", "بعد", "قبل", "حتى", "لم", "لن", "قد", "ثم",
    "او", "لكن", "غير", "بين", "عند", "دون", "عليه", "عليها", "منه",
    "منها", "فيه", "فيها", "هو", "هي", "نحن", "لها", "له", "بها", "به",
    "رسول", "النبي", "صلى", "وسلم", "رضي", "الذين", "الا", "اذا",
    "قلت", "يقول", "الحمد", "سبحانه", "تعالى",
}

# The old gibberish rule deleted any paragraph that did not contain one of
# the stopwords above. That removes valid titles, headings, poetry, and
# technical prose. Keep the detector available for future diagnostics, but
# never delete paragraphs based on this weak heuristic.
ENABLE_GIBBERISH_FILTER = False


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def remove_diacritics(text: str) -> str:
    """Strip tashkeel marks. Also fixes the common extraction artifact
    where a diacritic separates from its letter with a stray space
    (e.g. "محمدا ً" -> "محمدا") since we just remove the mark itself."""
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic_letters(text: str) -> str:
    """Collapse common spelling-variant letters to one form, and drop
    the purely decorative tatweel elongation character. This reduces
    noise for the embedding model without changing meaning."""
    text = text.replace(TATWEEL, "")
    # Alef variants -> plain alef
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Some PDFs render "الله" with an extra decorative alef stroke,
    # extracting as "االله" instead of "الله" -- very high frequency
    # in religious texts, worth fixing explicitly.
    text = text.replace("االله", "الله")
    return text


def is_gibberish_paragraph(paragraph: str, min_words: int = 2) -> bool:
    """
    Detect OCR noise from decorative covers/borders: sequences of
    Arabic-looking letter clusters that aren't real words.

    Runs on full (already line-joined) paragraphs, not raw short
    wrapped lines -- real content is also naturally short before
    joining, so judging gibberish pre-join produces unreliable
    results.

    Only stopwords of length >= 3 count as "reliable evidence" of
    real text. Short 2-letter function words (من, ان, لا, ...) are
    common enough that they occasionally appear by pure chance inside
    random gibberish letter clusters, which was causing real garbage
    blocks to slip through when they happened to contain one. Every
    real paragraph tested (even a 12-word one) contained at least one
    3+-letter stopword; the gibberish block tested contained none.
    """
    words = paragraph.split()
    if len(words) < min_words:
        return False  # too short to judge at all -- leave it alone

    reliable_hits = sum(
        1 for w in words
        if len(w) >= 3 and w.strip("،.؛:؟!»«\"'()[]") in ARABIC_STOPWORDS
    )
    return reliable_hits == 0


def strip_noise_lines(text: str) -> str:
    """Drop watermark lines and standalone page-number/footnote-marker
    lines. Runs before line-rejoining. Gibberish detection happens
    later, after joining, since it needs full paragraphs to judge
    reliably (see is_gibberish_paragraph)."""
    kept_lines = []
    for line in text.split("\n"):
        # extract_text.py uses form-feed as a page separator. str.strip()
        # would erase it, so preserve it before normal whitespace handling.
        if "\f" in line:
            parts = line.split("\f")
            for part_index, part in enumerate(parts):
                if part.strip():
                    kept_lines.append(part.strip())
                if part_index < len(parts) - 1:
                    kept_lines.append("\f")
            continue

        stripped = line.strip()

        if not stripped:
            kept_lines.append("")  # preserve blank lines as page/para breaks
            continue

        if URL_LINE.match(stripped):
            continue
        if STANDALONE_NUMBER_LINE.match(stripped):
            continue

        kept_lines.append(stripped)

    return "\n".join(kept_lines)


def join_wrapped_lines_to_paragraphs(text: str) -> list:
    """
    Rejoin PDF-wrapped lines into paragraphs, returned per-page as a
    list of paragraph strings (list[list[str]]) so gibberish filtering
    can run per-paragraph afterward.

    A blank line or form-feed is treated as a real paragraph/page
    break. Otherwise, lines are merged with a space UNLESS the
    previous line already ends in sentence-terminal punctuation, in
    which case a new paragraph starts.

    Known limitation: some PDFs occasionally split a single word
    across two wrapped lines (a PyMuPDF text-run quirk, not something
    fixable from text alone). Joining with a space in that case
    inserts one stray space inside a word rather than leaving the
    sentence permanently broken -- a much smaller quality hit, and
    one embedding models tolerate reasonably well since they tokenize
    at the sub-word level.
    """
    pages = text.split("\f")
    result = []

    for page in pages:
        paragraphs = []
        buffer = ""

        for raw_line in page.split("\n"):
            line = raw_line.strip()

            if not line:
                if buffer:
                    paragraphs.append(buffer.strip())
                    buffer = ""
                continue

            if buffer:
                buffer += " " + line
            else:
                buffer = line

            if SENTENCE_END.search(line):
                paragraphs.append(buffer.strip())
                buffer = ""

        if buffer:
            paragraphs.append(buffer.strip())

        result.append(paragraphs)

    return result


def collapse_whitespace(text: str) -> str:
    """Final pass: collapse repeated spaces and excess blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_arabic_text(raw_text: str) -> str:
    """
    Full cleaning pipeline. Order matters:
      1. Remove diacritics first -- cleans up stray-space-before-mark
         artifacts before any line-based analysis.
      2. Normalize letter variants / strip tatweel.
      3. Strip watermark / page-number lines (structural noise).
      4. Rejoin wrapped lines into real paragraphs.
      5. Drop gibberish paragraphs (decorative-page OCR misreads) --
         done AFTER joining, since full paragraphs give a much more
         reliable signal than short raw wrapped lines.
      6. Collapse leftover whitespace.
    """
    text = remove_diacritics(raw_text)
    text = normalize_arabic_letters(text)
    text = strip_noise_lines(text)

    pages = join_wrapped_lines_to_paragraphs(text)

    cleaned_pages = []
    for paragraphs in pages:
        kept = [
            p for p in paragraphs
            if not (ENABLE_GIBBERISH_FILTER and is_gibberish_paragraph(p))
        ]
        if kept:
            cleaned_pages.append("\n\n".join(kept))

    # Pages are joined with a clear separator. For sampled OCR books
    # these represent non-adjacent, unrelated chunks of the book —
    # deliberately kept visually distinct rather than merged into one
    # false-continuous narrative.
    text = "\n\n---\n\n".join(cleaned_pages)
    text = collapse_whitespace(text)
    return text


def clean_folder(input_dir: str, output_dir: str) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    for txt_path in txt_files:
        output_path = output_dir / txt_path.name

        if output_path.exists():
            print(f"Skipping {txt_path.name} (already cleaned)")
            continue

        raw = txt_path.read_text(encoding="utf-8")
        cleaned = clean_arabic_text(raw)
        output_path.write_text(cleaned, encoding="utf-8")

        reduction = 100 * (1 - len(cleaned) / len(raw)) if raw else 0
        print(f"Cleaned {txt_path.name} ({len(raw)} -> {len(cleaned)} chars, {reduction:.0f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description="Clean extracted Arabic book text.")
    parser.add_argument("--file", help="Path to a single .txt file")
    parser.add_argument("--folder", help="Path to a folder of .txt files")
    parser.add_argument("--out", default="data/cleaned_text", help="Output folder (used with --folder)")
    args = parser.parse_args()

    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
        print(clean_arabic_text(raw))
    elif args.folder:
        clean_folder(args.folder, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
