"""Shared CSV metadata normalization for embedding and display."""
import unicodedata
import pandas as pd
from project_config import project_path


def clean_value(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return None if not value or value.casefold() == "nan" else value


def normalize_book_id(filename: str) -> str:
    value = unicodedata.normalize("NFKC", filename).strip()
    if value.lower().endswith((".pdf", ".txt")):
        value = value[:-4]
    return value.strip()


def load_metadata(csv_path) -> dict:
    """Require only filename; ignore unrelated columns, including dates."""
    df = pd.read_csv(project_path(csv_path), dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if "filename" not in df.columns:
        raise ValueError("metadata CSV must have a 'filename' column")
    metadata = {}
    for _, row in df.iterrows():
        filename = clean_value(row["filename"])
        if not filename:
            continue
        book_id = normalize_book_id(filename)
        if not book_id:
            raise ValueError(f"Empty normalized filename: {filename!r}")
        if book_id in metadata:
            raise ValueError(f"Duplicate normalized filename in metadata CSV: {filename!r}")
        metadata[book_id] = {field: clean_value(row.get(field)) for field in
                             ("title", "author", "description", "category")}
    return metadata


def metadata_text(entry: dict) -> str:
    labels = (("title", "العنوان"), ("author", "المؤلف"),
              ("category", "التصنيف"), ("description", "الوصف"))
    return "\n".join(f"{label}: {clean_value(entry.get(field))}"
                     for field, label in labels if clean_value(entry.get(field)))
