"""
search.py

Step 5 of the book search pipeline: takes a user's free-text query,
embeds it, searches the FAISS index, and joins the results with book
metadata (title, author, publication date, summary) from the Google
Sheet export, returning results in the exact shape app.py will hand
to the website.

Exposes:
    search(query, index, id_lookup, metadata, model, top_k=3) -> list[dict]

Can also be run directly to test a query from the command line.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

sys.path.insert(0, str(Path(__file__).parent))
from build_index import load_index  # noqa: E402


MODEL_NAME = "intfloat/multilingual-e5-base"


def load_model(model_name: str = MODEL_NAME):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers")
    return SentenceTransformer(model_name)


def load_metadata(csv_path: str) -> dict:
    """
    Load the metadata CSV (exported from the Google Sheet) into a
    dict keyed by book_id (filename without extension), so it can be
    looked up the same way id_lookup.json identifies books.

    Missing/incomplete rows are fine -- a book with no metadata row
    at all still gets a result, just with only what's known filled in
    (see _metadata_for below).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "filename" not in df.columns:
        raise ValueError(f"metadata CSV must have a 'filename' column, got: {list(df.columns)}")

    metadata = {}
    for _, row in df.iterrows():
        raw_filename = str(row["filename"]).strip()
        book_id = Path(raw_filename).stem  # strips .pdf, matches book_id used elsewhere
        metadata[book_id] = {
            "title": str(row.get("title", "")).strip() or None,
            "author": str(row.get("author", "")).strip() or None,
            "publication_date": str(row.get("publication_date", "")).strip() or None,
            "summary": str(row.get("description", "")).strip() or None,
        }

    return metadata


def _metadata_for(book_id: str, metadata: dict) -> dict:
    """
    Look up a book's metadata, falling back gracefully when the sheet
    doesn't have a row for it yet (or a field is blank) -- so a book
    still shows up in results with at least a filename-derived title
    rather than crashing or silently vanishing.
    """
    entry = metadata.get(book_id, {})
    return {
        "title": entry.get("title") or book_id.replace("_", " "),
        "author": entry.get("author") or "Unknown",
        "publication_date": entry.get("publication_date") or "Unknown",
        "summary": entry.get("summary") or "No summary available yet.",
    }


def embed_query(query: str, model) -> np.ndarray:
    """
    Embed a user's search query. E5 models require a "query: " prefix
    here -- the mirror-image of the "passage: " prefix used when
    embedding book text in create_embeddings.py. Mixing these up
    doesn't error, it just quietly produces worse matches.
    """
    prefixed = f"query: {query}"
    vector = model.encode([prefixed], normalize_embeddings=True)[0]
    return vector.astype(np.float32)


def search(query: str, index, id_lookup: list, metadata: dict, model, top_k: int = 3) -> list:
    """
    Run a search and return results already shaped for the API:
    title, author, publication_date, summary, score -- ready for
    app.py to return as JSON with no further transformation.
    """
    query_vector = embed_query(query, model).reshape(1, -1)
    scores, positions = index.search(query_vector, top_k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue  # FAISS pads with -1 if there are fewer than top_k books total
        book_id = id_lookup[pos]
        meta = _metadata_for(book_id, metadata)
        results.append({
            "title": meta["title"],
            "author": meta["author"],
            "publication_date": meta["publication_date"],
            "summary": meta["summary"],
            "score": round(float(score), 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Search the book index from the command line.")
    parser.add_argument("--query", required=True, help="Search text")
    parser.add_argument("--index", default="data/index", help="Folder containing books.index + id_lookup.json")
    parser.add_argument("--metadata", default="data/metadata.csv", help="Path to the metadata CSV")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--dry-run", action="store_true", help="Use a random query vector instead of a real model")
    args = parser.parse_args()

    index, id_lookup = load_index(args.index)
    metadata = load_metadata(args.metadata)

    if args.dry_run:
        model = None
        query_vector = np.random.randn(index.d).astype(np.float32)
        query_vector /= np.linalg.norm(query_vector)
        scores, positions = index.search(query_vector.reshape(1, -1), args.top_k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            meta = _metadata_for(id_lookup[pos], metadata)
            results.append({**meta, "score": round(float(score), 4)})
    else:
        model = load_model()
        results = search(args.query, index, id_lookup, metadata, model, top_k=args.top_k)

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()