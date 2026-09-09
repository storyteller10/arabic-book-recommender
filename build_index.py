"""
build_index.py

Step 4 of the book search pipeline: loads all book embeddings (from
create_embeddings.py) into a FAISS index for fast similarity search,
and saves an id-lookup table so a raw FAISS result (just a position
number and a distance) can be turned back into an actual book.

Uses cosine similarity via a normalized-vector inner-product index
(IndexFlatIP on unit-length vectors is mathematically equivalent to
cosine similarity, and is what E5-family models are designed to be
compared with).

Exposes:
    build_index(embeddings_dir) -> (faiss_index, id_lookup)
    load_index(index_dir) -> (faiss_index, id_lookup)   # for search.py to use later

Can also be run directly to build (or rebuild) the index from a
folder of embeddings.
"""

import argparse
import json
from pathlib import Path
from project_config import project_path

import faiss
import numpy as np


def build_index(embeddings_dir: str):
    """
    Load every .npy embedding referenced in manifest.json and build a
    FAISS index from them.

    Returns:
        (index, id_lookup) where id_lookup is a list of book_ids such
        that id_lookup[i] is the book_id for the vector at FAISS
        position i. FAISS itself only knows positions, not book
        identities -- this list is what makes results meaningful.
    """
    embeddings_dir = project_path(embeddings_dir)
    manifest_path = embeddings_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found in {embeddings_dir} -- run create_embeddings.py first"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not manifest:
        raise ValueError(f"manifest.json in {embeddings_dir} is empty -- no books embedded yet")

    id_lookup = []
    vectors = []

    # Sorting keeps the index build deterministic/reproducible across
    # runs, which makes debugging much easier than arbitrary dict order.
    for book_id in sorted(manifest.keys()):
        entry = manifest[book_id]
        if entry.get("dry_run") is True:
            raise ValueError(f"Synthetic dry-run embedding cannot be indexed: {book_id}")
        vector_path = embeddings_dir / entry["embedding_file"]

        if not vector_path.exists():
            print(f"  [warning] {vector_path.name} listed in manifest but missing on disk, skipping {book_id}")
            continue

        vector = np.load(vector_path).astype(np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError(f"Embedding must be a finite nonzero vector: {vector_path}")
        vectors.append(vector)
        id_lookup.append(book_id)

    if not vectors:
        raise ValueError("No valid embeddings found to index")

    matrix = np.vstack(vectors)
    dimension = matrix.shape[1]

    # IndexFlatIP = exact (not approximate) inner-product search.
    # With normalized vectors, inner product == cosine similarity.
    # "Flat" means it checks every vector rather than using an
    # approximate shortcut -- completely fine (and simplest/most
    # accurate) at this scale; worth revisiting only if the
    # collection grows into the tens of thousands of books.
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    return index, id_lookup


def save_index(index, id_lookup: list, output_dir: str) -> None:
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "books.index"))
    (output_dir / "id_lookup.json").write_text(
        json.dumps(id_lookup, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: str):
    """For search.py to use later: loads a previously built index and
    its id_lookup table."""
    index_dir = project_path(index_dir)
    index = faiss.read_index(str(index_dir / "books.index"))
    id_lookup = json.loads((index_dir / "id_lookup.json").read_text(encoding="utf-8"))
    return index, id_lookup


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from book embeddings.")
    parser.add_argument("--embeddings", required=True, help="Folder containing manifest.json + .npy embedding files")
    parser.add_argument("--out", default="data/index_metadata", help="Output folder for the FAISS index + id lookup table")
    args = parser.parse_args()

    index, id_lookup = build_index(args.embeddings)
    save_index(index, id_lookup, args.out)

    print(f"Indexed {index.ntotal} books ({index.d}-dim vectors)")
    print(f"Saved to {args.out}/books.index and {args.out}/id_lookup.json")


if __name__ == "__main__":
    main()