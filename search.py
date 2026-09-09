"""Search the FAISS book index with E5 queries and optional book metadata."""
import argparse
import json
import numpy as np

from build_index import load_index
from create_embeddings import MODEL_NAME, load_model
from metadata import clean_value, load_metadata, normalize_book_id


def _metadata_for(book_id: str, metadata: dict) -> dict:
    entry = metadata.get(normalize_book_id(book_id), {})
    return {
        "title": clean_value(entry.get("title")) or book_id.replace("_", " "),
        "author": clean_value(entry.get("author")) or "Unknown",
        "category": clean_value(entry.get("category")) or "Uncategorized",
        "summary": clean_value(entry.get("description")) or "No summary available yet.",
    }


def embed_query(query: str, model) -> np.ndarray:
    """Encode a query with the E5 query: prefix."""
    return np.asarray(model.encode([f"query: {query}"], normalize_embeddings=True)[0],
                      dtype=np.float32)


def search(query: str, index, id_lookup: list, metadata: dict, model, top_k: int = 3) -> list:
    """Return ranked title, author, category, summary and score, without a cutoff."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer of at least 1")
    top_k = min(top_k, index.ntotal)
    if top_k == 0:
        return []
    vector = embed_query(query, model).reshape(1, -1)
    scores, positions = index.search(vector, top_k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        results.append({"rank": len(results) + 1,
                        **_metadata_for(id_lookup[pos], metadata),
                        "score": round(float(score), 4)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Nonblank search text")
    parser.add_argument("--index", default="data/index_metadata", help="Index folder (default: data/index_metadata)")
    parser.add_argument("--metadata", default="data/metadata.csv", help="CSV with filename and optional title/author/description/category")
    parser.add_argument("--model", default=MODEL_NAME, help="Model used to build the embeddings")
    parser.add_argument("--top-k", type=int, default=3, help="Positive result count, capped at indexed books")
    parser.add_argument("--dry-run", action="store_true", help="Use a random query vector for plumbing tests")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query cannot be empty")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    index, lookup = load_index(args.index)
    metadata = load_metadata(args.metadata)
    if args.dry_run:
        class RandomModel:
            def encode(self, texts, normalize_embeddings=True):
                vector = np.random.randn(len(texts), index.d).astype(np.float32)
                return vector / np.linalg.norm(vector, axis=1, keepdims=True)
        model = RandomModel()
    else:
        model = load_model(args.model)
    print(json.dumps({"results": search(args.query, index, lookup, metadata, model, args.top_k)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
