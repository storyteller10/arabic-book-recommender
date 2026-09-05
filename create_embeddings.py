"""
create_embeddings.py

Step 3 of the book search pipeline: converts each cleaned book's text
into a single vector (embedding) that captures what the book is about.

Why chunking is necessary: embedding models can only look at a small
window of text at once (roughly 256-512 words). Cleaned books range
from a few thousand characters (OCR'd, sampled) up to ~470,000
characters (a full native-text book) -- feeding the whole thing in
would silently truncate to just the first paragraph or two, badly
under-representing the book. Instead, each book is split into several
chunks spread across the document, each chunk is embedded separately,
and the results are averaged into one final "book-level" vector. This
is the same representative-sample philosophy used for OCR page
sampling in extract_text.py, applied one level up.

Exposes:
    embed_text(text, model) -> np.ndarray          # one chunk
    embed_document(text, model) -> np.ndarray       # whole book, chunked+averaged

Can also be run directly on a folder of cleaned .txt files.

IMPORTANT: uses the E5 model family, which requires a "query: " or
"passage: " prefix on all input text -- this is part of how the model
was trained, not optional formatting. Book text being indexed uses
"passage: "; user search queries (in search.py, built later) must use
"query: ". Mixing these up silently produces worse results rather
than an error, so it's worth remembering when search.py is built.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # allows --dry-run to work without the real dependency


MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_WORD_COUNT = 200      # safer for Arabic, where one word may be several tokens
NUM_CHUNKS_PER_BOOK = 8     # how many chunks to sample per book and average
MIN_WORDS_FOR_CHUNKING = CHUNK_WORD_COUNT * NUM_CHUNKS_PER_BOOK


def load_model(model_name: str = MODEL_NAME):
    """Loads the embedding model. Requires internet access to Hugging
    Face Hub the first time (downloads and caches the model weights;
    later runs use the local cache)."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers")
    return SentenceTransformer(model_name)


def split_into_chunks(text: str, chunk_word_count: int = CHUNK_WORD_COUNT, num_chunks: int = NUM_CHUNKS_PER_BOOK) -> list:
    """
    Split a book's text into up to num_chunks pieces of roughly
    chunk_word_count words each, spread evenly across the document
    (not just the beginning) -- same sampling idea as OCR page
    selection in extract_text.py.

    If the text is shorter than num_chunks * chunk_word_count words,
    just chunk what's there sequentially instead of trying to force
    artificial spacing across too little material.
    """
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    if total_words < MIN_WORDS_FOR_CHUNKING:
        # Not enough text to spread chunks meaningfully -- chunk
        # sequentially from the start instead.
        chunks = []
        for i in range(0, total_words, chunk_word_count):
            chunk_words = words[i:i + chunk_word_count]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
        return chunks[:num_chunks] if len(chunks) > num_chunks else chunks

    # Spread num_chunks starting points evenly across the document
    step = (total_words - chunk_word_count) / (num_chunks - 1) if num_chunks > 1 else 0
    chunks = []
    for i in range(num_chunks):
        start = int(i * step)
        end = start + chunk_word_count
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def embed_document(text: str, model) -> np.ndarray:
    """
    Embed a whole book: split into chunks, embed each chunk, average
    the results into one vector. Returns a single normalized vector.
    """
    chunks = split_into_chunks(text)

    if not chunks:
        return None

    # E5 models require this exact "passage: " prefix on indexed text
    # (as opposed to "query: " for search queries) -- part of how the
    # model was trained, not optional.
    prefixed_chunks = [f"passage: {c}" for c in chunks]

    chunk_embeddings = model.encode(prefixed_chunks, normalize_embeddings=True)

    # Average the chunk vectors into one book-level vector, then
    # re-normalize (averaging changes the vector's length, and FAISS
    # cosine-similarity search assumes normalized vectors).
    book_embedding = np.mean(chunk_embeddings, axis=0)
    book_embedding = book_embedding / np.linalg.norm(book_embedding)

    return book_embedding


def embed_folder(input_dir: str, output_dir: str, dry_run: bool = False) -> None:
    """
    Embed every cleaned .txt file in input_dir. Saves:
      - one .npy file per book (the embedding vector)
      - a manifest.json mapping filenames to their embedding files,
        which build_index.py and search.py both rely on

    dry_run=True uses random vectors instead of a real model -- for
    testing the pipeline's plumbing without needing model access.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    model = None
    if not dry_run:
        print(f"Loading embedding model ({MODEL_NAME})... this may take a while the first time.", file=sys.stderr)
        model = load_model()

    for txt_path in txt_files:
        book_id = txt_path.stem
        embedding_path = output_dir / f"{book_id}.npy"

        source_hash = hashlib.sha256(txt_path.read_bytes()).hexdigest()
        expected_config = {
            "model_name": MODEL_NAME,
            "chunk_word_count": CHUNK_WORD_COUNT,
            "num_chunks": NUM_CHUNKS_PER_BOOK,
            "dry_run": dry_run,
            "source_sha256": source_hash,
        }

        existing = manifest.get(book_id, {})
        if embedding_path.exists() and all(
            existing.get(key) == value for key, value in expected_config.items()
        ):
            print(f"Skipping {txt_path.name} (already embedded)")
            continue

        text = txt_path.read_text(encoding="utf-8")

        if dry_run:
            # Fake embedding for pipeline testing -- same shape/normalization
            # a real model would produce (768-dim, matches multilingual-e5-base).
            fake = np.random.randn(768).astype(np.float32)
            embedding = fake / np.linalg.norm(fake)
        else:
            embedding = embed_document(text, model)

        if embedding is None:
            print(f"  [warning] no text to embed in {txt_path.name}, skipping", file=sys.stderr)
            continue

        np.save(embedding_path, embedding)
        manifest[book_id] = {
            "filename": txt_path.name,
            "embedding_file": embedding_path.name,
            **expected_config,
        }
        print(f"Embedded {txt_path.name}")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path} ({len(manifest)} books)")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for cleaned book text.")
    parser.add_argument("--folder", required=True, help="Folder of cleaned .txt files")
    parser.add_argument("--out", default="data/embeddings", help="Output folder for embeddings + manifest")
    parser.add_argument("--dry-run", action="store_true", help="Use fake random vectors instead of a real model (for testing without model access)")
    args = parser.parse_args()

    embed_folder(args.folder, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
