"""Embed sampled cleaned text and one separate metadata passage per book.

E5 passage vectors are normalized separately, combined with metadata weight
(default 0.25), and normalized again. Missing metadata uses text alone; empty
text permits metadata alone. Use a separate output directory to preserve the
text-only experiment. Dry-run vectors are synthetic and never reused as real.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from project_config import project_path

import numpy as np
from metadata import load_metadata, metadata_text, normalize_book_id

MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_WORD_COUNT = 200
NUM_CHUNKS_PER_BOOK = 8
METADATA_WEIGHT = 0.25
TEXT_WEIGHT = 1.0 - METADATA_WEIGHT


def load_model(model_name: str = MODEL_NAME):
    """Load cached E5 weights, downloading on the first real run."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed") from exc
    return SentenceTransformer(model_name)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must be a finite nonzero vector")
    return vector / norm


def split_into_chunks(text: str, chunk_word_count: int = CHUNK_WORD_COUNT,
                      num_chunks: int = NUM_CHUNKS_PER_BOOK) -> list:
    """Sample up to num_chunks word windows evenly across a document."""
    if chunk_word_count < 1 or num_chunks < 1:
        raise ValueError("Chunk size and number of chunks must be positive")
    words = text.split()
    if not words:
        return []
    if len(words) < chunk_word_count * num_chunks:
        return [" ".join(words[i:i + chunk_word_count])
                for i in range(0, len(words), chunk_word_count)][:num_chunks]
    step = (len(words) - chunk_word_count) / (num_chunks - 1) if num_chunks > 1 else 0
    return [" ".join(words[int(i * step):int(i * step) + chunk_word_count])
            for i in range(num_chunks)]


def embed_document(text: str, model, chunk_word_count=CHUNK_WORD_COUNT,
                   num_chunks=NUM_CHUNKS_PER_BOOK):
    """Average normalized sampled text vectors into a normalized book vector."""
    chunks = split_into_chunks(text, chunk_word_count, num_chunks)
    if not chunks:
        return None
    vectors = model.encode([f"passage: {chunk}" for chunk in chunks], normalize_embeddings=True)
    return normalize(np.mean(vectors, axis=0))


def embed_book(text, representation, model, metadata_weight=METADATA_WEIGHT,
               chunk_word_count=CHUNK_WORD_COUNT, num_chunks=NUM_CHUNKS_PER_BOOK):
    """Blend text and a single metadata vector; allow either source alone."""
    if not 0 <= metadata_weight <= 1:
        raise ValueError("Metadata weight must be between 0 and 1")
    text_vector = embed_document(text, model, chunk_word_count, num_chunks)
    meta_vector = None
    if representation.strip():
        meta_vector = normalize(model.encode([f"passage: {representation}"],
                                             normalize_embeddings=True)[0])
    if text_vector is None:
        if meta_vector is not None:
            print("  [warning] cleaned text is empty; using metadata-only embedding", file=sys.stderr)
        return meta_vector
    if meta_vector is None:
        return text_vector
    return normalize((1 - metadata_weight) * text_vector + metadata_weight * meta_vector)


class _DryRunModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.stack([normalize(np.random.randn(768)) for _ in texts])


def embed_folder(input_dir: str, output_dir: str, dry_run: bool = False,
                 metadata_path=None, metadata_weight=METADATA_WEIGHT,
                 model_name=MODEL_NAME, chunk_word_count=CHUNK_WORD_COUNT,
                 num_chunks=NUM_CHUNKS_PER_BOOK) -> None:
    """Save vectors and cache fingerprints for all current usable cleaned files.

    Fingerprints include text bytes, relevant metadata, model, chunk settings,
    metadata weight and dry-run status. Legacy entries are regenerated.
    Omit metadata_path to explicitly run the text-only baseline.
    """
    if not 0 <= metadata_weight <= 1:
        raise ValueError("Metadata weight must be between 0 and 1")
    if chunk_word_count < 1 or num_chunks < 1:
        raise ValueError("Chunk size and number of chunks must be positive")
    input_dir, output_dir = project_path(input_dir), project_path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Cleaned text directory not found: {input_dir}")
    metadata = load_metadata(metadata_path) if metadata_path is not None else {}
    txt_files = sorted(input_dir.glob("*.txt"))
    seen = set()
    for path in txt_files:
        key = normalize_book_id(path.name)
        if key in seen:
            raise ValueError(f"Duplicate normalized cleaned filename: {path.name!r}")
        seen.add(key)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {}
    model = None
    for txt_path in txt_files:
        book_id = txt_path.stem
        embedding_path = output_dir / f"{book_id}.npy"
        source_bytes = txt_path.read_bytes()
        text = source_bytes.decode("utf-8-sig")
        representation = metadata_text(metadata.get(normalize_book_id(txt_path.name), {}))
        if not text.strip() and not representation:
            print(f"  [warning] no text or metadata for {txt_path.name}, skipping", file=sys.stderr)
            continue
        expected_config = {
            "model_name": model_name,
            "chunk_word_count": chunk_word_count,
            "num_chunks": num_chunks,
            "metadata_weight": metadata_weight,
            "dry_run": dry_run,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "metadata_sha256": hashlib.sha256(representation.encode("utf-8")).hexdigest(),
        }
        existing = previous.get(book_id, {})
        if embedding_path.exists() and all(existing.get(k) == v for k, v in expected_config.items()):
            manifest[book_id] = existing
            print(f"Skipping {txt_path.name} (already embedded)")
            continue
        if model is None:
            model = _DryRunModel() if dry_run else load_model(model_name)
        vector = embed_book(text, representation, model, metadata_weight, chunk_word_count, num_chunks)
        np.save(embedding_path, vector)
        manifest[book_id] = {"filename": txt_path.name, "embedding_file": embedding_path.name,
                             **expected_config}
        print(f"Embedded {txt_path.name}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path} ({len(manifest)} books)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Folder of cleaned .txt files")
    parser.add_argument("--out", default="data/embeddings_metadata", help="Output vectors and manifest (default: data/embeddings_metadata)")
    parser.add_argument("--metadata", help="Optional metadata CSV, e.g. data/metadata.csv; omitted means text-only")
    parser.add_argument("--metadata-weight", type=float, default=METADATA_WEIGHT, help="Metadata blend weight, 0 to 1 (default: 0.25)")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_WORD_COUNT, help="Words per chunk (default: 200)")
    parser.add_argument("--num-chunks", type=int, default=NUM_CHUNKS_PER_BOOK, help="Maximum sampled chunks (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic vectors for plumbing tests only")
    args = parser.parse_args()
    if not 0 <= args.metadata_weight <= 1 or args.chunk_size < 1 or args.num_chunks < 1:
        parser.error("Weight must be 0 to 1; chunk size and count must be positive")
    embed_folder(args.folder, args.out, args.dry_run, args.metadata, args.metadata_weight,
                 args.model, args.chunk_size, args.num_chunks)


if __name__ == "__main__":
    main()
