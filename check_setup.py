"""Read-only setup checks. Never loads/downloads model weights or rebuilds data."""
import sys
sys.dont_write_bytecode = True
import importlib
import json
import os
import hashlib
from project_config import BASE_DIR, configured_path


def main():
    failures = []
    def report(level, message):
        print(f"{level}: {message}")
        if level == "FAIL":
            failures.append(message)
    for name in ("numpy", "pandas", "faiss", "fastapi", "pydantic", "uvicorn",
                 "httpx", "sentence_transformers", "torch", "pymupdf", "pytesseract", "PIL", "regex"):
        try:
            importlib.import_module(name)
            report("PASS", f"Import {name}")
        except Exception as exc:
            report("FAIL", f"Import {name}: {exc}; install requirements.txt")
    index_dir = configured_path("BOOK_INDEX_DIR", "data/index_metadata")
    metadata_path = configured_path("BOOK_METADATA_PATH", "data/metadata.csv")
    for path in (metadata_path, index_dir / "books.index", index_dir / "id_lookup.json"):
        report("PASS" if path.is_file() else "FAIL", f"Required file: {path}")
    try:
        import numpy as np
        from build_index import load_index
        from metadata import load_metadata, normalize_book_id, metadata_text
        from create_embeddings import MODEL_NAME, METADATA_WEIGHT, CHUNK_WORD_COUNT, NUM_CHUNKS_PER_BOOK
        import extract_text
        import diagnose_pdfs
        model = os.environ.get("BOOK_MODEL_NAME", MODEL_NAME)
        expected = dict(model_name=model, metadata_weight=METADATA_WEIGHT,
                        chunk_word_count=CHUNK_WORD_COUNT, num_chunks=NUM_CHUNKS_PER_BOOK, dry_run=False)
        if (model, METADATA_WEIGHT, CHUNK_WORD_COUNT, NUM_CHUNKS_PER_BOOK) != ("intfloat/multilingual-e5-base", .25, 200, 8):
            report("FAIL", "Configuration differs from the preserved E5-base / 75:25 / 200 words / 8 chunks settings")
        else:
            report("PASS", "Model and embedding settings match the preserved configuration")
        for key in ("SAMPLE_PAGE_COUNT", "FRONT_MATTER_PAGES", "SCANNED_BOOK_THRESHOLD", "MIN_SUBSTANTIAL_CHARS", "MIN_ARABIC_LETTER_RATIO"):
            if getattr(extract_text, key) != getattr(diagnose_pdfs, key):
                report("FAIL", f"OCR/diagnostic configuration differs: {key}")
        report("PASS" if extract_text.SAMPLE_PAGE_COUNT == 30 else "FAIL", "OCR sample must be 30 pages")
        index, lookup = load_index(index_dir)
        if not isinstance(lookup, list) or any(not isinstance(k, str) for k in lookup):
            raise ValueError("id_lookup.json must be a list of strings")
        normalized = [normalize_book_id(k) for k in lookup]
        if index.ntotal == 0 or index.ntotal != len(lookup) or len(set(normalized)) != len(lookup):
            raise ValueError("Empty index, lookup count mismatch, or duplicate normalized IDs")
        report("PASS", f"{index.ntotal} indexed books; {index.d} dimensions; unique matching lookup")
        if index.d != 768:
            report("FAIL", "Expected 768 dimensions for multilingual-e5-base")
        metadata = load_metadata(metadata_path)
        missing = sorted(set(normalized) - set(metadata))
        extra = sorted(set(metadata) - set(normalized))
        report("WARNING" if missing else "PASS", f"Indexed IDs without metadata: {missing}")
        report("WARNING" if extra else "PASS", f"Metadata IDs not indexed: {extra}")
        # Optional provenance: API-only handoffs need just index, lookup, and CSV.
        manifest_path = BASE_DIR / "data/embeddings_metadata/manifest.json"
        if index_dir.resolve() != (BASE_DIR / "data/index_metadata").resolve() or not manifest_path.is_file():
            report("WARNING", "Embedding provenance unavailable for this index; dimensions alone cannot prove model identity")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if set(manifest) != set(lookup):
                report("FAIL", "Embedding manifest IDs differ from index lookup")
            for position, key in enumerate(lookup):
                entry = manifest.get(key, {})
                if any(entry.get(k) != v for k, v in expected.items()):
                    report("FAIL", f"Embedding configuration/dry-run mismatch: {key}")
                representation = metadata_text(metadata.get(normalize_book_id(key), {}))
                if entry.get("metadata_sha256") != hashlib.sha256(representation.encode("utf-8")).hexdigest():
                    report("WARNING", f"Metadata changed since embedding: {key}")
                source = BASE_DIR / "data/cleaned_text" / entry.get("filename", "")
                if source.is_file() and entry.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
                    report("WARNING", f"Cleaned text changed since embedding: {key}")
                vector_path = manifest_path.parent / entry.get("embedding_file", "")
                if vector_path.is_file():
                    vector = np.load(vector_path, allow_pickle=False)
                    if not np.array_equal(index.reconstruct(position), vector):
                        report("FAIL", f"Index vector differs from manifest embedding: {key}")
            report("PASS", "Optional embedding provenance inspection completed (see any warnings/failures above)")
        try:
            command = extract_text.configure_tesseract()
            report("PASS", f"Tesseract and ara language available: {command}")
        except RuntimeError as exc:
            report("WARNING", f"{exc} OCR is optional for API-only use.")
    except Exception as exc:
        report("FAIL", f"Artifact/configuration checks: {exc}")
    report("FAIL" if failures else "PASS", f"Setup finished with {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
