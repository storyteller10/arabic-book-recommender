"""Focused offline tests using deterministic embeddings and a real FAISS index."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from fastapi.testclient import TestClient

import app as api
import create_embeddings as embeddings
from build_index import build_index, save_index
from metadata import clean_value, load_metadata, metadata_text, normalize_book_id
from search import search


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.extend(texts)
        return np.array([[0., 1.] if "العنوان:" in text else [1., 0.]
                         for text in texts], dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return 2


class MetadataTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_book_id("  Ｂook.pdf  "), "Book")
        self.assertEqual(normalize_book_id("Book.txt"), "Book")
        self.assertEqual(normalize_book_id("book.part.pdf"), "book.part")

    def test_blank_nan(self):
        for value in (None, float("nan"), "", "  ", "nan", " NaN "):
            self.assertIsNone(clean_value(value))
        self.assertEqual(metadata_text({"title": float("nan")}), "")

    def test_load_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.csv"
            path.write_text("filename,title,author,description,category,publication_date\n Ｂook.pdf ,Title,NaN, Summary , ,1999\n", encoding="utf-8")
            data = load_metadata(path)
            self.assertEqual(data["Book"], dict(title="Title", author=None, description="Summary", category=None))
            with path.open("a", encoding="utf-8") as handle:
                handle.write("Book.pdf,Duplicate,,,,\n")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized filename"):
                load_metadata(path)
            path.write_text("title\nTitle\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filename"):
                load_metadata(path)


class EmbeddingTests(unittest.TestCase):
    def test_blend_and_fallbacks(self):
        model = FakeModel()
        representation = metadata_text({"title": "كتاب"})
        result = embeddings.embed_book("book text", representation, model)
        np.testing.assert_allclose(result, np.array([.75, .25]) / np.linalg.norm([.75, .25]))
        self.assertEqual(model.calls, ["passage: book text", "passage: العنوان: كتاب"])
        np.testing.assert_allclose(embeddings.embed_book("book text", "", model), [1, 0])
        np.testing.assert_allclose(embeddings.embed_book(" ", representation, model), [0, 1])
        self.assertIsNone(embeddings.embed_book("", "", model))

    def test_chunk_sampling(self):
        self.assertEqual(embeddings.split_into_chunks("0 1 2 3 4 5 6 7 8 9", 2, 3),
                         ["0 1", "4 5", "8 9"])

    def test_manifest_invalidation_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "clean", root / "vectors"
            source.mkdir()
            text_path = source / "Book.txt"
            text_path.write_text("text", encoding="utf-8")
            csv_path = root / "metadata.csv"
            csv_path.write_text("filename,title\nOther.pdf,Other\n", encoding="utf-8")
            kwargs = dict(metadata_path=csv_path)
            with patch.object(embeddings, "load_model", side_effect=lambda name: FakeModel()) as loader:
                def run():
                    embeddings.embed_folder(source, output, **kwargs)
                run()
                np.testing.assert_allclose(np.load(output / "Book.npy"), [1, 0])
                run()
                self.assertEqual(loader.call_count, 1)
                changes = [lambda: text_path.write_text("new text", encoding="utf-8"),
                           lambda: csv_path.write_text("filename,title\nBook.pdf,كتاب\n", encoding="utf-8"),
                           lambda: kwargs.update(model_name="another-model"),
                           lambda: kwargs.update(chunk_word_count=3),
                           lambda: kwargs.update(num_chunks=2),
                           lambda: kwargs.update(metadata_weight=.5)]
                for count, change in enumerate(changes, start=2):
                    change()
                    run()
                    self.assertEqual(loader.call_count, count)
                kwargs["dry_run"] = True
                run()
                self.assertTrue(json.loads((output / "manifest.json").read_text())["Book"]["dry_run"])
                kwargs["dry_run"] = False
                run()
                self.assertEqual(loader.call_count, 8)
                # Publication date is not a fingerprint input.
                csv_path.write_text("filename,title,publication_date\nBook.pdf,كتاب,2000\n", encoding="utf-8")
                run()
                self.assertEqual(loader.call_count, 8)
                text_path.write_text("", encoding="utf-8")
                run()
                np.testing.assert_allclose(np.load(output / "Book.npy"), [0, 1])
                index, lookup = build_index(output)
                self.assertEqual((index.ntotal, lookup), (1, ["Book"]))
                csv_path.write_text("filename,title\nOther.pdf,Other\n", encoding="utf-8")
                run()
                self.assertEqual(json.loads((output / "manifest.json").read_text()), {})

    def test_duplicate_cleaned_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("Book.txt", "Ｂook.txt"):
                (root / name).write_text("text", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized cleaned filename"):
                embeddings.embed_folder(root, root / "out", dry_run=True)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = faiss.IndexFlatIP(2)
        self.index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
        save_index(self.index, ["Book", "Missing_book"], self.root)
        self.metadata_path = self.root / "metadata.csv"
        self.metadata_path.write_text("filename,title,author,description,category\nBook.pdf,كتاب,,وصف,سيرة\n", encoding="utf-8")
        env = patch.dict("os.environ", {"BOOK_INDEX_DIR": str(self.root),
                                      "BOOK_METADATA_PATH": str(self.metadata_path),
                                      "BOOK_MODEL_NAME": embeddings.MODEL_NAME})
        env.start()
        self.addCleanup(env.stop)

    def test_health_search_validation_and_reuse(self):
        model = FakeModel()
        with patch.object(api, "load_model", return_value=model) as loader, \
             patch.object(api, "load_index", wraps=api.load_index) as index_loader, \
             patch.object(api, "load_metadata", wraps=api.load_metadata) as meta_loader:
            with TestClient(api.create_app()) as client:
                self.assertEqual(client.get("/health").json(), {
                    "status": "ok", "model": embeddings.MODEL_NAME, "indexed_books": 2})
                response = client.post("/search", json={"query": "  كتاب عن السيرة النبوية  ", "top_k": 10})
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["query"], "كتاب عن السيرة النبوية")
                self.assertEqual(len(body["results"]), 2)
                self.assertEqual(body["results"][0], {"rank": 1, "title": "كتاب", "author": "Unknown",
                                                    "category": "سيرة", "summary": "وصف", "score": 1.0})
                self.assertEqual(body["results"][1]["title"], "Missing book")
                self.assertEqual(body["results"][1]["rank"], 2)
                self.assertEqual(model.calls, ["query: كتاب عن السيرة النبوية"])
                self.assertEqual(client.post("/search", json={"query": "كتاب"}).status_code, 200)
                for query in ("", "   ", "\n\t"):
                    self.assertEqual(client.post("/search", json={"query": query}).status_code, 422)
                for top_k in (0, -1, 11, 1.5, True, "3"):
                    self.assertEqual(client.post("/search", json={"query": "كتاب", "top_k": top_k}).status_code, 422)
                response = client.options("/search", headers={"Origin": "http://localhost:3000",
                                                              "Access-Control-Request-Method": "POST"})
                self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")
                self.assertEqual(client.get("/docs").status_code, 200)
                with patch.object(api, "search", side_effect=RuntimeError("secret internal detail")):
                    response = client.post("/search", json={"query": "كتاب"})
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn("secret", response.text)
            for mocked in (loader, index_loader, meta_loader):
                mocked.assert_called_once()

    def test_startup_missing_files(self):
        for path in (self.metadata_path, self.root / "books.index", self.root / "id_lookup.json"):
            moved = path.with_suffix(".bak")
            path.rename(moved)
            try:
                with self.assertRaisesRegex(RuntimeError, "required file unavailable"):
                    with TestClient(api.create_app()):
                        pass
            finally:
                moved.rename(path)

    def test_cli_search_validation_and_fallback(self):
        for query, count in ((" ", 3), ("book", 0)):
            with self.assertRaises(ValueError):
                search(query, self.index, ["Book", "Missing"], {}, FakeModel(), count)
        results = search("book", self.index, ["Book", "Missing"], {}, FakeModel(), 99)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["category"], "Uncategorized")
        self.assertEqual(search("book", faiss.IndexFlatIP(2), [], {}, FakeModel()), [])

    def test_relative_defaults(self):
        with patch.dict("os.environ", {"BOOK_INDEX_DIR": "data/index_metadata"}):
            self.assertEqual(api.configured_path("BOOK_INDEX_DIR", "unused"),
                             api.BASE_DIR / "data/index_metadata")


if __name__ == "__main__":
    unittest.main()
