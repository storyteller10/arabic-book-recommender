"""Offline regressions for portable setup and unsafe index inputs."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from build_index import build_index, load_index
from project_config import BASE_DIR, project_path
import extract_text


class HandoffTests(unittest.TestCase):
    def test_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                self.assertEqual(project_path("data/metadata.csv"), BASE_DIR / "data/metadata.csv")
                index, lookup = load_index("data/index_metadata")
                self.assertEqual(index.ntotal, len(lookup))
            finally:
                os.chdir(previous)

    def test_rejects_synthetic_and_invalid_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"Book": {"embedding_file": "Book.npy", "dry_run": True}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.save(root / "Book.npy", np.array([1., 0.], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                build_index(root)
            manifest["Book"]["dry_run"] = False
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for vector in ([float("nan"), 1.], [0., 0.], [[1., 0.]]):
                np.save(root / "Book.npy", np.array(vector, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "finite nonzero"):
                    build_index(root)

    def test_tesseract_override_and_missing_language(self):
        with patch.dict(os.environ, {"TESSERACT_CMD": "custom-tesseract"}), \
             patch.object(extract_text.shutil, "which") as which, \
             patch.object(extract_text.pytesseract, "get_languages", return_value=["ara"]):
            self.assertEqual(extract_text.configure_tesseract(), "custom-tesseract")
            which.assert_not_called()
        with patch.dict(os.environ, {"TESSERACT_CMD": "custom-tesseract"}), \
             patch.object(extract_text.pytesseract, "get_languages", return_value=["eng"]):
            with self.assertRaisesRegex(RuntimeError, "ara.traineddata"):
                extract_text.configure_tesseract()


if __name__ == "__main__":
    unittest.main()
