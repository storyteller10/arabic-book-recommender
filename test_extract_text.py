"""Focused Arabic font-encoding regressions; OCR is mocked."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import unicodedata

import diagnose_pdfs as diagnostic
import extract_text as extractor

ARABIC = "\u062a\u0630\u0643\u0631\u062a \u0644\u064a\u0644\u0649 " * 10
BROKEN = "iilJjijiSiy byI\u00e1\u00e9j gilltJ & \u00e9 \u00e0 ~ \u00f6 " * 5


def document(texts):
    doc = MagicMock()
    pages = [MagicMock() for _ in texts]
    for page, text in zip(pages, texts):
        page.get_text.return_value = text
    doc.page_count = len(pages)
    doc.__iter__.side_effect = lambda: iter(pages)
    doc.__getitem__.side_effect = pages.__getitem__
    return doc, pages


class ArabicExtractionTests(unittest.TestCase):
    def test_validation_in_both_modules(self):
        cases = [
            (ARABIC, True), ("", False), (" \n ", False),
            ("\u0644\u064a\u0644\u0649", False),
            ("iilJjijiSiy byI\u00e1\u00e9j gilltJ", False),
            (BROKEN, False), (ARABIC + " English words", True),
            ("123 () & " * 20, False),
            # Arabic diacritics and digits must not inflate the letter ratio.
            ("abc " * 20 + "\u064e\u0651\u0661\u060c" * 100, False),
            ("\u0628" * 30 + "x" * 70, True),
            ("\u0628" * 29 + "x" * 71, False),
            # NFKC expands presentation-form ligatures before length checking.
            ("\ufefb" * 20, True),
        ]
        for module in (extractor, diagnostic):
            for text, expected in cases:
                with self.subTest(module=module.__name__, text=text):
                    self.assertEqual(module.has_usable_arabic_text(text), expected)

    def run_extraction(self, texts):
        doc, pages = document(texts)
        with patch.object(extractor.pymupdf, "open", return_value=doc), \
             patch.object(extractor, "ocr_page", return_value="\ufefb" * 30) as ocr, \
             contextlib.redirect_stderr(io.StringIO()):
            result = extractor.extract_text_from_pdf(__file__)
        doc.close.assert_called_once()
        called = {i for i, page in enumerate(pages)
                  if any(call.args[0] is page for call in ocr.call_args_list)}
        return result, called

    def diagnosis(self, texts):
        doc, _ = document(texts)
        with patch.object(diagnostic.pymupdf, "open", return_value=doc):
            return diagnostic.diagnose_pdf(Path(__file__))

    def test_sampled_ocr_clears_unsampled_corruption_and_keeps_native(self):
        texts = [BROKEN] * 30
        texts[0] = ARABIC  # valid page inside representative sample
        texts[29] = ARABIC  # valid page outside representative sample
        result, called = self.run_extraction(texts)
        expected_indices = extractor.get_sample_page_indices(30) - {0, 29}
        self.assertEqual(called, expected_indices)
        expected = [ARABIC if i in {0, 29} else "\ufefb" * 30
                    for i in range(30) if i in expected_indices | {0, 29}]
        self.assertEqual(result, unicodedata.normalize("NFKC", "\n\f\n".join(expected)))
        for corrupt in ("\u00f6", "\u00e9", "\u00e0", "iilJjijiSiy"):
            self.assertNotIn(corrupt, result)
        report = self.diagnosis(texts)
        self.assertIn(f"SAMPLED OCR", report["branch"])
        self.assertIn(f"{len(called)}/30", report["branch"])
        self.assertEqual(report["corrupt_text_pages"], 28)

    def test_per_page_ocr_only_replaces_unusable_pages(self):
        texts = [ARABIC, BROKEN, ARABIC, "", ARABIC]
        result, called = self.run_extraction(texts)
        self.assertEqual(called, {1, 3})
        self.assertEqual(result.split("\n\f\n")[::2], [ARABIC] * 3)
        self.assertIn("PER-PAGE FALLBACK", self.diagnosis(texts)["branch"])
        self.assertIn("2/5", self.diagnosis(texts)["branch"])

    def test_native_arabic_preserved_without_ocr(self):
        result, called = self.run_extraction([ARABIC, ARABIC])
        self.assertEqual(called, set())
        self.assertEqual(result, ARABIC + "\n\f\n" + ARABIC)
        self.assertIn("NATIVE TEXT ONLY", self.diagnosis([ARABIC] * 2)["branch"])

    def test_sampled_branch_at_threshold(self):
        texts = [BROKEN] * 7 + [ARABIC] * 3
        _, called = self.run_extraction(texts)
        self.assertEqual(called, set(range(7)))
        self.assertIn("SAMPLED OCR", self.diagnosis(texts)["branch"])


if __name__ == "__main__":
    unittest.main()
