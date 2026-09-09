# Arabic book recommender

A FastAPI backend for a separate frontend. It searches a prepared FAISS index
using `intfloat/multilingual-e5-base`. The current configuration is preserved:
75% PDF/text and 25% metadata, 200-word chunks, up to 8 sampled chunks,
and a 30-page OCR sample for scanned/broken books. There is no similarity cutoff.

## Quick API setup

Use Python 3.13 (the existing environment uses 3.13.7). From the cloned folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_setup.py
.\run_api.ps1
```

Or, with the environment active and terminal in the project directory:

```powershell
uvicorn app:app --reload
```

On Linux/macOS, activate with `source .venv/bin/activate`, then use the same
Python/pip commands and the direct uvicorn command. Package wheels/platform
compatibility still need checking on that operating system.

Open http://127.0.0.1:8000/docs for interactive requests and
http://127.0.0.1:8000/health for readiness. Stop with Ctrl+C.
The API needs only source/dependencies plus `data/metadata.csv`,
`data/index_metadata/books.index`, and `data/index_metadata/id_lookup.json`.
Tesseract, PDFs, extracted/cleaned text, and individual embeddings are not needed
to serve this prepared index. Do not rebuild simply to start the frontend.

`check_setup.py` prints PASS/WARNING/FAIL and exits nonzero on failures. It imports
packages and reads artifacts without loading model weights, downloading a model,
writing outputs, or rebuilding. Metadata mismatch warnings mean display fallbacks
may be used; missing OCR is only a warning for API-only use.

## Optional configuration

`.env.example` documents shell variables; neither the API nor launcher loads a
`.env` file automatically. For example, set `$env:BOOK_INDEX_DIR = 'data/index_metadata'`
in PowerShell before launch. On Bash use `export BOOK_INDEX_DIR=data/index_metadata`.

| Variable | Default / purpose |
| --- | --- |
| BOOK_INDEX_DIR | data/index_metadata |
| BOOK_METADATA_PATH | data/metadata.csv |
| BOOK_MODEL_NAME | intfloat/multilingual-e5-base; must match index model |
| BOOK_ALLOWED_ORIGINS | localhost and 127.0.0.1, ports 3000 and 5173; comma separated |
| TESSERACT_CMD | Optional executable; otherwise PATH, then standard Windows install |
| HF_TOKEN | Optional; improves Hugging Face download limits, never required |

All pipeline input/output paths resolve relative to the project folder, including
relative environment paths. Absolute paths remain supported. CLI `--model` is
explicit and defaults to E5-base; API BOOK_* variables configure the API.
Never put real tokens in Git. `.env` is ignored.

GPU is optional. CPU affects embedding-generation speed, not retrieval quality;
minor floating-point differences across hardware can occur. The first actual
model use may download/cache weights. Startup loads one model per API worker;
search requests reuse it. Reload restarts and multiple workers each load a model.
The setup checker does not warm the model cache. Restart after changing metadata,
index, or environment settings. Startup validates required files, index count,
and model dimensions; equal dimensions alone cannot establish model identity.

## API contract

```http
POST /search
Content-Type: application/json
```

```json
{"query": "كتاب عن السيرة النبوية", "top_k": 3}
```

Response is `{ "query": "...", "results": [...] }`. Each result contains:

| Field | Type / meaning |
| --- | --- |
| rank | integer, starts at 1 |
| title | string, metadata title or filename fallback |
| author | string, defaults to Unknown |
| category | string, defaults to Uncategorized |
| summary | string, from description; default No summary available yet. |
| score | number, inner-product similarity rounded to 4 decimal places |

Query whitespace is trimmed. Blank queries return HTTP 422. `top_k` defaults to
3 and must be a JSON integer from 1 to 10 (strings, booleans and fractions are
invalid); results are capped at the number of indexed books. Unexpected search
failures return HTTP 503 with a generic message and server-side logging.
Health returns `status`, `model`, and `indexed_books`. Dates are not response fields.

```javascript
const response = await fetch("http://127.0.0.1:8000/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "كتاب عن السيرة النبوية", top_k: 3 })
});
if (!response.ok) throw new Error(`Search failed: ${response.status}`);
const { query, results } = await response.json();
console.log(query, results);
```

## Full rebuild (only when inputs change)

Run in this exact order, from the project directory:

```powershell
python diagnose_pdfs.py --folder data/1_trial_books
python extract_text.py --folder data/1_trial_books --out data/extracted_text
python clean_text.py --folder data/extracted_text --out data/cleaned_text
python create_embeddings.py --folder data/cleaned_text --metadata data/metadata.csv --out data/embeddings_metadata
python build_index.py --embeddings data/embeddings_metadata --out data/index_metadata
python search.py --index data/index_metadata --metadata data/metadata.csv --query "كتاب عن السيرة النبوية" --top-k 3
.\run_api.ps1
```

Extraction and cleaning skip existing outputs. For changed PDFs or cleaning rules,
manually arrange fresh output folders and use those folders in downstream commands,
or deliberately refresh the affected output files yourself. Simply rerunning the
commands above does NOT refresh existing extracted/cleaned files. No automatic
removal, renaming, or cleanup is performed by the handoff changes.

| Change | Stages to rerun |
| --- | --- |
| Added PDF | Diagnose, extract, clean, embed, index, CLI test; restart API |
| Changed PDF | Same; ensure its extracted and cleaned outputs are refreshed first |
| Metadata title/author/category/description or filename | Embed with metadata, index, CLI test; restart API |
| Unused metadata column (e.g. publication_date) | No vector rebuild; restart if needed to reload CSV |
| Cleaning rules | Clean into fresh outputs, embed, index, CLI test; restart API |
| Embedding settings/model | Embed, index, CLI test; match API model and restart |
| Frontend only | No backend artifacts to rebuild; adjust CORS origins only if needed |

Metadata requires `filename`; optional fields are `title`, `author`, `description`,
and `category`. Extra columns are ignored. Save CSV as UTF-8 (BOM accepted).
Names match using NFKC normalization, trimming, and removal of .pdf/.txt.
Duplicate normalized names are errors. Blank rows and NaN fields are handled.
Missing metadata uses text-only embeddings; empty text can use metadata alone.
Book vectors average normalized E5 `passage:` chunks, blend a separate metadata
passage, then normalize again. Queries use `query:`; ranking uses exact IndexFlatIP.

The manifest fingerprints cleaned bytes, metadata, model, chunks, weight and
dry-run status. Only changed entries regenerate. Unreferenced NPY files are not
indexed. `--dry-run` produces synthetic vectors; keep these in separate scratch
folders. Index building rejects entries marked dry-run. A real embedding pass
regenerates synthetic vectors. Legacy manifests cannot prove provenance.
The old `data/index` and `data/embeddings` remain available for explicit experiments;
default CLI index paths now match the API's metadata-aware index.

Install Tesseract and Arabic `ara` language data only for OCR rebuilds. Resolution
order is TESSERACT_CMD, system PATH, then `C:\Program Files\Tesseract-OCR\tesseract.exe`.
The checker reports missing executable/languages; OCR errors explain how to fix
configuration. `TESSDATA_PREFIX` can point to installed language data if necessary.
Sampling and OCR language/DPI settings remain unchanged.

## Known artifact mismatch

The current index/cleaned files use `من معين السيرة صالح احمد الشامي`, while the
current PDF/metadata use `من معين السيرة`. This book currently receives metadata
fallbacks and its stored vector is text-only. The checker reports both unmatched
IDs. Resolving the names and rebuilding would change results, so the handoff
preserves the current artifacts. Before a future rebuild, choose one consistent
identifier and deliberately refresh that book's derived artifacts; avoid leaving
both names as active cleaned files.

## What to share on GitHub

Commit the source, tests, README, launch/setup files, requirements, .gitignore,
.env.example, and these small runtime artifacts together:

| Artifact | Current bytes |
| --- | ---: |
| data/metadata.csv | 27,401 |
| data/index_metadata/books.index | 30,765 |
| data/index_metadata/id_lookup.json | 435 |

PDFs and extracted/cleaned text are unnecessary for frontend API use; share them
separately when needed for rebuilding and when redistribution is permitted.
Individual NPY vectors are also unnecessary for serving; keep them plus their
manifest only if you want reproducibility/provenance inspection without re-embedding.
Old text-only indexes, backups, dry runs, environments, model caches, and personal
notes (`updates.txt`, `request_search_format.txt`, `Final Books/`) are not handoff
requirements. The notes describe older behavior; this README is the current contract.
Ignore rules do not untrack files already committed. Review tracked files yourself;
no Git mutations are performed by this review.

GitHub warns above 50 MiB and blocks files above 100 MiB in ordinary Git.
See [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
The current largest project PDF is 69,378,053 bytes (about 66.16 MiB), above the
warning size. Local environment binaries can exceed the hard limit and must remain
ignored. See HANDOFF_REVIEW.md for validation results and inventory.

## Tests

```powershell
python -m unittest discover -v
python check_setup.py
```

Existing tests use deterministic fake models and real FAISS indexes, exercising
metadata, NaN/duplicates, embedding fingerprints, API responses/validation, resource
reuse, and mocked OCR. Real-model validation is reported separately in the review.
