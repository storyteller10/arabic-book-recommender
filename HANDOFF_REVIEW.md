# Handoff review — 2026-09-09

## Outcome

Prepared local source and documentation for developer handoff. No commit, push,
repository-file deletion, rename, untracking, or real embedding rebuild was performed.
Pre-existing edits/untracked files and the pre-existing PDF filename change remain.
The API source itself was already correct and was not edited in this review.

## Files inspected

Read all project Python files: diagnose_pdfs.py, extract_text.py, clean_text.py,
create_embeddings.py, build_index.py, search.py, app.py, metadata.py,
test_extract_text.py and test_book_api.py. Read requirements.txt, .gitignore,
README.md, request_search_format.txt and updates.txt. Inspected the data directory
inventory and sizes, metadata CSV through its parser, embedding manifests and
vectors, index/lookup pairs, old artifact directories, Final Books directory
inventory, Git status and tracked-file inventory. No AGENTS.md was found in the
repository or checked ancestor directories. Reviewed all newly created files too.
PDFs/text corpora were inventoried as artifacts; this was not a content-by-content
literary review of every book. No production extraction/OCR was rerun.

## Files changed or created

| Files | Change |
| --- | --- |
| project_config.py (new) | Shared project-relative path resolver |
| extract_text.py | Portable paths; lazy Tesseract selection and language errors |
| diagnose_pdfs.py | Project-relative input paths |
| clean_text.py | Project-relative paths; corrected disabled-filter docstring |
| create_embeddings.py | Project-relative input/output paths only |
| metadata.py | Project-relative CSV path; explicit UTF-8/BOM decoding |
| build_index.py | Portable paths, metadata-aware default, rejects synthetic/invalid vectors |
| search.py | CLI index default now data/index_metadata |
| .gitignore | Environment/cache/editor/OCR temporary/backup/dry-run patterns |
| .env.example (new) | Optional documented variables, no credentials |
| run_api.ps1 (new) | Activates local environment, sets safe defaults, launches reload server |
| check_setup.py (new) | Read-only dependency/configuration/artifact/OCR checks |
| test_handoff.py (new) | Three focused portability and input-validation regressions |
| README.md | Beginner API/rebuild/frontend contract and artifact handoff instructions |
| HANDOFF_REVIEW.md (new) | This review and validation record |

app.py, requirements.txt, existing tests, personal notes, metadata data, PDFs,
extracted/cleaned text, and index/embedding files were preserved.

## Bugs and convenience improvements

Relative pipeline paths formerly depended on the terminal directory. They now
resolve against the repository, with absolute paths still supported. API path
behavior was already correct. CLI search/index output defaults now agree with the
metadata-aware API; explicitly passing the old paths still selects old experiments.

Extraction previously checked Tesseract at import time, preferred the Windows
fallback over PATH, and ignored TESSERACT_CMD. It now resolves explicit setting,
then PATH, then Windows fallback, and checks language data before OCR. Native-only
work and imports do not fail just because OCR is absent. OCR page selection,
DPI, language, thresholds and normalization are unchanged.

Index building previously allowed manifest entries explicitly marked dry-run and
nonfinite/zero/malformed vectors. It now rejects those inputs without changing
valid vectors or their order. Real embedding cache invalidation already prevented
reuse of dry-run vectors and was preserved. Legacy manifests without provenance
remain a limitation; dimensions alone do not prove model compatibility.

Metadata already handled NaN, missing fields, Unicode-normalized identifiers and
duplicates correctly. Explicit UTF-8/BOM parsing makes the file contract clear.
No token or personal absolute source path was found by the source/config pattern
scan. This is not a forensic audit of Git history or binary files. Historical
notes contain old sandbox links and obsolete contracts and remain ignored.

## Validation results

- All 13 project Python files passed python -m py_compile.
- All 19 unittest tests passed (16 existing, 3 added).
- check_setup.py: zero failures; imports, 10-book/768-dimensional index, lookup,
  model/75:25/200-word/8-chunk configuration, OCR sample and Arabic data passed.
- All 10 current index vectors exactly match their corresponding real NPY vectors.
  Manifest settings match the preserved configuration; text/metadata fingerprints
  match current inputs. No real data files were rebuilt.
- Metadata has 20 usable filename rows. Nine match current indexed books; one
  indexed filename has no metadata, and 11 metadata IDs are not indexed.
- Every requirement pin matches the installed version; pip check reports no broken
  requirements. No package installation or model download was needed.
- Real cached-model FastAPI lifespan and GET /health, GET /docs, POST /search
  passed through TestClient. Blank query and top_k 0, 11, string, boolean and
  fractional values returned 422. Repeated search reused a single model load.
- run_api.ps1 passed PowerShell parser validation. Socket-bound uvicorn reload
  and the friend's actual frontend were not exercised; endpoint tests used ASGI
  TestClient with the real model and data.
- Relative path operation was tested from a temporary working directory.
- Git diff whitespace check passed with CRLF accepted (core.whitespace=cr-at-eol).
- Arabic examples/source were checked for replacement characters and mojibake.
- Fresh dependency installation and Linux/macOS execution were not performed.

The real baseline query was:

> أريد كتابا يساعدني على التغلب على الكسل وزيادة النشاط والإنجاز

| Rank | Book | Before | After |
| --- | --- | ---: | ---: |
| 1 | الحرب على الكسل | 0.8791 | 0.8791 |
| 2 | الرجل النبيل | 0.8479 | 0.8479 |
| 3 | لأنك الله | 0.8476 | 0.8476 |

Ranking and API-rounded scores are exactly unchanged. Unrounded scores were not
captured for the baseline. Model choice, weights, chunking, normalization, ranking,
response schemas and threshold behavior are unchanged. Selecting the CLI default
now searches the metadata index instead of the old baseline; that intentional
convenience change can change CLI results when --index is omitted.

Non-blocking warnings: installed Starlette deprecates its HTTPX TestClient path,
and sentence-transformers deprecates get_sentence_embedding_dimension in favor
of get_embedding_dimension. Both current calls work and remain unchanged.
The first imports were slow while Transformers inspected package files on the
Drive-backed environment; imports and model loading ultimately completed.

## Remaining artifact mismatch

The index/cleaned filename is `من معين السيرة صالح احمد الشامي`; the current PDF
and metadata use `من معين السيرة`. The stored vector therefore uses text only,
and display metadata falls back for this indexed book. Fixing names and regenerating
its vector would affect results, so this review reports it instead of changing it.
Before a deliberate future rebuild, align that identity and refresh its derived
outputs without leaving both names active. The other unindexed metadata rows are
not searchable until their corresponding book inputs are processed.

Extraction and cleaning still skip existing outputs. The README explains that
changed inputs require refreshed outputs/fresh folders before downstream stages.

## GitHub handoff

Commit source, tests, launch/setup files, README/review, requirements and .env.example,
plus these runtime files as one consistent set:

| Artifact | Bytes |
| --- | ---: |
| data/metadata.csv | 27,401 |
| data/index_metadata/books.index | 30,765 |
| data/index_metadata/id_lookup.json | 435 |

Do not include .venv, caches/model weights, real .env/tokens, OCR scratch files,
backups, synthetic outputs or personal notes. PDFs, extracted/cleaned text and
individual NPY embeddings are not required by the frontend/API handoff. Prefer a
separate rebuild-data bundle; include NPY plus manifest only if useful for audit
and avoiding re-embedding. Share book content only when redistribution is permitted.
Do not automatically ignore the small working metadata index. Ignore rules leave
already tracked files tracked; the owner must decide what to stage later.

No project data file exceeds GitHub's ordinary 100 MiB limit. The largest PDF,
`data/1_trial_books/من معين السيرة.pdf`, is 69,378,053 bytes (66.16 MiB), above the
50 MiB warning size. The only local file above 100 MiB found in the recursive scan
is ignored `.venv/Lib/site-packages/torch/lib/torch_cpu.dll`, 305,081,856 bytes.
Limits verified in [GitHub documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Friend's exact commands after cloning

From the cloned repository directory, with Python 3.13 available:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_setup.py
.\run_api.ps1
```

Then visit http://127.0.0.1:8000/docs or http://127.0.0.1:8000/health.
The prepared runtime files must be included in the clone. No HF token or OCR
installation is required for API-only use. Model weights may download on first use.
