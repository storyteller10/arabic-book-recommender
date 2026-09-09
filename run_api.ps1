$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$activation = Join-Path $PSScriptRoot '.venv/Scripts/Activate.ps1'
if (Test-Path -LiteralPath $activation) { . $activation }
if (-not $env:BOOK_INDEX_DIR) { $env:BOOK_INDEX_DIR = 'data/index_metadata' }
if (-not $env:BOOK_METADATA_PATH) { $env:BOOK_METADATA_PATH = 'data/metadata.csv' }
$env:PYTHONIOENCODING = 'utf-8'
python -m uvicorn app:app --reload
exit $LASTEXITCODE
