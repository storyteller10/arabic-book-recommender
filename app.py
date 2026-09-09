"""FastAPI book search. Run: python -m uvicorn app:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from build_index import load_index
from search import MODEL_NAME, load_metadata, load_model, search

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank")
        return value


class SearchResult(BaseModel):
    rank: int
    title: str
    author: str
    category: str
    summary: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    indexed_books: int


@dataclass(frozen=True)
class SearchResources:
    model_name: str
    model: object
    index: object
    lookup: list
    metadata: dict
    inference_lock: Lock


def configured_path(variable, default):
    path = Path(os.environ.get(variable, default)).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def create_app():
    @asynccontextmanager
    async def lifespan(application):
        index_dir = configured_path("BOOK_INDEX_DIR", "data/index_metadata")
        metadata_path = configured_path("BOOK_METADATA_PATH", "data/metadata.csv")
        model_name = os.environ.get("BOOK_MODEL_NAME", MODEL_NAME)
        for path in (index_dir / "books.index", index_dir / "id_lookup.json", metadata_path):
            if not path.is_file():
                raise RuntimeError(f"Book search startup failed: required file unavailable: {path}")
        try:
            index, lookup = load_index(index_dir)
            metadata = load_metadata(metadata_path)
            if not isinstance(lookup, list) or any(not isinstance(key, str) for key in lookup):
                raise ValueError("Index lookup must be a list of book identifiers")
            if index.ntotal != len(lookup) or index.ntotal == 0:
                raise ValueError("Index must be nonempty and match its lookup table")
            model = load_model(model_name)
            if model.get_sentence_embedding_dimension() != index.d:
                raise ValueError("Model embedding dimension does not match the index")
        except Exception as exc:
            raise RuntimeError(f"Book search startup failed: {exc}") from exc
        application.state.resources = SearchResources(model_name, model, index, lookup, metadata, Lock())
        try:
            yield
        finally:
            del application.state.resources

    application = FastAPI(title="Book Search API", lifespan=lifespan)
    default_origins = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    origins = [origin.strip() for origin in os.environ.get("BOOK_ALLOWED_ORIGINS", default_origins).split(",") if origin.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins,
                               allow_credentials=False, allow_methods=["GET", "POST"],
                               allow_headers=["Content-Type"])

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        resources = request.app.state.resources
        return HealthResponse(status="ok", model=resources.model_name,
                              indexed_books=resources.index.ntotal)

    @application.post("/search", response_model=SearchResponse)
    def search_books(payload: SearchRequest, request: Request):
        resources = request.app.state.resources
        try:
            # Keep shared inference safe and bound simultaneous model work.
            # Sync endpoints run in FastAPI's thread pool, not the event loop.
            with resources.inference_lock:
                results = search(payload.query, resources.index, resources.lookup,
                                 resources.metadata, resources.model, payload.top_k)
            return SearchResponse(query=payload.query, results=results)
        except Exception:
            logger.exception("Book search request failed")
            raise HTTPException(status_code=503, detail="Search is temporarily unavailable. Please try again later.") from None

    return application


app = create_app()
