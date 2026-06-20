"""TypedDict shapes for documented responses.

These exist purely for editor autocomplete and static checking - every client
method returns the plain ``dict`` parsed from JSON, so unknown/extra fields a
future server adds are never dropped. Shapes that the server defines loosely
(file summaries, embeddings) are returned as ``dict`` without a TypedDict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ChatResponse(TypedDict):
    # The server's buffered (stream=false) JSON body. For response_format="json",
    # ``content`` is the model's raw JSON string - parse it yourself.
    session_id: str | None
    content: str


class SessionHistory(TypedDict):
    session_id: str
    history: list[dict[str, Any]]


class StatusMessage(TypedDict, total=False):
    # The server's status payloads vary by route: some carry ``message``
    # (collection/file deletes), others ``detail`` (session deletes).
    status: str
    message: str
    detail: str


class UploadedFileResult(TypedDict, total=False):
    filename: str | None
    status: str  # "success" or "error"
    detail: str  # present on per-file failures


class UploadResponse(TypedDict):
    collection_name: str
    processed: int
    succeeded: int
    results: list[UploadedFileResult]


class AskResponse(TypedDict):
    # The server's buffered (stream=false) JSON body. For response_format="json",
    # ``content`` is the model's raw JSON string - parse it yourself.
    session_id: str | None
    search_query: str | None
    sources: list[str]
    content: str


class SearchResult(TypedDict):
    # One ranked chunk from the retrieval-only search endpoint.
    source: str  # filename the chunk came from
    text: str  # the chunk's raw text
    score: float  # ranking score; read against ``score_type``


class SearchResponse(TypedDict):
    # Buffered body of the retrieval-only search endpoint. ``score_type`` is
    # "rrf" (hybrid pgvector backend; small values, higher is better) or
    # "similarity" (dense Chroma backend; 0-1), telling you how to read each
    # result's ``score``.
    collection_name: str
    query: str
    n_results: int
    results: list[SearchResult]
    score_type: str


class Summary(TypedDict):
    # Buffered body of file_summary and document summary endpoints.
    filename: str
    content: str


class Comparison(TypedDict):
    # Buffered body of the compare endpoint.
    file_1: str
    file_2: str
    content: str
