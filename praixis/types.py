"""TypedDict shapes for documented responses.

These exist purely for editor autocomplete and static checking - every client
method returns the plain ``dict`` parsed from JSON, so unknown/extra fields a
future server adds are never dropped. Shapes that the server defines loosely
(file summaries, embeddings) are returned as ``dict`` without a TypedDict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ChatResponse(TypedDict):
    # Shaped by the client from the streamed reply (the server sends
    # text/event-stream, not JSON).
    session_id: str | None
    response: str
    response_format: str


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
    # Shaped by the client from the streamed reply.
    answer: str
    sources: list[str]
    session_id: str | None
    search_query: str | None
