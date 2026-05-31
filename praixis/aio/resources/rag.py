"""Vector / RAG endpoints - prefix /rag-db (async)."""

from __future__ import annotations

from typing import Any, Literal

from ..._files import FileInput, to_parts
from ..._http import encode_path_segment, parse_event_stream
from ...types import AskResponse, StatusMessage, UploadResponse
from .._transport import AsyncTransport

_PREFIX = "/rag-db"


class AsyncRagResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def upload(
        self,
        files: FileInput | list[FileInput],
        *,
        collection_name: str = "main",
        chunk_size: int = 2000,
        chunk_overlap: int = 150,
        chunking_strategy: Literal["semantic", "character"] = "semantic",
    ) -> UploadResponse:
        """POST /rag-db/upload - ingest one or more documents into a collection.

        Each file may be a path, a ``(filename, content)`` pair, or a
        ``(filename, content, content_type)`` triple. Supports .pdf/.docx/.txt.
        """
        parts = to_parts(files, "files")
        fields = [
            ("collection_name", collection_name),
            ("chunk_size", str(chunk_size)),
            ("chunk_overlap", str(chunk_overlap)),
            ("chunking_strategy", chunking_strategy),
        ]
        return await self._t.upload(f"{_PREFIX}/upload", files=parts, fields=fields)

    async def ask(
        self,
        question: str,
        *,
        collection_name: str,
        session_id: str | None = None,
        n_results: int = 5,
        system_prompt: str | None = None,
        metadata_filter: dict | None = None,
    ) -> AskResponse:
        """POST /rag-db/ask - answer a question grounded in a collection.

        The endpoint streams ``text/event-stream``; we buffer it and return
        ``{"answer", "sources", "session_id", "search_query"}``.
        """
        body: dict[str, Any] = {
            "collection_name": collection_name,
            "question": question,
            "n_results": n_results,
        }
        if session_id is not None:
            body["session_id"] = session_id
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if metadata_filter is not None:
            body["metadata_filter"] = metadata_filter
        raw = await self._t.request_json("POST", f"{_PREFIX}/ask", json_body=body, parse="text")
        parsed = parse_event_stream(raw)
        return {
            "answer": parsed["text"],
            "sources": parsed["sources"] or [],
            "session_id": parsed["session_id"],
            "search_query": parsed["search_query"],
        }

    async def embed(self, text: str) -> dict:
        """POST /rag-db/embed - return the embedding vector for ``text``."""
        return await self._t.request_json("POST", f"{_PREFIX}/embed", json_body={"text": text})

    async def list_collections(self) -> list:
        """GET /rag-db/list - collections owned by the calling app."""
        data = await self._t.request_json("GET", f"{_PREFIX}/list")
        return (data or {}).get("active_collections", [])

    async def list_files(self, collection_name: str) -> dict:
        """GET /rag-db/{collection_name}/files - files in a collection."""
        coll = encode_path_segment(collection_name)
        return await self._t.request_json("GET", f"{_PREFIX}/{coll}/files")

    async def delete_collection(self, collection_name: str) -> StatusMessage:
        """DELETE /rag-db/delete/{collection_name} - remove an entire collection."""
        coll = encode_path_segment(collection_name)
        return await self._t.request_json("DELETE", f"{_PREFIX}/delete/{coll}")

    async def delete_file(self, collection_name: str, filename: str) -> StatusMessage:
        """DELETE /rag-db/{collection_name}/files/{filename} - remove one file."""
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return await self._t.request_json("DELETE", f"{_PREFIX}/{coll}/files/{name}")

    async def compare(self, collection_name: str, file_1: str, file_2: str) -> dict:
        """POST /rag-db/knowledge_base/compare - compare two stored documents."""
        body = {"collection_name": collection_name, "file_1": file_1, "file_2": file_2}
        return await self._t.request_json("POST", f"{_PREFIX}/knowledge_base/compare", json_body=body)

    async def summarize_document(self, collection_name: str, filename: str) -> dict:
        """GET /rag-db/knowledge_base/{collection_name}/files/{filename}/summary."""
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return await self._t.request_json(
            "GET", f"{_PREFIX}/knowledge_base/{coll}/files/{name}/summary"
        )
