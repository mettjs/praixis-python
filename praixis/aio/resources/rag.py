"""Vector / RAG endpoints - prefix /rag-db (async)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from ..._files import FileInput, to_parts
from ..._http import aiter_stream_events, encode_path_segment
from ...types import AskResponse, Comparison, SearchResponse, StatusMessage, StreamEvent, Summary, UploadResponse
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
        improved_search: bool = False,
    ) -> UploadResponse:
        """POST /rag-db/upload - ingest one or more documents into a collection.

        Each file may be a path, a ``(filename, content)`` pair, or a
        ``(filename, content, content_type)`` triple. Supports .pdf/.docx/.txt;
        the filename extension is the server's primary format signal, with the
        declared content type as the fallback for extension-less names.

        Set ``improved_search=True`` to enable hypothetical-question indexing:
        questions are generated in the background after the upload returns (the
        document is searchable immediately; natural-language matching improves
        once generation finishes).
        """
        parts = to_parts(files, "files")
        fields = [
            ("collection_name", collection_name),
            ("chunk_size", str(chunk_size)),
            ("chunk_overlap", str(chunk_overlap)),
            ("chunking_strategy", chunking_strategy),
            ("improved_search", str(improved_search).lower()),
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
        response_format: Literal["text", "json"] = "text",
    ) -> AskResponse:
        """POST /rag-db/ask - answer a question grounded in a collection.

        Sends ``stream=false`` and returns the server's buffered JSON body
        ``{"session_id", "search_query", "sources", "content"}``. For
        ``response_format="json"``, ``content`` is the model's raw JSON string.

        ``metadata_filter`` restricts retrieval to a single source document; the
        only honored key is ``source`` (e.g. ``{"source": "policy.pdf"}``). Any
        other keys are ignored, not an error.
        """
        body: dict[str, Any] = {
            "collection_name": collection_name,
            "question": question,
            "n_results": n_results,
            "response_format": response_format,
            "stream": False,
        }
        if session_id is not None:
            body["session_id"] = session_id
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if metadata_filter is not None:
            body["metadata_filter"] = metadata_filter
        return await self._t.request_json("POST", f"{_PREFIX}/ask", json_body=body)

    def ask_stream(
        self,
        question: str,
        *,
        collection_name: str,
        session_id: str | None = None,
        n_results: int = 5,
        system_prompt: str | None = None,
        metadata_filter: dict | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[StreamEvent]:
        """POST /rag-db/ask - stream the grounded answer incrementally.

        Yields ``{"type", "value"}`` events: ``"session_id"``, ``"search_query"``
        and ``"sources"`` markers first (``sources`` carries a ``list[str]``),
        then ``"token"`` events carrying the answer.
        """
        body: dict[str, Any] = {
            "collection_name": collection_name,
            "question": question,
            "n_results": n_results,
            "response_format": response_format,
            "stream": True,
        }
        if session_id is not None:
            body["session_id"] = session_id
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if metadata_filter is not None:
            body["metadata_filter"] = metadata_filter
        return aiter_stream_events(self._t.request_stream("POST", f"{_PREFIX}/ask", json_body=body))

    async def search(
        self,
        query: str,
        *,
        collection_name: str,
        n_results: int = 5,
    ) -> SearchResponse:
        """POST /rag-db/search - retrieval only: ranked raw chunks, no LLM.

        Returns the server's buffered JSON body ``{"collection_name", "query",
        "n_results", "results", "score_type"}``, where each result is
        ``{"source", "text", "score"}``. Unlike :meth:`ask` it does not
        reformulate the query or call the model, so pass a standalone query. Use
        it when you want the evidence and its scores to reason over yourself
        (e.g. fusing these chunks with another source) instead of a finished
        answer. ``score_type`` is ``"rrf"`` (hybrid pgvector backend) or
        ``"similarity"`` (dense Chroma backend).
        """
        return await self._t.request_json(
            "POST",
            f"{_PREFIX}/search",
            json_body={"collection_name": collection_name, "query": query, "n_results": n_results},
        )

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

    async def compare(
        self,
        collection_name: str,
        file_1: str,
        file_2: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> Comparison:
        """POST /rag-db/knowledge_base/compare - compare two stored documents.

        Returns the server's buffered JSON body ``{"file_1", "file_2", "content"}``.
        """
        body = {
            "collection_name": collection_name,
            "file_1": file_1,
            "file_2": file_2,
            "response_format": response_format,
        }
        return await self._t.request_json("POST", f"{_PREFIX}/knowledge_base/compare", json_body=body)

    def compare_stream(
        self,
        collection_name: str,
        file_1: str,
        file_2: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[StreamEvent]:
        """POST /rag-db/knowledge_base/compare - stream the comparison incrementally.

        Yields ``{"type", "value"}`` events: ``"progress"`` markers for large
        documents, an ``"error"`` event on an in-stream failure, and ``"token"``
        events carrying the comparison text.
        """
        body = {
            "collection_name": collection_name,
            "file_1": file_1,
            "file_2": file_2,
            "response_format": response_format,
            "stream": True,
        }
        return aiter_stream_events(
            self._t.request_stream("POST", f"{_PREFIX}/knowledge_base/compare", json_body=body)
        )

    async def summarize_document(
        self,
        collection_name: str,
        filename: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> Summary:
        """GET /rag-db/knowledge_base/{collection_name}/files/{filename}/summary.

        Returns the server's buffered JSON body ``{"filename", "content"}``.
        """
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return await self._t.request_json(
            "GET",
            f"{_PREFIX}/knowledge_base/{coll}/files/{name}/summary",
            params={"response_format": response_format},
        )

    def summarize_document_stream(
        self,
        collection_name: str,
        filename: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[StreamEvent]:
        """GET /rag-db/knowledge_base/{collection_name}/files/{filename}/summary -
        stream the document summary incrementally.

        Yields ``{"type", "value"}`` events: a ``"file"`` marker first, then
        ``"progress"`` events for large documents (may repeat), an ``"error"``
        event on an in-stream failure, and ``"token"`` events carrying the
        summary text.
        """
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return aiter_stream_events(self._t.request_stream(
            "GET",
            f"{_PREFIX}/knowledge_base/{coll}/files/{name}/summary",
            params={"response_format": response_format, "stream": "true"},
        ))
