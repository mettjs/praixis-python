"""Vector / RAG endpoints - prefix /rag-db."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from .._files import FileInput, to_parts
from .._http import encode_path_segment, iter_stream_events
from .._transport import Transport
from ..types import (
    AskResponse,
    Comparison,
    FileChunks,
    QuestionRegeneration,
    QuestionStatus,
    SearchResponse,
    StatusMessage,
    StreamEvent,
    Summary,
    TextUploadResponse,
    UploadResponse,
)

_PREFIX = "/rag-db"


class RagResource:
    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def upload(
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
        return self._t.upload(f"{_PREFIX}/upload", files=parts, fields=fields)

    def upload_text(
        self,
        text: str,
        filename: str,
        *,
        collection_name: str = "main",
        chunk_size: int = 2000,
        chunk_overlap: int = 150,
        chunking_strategy: Literal["semantic", "character"] = "semantic",
        improved_search: bool = False,
    ) -> TextUploadResponse:
        """POST /rag-db/upload_text - ingest raw text as a document.

        For callers that already hold the content (scraped pages, database
        records) - no file wrapping needed. Runs the same pipeline as
        :meth:`upload` after text extraction, so the stored document works with
        every other endpoint. ``filename`` becomes the document's stored
        identity; re-using one replaces the prior document. Returns
        ``{"status", "collection_name", "filename", "chunks_stored",
        "improved_search"}``.
        """
        body: dict[str, Any] = {
            "text": text,
            "filename": filename,
            "collection_name": collection_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
            "improved_search": improved_search,
        }
        return self._t.request_json("POST", f"{_PREFIX}/upload_text", json_body=body)

    def get_chunks(self, collection_name: str, filename: str) -> FileChunks:
        """GET /rag-db/{collection_name}/files/{filename}/chunks - stored chunks.

        Returns the document's chunks in order (``{"chunk_index", "content"}``
        each) - exactly what retrieval sees, useful for debugging why a search
        returned what it did.
        """
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return self._t.request_json("GET", f"{_PREFIX}/{coll}/files/{name}/chunks")

    def question_status(self, collection_name: str, filename: str) -> QuestionStatus:
        """GET /rag-db/{collection_name}/files/{filename}/questions - index status.

        Returns ``{"collection_name", "filename", "total_chunks",
        "questions_stored", "generation_pending"}``. ``generation_pending`` is
        True while a background generation pass is running - poll this after
        :meth:`regenerate_questions` (or an upload with
        ``improved_search=True``) to see when matching is fully improved.
        """
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return self._t.request_json("GET", f"{_PREFIX}/{coll}/files/{name}/questions")

    def regenerate_questions(self, collection_name: str, filename: str) -> QuestionRegeneration:
        """POST /rag-db/{collection_name}/files/{filename}/questions - backfill
        or rebuild the hypothetical-question index for a stored document.

        ``improved_search`` is no longer locked in at upload time: a fresh
        question set is generated in the background, and the existing index is
        replaced only once the new pass succeeds (poll :meth:`question_status`
        for progress). Returns ``{"status": "scheduled", "collection_name",
        "filename", "chunks"}``.

        Raises :class:`praixis.APIError` with status 409 while a pass is
        already running, or 400 when question indexing is disabled server-side.
        """
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return self._t.request_json("POST", f"{_PREFIX}/{coll}/files/{name}/questions")

    def ask(
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
        return self._t.request_json("POST", f"{_PREFIX}/ask", json_body=body)

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
    ) -> Iterator[StreamEvent]:
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
        return iter_stream_events(self._t.request_stream("POST", f"{_PREFIX}/ask", json_body=body))

    def search(
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
        return self._t.request_json(
            "POST",
            f"{_PREFIX}/search",
            json_body={"collection_name": collection_name, "query": query, "n_results": n_results},
        )

    def embed(self, text: str) -> dict:
        """POST /rag-db/embed - return the embedding vector for ``text``."""
        return self._t.request_json("POST", f"{_PREFIX}/embed", json_body={"text": text})

    def list_collections(self) -> list:
        """GET /rag-db/list - collections owned by the calling app."""
        data = self._t.request_json("GET", f"{_PREFIX}/list")
        return (data or {}).get("active_collections", [])

    def list_files(self, collection_name: str) -> dict:
        """GET /rag-db/{collection_name}/files - files in a collection."""
        coll = encode_path_segment(collection_name)
        return self._t.request_json("GET", f"{_PREFIX}/{coll}/files")

    def delete_collection(self, collection_name: str) -> StatusMessage:
        """DELETE /rag-db/delete/{collection_name} - remove an entire collection."""
        coll = encode_path_segment(collection_name)
        return self._t.request_json("DELETE", f"{_PREFIX}/delete/{coll}")

    def delete_file(self, collection_name: str, filename: str) -> StatusMessage:
        """DELETE /rag-db/{collection_name}/files/{filename} - remove one file."""
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return self._t.request_json("DELETE", f"{_PREFIX}/{coll}/files/{name}")

    def compare(
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
        return self._t.request_json("POST", f"{_PREFIX}/knowledge_base/compare", json_body=body)

    def compare_stream(
        self,
        collection_name: str,
        file_1: str,
        file_2: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> Iterator[StreamEvent]:
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
        return iter_stream_events(
            self._t.request_stream("POST", f"{_PREFIX}/knowledge_base/compare", json_body=body)
        )

    def summarize_document(
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
        return self._t.request_json(
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
    ) -> Iterator[StreamEvent]:
        """GET /rag-db/knowledge_base/{collection_name}/files/{filename}/summary -
        stream the document summary incrementally.

        Yields ``{"type", "value"}`` events: a ``"file"`` marker first, then
        ``"progress"`` events for large documents (may repeat), an ``"error"``
        event on an in-stream failure, and ``"token"`` events carrying the
        summary text.
        """
        coll = encode_path_segment(collection_name)
        name = encode_path_segment(filename)
        return iter_stream_events(self._t.request_stream(
            "GET",
            f"{_PREFIX}/knowledge_base/{coll}/files/{name}/summary",
            params={"response_format": response_format, "stream": "true"},
        ))
