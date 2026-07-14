"""Core AI endpoints - prefix /general-requests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from .._files import FileInput, to_part
from .._http import encode_path_segment, iter_stream_events
from .._transport import Transport
from ..types import (
    ChatResponse,
    CompactionResult,
    SessionHistory,
    SessionUsage,
    StatusMessage,
    StreamEvent,
    Summary,
    UndoResult,
)

_PREFIX = "/general-requests"


class ChatResource:
    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def send(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        session_id: str | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> ChatResponse:
        """POST /general-requests/chat - send a prompt and get the full reply.

        Sends ``stream=false`` and returns the server's buffered JSON body
        ``{"session_id", "content"}``. For ``response_format="json"``, ``content``
        is the model's raw JSON string - parse it yourself.

        Omit ``session_id`` to start a new conversation; the returned
        ``session_id`` identifies it for follow-ups.
        """
        body: dict[str, Any] = {"prompt": prompt, "response_format": response_format, "stream": False}
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if session_id is not None:
            body["session_id"] = session_id
        return self._t.request_json("POST", f"{_PREFIX}/chat", json_body=body)

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        session_id: str | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> Iterator[StreamEvent]:
        """POST /general-requests/chat - stream the reply incrementally.

        Yields ``{"type", "value"}`` events: a ``"session_id"`` marker first,
        then ``"token"`` events carrying pieces of the generated content::

            for event in client.chat.stream("Tell me a story"):
                if event["type"] == "token":
                    print(event["value"], end="")
        """
        body: dict[str, Any] = {"prompt": prompt, "response_format": response_format, "stream": True}
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if session_id is not None:
            body["session_id"] = session_id
        return iter_stream_events(self._t.request_stream("POST", f"{_PREFIX}/chat", json_body=body))

    def summarize_file(
        self,
        file: FileInput,
        *,
        task: str = "Summarize the key points of this document.",
        tone: str = "Professional and objective",
        response_format: Literal["text", "json"] = "text",
    ) -> Summary:
        """POST /general-requests/file_summary - summarize one uploaded file.

        ``file`` may be a path, a ``(filename, content)`` pair, or a
        ``(filename, content, content_type)`` triple.

        Sends ``stream=false`` and returns the server's buffered JSON body
        ``{"filename", "content"}``.
        """
        part = to_part(file, "file")
        return self._t.upload(
            f"{_PREFIX}/file_summary",
            files=[part],
            fields=[
                ("task", task),
                ("tone", tone),
                ("response_format", response_format),
                ("stream", "false"),
            ],
        )

    def summarize_file_stream(
        self,
        file: FileInput,
        *,
        task: str = "Summarize the key points of this document.",
        tone: str = "Professional and objective",
        response_format: Literal["text", "json"] = "text",
    ) -> Iterator[StreamEvent]:
        """POST /general-requests/file_summary - stream the summary incrementally.

        Yields ``{"type", "value"}`` events: a ``"file"`` marker first, then
        ``"progress"`` events for large documents (may repeat), an ``"error"``
        event on an in-stream failure (e.g. GPU busy), and ``"token"`` events
        carrying the summary text.
        """
        part = to_part(file, "file")
        return iter_stream_events(self._t.upload_stream(
            f"{_PREFIX}/file_summary",
            files=[part],
            fields=[
                ("task", task),
                ("tone", tone),
                ("response_format", response_format),
                ("stream", "true"),
            ],
        ))

    def list_sessions(self) -> list[str]:
        """GET /general-requests/chat/sessions/active - active session IDs."""
        data = self._t.request_json("GET", f"{_PREFIX}/chat/sessions/active")
        return (data or {}).get("active_sessions", [])

    def get_history(self, session_id: str) -> SessionHistory:
        """GET /general-requests/chat/{session_id} - a session's message history."""
        sid = encode_path_segment(session_id)
        return self._t.request_json("GET", f"{_PREFIX}/chat/{sid}")

    def get_usage(self, session_id: str) -> SessionUsage:
        """GET /general-requests/chat/{session_id}/usage - a session's token usage.

        Returns ``{"session_id", "requests", "prompt_tokens", "completion_tokens",
        "total_tokens", "estimated_context_tokens"}``. Counters cover the streamed
        answer (chat and RAG), RAG query reformulation, and compaction calls, and
        expire with the session. ``estimated_context_tokens`` is the current
        history size against the server's context budget - how close the session
        is to auto-compacting.
        """
        sid = encode_path_segment(session_id)
        return self._t.request_json("GET", f"{_PREFIX}/chat/{sid}/usage")

    def compact(self, session_id: str) -> CompactionResult:
        """POST /general-requests/chat/{session_id}/compact - compact a session now.

        Folds older exchanges into an LLM-written summary (the server also does
        this automatically near its context budget). Returns ``{"status",
        "session_id", "messages_before", "messages_after",
        "estimated_tokens_before", "estimated_tokens_after"}``.

        Raises :class:`praixis.APIError` with status 400 when there is nothing to
        fold yet, or 503 when the GPU pool is saturated.
        """
        sid = encode_path_segment(session_id)
        return self._t.request_json("POST", f"{_PREFIX}/chat/{sid}/compact")

    def undo_last_exchange(self, session_id: str) -> UndoResult:
        """DELETE /general-requests/chat/{session_id}/last - undo the last exchange.

        Removes the most recent user message and the assistant reply that
        followed it (or just the user message if generation failed), so you can
        retry or regenerate. Compaction summaries are kept. Returns ``{"status",
        "session_id", "removed_messages", "undone_prompt",
        "messages_remaining"}``; ``undone_prompt`` is the removed user message.

        Raises :class:`praixis.APIError` with status 400 when the session has no
        user messages left to undo.
        """
        sid = encode_path_segment(session_id)
        return self._t.request_json("DELETE", f"{_PREFIX}/chat/{sid}/last")

    def clear_history(self, session_id: str) -> StatusMessage:
        """DELETE /general-requests/chat/{session_id} - clear a session."""
        sid = encode_path_segment(session_id)
        return self._t.request_json("DELETE", f"{_PREFIX}/chat/{sid}")
