"""Core AI endpoints - prefix /general-requests (async)."""

from __future__ import annotations

from typing import Any, Literal

from ..._files import FileInput, to_part
from ..._http import encode_path_segment, parse_event_stream
from ...types import ChatResponse, SessionHistory, StatusMessage
from .._transport import AsyncTransport

_PREFIX = "/general-requests"


class AsyncChatResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def send(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        session_id: str | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> ChatResponse:
        """POST /general-requests/chat - send a prompt and get the full reply.

        The endpoint streams ``text/event-stream``; we buffer it and return the
        decoded reply as ``{"session_id", "response", "response_format"}``.

        Omit ``session_id`` to start a new conversation; the returned
        ``session_id`` identifies it for follow-ups.
        """
        body: dict[str, Any] = {"prompt": prompt, "response_format": response_format}
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if session_id is not None:
            body["session_id"] = session_id
        raw = await self._t.request_json("POST", f"{_PREFIX}/chat", json_body=body, parse="text")
        parsed = parse_event_stream(raw)
        return {
            "session_id": parsed["session_id"],
            "response": parsed["text"],
            "response_format": response_format,
        }

    async def summarize_file(
        self,
        file: FileInput,
        *,
        task: str = "Summarize the key points of this document.",
        tone: str = "Professional and objective",
    ) -> dict:
        """POST /general-requests/file_summary - summarize one uploaded file.

        ``file`` may be a path, a ``(filename, content)`` pair, or a
        ``(filename, content, content_type)`` triple.

        The endpoint streams ``text/event-stream``; we buffer it and return
        ``{"filename", "summary"}``.
        """
        part = to_part(file, "file")
        raw = await self._t.upload(
            f"{_PREFIX}/file_summary",
            files=[part],
            fields=[("task", task), ("tone", tone)],
            parse="text",
        )
        parsed = parse_event_stream(raw)
        return {"filename": parsed["file"], "summary": parsed["text"]}

    async def list_sessions(self) -> list[str]:
        """GET /general-requests/chat/sessions/active - active session IDs."""
        data = await self._t.request_json("GET", f"{_PREFIX}/chat/sessions/active")
        return (data or {}).get("active_sessions", [])

    async def get_history(self, session_id: str) -> SessionHistory:
        """GET /general-requests/chat/{session_id} - a session's message history."""
        sid = encode_path_segment(session_id)
        return await self._t.request_json("GET", f"{_PREFIX}/chat/{sid}")

    async def clear_history(self, session_id: str) -> StatusMessage:
        """DELETE /general-requests/chat/{session_id} - clear a session."""
        sid = encode_path_segment(session_id)
        return await self._t.request_json("DELETE", f"{_PREFIX}/chat/{sid}")
