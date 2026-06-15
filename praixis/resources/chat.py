"""Core AI endpoints - prefix /general-requests."""

from __future__ import annotations

from typing import Any, Literal

from .._files import FileInput, to_part
from .._http import encode_path_segment
from .._transport import Transport
from ..types import ChatResponse, SessionHistory, StatusMessage, Summary

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

    def list_sessions(self) -> list[str]:
        """GET /general-requests/chat/sessions/active - active session IDs."""
        data = self._t.request_json("GET", f"{_PREFIX}/chat/sessions/active")
        return (data or {}).get("active_sessions", [])

    def get_history(self, session_id: str) -> SessionHistory:
        """GET /general-requests/chat/{session_id} - a session's message history."""
        sid = encode_path_segment(session_id)
        return self._t.request_json("GET", f"{_PREFIX}/chat/{sid}")

    def clear_history(self, session_id: str) -> StatusMessage:
        """DELETE /general-requests/chat/{session_id} - clear a session."""
        sid = encode_path_segment(session_id)
        return self._t.request_json("DELETE", f"{_PREFIX}/chat/{sid}")
