"""HTTP primitives shared by the sync and async transports.

Keeping these in one place means the auth scheme and the request/response
shapes are defined once and reused, so the two transports can never drift
apart.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote


def encode_path_segment(value: str) -> str:
    """Percent-encode a single URL path segment (filename, session id, ...).

    Uses ``safe=""`` so reserved characters - spaces, ``/``, ``?``, ``#``, ``%``
    - in a value like a filename can't corrupt the URL. Values already limited to
    ``[A-Za-z0-9_.-]`` (collection names, the usual ids) pass through unchanged.
    """
    return quote(str(value), safe="")

# A single file part: (field_name, filename, content_bytes, content_type).
FilePart = tuple[str, str, bytes, str]
# A single form field: (field_name, value).
FormField = tuple[str, str]


def auth_headers(api_key: str) -> dict[str, str]:
    """Build the auth headers for a request.

    Every app endpoint (chat + RAG) authenticates with the app-level
    ``X-API-Key`` header. (Admin routes use HTTP Basic and a browser UI; they're
    deliberately out of scope for this SDK.)
    """
    return {"X-API-Key": api_key} if api_key else {}


# A streamed (text/event-stream) line of the form ``[KEY:value]``. The server
# prefixes its streamed chat / RAG / summary responses with these marker lines
# (e.g. [SESSION_ID:...], [SEARCH_QUERY:...], [SOURCES:a,b], [FILE:...],
# [PROGRESS:...], [ERROR:...]) before emitting the generated text.
_MARKER_RE = re.compile(r"^\[([A-Z_]+):(.*)\]$")


def parse_event_stream(text: str) -> dict[str, Any]:
    """Parse a buffered ``text/event-stream`` body into markers + generated text.

    The chat, RAG-ask and file-summary endpoints don't return JSON - they stream
    plain text whose leading lines are ``[KEY:value]`` markers followed by the
    model's output. Both transports buffer that body and hand it here so the two
    clients shape streamed responses identically.

    Returns a dict with the recognised markers (``session_id``, ``search_query``,
    ``sources`` as a list, ``file``), the joined generated ``text``, and the raw
    ``markers`` map for anything else (e.g. ``PROGRESS``/``ERROR``).
    """
    markers: dict[str, str] = {}
    body_lines: list[str] = []
    for line in text.split("\n"):
        m = _MARKER_RE.match(line)
        if m:
            markers[m.group(1)] = m.group(2)
        else:
            body_lines.append(line)

    sources_raw = markers.get("SOURCES")
    sources = [s for s in sources_raw.split(",") if s] if sources_raw is not None else None
    return {
        "session_id": markers.get("SESSION_ID"),
        "search_query": markers.get("SEARCH_QUERY"),
        "sources": sources,
        "file": markers.get("FILE"),
        "text": "\n".join(body_lines).strip("\n"),
        "markers": markers,
    }
