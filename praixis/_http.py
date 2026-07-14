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


# The server's streamed (text/event-stream) responses are NOT JSON. They begin
# with zero or more single-line ``[KEY:value]`` markers and are followed by the
# raw generated content:
#
#   [SESSION_ID:<id>]\n
#   [SEARCH_QUERY:<query>]\n     (RAG ask only)
#   [SOURCES:<a.txt,b.txt>]\n    (RAG ask only)
#   [FILE:<filename>]\n          (file / document summary only)
#   [PROGRESS:<message>]\n       (file summary, large docs; may repeat)
#   [ERROR:<message>]\n          (in-stream failure; always before content)
#   ...content tokens...
#
# Markers are emitted on their own ``\n``-terminated lines before any content,
# so we peel complete marker lines off the head of the stream and treat
# everything from the first non-marker byte onward as content.
#
# Items inside the comma-separated SOURCES value are escaped by the server
# (v2.3.0+): ``%`` -> ``%25``, ``,`` -> ``%2C``, ``]`` -> ``%5D``, ``\n`` ->
# ``%0A``, ``\r`` -> ``%0D``, so filenames containing those characters can't
# corrupt the list or the marker line. Decoding applies the reverse
# replacements with ``%25`` last.

_SOURCE_UNESCAPES = (("%2C", ","), ("%5D", "]"), ("%0A", "\n"), ("%0D", "\r"), ("%25", "%"))


def _decode_source(item: str) -> str:
    for escaped, raw in _SOURCE_UNESCAPES:
        item = item.replace(escaped, raw)
    return item

_STREAM_MARKER_KEYS = ("SESSION_ID", "SEARCH_QUERY", "SOURCES", "FILE", "PROGRESS", "ERROR")

# A complete leading marker line: ``[KEY:value]\n``.
_STREAM_MARKER_RE = re.compile(r"^\[(" + "|".join(_STREAM_MARKER_KEYS) + r"):([^\n]*)\]\n")

# A buffer that is still a possible (incomplete) marker line: no ``\n`` yet.
_PARTIAL_MARKER_RE = re.compile(r"^\[[A-Z_]*(:[^\n]*)?$")


def _marker_event(key: str, value: str) -> dict[str, Any]:
    if key == "SOURCES":
        return {"type": "sources", "value": [_decode_source(s) for s in value.split(",") if s]}
    return {"type": key.lower(), "value": value}


class StreamEventAssembler:
    """Incremental parser turning decoded text chunks into stream events.

    ``feed`` each chunk as it arrives and yield the returned events; call
    ``finish`` once the stream ends to flush anything still buffered. Events are
    ``{"type", "value"}`` dicts: the marker events above (``sources`` carries a
    ``list[str]``), then ``token`` events with the generated content. Shared by
    the sync and async clients so the two shape streams identically.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_content = False

    def _drain_markers(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while (m := _STREAM_MARKER_RE.match(self._buffer)) is not None:
            events.append(_marker_event(m.group(1), m.group(2)))
            self._buffer = self._buffer[m.end():]
        return events

    def feed(self, chunk: str) -> list[dict[str, Any]]:
        if self._in_content:
            return [{"type": "token", "value": chunk}] if chunk else []
        self._buffer += chunk
        events = self._drain_markers()
        # The buffer no longer starts with a complete marker. If it can't still
        # grow into one either, the marker section is over: the rest is content.
        if self._buffer and not _PARTIAL_MARKER_RE.match(self._buffer):
            events.append({"type": "token", "value": self._buffer})
            self._buffer = ""
            self._in_content = True
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self._drain_markers()
        if self._buffer:
            events.append({"type": "token", "value": self._buffer})
            self._buffer = ""
        return events


def iter_stream_events(chunks):
    """Turn an iterable of decoded text chunks into an iterator of events."""
    assembler = StreamEventAssembler()
    for chunk in chunks:
        yield from assembler.feed(chunk)
    yield from assembler.finish()


async def aiter_stream_events(chunks):
    """Async variant of :func:`iter_stream_events`."""
    assembler = StreamEventAssembler()
    async for chunk in chunks:
        for event in assembler.feed(chunk):
            yield event
    for event in assembler.finish():
        yield event
