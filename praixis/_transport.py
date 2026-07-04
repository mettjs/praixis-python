"""Low-level synchronous HTTP transport built entirely on the standard library.

This module depends on nothing outside ``urllib``, ``json`` and ``uuid`` (auth
and the file-part shape are shared via ``praixis._http``), so the sync client
cannot be broken by an upstream package release.

Every request authenticates with the app-level ``X-API-Key`` header.
"""

from __future__ import annotations

import codecs
import json
import uuid
from collections.abc import Iterator
from typing import Any, Literal
from urllib import error as urllib_error
from urllib import parse, request

ParseMode = Literal["json", "text"]

from .errors import APIConnectionError, error_for_status

# Re-exported so existing imports (e.g. ``from ._transport import FilePart``)
# keep working; the canonical definitions live in ``_http``.
from ._http import FilePart, FormField, auth_headers  # noqa: F401


class Transport:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # -- helpers ---------------------------------------------------------

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = self.base_url + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + parse.urlencode(clean)
        return url

    def _auth_headers(self) -> dict[str, str]:
        return auth_headers(self.api_key)

    def _decode_error(self, exc: urllib_error.HTTPError):
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except (ValueError, TypeError):
            pass
        return error_for_status(exc.code, raw, detail)

    def _send(self, req: request.Request, *, parse: ParseMode = "json") -> Any:
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib_error.HTTPError as exc:
            raise self._decode_error(exc) from None
        except urllib_error.URLError as exc:
            raise APIConnectionError(f"failed to reach {self.base_url}: {exc.reason}", cause=exc) from exc
        if parse == "text":
            # Streaming endpoints (chat / RAG ask / file summary) reply with a
            # text/event-stream body, not JSON. Return it decoded.
            return body.decode("utf-8") if body else ""
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _stream_body(self, req: request.Request) -> Iterator[str]:
        """Open ``req`` and yield its body incrementally as decoded text chunks.

        A generator, so the request is sent on the first iteration; connection
        and HTTP errors surface there with the same mapping as buffered calls.
        The configured ``timeout`` applies per read - an idle bound, not a cap
        on the stream's total duration.
        """
        try:
            resp = request.urlopen(req, timeout=self.timeout)
        except urllib_error.HTTPError as exc:
            raise self._decode_error(exc) from None
        except urllib_error.URLError as exc:
            raise APIConnectionError(f"failed to reach {self.base_url}: {exc.reason}", cause=exc) from exc
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                try:
                    # read1: return whatever bytes are available instead of
                    # blocking to fill the buffer - tokens flow as they arrive.
                    raw = resp.read1(65536)
                except OSError as exc:
                    raise APIConnectionError(
                        f"stream from {self.base_url} interrupted: {exc}", cause=exc
                    ) from exc
                if not raw:
                    break
                text = decoder.decode(raw)
                if text:
                    yield text
            tail = decoder.decode(b"", True)
            if tail:
                yield tail
        finally:
            resp.close()

    # -- requests --------------------------------------------------------

    def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        parse: ParseMode = "json",
    ) -> Any:
        headers = self._auth_headers()
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self._url(path, params), data=data, method=method, headers=headers)
        return self._send(req, parse=parse)

    def request_stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Like :meth:`request_json`, but yield the response body incrementally
        as decoded text chunks, for the server's streamed (text/event-stream)
        endpoints which are not JSON."""
        headers = self._auth_headers()
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self._url(path, params), data=data, method=method, headers=headers)
        return self._stream_body(req)

    def upload(
        self,
        path: str,
        *,
        files: list[FilePart],
        fields: list[FormField] | None = None,
        params: dict[str, Any] | None = None,
        parse: ParseMode = "json",
    ) -> Any:
        boundary = f"----praixis{uuid.uuid4().hex}"
        body = _encode_multipart(boundary, fields or [], files)
        headers = self._auth_headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = request.Request(self._url(path, params), data=body, method="POST", headers=headers)
        return self._send(req, parse=parse)

    def upload_stream(
        self,
        path: str,
        *,
        files: list[FilePart],
        fields: list[FormField] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Like :meth:`upload`, but yield the response body incrementally as
        decoded text chunks."""
        boundary = f"----praixis{uuid.uuid4().hex}"
        body = _encode_multipart(boundary, fields or [], files)
        headers = self._auth_headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = request.Request(self._url(path, params), data=body, method="POST", headers=headers)
        return self._stream_body(req)


def _encode_multipart(boundary: str, fields: list[FormField], files: list[FilePart]) -> bytes:
    """Build a multipart/form-data body supporting many fields and many files."""
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")
    for name, filename, content, content_type in files:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)
