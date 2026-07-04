"""Async HTTP transport built on httpx.

Mirrors the sync :class:`praixis._transport.Transport` method-for-method - the
only difference is that requests are awaited. Auth, the file-part shape, and the
error classes are shared with the sync client via ``praixis._http`` and
``praixis.errors`` so the two transports can never disagree on behavior.

httpx is an optional dependency, pulled in only when the async client is used
(see ``praixis[async]``). Its exceptions are caught here and re-raised as the
SDK's own ``APIConnectionError`` / ``APIError`` subclasses, so callers catch the
same ``praixis`` exceptions regardless of which client they chose.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from .._http import FilePart, FormField, auth_headers
from ..errors import APIConnectionError, error_for_status

ParseMode = Literal["json", "text"]


class AsyncTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    # -- helpers ---------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return auth_headers(self.api_key)

    @staticmethod
    def _clean(params: dict[str, Any] | None) -> dict[str, Any] | None:
        if not params:
            return None
        clean = {k: v for k, v in params.items() if v is not None}
        return clean or None

    def _handle(self, resp: httpx.Response, *, parse: ParseMode = "json") -> Any:
        if resp.status_code >= 400:
            raise self._error(resp)
        if parse == "text":
            # Streaming endpoints reply with text/event-stream, not JSON; return
            # it decoded.
            return resp.text
        if not resp.content:
            return None
        return resp.json()

    def _error(self, resp: httpx.Response):
        body = resp.text
        detail = None
        try:
            parsed = resp.json()
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except (ValueError, TypeError):
            pass
        return error_for_status(resp.status_code, body, detail)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, self.base_url + path, **kwargs)
        except httpx.TimeoutException as exc:
            raise APIConnectionError(f"failed to reach {self.base_url}: request timed out", cause=exc) from exc
        except httpx.HTTPError as exc:
            raise APIConnectionError(f"failed to reach {self.base_url}: {exc}", cause=exc) from exc

    # -- requests --------------------------------------------------------

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        parse: ParseMode = "json",
    ) -> Any:
        resp = await self._request(
            method,
            path,
            json=json_body,
            params=self._clean(params),
            headers=self._auth_headers(),
        )
        return self._handle(resp, parse=parse)

    async def _stream_response(self, method: str, path: str, **kwargs: Any) -> AsyncIterator[str]:
        """Send a request and yield its body incrementally as decoded text chunks.

        An async generator, so the request is sent on the first iteration;
        connection and HTTP errors surface there with the same mapping as
        buffered calls.
        """
        try:
            async with self._client.stream(method, self.base_url + path, **kwargs) as resp:
                if resp.status_code >= 400:
                    # The body hasn't been read in stream mode; pull it in so
                    # the error can carry the server's detail message.
                    await resp.aread()
                    raise self._error(resp)
                async for text in resp.aiter_text():
                    if text:
                        yield text
        except httpx.TimeoutException as exc:
            raise APIConnectionError(f"failed to reach {self.base_url}: request timed out", cause=exc) from exc
        except httpx.HTTPError as exc:
            raise APIConnectionError(f"failed to reach {self.base_url}: {exc}", cause=exc) from exc

    def request_stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Like :meth:`request_json`, but yield the response body incrementally
        as decoded text chunks, for the server's streamed (text/event-stream)
        endpoints which are not JSON."""
        return self._stream_response(
            method,
            path,
            json=json_body,
            params=self._clean(params),
            headers=self._auth_headers(),
        )

    async def upload(
        self,
        path: str,
        *,
        files: list[FilePart],
        fields: list[FormField] | None = None,
        params: dict[str, Any] | None = None,
        parse: ParseMode = "json",
    ) -> Any:
        # httpx builds the multipart body (and boundary) natively from these.
        httpx_files = [(field, (filename, content, content_type)) for field, filename, content, content_type in files]
        data = {name: value for name, value in (fields or [])}
        resp = await self._request(
            "POST",
            path,
            files=httpx_files,
            data=data,
            params=self._clean(params),
            headers=self._auth_headers(),
        )
        return self._handle(resp, parse=parse)

    def upload_stream(
        self,
        path: str,
        *,
        files: list[FilePart],
        fields: list[FormField] | None = None,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Like :meth:`upload`, but yield the response body incrementally as
        decoded text chunks."""
        httpx_files = [(field, (filename, content, content_type)) for field, filename, content, content_type in files]
        data = {name: value for name, value in (fields or [])}
        return self._stream_response(
            "POST",
            path,
            files=httpx_files,
            data=data,
            params=self._clean(params),
            headers=self._auth_headers(),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
