"""The async Praixis Engine client."""

from __future__ import annotations

from ._transport import AsyncTransport
from .resources import AsyncChatResource, AsyncRagResource


class AsyncPraixisClient:
    """Asynchronous client for the Praixis Engine API.

    Mirrors :class:`praixis.PraixisClient` method-for-method, but every call is
    a coroutine you ``await``. Built on httpx (an optional dependency, installed
    via ``praixis[async]``).

    The client owns an httpx connection pool, so close it when done - either
    explicitly with :meth:`aclose` or by using it as an async context manager::

        async with AsyncPraixisClient("http://localhost:8080", "app-key") as client:
            reply = await client.chat.send("Hello")
            print(reply["content"])

    Args:
        base_url: Root URL of the API, e.g. ``http://localhost:8080``.
        api_key: Sent as the ``X-API-Key`` header on every request.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 30.0,
    ) -> None:
        self._transport = AsyncTransport(base_url, api_key, timeout=timeout)
        self.chat = AsyncChatResource(self._transport)
        self.rag = AsyncRagResource(self._transport)

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    async def aclose(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._transport.aclose()

    async def __aenter__(self) -> "AsyncPraixisClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
