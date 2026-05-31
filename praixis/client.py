"""The top-level Praixis Engine client."""

from __future__ import annotations

from ._transport import Transport
from .resources import ChatResource, RagResource


class PraixisClient:
    """Synchronous client for the Praixis Engine API.

    Built on the Python standard library only - no third-party dependencies.

    Args:
        base_url: Root URL of the API, e.g. ``http://localhost:8080``.
        api_key: Sent as the ``X-API-Key`` header on every request.
        timeout: Per-request timeout in seconds.

    Resources are grouped as attributes::

        client = PraixisClient("http://localhost:8080", "app-key")
        reply = client.chat.send("Hello")
        print(reply["response"])
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 30.0,
    ) -> None:
        self._transport = Transport(base_url, api_key, timeout=timeout)
        self.chat = ChatResource(self._transport)
        self.rag = RagResource(self._transport)

    @property
    def base_url(self) -> str:
        return self._transport.base_url
