"""Praixis Engine Python client.

The synchronous :class:`PraixisClient` is built entirely on the standard
library, so it has zero dependencies and upstream package releases can never
break it::

    from praixis import PraixisClient

    client = PraixisClient("http://localhost:8080", "your-api-key")
    print(client.chat.send("Hello")["content"])

An :class:`AsyncPraixisClient` is also available for async/await code. It is
backed by httpx, the SDK's only (optional) dependency - install it with
``pip install praixis[async]``. The import is lazy, so httpx is required only if
you actually use the async client; ``import praixis`` alone never needs it::

    from praixis import AsyncPraixisClient

    async with AsyncPraixisClient("http://localhost:8080", "your-api-key") as client:
        print((await client.chat.send("Hello"))["content"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .client import PraixisClient
from .errors import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    NotFoundError,
    PraixisError,
    RateLimitError,
)

if TYPE_CHECKING:
    # Give type checkers / IDEs the symbol without importing httpx at runtime.
    from .aio import AsyncPraixisClient

__version__ = "1.3.0"

__all__ = [
    "PraixisClient",
    "AsyncPraixisClient",
    "PraixisError",
    "APIError",
    "APIConnectionError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "__version__",
]


def __getattr__(name: str) -> object:
    # Lazily resolve the async client so importing praixis never pulls in httpx
    # unless the async client is actually requested.
    if name == "AsyncPraixisClient":
        from .aio import AsyncPraixisClient

        return AsyncPraixisClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
