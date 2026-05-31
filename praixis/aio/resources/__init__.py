"""Async resource groups, one per API area.

Each mirrors its sync counterpart in ``praixis.resources``; the only difference
is that every method is a coroutine.
"""

from .chat import AsyncChatResource
from .rag import AsyncRagResource

__all__ = ["AsyncChatResource", "AsyncRagResource"]
