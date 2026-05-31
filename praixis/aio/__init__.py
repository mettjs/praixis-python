"""Async client package.

Importing this package (or ``AsyncPraixisClient``) requires httpx, the SDK's
only optional dependency. Sync-only users never touch this module and so never
need httpx installed.
"""

from .client import AsyncPraixisClient

__all__ = ["AsyncPraixisClient"]
