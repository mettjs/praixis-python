"""Exception types raised by the Praixis client.

The hierarchy is intentionally small so callers can catch broadly
(``PraixisError``) or narrowly (``AuthenticationError``) without depending on
any third-party exception classes.
"""

from __future__ import annotations


class PraixisError(Exception):
    """Base class for every error raised by this client."""


class APIConnectionError(PraixisError):
    """The request never reached the server (DNS, refused, timeout, TLS)."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class APIError(PraixisError):
    """The server returned a non-2xx response.

    Attributes:
        status_code: The HTTP status code.
        body: The raw, undecoded response body (may be empty).
        detail: The server-provided ``detail`` message when present, else the
            raw body.
    """

    def __init__(self, status_code: int, body: str, detail: str | None = None) -> None:
        self.status_code = status_code
        self.body = body
        self.detail = detail or body
        super().__init__(f"API error (status {status_code}): {self.detail}")


class AuthenticationError(APIError):
    """A 401 or 403 response - missing or invalid API key."""


class NotFoundError(APIError):
    """A 404 response - the requested resource does not exist."""


class RateLimitError(APIError):
    """A 429 response - the per-route rate limit was exceeded."""


def error_for_status(status_code: int, body: str, detail: str | None = None) -> APIError:
    """Return the most specific ``APIError`` subclass for a status code."""
    if status_code in (401, 403):
        return AuthenticationError(status_code, body, detail)
    if status_code == 404:
        return NotFoundError(status_code, body, detail)
    if status_code == 429:
        return RateLimitError(status_code, body, detail)
    return APIError(status_code, body, detail)
