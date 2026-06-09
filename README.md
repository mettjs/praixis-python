# Praixis Engine — Python Client

A lightweight Python client for the Praixis Engine API, in both **sync** and
**async** flavors.

- **`PraixisClient`** — synchronous, **zero dependencies**, built entirely on
  the standard library (`urllib`, `json`, `uuid`), so an upstream package
  release can never break it.
- **`AsyncPraixisClient`** — async/await, built on `httpx` (the only optional
  dependency). Imported lazily, so the sync client stays dependency-free.
- Same surface in both: resource-grouped `client.chat` and `client.rag` — chat
  + file summary, and RAG ingest/ask/embed/compare.

> The companion Node.js client lives in its own repository.

## Installation

```bash
pip install praixis            # sync client only, zero dependencies
pip install "praixis[async]"   # also installs httpx for the async client
```

Or vendor the `praixis/` package directly into your project — the sync client
has no deps.

Requires Python 3.10+.

## Authentication

Every endpoint uses an app-level API key sent as the `X-API-Key` header.

```python
from praixis import PraixisClient

client = PraixisClient("http://localhost:8080", "your-api-key")
```

> Admin/system routes (`/api/system/*`, HTTP Basic auth) are intentionally **not**
> part of this SDK. They already have a browser UI, and baking admin credentials
> into application code would be a security anti-pattern. The `X-API-Key` this
> SDK uses is an app-level credential, not an admin one.

## Chat

```python
# Start a conversation
reply = client.chat.send("Hello, world!")
print(reply["session_id"], reply["response"])

# Continue it
client.chat.send("And again?", session_id=reply["session_id"])

# JSON-mode response, custom system prompt
client.chat.send("List 3 colors", response_format="json", system_prompt="Be terse")

# Sessions
client.chat.list_sessions()          # -> [session_id, ...]
client.chat.get_history(session_id)  # -> {"session_id", "history": [...]}
client.chat.clear_history(session_id)

# Summarize an uploaded file (path, or (filename, content[, content_type]))
client.chat.summarize_file("report.pdf")
client.chat.summarize_file(("notes.txt", "raw text here"))
```

> **Note on streaming:** the server streams chat and RAG answers as
> `text/event-stream`, not JSON. This client buffers the full response and
> parses the leading marker lines (`[SESSION_ID:...]`, and for RAG
> `[SEARCH_QUERY:...]` / `[SOURCES:...]`) out of the body for you, so
> `chat.send` returns `{"session_id", "response", "response_format"}` and
> `rag.ask` returns `{"answer", "sources", "session_id", "search_query"}`.
> Buffering is the right default for scripts and backends; token-by-token
> iteration is not yet exposed.

## RAG

```python
# Ingest one or many documents into a collection
client.rag.upload("manual.pdf", collection_name="docs")
client.rag.upload([("a.txt", "..."), ("b.txt", "...")], collection_name="docs")

# Improved search: generate hypothetical questions in the background so plain,
# conversational queries match formal/technical text better. The document is
# searchable immediately; matching improves once generation finishes.
client.rag.upload("ley.pdf", collection_name="docs", improved_search=True)

# Ask a question grounded in a collection
ans = client.rag.ask("What does the manual say about setup?", collection_name="docs")
print(ans)

# Embeddings, listing, deletion, compare, summarize
client.rag.embed("some text")
client.rag.list_collections()
client.rag.list_files("docs")
client.rag.delete_file("docs", "a.txt")
client.rag.delete_collection("docs")
client.rag.compare("docs", "a.txt", "b.txt")
client.rag.summarize_document("docs", "manual.pdf")
```

## Error handling

```python
from praixis import (
    APIError, AuthenticationError, NotFoundError,
    RateLimitError, APIConnectionError,
)

try:
    client.chat.send("hi")
except AuthenticationError:
    ...          # 401 / 403
except NotFoundError:
    ...          # 404
except RateLimitError:
    ...          # 429 (per-route limits)
except APIError as e:
    print(e.status_code, e.detail)
except APIConnectionError:
    ...          # never reached the server
```

All exceptions inherit from `praixis.PraixisError`.

## Testing

Both suites run against a standard-library mock HTTP server — no network needed.

```bash
# Sync client — zero dependencies
uv run --no-project python tests/test_client.py

# Async client — needs httpx
uv run --with httpx python tests/test_async_client.py
```

The async suite also asserts that the sync and async resources expose an
identical set of methods, so the two clients can't silently drift apart.

## Privacy note

This client transmits whatever you pass to it (prompts, documents, session IDs)
to the configured Praixis Engine server. Those payloads may contain personal
data — handle them according to your own privacy obligations. The client stores
nothing locally and adds no telemetry.

## License

MIT — see [LICENSE](./LICENSE).
