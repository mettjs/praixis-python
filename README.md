# Praixis Engine — Python Client

A lightweight Python client for the Praixis Engine API, in both **sync** and
**async** flavors.

- **`PraixisClient`** — synchronous, **zero dependencies**, built entirely on
  the standard library (`urllib`, `json`, `uuid`), so an upstream package
  release can never break it.
- **`AsyncPraixisClient`** — async/await, built on `httpx` (the only optional
  dependency). Imported lazily, so the sync client stays dependency-free.
- Same surface in both: resource-grouped `client.chat` and `client.rag` — chat
  + file summary, and RAG ingest/ask/search/embed/compare.

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
print(reply["session_id"], reply["content"])

# Continue it
client.chat.send("And again?", session_id=reply["session_id"])

# JSON-mode response, custom system prompt. "content" is still a string — the
# model's raw JSON text — which you parse yourself.
import json
r = client.chat.send("List 3 colors", response_format="json", system_prompt="Be terse")
colors = json.loads(r["content"])

# Sessions
client.chat.list_sessions()          # -> [session_id, ...]
client.chat.get_history(session_id)  # -> {"session_id", "history": [...]}
client.chat.clear_history(session_id)

# Per-session token usage. Counts the streamed answers (chat and RAG), RAG query
# reformulation, and compaction calls; counters expire with the session.
# estimated_context_tokens shows how close the session is to auto-compacting.
usage = client.chat.get_usage(session_id)
# -> {"session_id", "requests", "prompt_tokens", "completion_tokens",
#     "total_tokens", "estimated_context_tokens"}

# Compact a session on demand: fold older exchanges into an LLM-written summary
# (the server also does this automatically near its context budget). Raises
# APIError 400 when there's nothing to fold yet.
client.chat.compact(session_id)
# -> {"status", "session_id", "messages_before", "messages_after",
#     "estimated_tokens_before", "estimated_tokens_after"}

# Undo the last exchange: removes the most recent user message and the assistant
# reply that followed it, so you can retry or regenerate. Compaction summaries
# are kept. Raises APIError 400 when there's no user message left to undo.
undone = client.chat.undo_last_exchange(session_id)
# -> {"status", "session_id", "removed_messages", "undone_prompt",
#     "messages_remaining"}  — undone_prompt is the removed user message

# Summarize an uploaded file (path, or (filename, content[, content_type])).
# Give the filename a .pdf/.docx/.txt extension — it's the primary format
# signal; content_type is only the fallback for extension-less names.
client.chat.summarize_file("report.pdf")
client.chat.summarize_file(("notes.txt", "raw text here"))
```

### Streaming

The server's generative endpoints accept a `stream` toggle. The buffered methods
(`send`, `ask`, `summarize_file`, `compare`, `summarize_document`) send
`stream=false` and return the server's native JSON — the right default for
scripts and backends. The answer is always under `content`: `chat.send` returns
`{"session_id", "content"}`, `rag.ask` returns `{"session_id", "search_query",
"sources", "content"}`, and `compare` / `summarize_document` / `summarize_file`
return `{..., "content"}`.

For token-by-token output, use the streaming variants, which return an iterator
of `{"type", "value"}` events. Marker events (`session_id`, `search_query`,
`sources`, `file`, `progress`, `error`) arrive before the `token` events that
carry content:

```python
for event in client.chat.stream("Tell me a story"):
    if event["type"] == "token":
        print(event["value"], end="", flush=True)
    elif event["type"] == "session_id":
        session_id = event["value"]

# Every buffered generative method has a streaming sibling:
#   client.rag.ask_stream(question, collection_name=...)   # session_id, search_query, sources, then tokens
#   client.chat.summarize_file_stream(file)                 # file, [progress...], then tokens
#   client.rag.compare_stream(coll, f1, f2)                 # tokens
#   client.rag.summarize_document_stream(coll, filename)    # file, then tokens
```

On the async client the same methods yield events with `async for`:

```python
async for event in client.chat.stream("Tell me a story"):
    if event["type"] == "token":
        print(event["value"], end="", flush=True)
```

The request is sent on the first iteration, so connection and HTTP errors
surface there. On the sync client the configured `timeout` bounds each read
(an idle limit), not the stream's total duration.

## RAG

```python
# Ingest one or many documents into a collection
client.rag.upload("manual.pdf", collection_name="docs")
client.rag.upload([("a.txt", "..."), ("b.txt", "...")], collection_name="docs")

# Improved search: generate hypothetical questions in the background so plain,
# conversational queries match formal/technical text better. The document is
# searchable immediately; matching improves once generation finishes.
client.rag.upload("ley.pdf", collection_name="docs", improved_search=True)

# Ingest raw text directly — no file wrapping. Same pipeline and options as
# upload; the filename becomes the document's stored identity.
client.rag.upload_text("full document text…", "faq-2026.txt", collection_name="docs")
# -> {"status", "collection_name", "filename", "chunks_stored", "improved_search"}
```

> **File inputs.** Each file may be a path, a `(filename, content)` pair, or a
> `(filename, content, content_type)` triple. The filename is the document's
> stored identity and the server's primary format signal, so prefer a
> `.pdf`/`.docx`/`.txt` extension. For extension-less names the server falls
> back to the declared content type (inferred from the extension when omitted),
> then to the file's magic bytes.

```python
# Ask a question grounded in a collection
ans = client.rag.ask("What does the manual say about setup?", collection_name="docs")
print(ans["content"], ans["sources"], ans["search_query"])

# Restrict retrieval to one source document. Only the "source" key is honored;
# any other keys are ignored (not an error).
client.rag.ask("What is the notice period?", collection_name="docs",
               metadata_filter={"source": "policy.pdf"})

# Retrieval only: ranked raw chunks, no LLM. Pass a standalone query (not reformulated).
hits = client.rag.search("setup steps", collection_name="docs")
for r in hits["results"]:
    print(r["source"], r["score"], r["text"])  # hits["score_type"] is "rrf" or "similarity"

# Embeddings, listing, deletion, compare, summarize. compare/summarize_document
# return {..., "content"} and accept an optional response_format.
client.rag.embed("some text")
client.rag.list_collections()
client.rag.list_files("docs")
client.rag.delete_file("docs", "a.txt")
client.rag.delete_collection("docs")
client.rag.compare("docs", "a.txt", "b.txt")              # -> {"file_1", "file_2", "content"}
client.rag.summarize_document("docs", "manual.pdf")       # -> {"filename", "content"}

# Inspect how a document was chunked — exactly what retrieval sees.
ch = client.rag.get_chunks("docs", "manual.pdf")
# -> {"status", "collection_name", "filename", "total_chunks",
#     "chunks": [{"chunk_index", "content"}, ...]}

# Question-index management: improved_search is no longer locked in at upload.
# regenerate_questions rebuilds the index in the background, replacing the old
# questions only once the new pass succeeds; poll question_status until
# generation_pending is False. Raises APIError 409 while a pass is already
# running, 400 when indexing is disabled server-side.
client.rag.regenerate_questions("docs", "manual.pdf")
# -> {"status": "scheduled", "collection_name", "filename", "chunks"}
client.rag.question_status("docs", "manual.pdf")
# -> {"collection_name", "filename", "total_chunks", "questions_stored",
#     "generation_pending"}
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
