"""Tests for the Praixis Python client against a stdlib mock server.

Run with: uv run --no-project python tests/test_client.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praixis import AuthenticationError, NotFoundError, PraixisClient  # noqa: E402
from praixis._http import iter_stream_events  # noqa: E402

_API_KEY = "app-key"


def _tokens(events):
    return "".join(e["value"] for e in events if e["type"] == "token")


def test_stream_parser() -> None:
    # A marker split across chunk boundaries, then content in pieces.
    events = list(iter_stream_events(iter(["[SESSION", "_ID:abc]", "\nHel", "lo"])))
    assert events == [
        {"type": "session_id", "value": "abc"},
        {"type": "token", "value": "Hel"},
        {"type": "token", "value": "lo"},
    ], events
    # RAG markers: comma-split sources.
    events = list(iter_stream_events(iter(["[SESSION_ID:s1]\n[SEARCH_QUERY:what is x?]\n[SOURCES:a.txt,b.txt]\n42"])))
    assert events[1] == {"type": "search_query", "value": "what is x?"}, events
    assert events[2] == {"type": "sources", "value": ["a.txt", "b.txt"]}, events
    # Escaped source items: commas / percents / brackets in filenames survive.
    events = list(iter_stream_events(iter(["[SOURCES:Q3%2C Final.pdf,50%25 off%5D.txt]\nok"])))
    assert events[0] == {"type": "sources", "value": ["Q3, Final.pdf", "50% off].txt"]}, events
    # Brackets inside the content are not mistaken for markers.
    events = list(iter_stream_events(iter(["[SESSION_ID:z]\nSee item [3]."])))
    assert _tokens(events) == "See item [3].", events
    # An [ERROR] marker (always emitted before content) is surfaced.
    events = list(iter_stream_events(iter(["[FILE:r.txt]\n[ERROR:GPU busy]\n"])))
    assert {"type": "error", "value": "GPU busy"} in events, events
    # No markers => everything is a token.
    events = list(iter_stream_events(iter(["plain text"])))
    assert events == [{"type": "token", "value": "plain text"}], events


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, text):
        # Mirror the real server: chat / RAG-ask / file-summary stream plain
        # text (marker lines + tokens), not JSON.
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_ok(self):
        return self.headers.get("X-API-Key") == _API_KEY

    def _read(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_GET(self):
        p = self.path.split("?")[0]
        if not self._api_ok():
            return self._json(403, {"detail": "API Key header missing."})
        if p == "/general-requests/chat/sessions/active":
            return self._json(200, {"active_sessions": ["s1", "s2"]})
        if p == "/general-requests/chat/missing":
            return self._json(404, {"detail": "not found"})
        if p.startswith("/general-requests/chat/") and p.endswith("/usage"):
            return self._json(200, {
                "session_id": p.rsplit("/", 2)[-2],
                "requests": 3,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "estimated_context_tokens": 40,
            })
        if p.startswith("/general-requests/chat/"):
            return self._json(200, {"session_id": p.rsplit("/", 1)[-1], "history": [{"role": "user", "content": "hi"}]})
        if p == "/rag-db/list":
            return self._json(200, {"status": "success", "total_documents": 1, "active_collections": ["main"]})
        if p.endswith("/summary"):
            if "stream=true" in self.path:
                return self._stream("[FILE:policy.pdf]\nsummary text")
            return self._json(200, {"filename": "policy.pdf", "content": "summary text"})
        if p.endswith("/chunks"):
            return self._json(200, {
                "status": "success",
                "collection_name": "docs",
                "filename": "policy.pdf",
                "total_chunks": 2,
                "chunks": [
                    {"chunk_index": 0, "content": "part one"},
                    {"chunk_index": 1, "content": "part two"},
                ],
            })
        if p.endswith("/questions"):
            return self._json(200, {
                "collection_name": "docs",
                "filename": "policy.pdf",
                "total_chunks": 2,
                "questions_stored": 10,
                "generation_pending": False,
            })
        return self._json(404, {"detail": "not found"})

    def do_DELETE(self):
        if not self._api_ok():
            return self._json(403, {"detail": "no key"})
        # Path segments must be percent-encoded; a raw space would corrupt the
        # HTTP request line and never reach here intact.
        assert " " not in self.path, f"unencoded path segment: {self.path}"
        p = self.path.split("?")[0]
        if p.startswith("/general-requests/") and p.endswith("/last"):
            return self._json(200, {
                "status": "success",
                "session_id": p.rsplit("/", 2)[-2],
                "removed_messages": 2,
                "undone_prompt": "hi",
                "messages_remaining": 1,
            })
        if p.startswith("/general-requests/"):
            return self._json(200, {"status": "success", "detail": "Session deleted."})
        return self._json(200, {"status": "success", "message": "deleted"})

    def do_POST(self):
        p = self.path.split("?")[0]
        raw = self._read()
        if not self._api_ok():
            return self._json(403, {"detail": "bad key"})
        if p == "/general-requests/chat":
            body = json.loads(raw)
            sid = body.get("session_id") or "new-id"
            # Buffered path sends stream=false and expects native JSON.
            if body.get("stream") is False:
                return self._json(200, {"session_id": sid, "content": f"echo:{body['prompt']}"})
            return self._stream(f"[SESSION_ID:{sid}]\necho:{body['prompt']}")
        if p.startswith("/general-requests/chat/") and p.endswith("/compact"):
            return self._json(200, {
                "status": "success",
                "session_id": p.rsplit("/", 2)[-2],
                "messages_before": 12,
                "messages_after": 5,
                "estimated_tokens_before": 900,
                "estimated_tokens_after": 300,
            })
        if p == "/general-requests/file_summary":
            assert b"multipart/form-data" in self.headers.get("Content-Type", "").encode()
            assert b"report.txt" in raw and b"hello doc" in raw
            assert b'name="response_format"' in raw
            # Buffered sends the stream field as "false"; streaming as "true".
            if b'name="stream"\r\n\r\nfalse' in raw:
                return self._json(200, {"filename": "report.txt", "content": "short"})
            return self._stream("[FILE:report.txt]\n[PROGRESS:reducing 3 chunks]\nshort")
        if p == "/rag-db/upload":
            assert raw.count(b'name="files"') == 2, "expected two file parts"
            assert b"a.txt" in raw and b"b.txt" in raw
            assert b'name="collection_name"' in raw
            return self._json(200, {
                "collection_name": "docs",
                "processed": 2,
                "succeeded": 2,
                "results": [
                    {"filename": "a.txt", "status": "success"},
                    {"filename": "b.txt", "status": "success"},
                ],
            })
        if p == "/rag-db/ask":
            body = json.loads(raw)
            sid = body.get("session_id") or "new-id"
            if body.get("stream") is False:
                return self._json(200, {
                    "session_id": sid,
                    "search_query": body["question"],
                    "sources": ["a.txt", "b.txt"],
                    "content": "42",
                })
            return self._stream(f"[SESSION_ID:{sid}]\n[SEARCH_QUERY:{body['question']}]\n[SOURCES:a.txt,b.txt]\n42")
        if p == "/rag-db/knowledge_base/compare":
            body = json.loads(raw)
            if body.get("stream"):
                return self._stream("v1 vs v2")
            return self._json(200, {"file_1": body["file_1"], "file_2": body["file_2"], "content": "v1 vs v2"})
        if p == "/rag-db/search":
            body = json.loads(raw)
            return self._json(200, {
                "collection_name": body["collection_name"],
                "query": body["query"],
                "n_results": body["n_results"],
                "results": [{"source": "a.txt", "text": "chunk text", "score": 0.9}],
                "score_type": "similarity",
            })
        if p == "/rag-db/embed":
            return self._json(200, {"text": "hello", "dimensions": 2, "embedding": [0.1, 0.2]})
        if p == "/rag-db/upload_text":
            body = json.loads(raw)
            assert body["chunking_strategy"] in ("semantic", "character"), body
            return self._json(200, {
                "status": "success",
                "collection_name": body["collection_name"],
                "filename": body["filename"],
                "chunks_stored": 3,
                "improved_search": body["improved_search"],
            })
        if p.endswith("/questions"):
            return self._json(200, {
                "status": "scheduled",
                "collection_name": "docs",
                "filename": "policy.pdf",
                "chunks": 2,
            })
        return self._json(404, {"detail": "not found"})


def main() -> int:
    test_stream_parser()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    client = PraixisClient(base, _API_KEY)

    # chat (buffered: stream=false, native {session_id, content})
    r = client.chat.send("hi", system_prompt="be brief")
    assert r["content"] == "echo:hi" and r["session_id"] == "new-id", r
    assert client.chat.send("again", session_id="s9")["session_id"] == "s9"
    assert client.chat.list_sessions() == ["s1", "s2"]
    h = client.chat.get_history("abc")
    assert h["session_id"] == "abc" and len(h["history"]) == 1, h
    u = client.chat.get_usage("abc")
    assert u["session_id"] == "abc" and u["total_tokens"] == 150, u
    assert u["estimated_context_tokens"] == 40, u
    c = client.chat.compact("abc")
    assert c["status"] == "success" and c["messages_after"] == 5, c
    undo = client.chat.undo_last_exchange("abc")
    assert undo["removed_messages"] == 2 and undo["undone_prompt"] == "hi", undo
    assert undo["session_id"] == "abc" and undo["messages_remaining"] == 1, undo
    assert client.chat.clear_history("abc")["status"] == "success"
    summ = client.chat.summarize_file(("report.txt", "hello doc"))
    assert summ["content"] == "short" and summ["filename"] == "report.txt", summ

    # streaming chat: markers arrive as events, content as tokens
    events = list(client.chat.stream("hi"))
    assert events[0] == {"type": "session_id", "value": "new-id"}, events
    assert _tokens(events) == "echo:hi", events

    # streaming file summary
    events = list(client.chat.summarize_file_stream(("report.txt", "hello doc")))
    assert events[0] == {"type": "file", "value": "report.txt"}, events
    assert {"type": "progress", "value": "reducing 3 chunks"} in events, events
    assert _tokens(events) == "short", events

    # rag
    up = client.rag.upload([("a.txt", "aaa"), ("b.txt", "bbb")], collection_name="docs")
    assert up["succeeded"] == 2 and up["processed"] == 2, up
    ans = client.rag.ask("q?", collection_name="docs", session_id="s2")
    assert ans["content"] == "42" and ans["session_id"] == "s2", ans
    assert ans["sources"] == ["a.txt", "b.txt"] and ans["search_query"] == "q?", ans
    # streaming ask: session/query/sources markers then answer tokens
    events = list(client.rag.ask_stream("q?", collection_name="docs", session_id="s2"))
    assert {"type": "sources", "value": ["a.txt", "b.txt"]} in events, events
    assert _tokens(events) == "42", events

    hits = client.rag.search("setup steps", collection_name="docs", n_results=3)
    assert hits["score_type"] == "similarity" and hits["n_results"] == 3, hits
    assert hits["results"][0]["source"] == "a.txt", hits
    cmp = client.rag.compare("docs", "v1.pdf", "v2.pdf")
    assert cmp["content"] == "v1 vs v2" and cmp["file_1"] == "v1.pdf", cmp
    doc_sum = client.rag.summarize_document("docs", "policy.pdf")
    assert doc_sum["content"] == "summary text" and doc_sum["filename"] == "policy.pdf", doc_sum

    # streaming compare / document summary
    assert _tokens(client.rag.compare_stream("docs", "v1.pdf", "v2.pdf")) == "v1 vs v2"
    events = list(client.rag.summarize_document_stream("docs", "policy.pdf"))
    assert events[0] == {"type": "file", "value": "policy.pdf"}, events
    assert _tokens(events) == "summary text", events
    assert client.rag.embed("hello")["dimensions"] == 2
    assert client.rag.list_collections() == ["main"]

    # text ingestion + chunk inspection + question index management
    txt = client.rag.upload_text("raw text here", "notes.txt", collection_name="docs", improved_search=True)
    assert txt["chunks_stored"] == 3 and txt["filename"] == "notes.txt", txt
    assert txt["improved_search"] is True, txt
    ch = client.rag.get_chunks("docs", "policy.pdf")
    assert ch["total_chunks"] == 2 and ch["chunks"][0]["chunk_index"] == 0, ch
    assert ch["chunks"][1]["content"] == "part two", ch
    qs = client.rag.question_status("docs", "policy.pdf")
    assert qs["questions_stored"] == 10 and qs["generation_pending"] is False, qs
    rq = client.rag.regenerate_questions("docs", "policy.pdf")
    assert rq["status"] == "scheduled" and rq["chunks"] == 2, rq
    assert client.rag.delete_collection("docs")["status"] == "success"
    assert client.rag.delete_file("docs", "a.txt")["status"] == "success"
    # filename with a space must be percent-encoded into the path
    assert client.rag.delete_file("docs", "my report.pdf")["status"] == "success"

    # api-key failure
    badkey = PraixisClient(base, "wrong")
    try:
        badkey.chat.send("hi")
        raise AssertionError("expected AuthenticationError")
    except AuthenticationError as e:
        assert e.status_code == 403, e

    # 404 mapping
    try:
        client.chat.get_history("missing")
        raise AssertionError("expected NotFoundError")
    except NotFoundError as e:
        assert e.status_code == 404, e

    server.shutdown()
    print("all python client tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
