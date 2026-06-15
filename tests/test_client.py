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

_API_KEY = "app-key"


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
        if p.startswith("/general-requests/chat/"):
            return self._json(200, {"session_id": p.rsplit("/", 1)[-1], "history": [{"role": "user", "content": "hi"}]})
        if p == "/rag-db/list":
            return self._json(200, {"status": "success", "total_documents": 1, "active_collections": ["main"]})
        if p.endswith("/summary"):
            return self._json(200, {"filename": "policy.pdf", "content": "summary text"})
        return self._json(404, {"detail": "not found"})

    def do_DELETE(self):
        if not self._api_ok():
            return self._json(403, {"detail": "no key"})
        # Path segments must be percent-encoded; a raw space would corrupt the
        # HTTP request line and never reach here intact.
        assert " " not in self.path, f"unencoded path segment: {self.path}"
        if self.path.split("?")[0].startswith("/general-requests/"):
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
        if p == "/general-requests/file_summary":
            assert b"multipart/form-data" in self.headers.get("Content-Type", "").encode()
            assert b"report.txt" in raw and b"hello doc" in raw
            assert b'name="response_format"' in raw
            if b'name="stream"' in raw:
                return self._json(200, {"filename": "report.txt", "content": "short"})
            return self._stream("[FILE:report.txt]\nshort")
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
            return self._json(200, {"file_1": body["file_1"], "file_2": body["file_2"], "content": "v1 vs v2"})
        if p == "/rag-db/embed":
            return self._json(200, {"text": "hello", "dimensions": 2, "embedding": [0.1, 0.2]})
        return self._json(404, {"detail": "not found"})


def main() -> int:
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
    assert client.chat.clear_history("abc")["status"] == "success"
    summ = client.chat.summarize_file(("report.txt", "hello doc"))
    assert summ["content"] == "short" and summ["filename"] == "report.txt", summ

    # rag
    up = client.rag.upload([("a.txt", "aaa"), ("b.txt", "bbb")], collection_name="docs")
    assert up["succeeded"] == 2 and up["processed"] == 2, up
    ans = client.rag.ask("q?", collection_name="docs", session_id="s2")
    assert ans["content"] == "42" and ans["session_id"] == "s2", ans
    assert ans["sources"] == ["a.txt", "b.txt"] and ans["search_query"] == "q?", ans
    cmp = client.rag.compare("docs", "v1.pdf", "v2.pdf")
    assert cmp["content"] == "v1 vs v2" and cmp["file_1"] == "v1.pdf", cmp
    doc_sum = client.rag.summarize_document("docs", "policy.pdf")
    assert doc_sum["content"] == "summary text" and doc_sum["filename"] == "policy.pdf", doc_sum
    assert client.rag.embed("hello")["dimensions"] == 2
    assert client.rag.list_collections() == ["main"]
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
