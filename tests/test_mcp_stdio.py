"""Drive the MCP server over real stdio, the way an MCP client does."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def rpc(proc, i, method, params=None):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def test_stdio_roundtrip(tmp_path: Path):
    (tmp_path / "a.py").write_text("def parse_header(x):\n    return x\n")
    (tmp_path / "b.md").write_text("windy weather in London today\n")
    env = {**os.environ, "RAPIDBM25_ROOTS": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, "-m", "rapidbm25.mcp_server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=env)
    try:
        init = rpc(proc, 1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "rapidbm25"
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        names = {t["name"] for t in rpc(proc, 2, "tools/list")["result"]["tools"]}
        assert {"bm25_index_texts", "bm25_index_path", "bm25_search"} <= names
        r = rpc(proc, 3, "tools/call", {"name": "bm25_index_path", "arguments": {"index": "repo", "path": str(tmp_path)}})
        assert r["result"]["structuredContent"]["chunks"] == 2
        r = rpc(proc, 4, "tools/call", {"name": "bm25_search", "arguments": {"index": "repo", "query": "london weather", "k": 1}})
        assert r["result"]["structuredContent"]["results"][0]["id"].startswith("b.md")
        r = rpc(proc, 5, "tools/call", {"name": "bm25_index_path", "arguments": {"index": "x", "path": "/etc"}})
        assert r["result"]["isError"] and "outside the allowed roots" in r["result"]["content"][0]["text"]
        r = rpc(proc, 6, "tools/call", {"name": "bm25_index_texts", "arguments": {"index": "t", "documents": ["red fox", "blue sky"]}})
        r = rpc(proc, 7, "tools/call", {"name": "bm25_search", "arguments": {"index": "t", "query": "fox"}})
        assert r["result"]["structuredContent"]["results"][0]["id"] == "0"
        init_old = rpc(proc, 8, "initialize", {"protocolVersion": "1999-01-01"})
        assert init_old["result"]["protocolVersion"] == "2025-06-18"
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


def test_tool_arguments_are_validated():
    from rapidbm25.mcp_server import PROTOCOL, call_tool, handle

    refused = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert refused["result"]["protocolVersion"] == PROTOCOL
    bad_params = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": []})
    assert bad_params["error"]["code"] == -32602
    bad_args = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "bm25_search", "arguments": []}})
    assert bad_args["result"]["isError"] is True
    assert "object" in bad_args["result"]["content"][0]["text"]

    with pytest.raises(TypeError):
        call_tool("bm25_index_texts", {"index": "t", "documents": "hello"})
    with pytest.raises(TypeError):
        call_tool("bm25_search", {"index": "t", "query": "q", "k": True})
    with pytest.raises(ValueError):
        call_tool("bm25_search", {"index": "t", "query": "q", "k": 0})
    with pytest.raises(ValueError):
        call_tool("bm25_search", {"index": "t", "query": "q", "k": 1001})
    with pytest.raises(ValueError):
        call_tool("bm25_index_texts", {"index": "t", "documents": ["a"], "k1": float("nan")})
    with pytest.raises(ValueError):
        call_tool("bm25_index_path", {"index": "t", "path": ".", "chunk_lines": 0})
    with pytest.raises(ValueError):
        call_tool("bm25_index_path", {"index": "t", "path": ".", "chunk_lines": 4, "overlap": 4})


def test_mcp_rejects_oversize_input():
    from rapidbm25.mcp_server import _MAX_DOC_BYTES, _MAX_LINE_BYTES, call_tool

    with pytest.raises(ValueError, match=str(_MAX_DOC_BYTES)):
        call_tool("bm25_index_texts", {"index": "t", "documents": ["x" * (_MAX_DOC_BYTES + 1)]})

    proc = subprocess.Popen(
        [sys.executable, "-m", "rapidbm25.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    payload = b"x" * (_MAX_LINE_BYTES + 1) + b'\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
    try:
        out, err = proc.communicate(payload, timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines, err
    too_long = json.loads(lines[0])
    assert too_long["id"] is None
    assert too_long["error"]["code"] == -32600
    assert str(_MAX_LINE_BYTES) in too_long["error"]["message"]
    assert json.loads(lines[1])["result"] == {}
