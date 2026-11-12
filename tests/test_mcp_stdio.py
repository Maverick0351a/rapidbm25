"""Drive the MCP server over real stdio, the way an MCP client does."""
import json
import os
import subprocess
import sys
from pathlib import Path


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
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
