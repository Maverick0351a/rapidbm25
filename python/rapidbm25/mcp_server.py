"""rapidbm25 MCP server (stdio, JSON-RPC 2.0, no dependencies beyond rapidbm25).

Run:  rapidbm25-mcp          (or: python -m rapidbm25.mcp_server)
Indexes live in memory for the life of the process. Path indexing is limited
to RAPIDBM25_ROOTS (os.pathsep-separated; default: the working directory).
"""
from __future__ import annotations

import json
import sys
import traceback

from . import __version__
from .textindex import VARIANTS, TextIndex, chunk_path

PROTOCOL = "2025-06-18"
INDEXES: dict[str, TextIndex] = {}

_params = {
    "variant": {"type": "string", "enum": list(VARIANTS), "default": "okapi"},
    "k1": {"type": "number", "default": 1.5},
    "b": {"type": "number", "default": 0.75},
}
TOOLS = [
    {"name": "bm25_index_texts",
     "description": "Build (or replace) a named in-memory BM25 index from a list of text documents.",
     "inputSchema": {"type": "object", "required": ["index", "documents"], "properties": {
         "index": {"type": "string"}, "documents": {"type": "array", "items": {"type": "string"}},
         "ids": {"type": "array", "items": {"type": "string"}, "description": "optional ids, same length"},
         **_params}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}},
    {"name": "bm25_index_path",
     "description": "Index text files under a local directory (or one file) as line-window chunks, for keyword search over a codebase or docs.",
     "inputSchema": {"type": "object", "required": ["index", "path"], "properties": {
         "index": {"type": "string"}, "path": {"type": "string"},
         "glob": {"type": "string", "default": "*", "description": "filename pattern, e.g. *.py"},
         "chunk_lines": {"type": "integer", "default": 40}, "overlap": {"type": "integer", "default": 10},
         **_params}},
     "annotations": {"readOnlyHint": True, "openWorldHint": False}},
    {"name": "bm25_search",
     "description": "Search a named index; returns the top-k documents by BM25 score.",
     "inputSchema": {"type": "object", "required": ["index", "query"], "properties": {
         "index": {"type": "string"}, "query": {"type": "string"},
         "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 1000}}},
     "annotations": {"readOnlyHint": True, "openWorldHint": False}},
    {"name": "bm25_list_indexes", "description": "List in-memory indexes and their sizes.",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True}},
    {"name": "bm25_drop_index", "description": "Delete a named in-memory index.",
     "inputSchema": {"type": "object", "required": ["index"], "properties": {"index": {"type": "string"}}},
     "annotations": {"destructiveHint": True}},
]


def _build_params(a):
    p = {"k1": float(a.get("k1", 1.5)), "b": float(a.get("b", 0.75))}
    return a.get("variant", "okapi"), p


def call_tool(name: str, a: dict):
    if name == "bm25_index_texts":
        docs = list(a["documents"])
        ids = a.get("ids") or [str(i) for i in range(len(docs))]
        if len(ids) != len(docs):
            raise ValueError("ids and documents must have the same length")
        variant, p = _build_params(a)
        INDEXES[a["index"]] = TextIndex(ids, docs, variant, p)
        return {"index": a["index"], "documents": len(docs)}
    if name == "bm25_index_path":
        ids, texts = chunk_path(a["path"], a.get("glob", "*"), int(a.get("chunk_lines", 40)),
                                int(a.get("overlap", 10)))
        if not texts:
            raise ValueError("no text chunks found")
        variant, p = _build_params(a)
        INDEXES[a["index"]] = TextIndex(ids, texts, variant, p)
        return {"index": a["index"], "chunks": len(texts)}
    if name == "bm25_search":
        idx = INDEXES.get(a["index"])
        if idx is None:
            raise KeyError(f"no index named {a['index']!r}")
        return {"results": idx.search(a["query"], int(a.get("k", 10)))}
    if name == "bm25_list_indexes":
        return {"indexes": [{"index": k, "documents": len(v.ids), "variant": v.variant}
                            for k, v in INDEXES.items()]}
    if name == "bm25_drop_index":
        return {"dropped": INDEXES.pop(a["index"], None) is not None}
    raise KeyError(f"unknown tool {name}")


def handle(msg: dict):
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return None  # notification
    try:
        if method == "initialize":
            res = {"protocolVersion": params.get("protocolVersion", PROTOCOL),
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "rapidbm25", "version": __version__}}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                out = call_tool(params["name"], params.get("arguments") or {})
                res = {"content": [{"type": "text", "text": json.dumps(out)}],
                       "structuredContent": out, "isError": False}
            except Exception as e:  # tool errors are results, not protocol errors
                res = {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": res}
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
