"""rapidbm25 MCP server (stdio, JSON-RPC 2.0, no dependencies beyond rapidbm25).

Run:  rapidbm25-mcp          (or: python -m rapidbm25.mcp_server)
Text indexes live in memory for the life of the process. Folder indexes are
cached under RAPIDBM25_CACHE (default: ~/.cache/rapidbm25) and only files
whose size or mtime changed are reindexed. Path indexing is limited to
RAPIDBM25_ROOTS (os.pathsep-separated; default: the working directory).
"""
from __future__ import annotations

import json
import sys
import traceback

from . import __version__
from .textindex import VARIANTS, TextIndex, index_folder

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
     "description": "Index text files under a local directory (or one file) as line-window chunks. The folder index is cached on disk; only files that changed are reindexed.",
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


# shortcut: MCP text indexes accept at most 100_000 documents and queries of
# 100_000 characters. Raise the caps when a caller indexes more through the server.
_MAX_DOCS = 100_000
_MAX_QUERY_CHARS = 100_000
_MAX_NAME = 256
_MAX_CHUNK_LINES = 10_000


def _name(value, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_NAME:
        raise ValueError(f"{what} must be a non-empty string of at most {_MAX_NAME} characters")
    return value


def _finite(value, name: str, default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return out


def _int_param(value, name: str, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"{name} must be an integer")
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < lo or value > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _build_params(a):
    variant = a.get("variant", "okapi")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    return variant, {"k1": _finite(a.get("k1"), "k1", 1.5), "b": _finite(a.get("b"), "b", 0.75)}


def call_tool(name: str, a: dict):
    if not isinstance(a, dict):
        raise TypeError("tool arguments must be an object")
    if name == "bm25_index_texts":
        index = _name(a.get("index"), "index")
        docs = a.get("documents")
        if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
            raise TypeError("documents must be a list of strings")
        if len(docs) > _MAX_DOCS:
            raise ValueError(f"documents must contain at most {_MAX_DOCS} strings")
        ids = a.get("ids")
        if ids is None:
            ids = [str(i) for i in range(len(docs))]
        elif not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise TypeError("ids must be a list of strings")
        if len(ids) != len(docs):
            raise ValueError("ids and documents must have the same length")
        variant, p = _build_params(a)
        INDEXES[index] = TextIndex(ids, docs, variant, p)
        return {"index": index, "documents": len(docs)}
    if name == "bm25_index_path":
        index = _name(a.get("index"), "index")
        path = a.get("path")
        if not isinstance(path, str) or not path:
            raise TypeError("path must be a non-empty string")
        glob = a.get("glob", "*")
        if not isinstance(glob, str) or not glob or len(glob) > _MAX_NAME:
            raise ValueError("glob must be a non-empty string")
        chunk_lines = _int_param(a.get("chunk_lines"), "chunk_lines", 40, 1, _MAX_CHUNK_LINES)
        # The default overlap is 10, but a one-line window cannot overlap by 10.
        if "overlap" not in a or a.get("overlap") is None:
            overlap = min(10, chunk_lines - 1)
        else:
            overlap = _int_param(a.get("overlap"), "overlap", 10, 0, chunk_lines - 1)
        variant, p = _build_params(a)
        ix = index_folder(path, glob, chunk_lines, overlap, variant, p)
        if ix.live.corpus_size == 0:
            raise ValueError("no text chunks found")
        INDEXES[index] = ix
        return {"index": index, "chunks": ix.live.corpus_size,
                "files_reindexed": ix.last_stats.get("files_reindexed", 0),
                "from_cache": bool(ix.last_stats.get("from_cache"))}
    if name == "bm25_search":
        index = _name(a.get("index"), "index")
        query = a.get("query")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if len(query) > _MAX_QUERY_CHARS:
            raise ValueError(f"query must be at most {_MAX_QUERY_CHARS} characters")
        k = _int_param(a.get("k"), "k", 10, 1, 1000)
        idx = INDEXES.get(index)
        if idx is None:
            raise KeyError(f"no index named {index!r}")
        return {"results": idx.search(query, k)}
    if name == "bm25_list_indexes":
        return {"indexes": [{"index": k, "documents": len(v.ids), "variant": v.variant}
                            for k, v in INDEXES.items()]}
    if name == "bm25_drop_index":
        index = _name(a.get("index"), "index")
        return {"dropped": INDEXES.pop(index, None) is not None}
    raise KeyError(f"unknown tool {name}")


def handle(msg: dict):
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        return None  # notification
    raw_params = msg.get("params")
    # An omitted params object is fine. An empty list is not: `or {}` would
    # hide it, because a list is only false when it has no items.
    if raw_params is None:
        params = {}
    elif not isinstance(raw_params, dict):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "invalid params"}}
    else:
        params = raw_params
    try:
        if method == "initialize":
            # Advertise the protocol this server implements. Echoing an unknown
            # client version would claim support we do not have.
            res = {"protocolVersion": PROTOCOL,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "rapidbm25", "version": __version__}}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                raw_args = params.get("arguments")
                # Same trap as params: an empty list is false, so `or {}` would
                # accept it and then fail later on a missing field.
                if raw_args is None:
                    arguments = {}
                elif not isinstance(raw_args, dict):
                    raise TypeError("tool arguments must be an object")
                else:
                    arguments = raw_args
                out = call_tool(params["name"], arguments)
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
