# rapidbm25 MCP server

Stdio MCP server shipped inside the `rapidbm25` wheel (`rapidbm25-mcp` entry point,
or `python -m rapidbm25.mcp_server`). Pure Python on top of the Rust core; no MCP SDK dependency.

| Tool | What it does |
|---|---|
| `bm25_index_texts` | Build a named in-memory index from a list of strings (optional ids) |
| `bm25_index_path` | Index text files under a folder as 40-line windows (skips `.git`, `node_modules`, `target`, binaries, files > 2 MB) |
| `bm25_search` | Top-k results with id, score, and a text snippet |
| `bm25_list_indexes` / `bm25_drop_index` | Housekeeping |

`bm25_index_path` only reads inside `RAPIDBM25_ROOTS` (default: the working directory).
Nothing leaves the machine. Text indexes live in memory. A folder index is also written
under `RAPIDBM25_CACHE` (default `~/.cache/rapidbm25`). The next `bm25_index_path` for the
same folder, glob, and parameters loads that cache and reindexes only files whose size
or mtime changed; removed files lose their chunks. The folder index is a `LiveIndex`,
so `fixes=True` (query-token dedupe and the exact-zero Okapi idf floor). Symlinks
are not followed, hardlinks (`st_nlink > 1`) and dotfiles are skipped unless the
glob's filename starts with `.`, and the cache directory is mode `0700` with files
mode `0600`. A file is reused only when size, mtime, and the SHA-256 of the indexed
bytes still match. The tool result includes `chunks`, `files_reindexed`, and `from_cache`.

Client config:

```json
{ "mcpServers": { "rapidbm25": { "command": "uvx", "args": ["--from", "rapidbm25", "rapidbm25-mcp"],
  "env": { "RAPIDBM25_ROOTS": "/path/to/repo" } } } }
```

MCP Registry metadata is in [`../server.json`](../server.json) (name `io.github.Maverick0351a/rapidbm25`).
An MCPB bundle manifest for Smithery and desktop clients is in [`../mcpb/`](../mcpb/).
