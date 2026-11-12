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
Nothing leaves the machine; indexes live in memory for the life of the process.

Client config:

```json
{ "mcpServers": { "rapidbm25": { "command": "uvx", "args": ["--from", "rapidbm25", "rapidbm25-mcp"],
  "env": { "RAPIDBM25_ROOTS": "/path/to/repo" } } } }
```

MCP Registry metadata is in [`../server.json`](../server.json) (name `io.github.Maverick0351a/rapidbm25`).
An MCPB bundle manifest for Smithery and desktop clients is in [`../mcpb/`](../mcpb/).
