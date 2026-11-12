# Security policy

## Supported versions

Only the latest release of rapidbm25 receives fixes.

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/Maverick0351a/rapidbm25/security/advisories/new)
rather than opening a public issue. Include the version, platform, and a minimal reproduction.
You should get an acknowledgement within 7 days.

## Scope notes

- The library does no network or file I/O; it scores token lists in memory.
- The MCP server (`rapidbm25-mcp`) speaks JSON-RPC over stdio only and reads files only through
  `bm25_index_path`, which is restricted to the directories in `RAPIDBM25_ROOTS` (default: the
  working directory). A way to read outside those roots is a security bug.
- Release wheels are built in GitHub Actions with `cargo-auditable` (Rust dependency list embedded
  in the binary, readable with `cargo audit bin`) and uploaded to PyPI with trusted publishing.
