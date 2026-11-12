# Security policy

## Supported versions

Only the latest release of rapidbm25 receives fixes.

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/Maverick0351a/rapidbm25/security/advisories/new)
rather than opening a public issue. Include the version, platform, and a minimal reproduction.
You should get an acknowledgement within 7 days.

## Scope notes

- The library does no network I/O. `save` and `load` read and write index files you pass in.
  The default `load` copies the file into memory. `load(mmap=True)` maps that file and is unsafe
  if the file is truncated or written in place while the index is alive.
- The MCP server (`rapidbm25-mcp`) speaks JSON-RPC over stdio only and reads files only through
  `bm25_index_path`, which is restricted to the directories in `RAPIDBM25_ROOTS` (default: the
  working directory). A way to read outside those roots is a security bug. Symlinks are not
  followed (`lstat` and `O_NOFOLLOW`, and the resolved path must stay under the root). Files
  with more than one hard link are skipped, because another name for the inode can sit outside
  the root. Dotfiles are skipped unless the glob's filename starts with `.`. The on-disk folder
  cache is created mode `0700` with files mode `0600`.
- Release wheels are built in GitHub Actions with `cargo-auditable` (Rust dependency list embedded
  in the binary, readable with `cargo audit bin`) and uploaded to PyPI with trusted publishing.
