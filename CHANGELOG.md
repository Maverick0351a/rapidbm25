# Changelog

All notable changes to this project are documented here. Versions follow [SemVer](https://semver.org/).

## 0.1.0

First release.

- `BM25Okapi`, `BM25L`, `BM25Plus` with rank_bm25's constructors, attributes and methods
  (`get_scores`, `get_batch_scores`, `get_top_n`), scoring bit for bit identical to rank_bm25.
- `compat=` keyword on `BM25L` / `BM25Plus`: `"0.2.2"` (default, the PyPI release) or
  `"47aa3dd"` (rank_bm25 git main), whose BM25L / BM25Plus formulas differ.
- `top_n(query, n)` and `top_k_of(scores, k)`: Rust partial select, deterministic tie order
  (score descending, then lower doc id).
- Index built in Rust (about 3.6× faster than rank_bm25); `get_scores` about 1,470× faster on a
  16,000-chunk code/docs corpus.
- `rapidbm25-mcp`: stdio MCP server with `bm25_index_texts`, `bm25_index_path`, `bm25_search`,
  `bm25_list_indexes`, `bm25_drop_index`; path indexing limited to `RAPIDBM25_ROOTS`.
- Agent skill `skill/rapidbm25-search`.
- Behavior contracts (407 observed vectors per rank_bm25 version) with exact replay.
- Wheels for CPython 3.9–3.15 on Linux (glibc and musl, x86_64 and aarch64), macOS (x86_64,
  arm64) and Windows (x86_64), built with cargo-auditable.

Known differences from rank_bm25: `get_top_n` tie order is deterministic; tokens must be `str`;
editing `.idf` / `.doc_freqs` after construction has no effect; `tokenizer=` runs in-process;
objects are not picklable yet.
