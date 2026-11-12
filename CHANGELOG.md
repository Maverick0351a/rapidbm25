# Changelog

All notable changes to this project are documented here. Versions follow [SemVer](https://semver.org/).

## Unreleased 0.2.0

Not published. The default `BM25Okapi` / `BM25L` / `BM25Plus` scores stay bit-identical to rank_bm25.

- **Persistence.** Every compat class pickles. `save(path)` / `load(path)` write a versioned little-endian file (`RBM25` format 1). Scalar floats are stored as IEEE-754 hex, so a round trip is bit-identical. `load(mmap=True)` (the default) scores a static index from a read-only mapping on 64-bit little-endian hosts.
- **`LiveIndex`.** Add, delete, and update documents by stable string id, with optional metadata and a `where` filter (`$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, `$exists`, or a callable). Global statistics update on each mutation; posting lists are not rebuilt from scratch. `get_scores` matches a static index built on `corpus()` (same variant, `compat`, parameters, and `fixes`). An empty live index is valid. A `threading.Lock` plus an internal mutex make concurrent add/search safe; a filter must not call back into the index.
- **Tokenizers (opt-in).** `Tokenizer(lowercase=, stopwords="english"|list, stemmer=<snowball language>, cjk=True)`. Unicode word split, a small original English stop list, Snowball stemming via `rust-stemmers` (MIT OR BSD-3-Clause), and overlapping CJK bigrams. Compat classes do not tokenize unless you pass `tokenizer=`.
- **Batch search.** `get_scores_many` and `top_n_many` release the GIL and score queries on a thread pool. Each row matches the single-query call.
- **Integrations.** `rapidbm25.integrations.langchain.BM25Retriever` (`from_texts`, `from_documents`, `k`, `invoke`, `save`, `load`) and `rapidbm25.integrations.llama_index.BM25Retriever`. Optional extras `langchain` and `llama-index`. Tests skip when the packages are absent.
- **`fuse`.** Reciprocal rank fusion and min-max weighted fusion of BM25 hits with any other ranked list.
- **`fixes=True`.** Opt-in for rank_bm25 #19 (repeated query tokens counted once, first-seen order), #36 (empty corpus), and #39 (Okapi idf of exactly 0, a term in half the documents, uses the existing epsilon floor when that floor is positive). Other idf values stay put. The default remains bit-exact.
- **MCP folder cache.** `bm25_index_path` stores the folder index under `RAPIDBM25_CACHE` (default `~/.cache/rapidbm25`) and reindexes only files whose size or mtime changed.

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
