# Changelog

All notable changes to this project are documented here. Versions follow [SemVer](https://semver.org/).

## Unreleased 0.2.0

Not published. The default `BM25Okapi` / `BM25L` / `BM25Plus` scores stay bit-identical to rank_bm25.

- **Persistence.** Every compat class pickles. `save(path)` / `load(path)` write a versioned little-endian file (`RBM25` format 1). Scalar floats are stored as IEEE-754 hex, so a round trip is bit-identical. `load` copies the file into memory. `mmap=True` is opt-in and unsafe if that file changes. `save` creates its temp file with `O_EXCL` and `O_NOFOLLOW`, so a symlink at the temp name is not followed.
- **`LiveIndex`.** Add, delete, and update documents by stable string id, with optional metadata and a `where` filter (`$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, `$exists`, or a callable). Global statistics update on each mutation; posting lists are not rebuilt from scratch. `get_scores` matches a static index built on `corpus()` (same variant, `compat`, parameters, and `fixes`). `fixes` defaults to `True` (pass `fixes=False` for bit-exact rank_bm25 bugs). An empty live index is valid. A `threading.Lock` plus an internal mutex make concurrent add/search safe; a filter must not call back into the index. The MCP folder index uses `LiveIndex`, so it follows this default.
- **Tokenizers (opt-in).** `Tokenizer(lowercase=, stopwords="english"|list, stemmer=<snowball language>, cjk=True)`. Unicode word split, a small original English stop list, Snowball stemming via `rust-stemmers` (MIT OR BSD-3-Clause), and overlapping CJK bigrams. Compat classes do not tokenize unless you pass `tokenizer=`.
- **Batch search.** `get_scores_many` and `top_n_many` release the GIL and score queries on a thread pool. Each row matches the single-query call.
- **Integrations.** `rapidbm25.integrations.langchain.BM25Retriever` (`from_texts`, `from_documents`, `k`, `invoke`, `save`, `load`) and `rapidbm25.integrations.llama_index.BM25Retriever`. Optional extras `langchain` and `llama-index`. Tests skip when the packages are absent.
- **`fuse`.** Reciprocal rank fusion (`k=60` by default) and min-max weighted fusion of BM25 hits with any other ranked list.
- **`fixes=True`.** Opt-in on the compat classes (their default stays `fixes=False`, bit-exact). Repeated query tokens are counted once, first-seen order (#19). An empty corpus is allowed (#36). Okapi idf that is exactly 0, a term in half the documents, uses the existing `epsilon * average_idf` floor when that floor is positive (#39). The IDF formula is unchanged, and no other idf moves. On a 2-document corpus the floor is not positive, so a term in one of the two documents still has idf 0.
- **MCP folder cache.** `bm25_index_path` stores the folder index under `RAPIDBM25_CACHE` (default `~/.cache/rapidbm25`, directory mode `0700`, files mode `0600`). A file is reused only when size, mtime, and a SHA-256 of the bytes that were indexed still match, and the path is re-stat'ed after the read. Symlinks and hardlinks (`st_nlink > 1`) are not indexed. Dotfiles are skipped unless the glob's filename starts with `.`.
- **Safe `load`.** The default copies the index into memory. `mmap=True` is an explicit opt-in that maps the caller's file; truncation or in-place writes of that file are unsafe. Load rejects a bad endian marker, a misaligned section, a non-monotonic offset table, or a posting id past `n_docs` with `ValueError` instead of panicking.
- **Retrieval filters.** LangChain and LlamaIndex retrievers drop documents that share no query term. The full ranking is filtered, so a zero-score match past the first `k + 50` documents is still returned. Metadata `where` callbacks run outside the index lock, and calling back into the same index raises `RuntimeError` before the index changes. `$in` requires a list. Booleans do not match `0` or `1`.

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
