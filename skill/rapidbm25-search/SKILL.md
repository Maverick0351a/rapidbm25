---
name: rapidbm25-search
description: Keyword (BM25) search over a local folder of code or docs, or over a list of texts. Use when you need to find which files or chunks mention specific identifiers, error strings, or terms, faster and more ranked than grep, with no network and no embeddings.
---

# rapidbm25-search

Ranked lexical search with a Rust BM25 core. Good for exact identifiers, error
messages, config keys, and rare terms, where embedding search is weak.

## Use

```bash
pip install rapidbm25            # or: uv pip install rapidbm25
python scripts/bm25_search.py --path ./repo --query "parse_header timeout" -k 10
python scripts/bm25_search.py --path ./docs --glob "*.md" --query "rate limit" --json
```

Output: rank, `file:start-end` line range, score, and the first line of the chunk.
Open the reported line range to read the hit.

## Tips

- Query with the words you expect in the answer, not a question. Identifiers work best.
- Narrow with `--glob "*.py"`; adjust `--chunk-lines` (default 40) for longer context.
- For an MCP client, run the bundled server instead: `rapidbm25-mcp`.
  `bm25_index_path` caches the folder under `RAPIDBM25_CACHE` and only reindexes files
  whose size or mtime changed.
- In Python: `from rapidbm25 import BM25Okapi` is a drop-in for `rank_bm25.BM25Okapi`,
  same scores bit for bit; `bm25.top_n(tokens, n)` gives fast (ids, scores).
- `bm25.save(path)` / `BM25Okapi.load(path)` round-trip scores. `LiveIndex` adds, deletes,
  and updates documents by string id without a full rebuild (`get_scores` matches a
  static rebuild). `get_scores_many` batches queries on a thread pool.
- `Tokenizer(...)` (lowercase, stopwords, Snowball stemmer, CJK bigrams) is opt-in.
- `fuse([bm25_hits, vector_hits], method="rrf")` mixes BM25 with any other ranking.
- `fixes=True` opts into three rank_bm25 bugfixes; the default stays bit-exact.
