# rapidbm25

<!-- mcp-name: io.github.Maverick0351a/rapidbm25 -->

[![wheels](https://github.com/Maverick0351a/rapidbm25/actions/workflows/wheels.yml/badge.svg)](https://github.com/Maverick0351a/rapidbm25/actions/workflows/wheels.yml)

Drop-in, Rust-native replacement for `rank_bm25`'s `BM25Okapi`, `BM25L`, and `BM25Plus`.
Same constructor, same attributes, **bit-for-bit identical scores**, about 1,470× faster
`get_scores`, plus a fast `top_n`, save/load, an incremental index, batch search,
optional tokenizers, hybrid fusion, a ready-to-run MCP server, and an agent skill.

## Install

```bash
pip install rapidbm25
```

Prebuilt wheels: CPython 3.9–3.15 on Linux (manylinux_2_28 and musllinux_1_2, x86_64 and aarch64),
macOS (x86_64 10.13+ for CPython 3.9–3.13 and 10.15+ for 3.14–3.15, arm64 11.0+), and Windows
(x86_64). Other platforms build from the sdist and need a Rust toolchain (1.85+).

## Use

```python
# from rank_bm25 import BM25Okapi
from rapidbm25 import BM25Okapi

docs = ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]
bm25 = BM25Okapi([d.split(" ") for d in docs])        # also: tokenizer=..., k1, b, epsilon

query = "windy London".split(" ")
scores = bm25.get_scores(query)                        # numpy float64, identical to rank_bm25
top = bm25.get_top_n(query, docs, n=1)                 # ['It is quite windy in London']
ids, best = bm25.top_n(query, n=2)                     # new: Rust partial select, no full sort
```

### Which rank_bm25?

`pip install rank_bm25` gives **0.2.2**, and its `BM25L` and `BM25Plus` differ from the project's
later git main (47aa3dd): 0.2.2's BM25L multiplies by term frequency (docs without the term score 0),
and its BM25Plus idf is `ln((N+1)/df)` instead of `ln(N+1) - ln(df)`. `BM25Okapi` is identical.
rapidbm25 matches **0.2.2 by default**; pass `compat="47aa3dd"` to `BM25L` / `BM25Plus` for git-main
behavior. Both are tested bit for bit.

```python
from rapidbm25 import BM25L
BM25L([d.split(" ") for d in docs], compat="47aa3dd")
```

## What is the same as rank_bm25

- Classes `BM25Okapi(corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25)`,
  `BM25L(..., delta=0.5)`, `BM25Plus(..., delta=1)`.
- Attributes `corpus_size`, `avgdl`, `doc_len`, `doc_freqs`, `idf`, `average_idf` (Okapi), `k1`, `b`, `epsilon`/`delta`.
- `get_scores`, `get_batch_scores`, `get_top_n` with the same arguments and asserts.
- Scores match bit for bit, including repeated query terms, unknown terms, empty documents,
  the Okapi negative-idf floor, and the NaN cases (`k1=0`, `b=1` with empty docs).

## What is different

| | rank_bm25 | rapidbm25 |
|---|---|---|
| `get_top_n` tie order | reverse of numpy's unstable argsort (varies by numpy build) | score descending, then lower doc id (deterministic) |
| Tokens | any hashable | `str` (non-str corpus tokens raise `TypeError`) |
| Editing `.idf` / `.doc_freqs` after build | changes later scores | no effect (scores come from the native index) |
| `tokenizer=` | runs in a multiprocessing pool | runs in-process |
| `compat=` keyword on BM25L / BM25Plus | n/a | `"0.2.2"` (default) or `"47aa3dd"` |
| Pickling / `save` / `load` | pickle only | pickle, plus a versioned file (load copies; `mmap=True` is opt-in) |
| `fixes=` | n/a | compat classes default `False` (bit-exact). `LiveIndex` defaults `True` |
| Lone surrogate in a token | accepted (`str` can hold it) | raises `UnicodeEncodeError` |

## Save, load, and pickle

```python
okapi_ix = BM25Okapi([d.split(" ") for d in docs])
okapi_ix.save("/tmp/rapidbm25-okapi.rbm25")
okapi_loaded = BM25Okapi.load("/tmp/rapidbm25-okapi.rbm25")   # copies into memory
```

`save` writes format 1: magic `RBM25`, a JSON header, then little-endian numeric columns.
Floats are stored as raw IEEE-754 bits, so `get_scores` after `load` matches the index that
was saved, bit for bit. The default ``load`` copies the columns into memory, so later truncation or
in-place edits of the file cannot change scores or kill the process.
``load(mmap=True)`` maps the caller's file and is unsafe if that file changes:
truncation can raise SIGBUS, and an in-place write can change scores. Pickle
keeps a picklable `tokenizer`; the file
format stores the tokens already in the index, not an arbitrary callable.

## Live index

`LiveIndex` is a new class (the compat classes stay immutable and bit-exact). It assigns
stable string ids, updates `N`, document frequency, `avgdl`, and idf on every `add`,
`delete`, and `update`, and patches posting lists instead of rebuilding them. Score-vector
order is insertion order; `update` keeps a document's place; `delete` closes the gap.
`get_scores` matches `BM25Okapi` / `BM25L` / `BM25Plus` built on `live.corpus()` with the
same `k1`, `b`, `epsilon` or `delta`, `compat`, and `fixes`. An empty live index is valid.
`fixes` defaults to `True` on `LiveIndex` and to `False` on the compat classes. Pass
`fixes=False` for rank_bm25's exact bugs as well as its formulas. The folder index used
by `bm25_index_path` is a `LiveIndex`, so it uses that default.

```python
from rapidbm25 import LiveIndex, Tokenizer, fuse
words = Tokenizer(lowercase=True, stopwords="english", stemmer="english")
live_ix = LiveIndex(tokenizer=words)
live_ix.add("dogs", "The running dogs chased the ball")
live_ix.add("cats", "A sleeping cat", metadata={"kind": "pet"})
live_ix.add("weather", "It is quite windy in London")
pet_hits = live_ix.search("sleeping cats", k=1, where={"kind": "pet"})
live_ix.save("/tmp/rapidbm25-live.rbm25")
reloaded = LiveIndex.load("/tmp/rapidbm25-live.rbm25")
batch_scores = reloaded.get_scores_many(["dogs", "london wind"])
fused = fuse([["dogs", "cats"], [("cats", 0.9), ("weather", 0.2)]], method="rrf")
```

`where` is a dict (`$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, `$exists`)
or a callable. `$in` and `$nin` take a list, not a string. A boolean does not match
`0` or `1`. The callable runs outside the index lock. `add` / `delete` / `update`
take a lock, so several threads can share one `LiveIndex`.

`variant` is `"okapi"` (default), `"l"`, or `"plus"`. `compat` is `"0.2.2"` or
`"47aa3dd"`, same as the static classes. `delta` defaults to 0.5 for L and 1 for Plus.

## Tokenizers

`Tokenizer` is opt-in. Unicode words (UAX #29), optional lowercase, an original English
stop list (or your own), a Snowball stemmer (`stemmer="english"` is Porter2; other
Snowball languages are accepted), and `cjk=True` for overlapping character bigrams over
Han, Hiragana, Katakana, and Hangul. Passing `tokenizer=` to a compat class is the only
way it tokenizes; omitting it leaves those classes unchanged.

## Batch search

`get_scores_many(queries)` returns a `(n_queries, corpus_size)` float64 array.
`top_n_many(queries, n)` returns one `(ids, scores)` pair per query. Both release the
GIL and score queries on a thread pool. Row `i` is bit-identical to the single-query call.
`LiveIndex` has the same two methods.

## fixes=True

On `BM25Okapi`, `BM25L`, and `BM25Plus`, `fixes=False` is the default and matches
rank_bm25, bugs included. `LiveIndex` defaults to `fixes=True`. The Okapi IDF formula
stays `ln(N - df + 0.5) - ln(df + 0.5)` either way. There is no smoothed replacement.

| issue | `fixes=False` | `fixes=True` |
|---|---|---|
| #19 repeated query tokens | each repeat adds the term again | the term is counted once, first-seen order |
| #36 empty corpus | `ZeroDivisionError` on the compat classes | empty static index, `get_scores` returns shape `(0,)` |
| #39 idf exactly 0 | Okapi idf is 0 when a term is in exactly half the documents | that idf uses the existing `epsilon * average_idf` floor when the floor is positive |

#39 does not change any other idf. It does not apply when `epsilon * average_idf` is
zero or negative. That is the usual 2-document case: each term that appears in one
document is in half the corpus, the average idf is not positive, and the term still
scores 0. `BM25L` and `BM25Plus` do not have this zero. The fix is Okapi-only. L and
Plus idf never hit exactly 0 for a real document frequency.

## Hybrid fusion

`fuse(rankings, method="rrf", k=60)` is reciprocal rank fusion: `weight / (k + rank)`,
ranks starting at 1. `k=60` is the default (the usual Cormack constant). Input scores
are ignored, so a BM25 list and a vector-store list do not need a shared scale.
`method="weighted"` min-max normalizes each list to `[0, 1]`
and returns the weighted sum. A list of equal positive scores contributes its weight;
a list of zeros contributes nothing.

## Integrations

Optional extras. Both are thin retrievers over `LiveIndex`, with `from_texts`, `k`,
and `save` / `load`. Tests skip when the dependency is not installed.

```bash
pip install "rapidbm25[langchain]"      # langchain-core
pip install "rapidbm25[llama-index]"    # llama-index-core
```

```python
from rapidbm25.integrations.langchain import BM25Retriever
retriever = BM25Retriever.from_texts(["sunny Paris", "windy London"], k=1)
assert retriever.invoke("windy")[0].page_content == "windy London"
```

`BM25Retriever.from_documents` accepts LangChain `Document` objects.
`rapidbm25.integrations.llama_index.BM25Retriever` implements `_retrieve` and returns
`NodeWithScore` nodes.

## Benchmarks

Workload: 16,000 chunks of Python stdlib sources and docs (mean 193 tokens, vocab 58,073), 200
held-out queries (mean 8.7 tokens), `BM25Okapi`. Machine: Intel Xeon (Sapphire Rapids) KVM guest,
8 vCPU, Linux x86_64, CPython 3.13.5, numpy 2.5.4. Shared VM, so treat small gaps as noise.

5 rounds; in each round every implementation runs once in a fresh process, in rotated order.
Medians; rapidbm25 rows time 20 batches per process.

| Implementation | Index build | `get_scores` × 200 | End to end × 200 (tokenize + score + top-10) |
|---|---|---|---|
| rank_bm25 0.2.2 | 0.433 s | 4.225 s | 4.322 s |
| rapidbm25, top-10 via `np.lexsort` | **0.119 s (3.6×)** | **2.87 ms (1,470×)** | 53.6 ms (81×) |
| rapidbm25, `top_n(q, 10)` | 0.124 s | 2.90 ms | **16.4 ms (263×)** |

End to end with `np.lexsort` is dominated by sorting 16,000 scores per query; `top_n` does a
partial select in Rust and is 3.3× faster.

### 0.2 save, incremental updates, batch search

`python bench/bench_v02.py`, one process, median of 5 after a warmup. Machine: Intel Xeon,
4 vCPU, Linux x86_64, CPython 3.12.3, release build. Synthetic corpus unless noted:
8,000 documents, 48 tokens each, vocabulary 4,000. The saved file is 7,924,566 bytes.

| operation | n | median |
|---|---:|---:|
| static `BM25Okapi` build | 8,000 | 10.7 ms |
| `save` | 8,000 | 15.9 ms |
| `load(mmap=True)` + one query | 8,000 | 2.01 ms |
| `load(mmap=False)` + one query | 8,000 | 2.81 ms |
| `LiveIndex.add` of every document | 8,000 | 85.3 ms |
| `LiveIndex.delete` | 800 | 6.82 ms |
| static rebuild of the documents still left | 7,200 | 8.53 ms |
| `LiveIndex.add` of 200 more, index already built | 200 | 1.57 ms |
| static rebuild of that whole set | 8,200 | 9.58 ms |

Building 8,000 documents in one `BM25Okapi(...)` call is faster than 8,000 individual
`add`s. Once the live index exists, adding 200 documents is 1.57 ms against 9.58 ms to
rebuild all 8,200, and deleting 800 is 6.82 ms against 8.53 ms to rebuild the rest.

Batch search is the same comparison, end to end (`get_scores` in a loop vs `get_scores_many`),
median of 5:

| workload | sequential | `get_scores_many` | speedup |
|---|---:|---:|---:|
| Okapi, 8,000×48, vocab 4,000, 64 queries (short postings) | 0.331 ms | 1.021 ms | 0.32× |
| Okapi, 30,000×40, vocab 200, 64 queries (long postings) | 4.94 ms | 3.64 ms | 1.36× |
| BM25Plus, 30,000×32, vocab 4,000, 48 queries (every document is scored) | 7.59 ms | 3.66 ms | 2.08× |

On this 4-vCPU guest the sparse Okapi kernel is cheap enough that the thread pool loses
when each query only walks short postings. It wins once a query walks long postings, and
by about 2× for BM25Plus, which adds a term's contribution to every document. Rows match
the single-query scores bit for bit either way.

Reproduce the 0.1 comparison:

```bash
pip install rapidbm25 rank-bm25==0.2.2
python bench/gen_workload.py --out /tmp/bm25-wl          # chunks your Python stdlib (numbers will differ by machine and corpus)
python bench/bench_interleaved.py --workloads /tmp/bm25-wl --out /tmp/bench.json
```

## Correctness evidence

- **Contract replay**: `contract/bm25.contract.json` holds 407 input/output vectors observed by
  running rank_bm25 0.2.2 from PyPI; `contract/bm25-47aa3dd.contract.json` holds 407 from git
  47aa3dd. `tests/replay_contract.py` replays them with exact comparison (floats as hex): 407/407 each.
  A control build that nudges scores by 1 ulp (`tests/control_ulp_off.py`) fails 263 of them, so
  the replay is not blind.
- **Fuzz** (`tests/fuzz_equivalence.py`): 1,000 random cases against each reference, across all
  three variants and edge parameters, compared bit for bit on scores, batch scores, idf,
  doc_freqs, avgdl, average_idf, and top-n: 0 failures.
- **Every wheel in CI** is installed on its own platform and runs the test suite, a 300-case fuzz
  against rank_bm25 0.2.2, and both contract replays.
- Wheels are built with `cargo-auditable` (dependency list embedded in the `.dep-v0` section).

## MCP server

`rapidbm25-mcp` is a stdio MCP server shipped in the wheel (no MCP SDK dependency):
`bm25_index_texts`, `bm25_index_path`, `bm25_search`, `bm25_list_indexes`, `bm25_drop_index`.
Text indexes live in memory. `bm25_index_path` caches the folder index under
`RAPIDBM25_CACHE` (default `~/.cache/rapidbm25`, directory mode `0700`, files mode `0600`).
A cached file is reused only when its size, mtime, and the SHA-256 of the bytes that
were indexed still match; the path is statted again after the read. Symlinks are not
followed. Files with another hard link (`st_nlink > 1`) are skipped. Dotfiles are
skipped unless the glob's filename starts with `.`. Reads stay inside `RAPIDBM25_ROOTS`
(default: the working directory). Nothing leaves the machine.

```json
{ "mcpServers": { "rapidbm25": { "command": "uvx", "args": ["--from", "rapidbm25", "rapidbm25-mcp"],
  "env": { "RAPIDBM25_ROOTS": "/path/to/repo" } } } }
```

Details: [mcp/README.md](mcp/README.md). Registry metadata: [server.json](server.json).

## Agent skill

`skill/rapidbm25-search/` (SKILL.md + `scripts/bm25_search.py`) for agents that run shell commands:

```bash
python skill/rapidbm25-search/scripts/bm25_search.py --path ./src --query "parse_header timeout" -k 10
```

## Agent eval

`eval/agent/` asks 20 code-search questions against the urllib3 commit pinned in
`eval/agent/tasks.json` (release 2.8.0). Each answer is one file and one symbol, checked
automatically. The harness asks the same model the same prompt twice: once with the
rapidbm25 MCP tools, once with grep and read-file. It writes `runs.json` and `summary.md`
(correctness rate and medians) under `eval/agent/out/`.

Dry-run, with a scripted model and no API key:

```bash
python -m venv .venv && . .venv/bin/activate
pip install "maturin>=1.7,<2" && maturin develop
python eval/agent/harness.py --dry-run
```

Live run, against an https OpenAI-compatible chat completions endpoint
(`http` is accepted only for `localhost` and `127.0.0.1`):

```bash
export EVAL_API_KEY=...          # secret; never commit it
export EVAL_MODEL=...
export EVAL_BASE_URL=https://api.openai.com/v1   # root that serves /chat/completions
python eval/agent/harness.py
```

The GitHub workflow `agent-eval` runs the live comparison only from Actions → Run workflow,
and only if the `EVAL_API_KEY` repository secret is set. The form supplies the model and
base URL. Pull requests run the dry-run and do not call a model.

## Build from source

```bash
pip install maturin && maturin build --release      # or scripts/build_auditable.sh
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (API compatibility with rank_bm25 is credited there).
