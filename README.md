# rapidbm25

<!-- mcp-name: io.github.Maverick0351a/rapidbm25 -->

[![wheels](https://github.com/Maverick0351a/rapidbm25/actions/workflows/wheels.yml/badge.svg)](https://github.com/Maverick0351a/rapidbm25/actions/workflows/wheels.yml)

Drop-in, Rust-native replacement for `rank_bm25`'s `BM25Okapi`, `BM25L`, and `BM25Plus`.
Same constructor, same attributes, **bit-for-bit identical scores**, about 1,470× faster
`get_scores`, plus a fast `top_n`, a ready-to-run MCP server, and an agent skill.

## Install

```bash
pip install rapidbm25
```

Prebuilt wheels: CPython 3.9–3.15 on Linux (manylinux_2_28 and musllinux_1_2, x86_64 and aarch64),
macOS (x86_64 10.13+, arm64 11.0+), and Windows (x86_64). Other platforms build from the sdist
and need a Rust toolchain (1.85+).

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
| Pickling | yes | not yet |

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
partial select in Rust and is 3.3× faster. Reproduce:

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
Indexes live in memory; nothing leaves the machine. `bm25_index_path` only reads inside
`RAPIDBM25_ROOTS` (default: the working directory).

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

## Build from source

```bash
pip install maturin && maturin build --release      # or scripts/build_auditable.sh
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (API compatibility with rank_bm25 is credited there).
