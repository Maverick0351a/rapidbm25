# Contributing

Issues and pull requests are welcome.

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest numpy rank-bm25==0.2.2
maturin develop --release
pytest -q tests
python tests/ci_check.py --cases 300            # fuzz vs rank_bm25 0.2.2 + both contract replays
```

Rules for changes:

- Scores must stay bit for bit identical to rank_bm25 (0.2.2 by default, 47aa3dd with
  `compat="47aa3dd"`). Keep IEEE operation order; do not introduce FMA or reassociation.
- `tests/replay_contract.py` must pass on both `contract/*.contract.json` files. Do not edit the
  recorded vectors to make a change pass; they were observed by running rank_bm25.
- Do not copy rank_bm25 source into this repository. It is used only as a test oracle.
- Performance claims need a run of `bench/bench_interleaved.py` with machine details.

By contributing you agree that your contributions are licensed under Apache-2.0.
