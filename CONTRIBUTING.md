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

## Release

0.2.0 is on PyPI. Ship the next version by tagging, not by pasting a token:

1. Keep the version identical in `pyproject.toml`, `Cargo.toml`, `python/rapidbm25/__init__.py`, `server.json`, and `mcpb/` (tests check this).
2. Record the version in `CHANGELOG.md`.
3. Merge to `main` and push a tag `vX.Y.Z` whose name matches that version.
4. `.github/workflows/wheels.yml` builds and tests the wheels, then publishes to PyPI with trusted publishing (the `pypi` environment, OIDC, no API token). A tag that does not match `pyproject.toml` is refused.

Publishing `server.json` to the MCP Registry stays a manual run of the same workflow (`publish` and `mcp_registry`) after the PyPI release is visible. The PyPI trusted publisher must allow `wheels.yml` in the `pypi` environment.
