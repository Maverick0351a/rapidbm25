"""Docs stay true: README examples run, and version / MCP name metadata agree."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _python_blocks():
    return re.findall(r"```python\n(.*?)```", README, re.S)


def test_readme_examples_run():
    blocks = _python_blocks()
    assert len(blocks) >= 2
    ns = {}
    for code in blocks:
        exec(compile(code, "README.md", "exec"), ns)
    assert ns["top"] == ["It is quite windy in London"]
    assert list(ns["ids"])[0] == 1 and len(ns["ids"]) == 2
    assert ns["scores"].dtype.name == "float64" and ns["scores"].shape == (3,)


def test_readme_matches_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    ns = {}
    exec(compile(_python_blocks()[0], "README.md", "exec"), ns)
    ref = rank_bm25.BM25Okapi([d.split(" ") for d in ns["docs"]])
    assert ref.get_scores(ns["query"]).tobytes() == ns["scores"].tobytes()


def _toml(path):
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_versions_and_names_agree():
    import rapidbm25

    version = _toml(ROOT / "pyproject.toml")["project"]["version"]
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "mcpb" / "manifest.json").read_text(encoding="utf-8"))
    cargo = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert rapidbm25.__version__ == version == cargo == server["version"] == manifest["version"]
    assert all(p["version"] == version and p["identifier"] == "rapidbm25" for p in server["packages"])
    assert f"<!-- mcp-name: {server['name']} -->" in README
    assert len(server["description"]) <= 100
    assert f"## {version}" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_mcp_tool_names_documented():
    from rapidbm25.mcp_server import TOOLS

    names = {t["name"] for t in TOOLS}
    manifest = json.loads((ROOT / "mcpb" / "manifest.json").read_text(encoding="utf-8"))
    assert names == {t["name"] for t in manifest["tools"]}
    for n in names:
        assert n in README
