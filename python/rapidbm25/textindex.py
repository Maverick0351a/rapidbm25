"""Small helpers shared by the MCP server and the agent skill script."""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import BM25L, BM25Okapi, BM25Plus

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
VARIANTS = {"okapi": BM25Okapi, "l": BM25L, "plus": BM25Plus}
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", ".venv", "venv", "__pycache__"}
MAX_FILE_BYTES = 2_000_000


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class TextIndex:
    ids: list[str]
    texts: list[str]
    variant: str = "okapi"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        cls = VARIANTS[self.variant]
        toks = [tokenize(t) for t in self.texts]
        self._sets = [frozenset(t) for t in toks]
        self.bm25 = cls(toks, **self.params)

    def search(self, query: str, k: int = 10, snippet_chars: int = 400) -> list[dict]:
        q = tokenize(query)
        qs = set(q)
        # over-fetch, then drop docs that share no query term (tiny corpora can
        # give Okapi idf <= 0, and L/Plus give every doc a delta floor)
        ids, scores = self.bm25.top_n(q, min(len(self.ids), max(k * 4, k + 50)))
        out = []
        for i, s in zip(ids.tolist(), scores.tolist()):
            if len(out) >= k:
                break
            if qs.isdisjoint(self._sets[i]):
                continue
            out.append({"rank": len(out) + 1, "id": self.ids[i], "score": round(s, 6),
                        "text": self.texts[i][:snippet_chars]})
        return out


def allowed_roots() -> list[Path]:
    env = os.environ.get("RAPIDBM25_ROOTS")
    roots = env.split(os.pathsep) if env else [os.getcwd()]
    return [Path(r).resolve() for r in roots if r]


def check_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    for r in allowed_roots():
        if p == r or r in p.parents:
            return p
    raise PermissionError(f"{p} is outside the allowed roots (set RAPIDBM25_ROOTS)")


def chunk_path(path: str, glob: str = "*", chunk_lines: int = 40, overlap: int = 10,
               max_files: int = 5000) -> tuple[list[str], list[str]]:
    root = check_path(path)
    files = [root] if root.is_file() else []
    if root.is_dir():
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in fns:
                if fnmatch.fnmatch(fn, glob):
                    files.append(Path(dp) / fn)
                    if len(files) >= max_files:
                        break
    ids, texts = [], []
    step = max(1, chunk_lines - overlap)
    for f in files:
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = f.read_bytes()
            if b"\0" in raw[:4096]:
                continue  # binary
            lines = raw.decode("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = f.relative_to(root) if root.is_dir() else Path(f.name)
        for start in range(0, max(1, len(lines)), step):
            body = "\n".join(lines[start:start + chunk_lines])
            if body.strip():
                ids.append(f"{rel}:{start + 1}-{min(start + chunk_lines, len(lines))}")
                texts.append(body)
            if start + chunk_lines >= len(lines):
                break
    return ids, texts
