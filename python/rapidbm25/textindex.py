"""Small helpers shared by the MCP server and the agent skill script."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

from . import BM25L, BM25Okapi, BM25Plus
from .live import LiveIndex

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
VARIANTS = {"okapi": BM25Okapi, "l": BM25L, "plus": BM25Plus}
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", ".venv", "venv", "__pycache__"}
MAX_FILE_BYTES = 2_000_000
MAX_CHUNKS = 200_000


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
        ids, scores = self.bm25.top_n(q, len(self.ids))
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


def _glob_asks_for_dotfile(pattern: str) -> bool:
    """True when the filename part of the glob itself starts with a dot.

    Python's ``fnmatch("*", ".env")`` is true. A shell glob does not match
    dotfiles, and neither does a folder index, unless the pattern asks.
    """
    name = pattern.replace("\\", "/").split("/")[-1]
    return name.startswith(".")


def _name_selected(name: str, pattern: str) -> bool:
    if name.startswith(".") and not _glob_asks_for_dotfile(pattern):
        return False
    return fnmatch.fnmatch(name, pattern)


def _under_root(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def read_regular_under_root(path: Path, root: Path) -> bytes | None:
    """Read a regular file that stays inside ``root``.

    Symlinks are skipped (``lstat`` plus ``O_NOFOLLOW``). Hardlinks
    (``st_nlink > 1``) are skipped too: another name for the same inode can
    sit outside the allowed root, and ``resolve()`` does not reveal it.
    The resolved path of a symlink must also stay under ``root``.
    """
    try:
        st = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1 or st.st_size > MAX_FILE_BYTES:
        return None
    if not _under_root(path, root):
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError:
        return None
    try:
        st2 = os.fstat(fd)
        if not stat.S_ISREG(st2.st_mode) or st2.st_nlink > 1 or st2.st_size > MAX_FILE_BYTES:
            return None
        chunks = []
        remaining = st2.st_size
        while remaining > 0:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)


def snapshot_file(path: Path, root: Path):
    """Return ``(stat, bytes, sha256)`` for bytes that still match a later stat.

    Retries when mtime, size, or inode move between the read and the re-stat,
    so a rewrite cannot pin a stale hash to a new timestamp.
    """
    for _ in range(4):
        try:
            before = path.lstat()
        except OSError:
            return None
        raw = read_regular_under_root(path, root)
        try:
            after = path.lstat()
        except OSError:
            return None
        if raw is None:
            return None
        stable = (
            stat.S_ISREG(before.st_mode)
            and stat.S_ISREG(after.st_mode)
            and before.st_nlink == 1
            and after.st_nlink == 1
            and before.st_mtime_ns == after.st_mtime_ns
            and before.st_size == after.st_size
            and before.st_ino == after.st_ino
        )
        if stable:
            return after, raw, hashlib.sha256(raw).hexdigest()
    return None


def iter_files(root: Path, glob: str, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]
    files = []
    if root.is_dir():
        for dp, dns, fns in os.walk(root, followlinks=False):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in fns:
                if _name_selected(fn, glob):
                    files.append(Path(dp) / fn)
                    if len(files) >= max_files:
                        return files
    return files


def _chunk_cap(rel: str) -> ValueError:
    return ValueError(f"chunk cap exceeded ({MAX_CHUNKS}) at {rel}")


def chunk_bytes(raw: bytes, rel: str, chunk_lines: int, overlap: int,
                budget: list[int] | None = None) -> tuple[list[str], list[str]]:
    """Split ``raw`` into line windows. ``budget`` is ``[windows_left]`` for this build."""
    if budget is None:
        budget = [MAX_CHUNKS]
    if b"\0" in raw[:4096]:
        return [], []
    step = max(1, chunk_lines - overlap)
    # Lower bound: `splitlines` yields at least one line per `\n`. Step 1
    # visits every line, so this cannot under-count the windows we would walk.
    if step == 1 and raw.count(b"\n") > budget[0]:
        raise _chunk_cap(rel)
    lines = raw.decode("utf-8", errors="replace").splitlines()
    ids, texts = [], []
    seen = 0
    for start in range(0, max(1, len(lines)), step):
        seen += 1
        if seen > budget[0]:
            raise _chunk_cap(rel)
        body = "\n".join(lines[start:start + chunk_lines])
        if body.strip():
            ids.append(f"{rel}:{start + 1}-{min(start + chunk_lines, len(lines))}")
            texts.append(body)
        if start + chunk_lines >= len(lines):
            break
    budget[0] -= seen
    return ids, texts


def chunk_file(path: Path, root: Path, chunk_lines: int, overlap: int,
               budget: list[int] | None = None) -> tuple[list[str], list[str]]:
    raw = read_regular_under_root(path, root)
    if raw is None:
        return [], []
    rel = str(path.relative_to(root) if root.is_dir() else Path(path.name))
    return chunk_bytes(raw, rel, chunk_lines, overlap, budget)


def chunk_path(path: str, glob: str = "*", chunk_lines: int = 40, overlap: int = 10,
               max_files: int = 5000) -> tuple[list[str], list[str]]:
    root = check_path(path)
    ids, texts = [], []
    budget = [MAX_CHUNKS]
    for f in iter_files(root, glob, max_files):
        more_ids, more_texts = chunk_file(f, root, chunk_lines, overlap, budget)
        ids.extend(more_ids)
        texts.extend(more_texts)
    return ids, texts


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path, 0o700)


def _restrict(path: Path) -> None:
    if os.name != "nt":
        os.chmod(path, 0o600)


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(os.fspath(path), flags, 0o600)
    try:
        os.write(fd, data)
        if os.name != "nt":
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def cache_root() -> Path:
    env = os.environ.get("RAPIDBM25_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "rapidbm25"


def _cache_key(root: Path, glob: str, chunk_lines: int, overlap: int, variant: str, params: dict) -> str:
    payload = json.dumps(
        {"root": str(root), "glob": glob, "chunk_lines": chunk_lines, "overlap": overlap,
         "variant": variant, "k1": params.get("k1", 1.5), "b": params.get("b", 0.75)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class PathIndex:
    """Folder index that caches on disk and reindexes only files that changed.

    Chunks live in a :class:`LiveIndex`, so a changed file drops its old chunk
    ids and adds the new ones without rebuilding the rest of the folder.
    ``search`` matches :class:`TextIndex`: same tokenizer, same over-fetch, and
    hits that share no query term are dropped.
    """

    def __init__(self, variant: str = "okapi", params: dict | None = None):
        self.variant = variant
        self.params = dict(params or {})
        self.live = LiveIndex(variant=variant, tokenizer=tokenize, k1=float(self.params.get("k1", 1.5)),
                              b=float(self.params.get("b", 0.75)))
        self.files: dict[str, dict] = {}
        self.sets: dict[str, frozenset] = {}
        self.last_stats = {"files_reindexed": 0, "from_cache": False, "chunks": 0}

    @property
    def ids(self) -> list[str]:
        return self.live.ids()

    def _remember(self, chunk_id: str, text: str) -> None:
        self.sets[chunk_id] = frozenset(tokenize(text))

    def sync(self, root: Path, glob: str, chunk_lines: int, overlap: int, max_files: int = 5000) -> dict:
        seen = set()
        reindexed = 0
        budget = [max(0, MAX_CHUNKS - self.live.corpus_size)]
        for f in iter_files(root, glob, max_files):
            rel = str(f.relative_to(root) if root.is_dir() else Path(f.name))
            snap = snapshot_file(f, root)
            if snap is None:
                continue
            st, raw, digest = snap
            seen.add(rel)
            prev = self.files.get(rel)
            if (prev and prev.get("mtime_ns") == st.st_mtime_ns and prev.get("size") == st.st_size
                    and prev.get("sha256") == digest):
                continue
            reindexed += 1
            if prev:
                old_ids = prev["chunk_ids"]
                for cid in old_ids:
                    self.live.delete(cid)
                    self.sets.pop(cid, None)
                budget[0] += len(old_ids)
            ids, texts = chunk_bytes(raw, rel, chunk_lines, overlap, budget)
            for cid, text in zip(ids, texts):
                self.live.add(cid, text, metadata={"path": rel})
                self._remember(cid, text)
            self.files[rel] = {
                "mtime_ns": st.st_mtime_ns, "size": st.st_size, "sha256": digest,
                "ino": int(st.st_ino), "chunk_ids": ids,
            }
        for rel in list(self.files):
            if rel not in seen:
                for cid in self.files[rel]["chunk_ids"]:
                    self.live.delete(cid)
                    self.sets.pop(cid, None)
                del self.files[rel]
                reindexed += 1
        stats = {"files_reindexed": reindexed, "chunks": self.live.corpus_size}
        self.last_stats.update(stats)
        return stats

    def save_cache(self, loc: Path) -> None:
        _ensure_private_dir(loc.parent)
        _ensure_private_dir(loc)
        previous = os.umask(0o077)
        try:
            self.live.save(loc / "index.rbm25")
            manifest = {"format": 1, "variant": self.variant, "params": self.params, "files": self.files}
            _write_private(loc / "manifest.json", json.dumps(manifest).encode("utf-8"))
            _restrict(loc / "index.rbm25")
        finally:
            os.umask(previous)

    def load_cache(self, loc: Path) -> None:
        self.live = LiveIndex.load(loc / "index.rbm25")
        manifest = json.loads((loc / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != 1:
            raise ValueError("unsupported folder-index cache")
        self.files = manifest["files"]
        self.sets = {}
        for doc_id, text in zip(self.live.ids(), self.live.texts()):
            self._remember(doc_id, text)

    def search(self, query: str, k: int = 10, snippet_chars: int = 400) -> list[dict]:
        q = tokenize(query)
        qs = set(q)
        n = self.live.corpus_size
        if n == 0:
            return []
        raw = self.live.search(q, k=n)
        out = []
        for hit in raw:
            if len(out) >= k:
                break
            if qs.isdisjoint(self.sets.get(hit["id"], ())):
                continue
            out.append({"rank": len(out) + 1, "id": hit["id"], "score": round(hit["score"], 6),
                        "text": hit["text"][:snippet_chars]})
        return out


def index_folder(path: str, glob: str = "*", chunk_lines: int = 40, overlap: int = 10,
                 variant: str = "okapi", params: dict | None = None, *, use_cache: bool = True,
                 max_files: int = 5000) -> PathIndex:
    """Index ``path``, reusing a disk cache and skipping files whose size and mtime are unchanged."""
    root = check_path(path)
    params = dict(params or {})
    ix = PathIndex(variant, params)
    from_cache = False
    loc = None
    if use_cache:
        loc = cache_root() / _cache_key(root, glob, chunk_lines, overlap, variant, params)
        if (loc / "manifest.json").is_file() and (loc / "index.rbm25").is_file():
            try:
                ix.load_cache(loc)
                from_cache = True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                ix = PathIndex(variant, params)
    stats = ix.sync(root, glob, chunk_lines, overlap, max_files=max_files)
    stats["from_cache"] = from_cache
    ix.last_stats = stats
    if use_cache and loc is not None:
        try:
            ix.save_cache(loc)
        except OSError:
            pass
    return ix
