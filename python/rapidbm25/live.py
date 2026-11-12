"""Incremental BM25 index with stable string ids.

Scores use the same formulas as :class:`rapidbm25.BM25Okapi`, :class:`BM25L`
and :class:`BM25Plus` (including ``compat`` and ``fixes``). ``get_scores`` is
bit-identical to building that static class on :meth:`corpus`.

Document order, which is the order of the score vector, is insertion order.
:meth:`update` replaces a document in place. :meth:`delete` removes it and
closes the gap. Ids are strings and do not move. ``N``, document frequency,
``avgdl`` and idf all change when the set of documents changes; callers do not
rebuild the index themselves. Posting lists are updated per document.

An empty :class:`LiveIndex` is valid (``corpus_size == 0``). That differs from
the static classes, which still raise ``ZeroDivisionError`` on an empty corpus
unless ``fixes=True``.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Optional, Sequence

from . import _native

_COMPAT = {"0.2.2": "022", "47aa3dd": ""}
_MISSING = object()


def _native_variant(variant: str, compat: str) -> str:
    if variant == "okapi":
        return "okapi"
    if variant not in ("l", "plus"):
        raise ValueError("variant must be 'okapi', 'l' or 'plus'")
    if compat not in _COMPAT:
        raise ValueError(f"compat must be one of {sorted(_COMPAT)}, got {compat!r}")
    return variant + _COMPAT[compat]


def _public_variant(native: str) -> str:
    if native.startswith("l"):
        return "l"
    if native.startswith("plus"):
        return "plus"
    return "okapi"


def _match(meta: dict, where) -> bool:
    """Structured metadata filter.

    ``where`` is a callable ``meta -> bool``, or a dict of field to either a
    value (equality) or an operator dict. Operators: ``$eq $ne $in $nin $gt
    $gte $lt $lte $exists``. Comparisons other than ``$exists`` fail when the
    field is absent. A callable must not call back into the index.
    """
    if callable(where):
        return bool(where(meta))
    if not isinstance(where, dict):
        raise TypeError("where must be a dict or a callable")
    for key, cond in where.items():
        has = key in meta
        val = meta[key] if has else _MISSING
        if isinstance(cond, dict):
            for op, operand in cond.items():
                if op == "$exists":
                    if bool(has) != bool(operand):
                        return False
                    continue
                if not has:
                    return False
                if op == "$eq":
                    ok = val == operand
                elif op == "$ne":
                    ok = val != operand
                elif op == "$in":
                    ok = val in operand
                elif op == "$nin":
                    ok = val not in operand
                elif op == "$gt":
                    ok = val > operand
                elif op == "$gte":
                    ok = val >= operand
                elif op == "$lt":
                    ok = val < operand
                elif op == "$lte":
                    ok = val <= operand
                else:
                    raise ValueError(f"unknown filter operator {op!r}")
                if not ok:
                    return False
        elif not has or val != cond:
            return False
    return True


def _tokenizer_tag(tokenizer) -> Optional[str]:
    if tokenizer is None:
        return None
    if isinstance(tokenizer, _native.Tokenizer):
        spec = {
            "type": "tokenizer",
            "lowercase": bool(tokenizer.lowercase),
            "stemmer": tokenizer.stemmer,
            "cjk": bool(tokenizer.cjk),
            "stop_preset": tokenizer.stop_preset,
            "stop_custom": list(tokenizer.stop_custom),
        }
        return json.dumps(spec, separators=(",", ":"))
    # Local import: textindex imports LiveIndex.
    from .textindex import tokenize as regex_lower

    if tokenizer is regex_lower:
        return json.dumps({"type": "regex_lower"}, separators=(",", ":"))
    raise TypeError(
        "save() stores Tokenizer or the built-in regex tokenizer. "
        "A custom callable is kept by pickle, not by the on-disk format."
    )


def _tokenizer_from_tag(tag: Optional[str]):
    if not tag:
        return None
    spec = json.loads(tag)
    if not spec:
        return None
    kind = spec.get("type")
    if kind == "regex_lower":
        from .textindex import tokenize as regex_lower
        return regex_lower
    if kind == "tokenizer":
        stop = spec.get("stop_preset")
        if not stop:
            custom = spec.get("stop_custom") or []
            stop = custom or None
        return _native.Tokenizer(
            lowercase=bool(spec.get("lowercase", True)),
            stemmer=spec.get("stemmer"),
            stopwords=stop,
            cjk=bool(spec.get("cjk", False)),
        )
    raise ValueError(f"unknown tokenizer tag {spec!r}")


def _restore_live(state: dict) -> "LiveIndex":
    idx = LiveIndex(
        variant=state["variant"],
        k1=state["k1"],
        b=state["b"],
        epsilon=state["epsilon"],
        delta=state["delta"],
        compat=state["compat"],
        tokenizer=state["tokenizer"],
        fixes=state["fixes"],
    )
    for doc in state["docs"]:
        idx._native.add(doc["id"], doc["tokens"], doc["text"], json.dumps(doc["metadata"], separators=(",", ":")))
    return idx


class _Guard:
    def __init__(self, owner: "LiveIndex"):
        self.owner = owner

    def __enter__(self):
        if getattr(self.owner._tls, "depth", 0):
            raise RuntimeError(
                "LiveIndex methods cannot be re-entered (do not modify or query the index from a metadata filter)"
            )
        self.owner._lock.acquire()
        self.owner._tls.depth = 1
        return self

    def __exit__(self, *exc):
        self.owner._tls.depth = 0
        self.owner._lock.release()


class LiveIndex:
    """Mutable BM25 index.

    Parameters match the static classes: ``variant`` is ``"okapi"``, ``"l"`` or
    ``"plus"``; ``compat`` is ``"0.2.2"`` (default) or ``"47aa3dd"`` for L and
    Plus. ``delta`` defaults to 0.5 for L and 1 for Plus. ``tokenizer`` is
    applied to string documents and string queries. A list of strings is
    already tokenized. ``fixes`` has the same meaning as on :class:`BM25Okapi`.
    """

    def __init__(self, variant: str = "okapi", *, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25,
                 delta: Optional[float] = None, compat: str = "0.2.2",
                 tokenizer: Optional[Callable[[str], Sequence[str]]] = None, fixes: bool = False):
        self.variant = variant
        self.compat = compat if variant != "okapi" else "0.2.2"
        if delta is None:
            delta = 1.0 if variant == "plus" else 0.5
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.fixes = bool(fixes)
        self.tokenizer = tokenizer
        native = _native_variant(variant, self.compat)
        self._native = _native.LiveIndex(native, self.k1, self.b, self.epsilon, self.delta, self.fixes)
        self._lock = threading.Lock()
        self._tls = threading.local()

    def _guard(self) -> _Guard:
        return _Guard(self)

    def _bind(self, native) -> "LiveIndex":
        self._native = native
        self.k1 = float(native.k1)
        self.b = float(native.b)
        self.epsilon = float(native.epsilon)
        self.delta = float(native.delta)
        self.fixes = bool(native.fixes)
        self.variant = _public_variant(native.variant)
        self.compat = native.compat if self.variant != "okapi" else "0.2.2"
        return self

    def _prepare_doc(self, text) -> tuple[list[str], str]:
        if isinstance(text, str):
            tokens = list(self.tokenizer(text)) if self.tokenizer else text.split()
            return tokens, text
        tokens = [str(t) for t in text]
        return tokens, " ".join(tokens)

    def _prepare_query(self, query) -> list[str]:
        if isinstance(query, str):
            if self.tokenizer:
                return list(self.tokenizer(query))
            return query.split()
        return [str(t) for t in query]

    @property
    def corpus_size(self) -> int:
        return int(self._native.corpus_size)

    def avgdl(self) -> float:
        with self._guard():
            return float(self._native.avgdl())

    def average_idf(self) -> float:
        with self._guard():
            return float(self._native.average_idf())

    def ids(self) -> list[str]:
        """Document ids in score-vector order."""
        with self._guard():
            return list(self._native.ids())

    def texts(self) -> list[str]:
        with self._guard():
            return list(self._native.texts())

    def metadatas(self) -> list[dict]:
        with self._guard():
            return [json.loads(m) if m else {} for m in self._native.metas()]

    def corpus(self) -> list[list[str]]:
        """Token lists in score-vector order. A static index on this corpus matches :meth:`get_scores`."""
        with self._guard():
            return [list(doc) for doc in self._native.corpus()]

    def idf(self) -> dict:
        """Term -> idf, in first-seen order, for the documents currently stored."""
        with self._guard():
            return dict(self._native.idf_items())

    def doc_freqs(self) -> list[dict]:
        with self._guard():
            return [dict(items) for items in self._native.doc_freq_items()]

    def doc_len(self) -> list[int]:
        with self._guard():
            return [int(x) for x in self._native.doc_lens()]

    def add(self, id: str, text, metadata: Optional[dict] = None) -> None:
        """Insert a document. ``id`` must be new. ``text`` is a string or a list of tokens."""
        tokens, raw = self._prepare_doc(text)
        meta = json.dumps({} if metadata is None else metadata, separators=(",", ":"), ensure_ascii=False)
        with self._guard():
            self._native.add(str(id), tokens, raw, meta)

    def delete(self, id: str) -> bool:
        """Remove a document. Returns False if the id is unknown. Other ids stay put."""
        with self._guard():
            return bool(self._native.delete(str(id)))

    def update(self, id: str, text, metadata: Optional[dict] = None) -> None:
        """Replace a document's text, keeping its position. ``metadata=None`` keeps the old metadata."""
        tokens, raw = self._prepare_doc(text)
        meta = None if metadata is None else json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)
        with self._guard():
            self._native.update(str(id), tokens, raw, meta)

    def get_scores(self, query):
        """float64 scores in insertion order. Bit-identical to the matching static class on :meth:`corpus`."""
        with self._guard():
            return self._native.get_scores(self._prepare_query(query))

    def get_scores_many(self, queries: Sequence):
        """``(n_queries, corpus_size)`` float64. GIL released; one thread per query."""
        with self._guard():
            tokenized = [self._prepare_query(q) for q in queries]
            return self._native.get_scores_many(tokenized)

    def search(self, query, k: int = 10, where=None) -> list[dict]:
        """Top-k hits as dicts: ``rank``, ``id``, ``score``, ``text``, ``metadata``.

        ``where`` filters metadata before ranking (see module docs). Scores are
        the raw BM25 values, not rounded.
        """
        with self._guard():
            tokens = self._prepare_query(query)
            n = int(self._native.corpus_size)
            if n == 0 or k <= 0:
                return []
            mask = None
            if where is not None:
                metas = [json.loads(m) if m else {} for m in self._native.metas()]
                mask = [_match(m, where) for m in metas]
            rows = self._native.search(tokens, int(k), mask)
        out = []
        for rank, (doc_id, score, text, meta) in enumerate(rows, start=1):
            out.append({
                "rank": rank,
                "id": doc_id,
                "score": float(score),
                "text": text,
                "metadata": json.loads(meta) if meta else {},
            })
        return out

    def save(self, path: str) -> None:
        """Write a versioned live index. Load rebuilds postings by replaying adds; scores match."""
        tag = _tokenizer_tag(self.tokenizer)
        with self._guard():
            self._native.save(os.fspath(path), tag)

    @classmethod
    def load(cls, path: str) -> "LiveIndex":
        native, tag = _native.LiveIndex.load(os.fspath(path))
        self = cls.__new__(cls)
        self.tokenizer = _tokenizer_from_tag(tag)
        self._lock = threading.Lock()
        self._tls = threading.local()
        return self._bind(native)

    def __reduce__(self):
        with self._guard():
            ids = list(self._native.ids())
            texts = list(self._native.texts())
            metas = [json.loads(m) if m else {} for m in self._native.metas()]
            corpus = [list(doc) for doc in self._native.corpus()]
            state = {
                "variant": self.variant,
                "compat": self.compat,
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "fixes": self.fixes,
                "tokenizer": self.tokenizer,
                "docs": [
                    {"id": i, "text": t, "metadata": m, "tokens": tok}
                    for i, t, m, tok in zip(ids, texts, metas, corpus)
                ],
            }
        return (_restore_live, (state,))

    def __repr__(self) -> str:
        return f"LiveIndex(variant={self.variant!r}, corpus_size={self.corpus_size}, fixes={self.fixes})"
