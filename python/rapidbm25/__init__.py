"""rapidbm25: Rust-native BM25 with a rank_bm25-compatible API.

    from rapidbm25 import BM25Okapi          # instead of: from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([doc.split() for doc in docs])
    bm25.get_scores("windy london".split())  # same float64 scores, bit for bit
    bm25.top_n("windy london".split(), n=10) # fast partial-select top-n (ids, scores)
    bm25.save("index.rbm25")                 # versioned file; load() memory-maps it
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from . import _native
from .hybrid import fuse
from .live import LiveIndex

__all__ = [
    "BM25",
    "BM25Okapi",
    "BM25L",
    "BM25Plus",
    "LiveIndex",
    "Tokenizer",
    "fuse",
    "top_k_of",
    "COMPAT_DEFAULT",
    "__version__",
]
__version__ = "0.2.0"

top_k_of = _native.top_k_of
Tokenizer = _native.Tokenizer

# Which rank_bm25 behavior to reproduce for BM25L / BM25Plus (BM25Okapi is the
# same in both). "0.2.2" is the PyPI release that `pip install rank_bm25`
# gives you; "47aa3dd" is the later git main, which changed both formulas.
COMPAT_DEFAULT = "0.2.2"
_COMPAT = {"0.2.2": "022", "47aa3dd": ""}
_BY_NAME = {}


def _variant(base: str, compat: str) -> str:
    if compat not in _COMPAT:
        raise ValueError(f"compat must be one of {sorted(_COMPAT)}, got {compat!r}")
    return base + _COMPAT[compat]


def _restore_bm25(cls_name: str, blob: bytes, tokenizer):
    index = _native.Index.from_bytes(blob)
    obj = _BY_NAME[cls_name]._from_index(index)
    obj.tokenizer = tokenizer
    return obj


class BM25:
    """Shared base. Subclasses set ``_variant`` and their own defaults.

    ``fixes=False`` (the default on these classes) reproduces rank_bm25,
    including its known bugs. ``fixes=True`` dedupes repeated query tokens
    (#19), accepts an empty corpus (#36), and gives an Okapi term whose idf is
    exactly 0 the same positive ``epsilon * average_idf`` floor already used
    for negative idf (#39). The IDF formula itself does not change. On a
    2-document corpus that floor is not positive, so a term in one of the two
    documents still has idf 0. :class:`LiveIndex` defaults to ``fixes=True``.
    """

    _variant = "okapi"

    def __init__(self, corpus: Iterable[Sequence[str]], tokenizer: Optional[Callable[[Any], Sequence[str]]] = None,
                 *, _k1: float = 1.5, _b: float = 0.75, _epsilon: float = 0.25, _delta: float = 0.5,
                 _variant: Optional[str] = None, _fixes: bool = False):
        self.tokenizer = tokenizer
        self.fixes = bool(_fixes)
        if tokenizer:
            corpus = [tokenizer(doc) for doc in corpus]
        self._index = _native.Index(corpus, _variant or self._variant, float(_k1), float(_b),
                                    float(_epsilon), float(_delta), fixes=self.fixes)
        self.corpus_size = self._index.corpus_size
        self.avgdl = self._index.avgdl
        self.doc_len = self._index.doc_len_list()
        self._idf = None
        self._doc_freqs = None

    @classmethod
    def _from_index(cls, index):
        self = cls.__new__(cls)
        self._index = index
        self.tokenizer = None
        self.fixes = bool(index.fixes)
        self._idf = None
        self._doc_freqs = None
        self.corpus_size = index.corpus_size
        self.avgdl = index.avgdl
        self.doc_len = index.doc_len_list()
        self.k1 = index.k1
        self.b = index.b
        if cls is BM25Okapi:
            self.epsilon = index.epsilon
            self.average_idf = index.average_idf
        else:
            self.delta = index.delta
            self.compat = index.compat
        return self

    # rank_bm25 exposes these as plain attributes; we build them on first use.
    @property
    def idf(self) -> dict:
        if self._idf is None:
            self._idf = self._index.idf_dict()
        return self._idf

    @property
    def doc_freqs(self) -> list:
        if self._doc_freqs is None:
            self._doc_freqs = self._index.doc_freqs_list()
        return self._doc_freqs

    @property
    def mapped(self) -> bool:
        """True when scores are read from a memory-mapped index file."""
        return bool(self._index.mapped)

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        """float64 array of length corpus_size, equal to rank_bm25's get_scores."""
        return self._index.get_scores(query)

    def get_scores_many(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        """``(n_queries, corpus_size)`` float64. The GIL is released and queries run on a thread pool.

        Row ``i`` is bit-identical to ``get_scores(queries[i])``.
        """
        return self._index.get_scores_many(list(queries))

    def get_batch_scores(self, query: Sequence[str], doc_ids: Sequence[int]) -> list:
        assert all(di < self.corpus_size for di in doc_ids)
        if len(doc_ids) == 0:
            return []
        return self.get_scores(query)[np.asarray(doc_ids, dtype=np.int64)].tolist()

    def top_n(self, query: Sequence[str], n: int = 5):
        """(doc_ids, scores) of the n best docs. Score descending; ties go to the lower doc id."""
        return self._index.top_k(query, int(n))

    def top_n_many(self, queries: Sequence[Sequence[str]], n: int = 5):
        """List of ``(doc_ids, scores)``, one per query. GIL released; threaded."""
        return list(self._index.top_n_many(list(queries), int(n)))

    def get_top_n(self, query: Sequence[str], documents: Sequence[Any], n: int = 5) -> list:
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        ids, _ = self._index.top_k(query, int(n))
        return [documents[i] for i in ids.tolist()]

    def save(self, path: str) -> None:
        """Write a versioned little-endian index. Scores reload bit-identical."""
        self._index.save(os.fspath(path))

    @classmethod
    def load(cls, path: str, *, mmap: bool = True):
        """Load an index written by :meth:`save`.

        ``mmap=True`` (default) scores from a read-only mapping on 64-bit
        little-endian hosts. Do not overwrite ``path`` while the returned
        object is alive. A tokenizer callable is not stored in the file; the
        index already holds the tokens it was built from. Pickle keeps a
        picklable tokenizer.
        """
        index = _native.Index.load(os.fspath(path), mmap=bool(mmap))
        name = index.class_name
        target = _BY_NAME[name]
        if cls is not BM25 and cls is not target:
            raise TypeError(f"{cls.__name__}.load cannot read a {name} index")
        return target._from_index(index)

    def __reduce__(self):
        return (_restore_bm25, (type(self).__name__, bytes(self._index.to_bytes()), self.tokenizer))


class BM25Okapi(BM25):
    _variant = "okapi"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, *, fixes=False):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer, _k1=k1, _b=b, _epsilon=epsilon, _fixes=fixes)
        self.average_idf = self._index.average_idf


class BM25L(BM25):
    _variant = "l"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5, *, compat=COMPAT_DEFAULT, fixes=False):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.compat = compat
        super().__init__(corpus, tokenizer, _k1=k1, _b=b, _delta=delta, _variant=_variant("l", compat), _fixes=fixes)


class BM25Plus(BM25):
    _variant = "plus"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1, *, compat=COMPAT_DEFAULT, fixes=False):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.compat = compat
        super().__init__(corpus, tokenizer, _k1=k1, _b=b, _delta=delta, _variant=_variant("plus", compat), _fixes=fixes)


_BY_NAME.update({"BM25Okapi": BM25Okapi, "BM25L": BM25L, "BM25Plus": BM25Plus})
