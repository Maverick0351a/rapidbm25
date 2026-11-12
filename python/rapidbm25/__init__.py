"""rapidbm25: Rust-native BM25 with a rank_bm25-compatible API.

    from rapidbm25 import BM25Okapi          # instead of: from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([doc.split() for doc in docs])
    bm25.get_scores("windy london".split())  # same float64 scores, bit for bit
    bm25.top_n("windy london".split(), n=10) # fast partial-select top-n (ids, scores)
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from . import _native

__all__ = ["BM25", "BM25Okapi", "BM25L", "BM25Plus", "top_k_of", "COMPAT_DEFAULT", "__version__"]
__version__ = "0.1.0"

top_k_of = _native.top_k_of

# Which rank_bm25 behavior to reproduce for BM25L / BM25Plus (BM25Okapi is the
# same in both). "0.2.2" is the PyPI release that `pip install rank_bm25`
# gives you; "47aa3dd" is the later git main, which changed both formulas.
COMPAT_DEFAULT = "0.2.2"
_COMPAT = {"0.2.2": "022", "47aa3dd": ""}


def _variant(base: str, compat: str) -> str:
    if compat not in _COMPAT:
        raise ValueError(f"compat must be one of {sorted(_COMPAT)}, got {compat!r}")
    return base + _COMPAT[compat]


class BM25:
    """Shared base. Subclasses set ``_variant`` and their own defaults."""

    _variant = "okapi"

    def __init__(self, corpus: Iterable[Sequence[str]], tokenizer: Optional[Callable[[Any], Sequence[str]]] = None,
                 *, _k1: float = 1.5, _b: float = 0.75, _epsilon: float = 0.25, _delta: float = 0.5,
                 _variant: Optional[str] = None):
        self.tokenizer = tokenizer
        if tokenizer:
            corpus = [tokenizer(doc) for doc in corpus]
        self._index = _native.Index(corpus, _variant or self._variant, float(_k1), float(_b),
                                    float(_epsilon), float(_delta))
        self.corpus_size = self._index.corpus_size
        self.avgdl = self._index.avgdl
        self.doc_len = self._index.doc_len_list()
        self._idf = None
        self._doc_freqs = None

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

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        """float64 array of length corpus_size, equal to rank_bm25's get_scores."""
        return self._index.get_scores(query)

    def get_batch_scores(self, query: Sequence[str], doc_ids: Sequence[int]) -> list:
        assert all(di < self.corpus_size for di in doc_ids)
        if len(doc_ids) == 0:
            return []
        return self.get_scores(query)[np.asarray(doc_ids, dtype=np.int64)].tolist()

    def top_n(self, query: Sequence[str], n: int = 5):
        """(doc_ids, scores) of the n best docs. Score descending; ties go to the lower doc id."""
        return self._index.top_k(query, int(n))

    def get_top_n(self, query: Sequence[str], documents: Sequence[Any], n: int = 5) -> list:
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        ids, _ = self._index.top_k(query, int(n))
        return [documents[i] for i in ids.tolist()]


class BM25Okapi(BM25):
    _variant = "okapi"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer, _k1=k1, _b=b, _epsilon=epsilon)
        self.average_idf = self._index.average_idf


class BM25L(BM25):
    _variant = "l"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5, *, compat=COMPAT_DEFAULT):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.compat = compat
        super().__init__(corpus, tokenizer, _k1=k1, _b=b, _delta=delta, _variant=_variant("l", compat))


class BM25Plus(BM25):
    _variant = "plus"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1, *, compat=COMPAT_DEFAULT):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.compat = compat
        super().__init__(corpus, tokenizer, _k1=k1, _b=b, _delta=delta, _variant=_variant("plus", compat))
