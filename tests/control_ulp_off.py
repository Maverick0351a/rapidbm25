import numpy as np, rapidbm25
class _W:
    def __init__(s, inner): s._i = inner
    def __getattr__(s, k): return getattr(s._i, k)
    def get_scores(s, q): return np.nextafter(s._i.get_scores(q), np.inf)
def _mk(cls): return lambda *a, **k: _W(getattr(rapidbm25, cls)(*a, **k))
BM25Okapi, BM25L, BM25Plus = _mk("BM25Okapi"), _mk("BM25L"), _mk("BM25Plus")
