"""Fuse BM25 hits with hits from any other ranker (typically a vector store).

Both methods are pure functions of the ranked lists you pass in. They do not
import a vector library.

* ``method="rrf"`` is reciprocal rank fusion: ``score += weight / (k + rank)``
  with ranks starting at 1. Scores on the input lists are ignored, so the two
  systems do not need a shared scale. ``k`` defaults to 60.
* ``method="weighted"`` min-max normalizes each list's scores to ``[0, 1]``
  and returns the weighted sum. A list whose scores are all equal contributes
  its weight when that score is positive, and nothing when it is zero.

Ties break by descending fused score, then by ``(type name, id)`` so the order
is stable for strings and ints.
"""
from __future__ import annotations

from typing import Sequence


def _as_id(item):
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return item[0]
    return item


def _tie(doc_id):
    if isinstance(doc_id, (str, int)):
        return (type(doc_id).__name__, doc_id)
    return (type(doc_id).__name__, repr(doc_id))


def fuse(rankings: Sequence[Sequence], method: str = "rrf", *, k: int = 60, weights: Sequence[float] | None = None):
    """Combine ranked lists into ``[(id, score), ...]``, best first.

    Each list is a sequence of ids, or of ``(id, score)`` pairs. ``rrf`` accepts
    either shape. ``weighted`` requires scores.
    """
    if method not in ("rrf", "weighted"):
        raise ValueError("method must be 'rrf' or 'weighted'")
    lists = [list(r) for r in rankings]
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights must have one value per ranking")
    scores: dict = {}
    if method == "rrf":
        kk = float(k)
        for weight, ranking in zip(weights, lists):
            for rank, item in enumerate(ranking, start=1):
                doc_id = _as_id(item)
                scores[doc_id] = scores.get(doc_id, 0.0) + float(weight) / (kk + rank)
    else:
        for weight, ranking in zip(weights, lists):
            pairs = []
            for item in ranking:
                if not (isinstance(item, (tuple, list)) and len(item) == 2):
                    raise TypeError("weighted fusion requires (id, score) pairs")
                pairs.append((item[0], float(item[1])))
            if not pairs:
                continue
            vals = [s for _, s in pairs]
            lo, hi = min(vals), max(vals)
            span = hi - lo
            w = float(weight)
            for doc_id, s in pairs:
                if span == 0.0:
                    norm = 1.0 if hi > 0.0 else 0.0
                else:
                    norm = (s - lo) / span
                scores[doc_id] = scores.get(doc_id, 0.0) + w * norm
    return sorted(scores.items(), key=lambda kv: (-kv[1], _tie(kv[0])))
