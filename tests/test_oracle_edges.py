"""Deterministic bit-for-bit checks against installed rank_bm25 0.2.2.

The fuzz harness draws unicode, duplicate documents, and empty queries only
some of the time, and it used to refuse an empty corpus entirely. These cases
are the ones a drop-in replacement has to get right on every run.
"""
import numpy as np
import pytest

import rank_bm25
from rapidbm25 import BM25L, BM25Okapi, BM25Plus

PAIRS = [
    (BM25Okapi, rank_bm25.BM25Okapi),
    (BM25L, rank_bm25.BM25L),
    (BM25Plus, rank_bm25.BM25Plus),
]

CORPORA = [
    [["Ünïcöde", "東京"], ["東京", "東京"], ["café"]],
    [["red", "fox"], ["red", "fox"], ["blue"]],
    [[], ["a"], ["a", "a"]],
    [["", "a"], ["a"]],
]
QUERIES = [[], ["東京"], ["Ünïcöde", "Ünïcöde"], ["red"], ["missing"], [""], ["fox", "fox"]]


def bits_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64), equal_nan=True))


@pytest.mark.parametrize("ours_cls,ref_cls", PAIRS)
def test_edges_match_rank_bm25(ours_cls, ref_cls):
    for docs in CORPORA:
        ref = ref_cls(docs)
        ours = ours_cls(docs)
        assert ours.corpus_size == ref.corpus_size
        assert ours.doc_len == list(ref.doc_len)
        assert bits_equal([ours.avgdl], [ref.avgdl])
        assert list(ours.idf) == list(ref.idf)
        assert bits_equal(list(ours.idf.values()), list(ref.idf.values()))
        assert [dict(d) for d in ours.doc_freqs] == [dict(d) for d in ref.doc_freqs]
        if ours_cls is BM25Okapi:
            assert bits_equal([ours.average_idf], [ref.average_idf])
        for query in QUERIES:
            assert bits_equal(ours.get_scores(query), ref.get_scores(query))
        # Duplicate documents tie. Both copies score the same bits.
        if docs[0] == docs[1]:
            scores = ours.get_scores(["red", "fox"])
            assert bits_equal(scores[:2], ref.get_scores(["red", "fox"])[:2])
            assert scores[0] == scores[1]


@pytest.mark.parametrize("ours_cls,ref_cls", PAIRS)
def test_empty_corpus_raises_the_same_error(ours_cls, ref_cls):
    with pytest.raises(ZeroDivisionError):
        ref_cls([])
    with pytest.raises(ZeroDivisionError):
        ours_cls([])


def test_non_str_corpus_token_is_rejected():
    with pytest.raises(TypeError):
        BM25Okapi([["ok", 1]])
