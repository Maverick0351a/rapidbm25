"""API-compat tests. The first two mirror the upstream loading test's behavior."""
import subprocess
import sys

import numpy as np
import pytest

from rapidbm25 import BM25L, BM25Okapi, BM25Plus, top_k_of

corpus = ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]
tokenized = [d.split(" ") for d in corpus]
ALGS = [BM25Okapi, BM25L, BM25Plus]


@pytest.mark.parametrize("cls", ALGS)
def test_corpus_loading(cls):
    a = cls(tokenized)
    assert a.corpus_size == 3 and a.avgdl == 5 and a.doc_len == [4, 6, 5]


def test_tokenizer():
    a = BM25Okapi(corpus, tokenizer=lambda d: d.split(" "))
    assert a.corpus_size == 3 and a.avgdl == 5 and a.doc_len == [4, 6, 5]


@pytest.mark.parametrize("cls", ALGS)
def test_scores_shape_and_top_n(cls):
    a = cls(tokenized)
    s = a.get_scores(["windy", "London"])
    assert s.dtype == np.float64 and s.shape == (3,)
    assert a.get_top_n(["windy", "London"], corpus, n=1) == [corpus[1]]
    ids, sc = a.top_n(["windy", "London"], n=2)
    assert ids[0] == 1 and sc[0] == s[1]


def test_batch_scores_and_attrs():
    a = BM25Okapi(tokenized)
    full = a.get_scores(["weather"])
    assert a.get_batch_scores(["weather"], [2, 0]) == [full[2], full[0]]
    assert a.get_batch_scores(["weather"], []) == []
    assert set(a.idf) >= {"Hello", "London"} and a.doc_freqs[0]["Hello"] == 1
    assert isinstance(a.average_idf, float)


def test_get_top_n_mismatch_asserts():
    with pytest.raises(AssertionError):
        BM25Okapi(tokenized).get_top_n(["x"], corpus[:2])


def test_length_checks_survive_python_optimize():
    code = (
        "from rapidbm25 import BM25Okapi\n"
        "a = BM25Okapi([['a'], ['b']])\n"
        "try:\n"
        "    a.get_top_n(['a'], ['only'])\n"
        "except AssertionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('get_top_n')\n"
        "try:\n"
        "    a.get_batch_scores(['a'], [9])\n"
        "except AssertionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('get_batch_scores')\n"
    )
    subprocess.run([sys.executable, "-O", "-c", code], check=True)


def test_empty_corpus_raises_like_reference():
    with pytest.raises(ZeroDivisionError):
        BM25Okapi([])


def test_top_k_of_ties_lower_id_first():
    assert top_k_of(np.array([1.0, 3.0, 3.0, 2.0]), 3).tolist() == [1, 2, 3]


def test_compat_modes():
    docs = [["a", "b"], ["b", "c", "c"], ["d"]]
    assert BM25L(docs).compat == "0.2.2"
    # 0.2.2 BM25L multiplies by tf, so a doc without the term scores exactly 0
    assert BM25L(docs).get_scores(["a"])[2] == 0.0
    assert BM25L(docs, compat="47aa3dd").get_scores(["a"])[2] > 0.0
    with pytest.raises(ValueError):
        BM25Plus(docs, compat="nope")
