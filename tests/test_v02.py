"""0.2 features: persistence, live index, tokenizers, batch search, fixes, fusion, integrations, MCP cache."""
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path

import numpy as np
import pytest

from rapidbm25 import BM25L, BM25Okapi, BM25Plus, LiveIndex, Tokenizer, fuse

CORPUS = ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]
TOKS = [d.split(" ") for d in CORPUS]
ALGS = [BM25Okapi, BM25L, BM25Plus]


def bits_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64), equal_nan=True))


def _static_like(live: LiveIndex):
    kw = {"k1": live.k1, "b": live.b, "fixes": live.fixes}
    if live.variant == "okapi":
        return BM25Okapi(live.corpus(), epsilon=live.epsilon, **kw)
    if live.variant == "l":
        return BM25L(live.corpus(), delta=live.delta, compat=live.compat, **kw)
    return BM25Plus(live.corpus(), delta=live.delta, compat=live.compat, **kw)


# ----- persistence ----------------------------------------------------------

@pytest.mark.parametrize("cls", ALGS)
def test_pickle_and_save_roundtrip_bits(cls, tmp_path: Path):
    kw = {"compat": "47aa3dd"} if cls is not BM25Okapi else {}
    a = cls(TOKS, **kw)
    b = pickle.loads(pickle.dumps(a))
    q = ["windy", "London", "windy"]
    assert bits_equal(a.get_scores(q), b.get_scores(q))
    assert a.corpus_size == b.corpus_size and a.doc_len == b.doc_len
    assert bits_equal(list(a.idf.values()), list(b.idf.values()))
    path = tmp_path / "idx.rbm25"
    a.save(path)
    c = cls.load(path, mmap=True)
    d = cls.load(path, mmap=False)
    assert bits_equal(a.get_scores(q), c.get_scores(q))
    assert bits_equal(a.get_scores(q), d.get_scores(q))
    assert c.mapped is True
    assert d.mapped is False
    assert type(c) is cls
    assert b.k1 == a.k1 and b.b == a.b


def test_load_rejects_wrong_class(tmp_path: Path):
    path = tmp_path / "okapi.rbm25"
    BM25Okapi(TOKS).save(path)
    with pytest.raises(TypeError):
        BM25L.load(path)


def test_pickle_with_tokenizer():
    tok = Tokenizer(lowercase=True)
    a = BM25Okapi(["Hello World", "hello there"], tokenizer=tok)
    b = pickle.loads(pickle.dumps(a))
    assert b.get_scores(["hello"])[0] == a.get_scores(["hello"])[0]
    assert list(b.tokenizer("Hello Worlds")) == list(tok("Hello Worlds"))


# ----- fixes ----------------------------------------------------------------

def test_fixes_default_matches_rank_bm25_repeated_query():
    rank_bm25 = pytest.importorskip("rank_bm25")
    # Odd corpus size so no term sits at exactly 50% df (#39 is tested separately).
    docs = [["a", "b", "b"], ["c", "a"], ["d"], ["e", "b"], ["f"]]
    q = ["b", "b", "a"]
    ref = rank_bm25.BM25Okapi(docs)
    ours = BM25Okapi(docs)
    assert bits_equal(ours.get_scores(q), ref.get_scores(q))
    # #19: fixes dedupes query tokens, first occurrence wins.
    fixed = BM25Okapi(docs, fixes=True)
    assert bits_equal(fixed.get_scores(q), ours.get_scores(["b", "a"]))
    assert not bits_equal(ours.get_scores(q), ours.get_scores(["b", "a"]))


def test_fixes_empty_corpus():
    with pytest.raises(ZeroDivisionError):
        BM25Okapi([])
    a = BM25Okapi([], fixes=True)
    assert a.corpus_size == 0 and a.get_scores(["x"]).shape == (0,)
    assert a.get_top_n(["x"], [], n=3) == []
    assert a.idf == {} and a.average_idf == 0.0
    b = pickle.loads(pickle.dumps(a))
    assert b.corpus_size == 0


def test_live_defaults_to_fixes_and_two_doc_idf_stays_zero():
    assert BM25Okapi([["a"], ["b"]]).fixes is False
    live = LiveIndex()
    assert live.fixes is True
    live.add("a", ["windy", "london"])
    live.add("b", ["sunny", "paris"])
    # N=2: each of these terms is in exactly half the documents, and the
    # epsilon floor is not positive, so fixes leave the original idf of 0.
    plain = BM25Okapi(live.corpus())
    fixed = BM25Okapi(live.corpus(), fixes=True)
    assert plain.idf["windy"] == 0.0
    assert fixed.idf["windy"] == 0.0
    assert plain.epsilon * plain.average_idf <= 0.0
    assert bits_equal(live.get_scores(["windy", "london"]), fixed.get_scores(["windy", "london"]))
    exact = LiveIndex(fixes=False)
    exact.add("a", ["windy", "windy"])
    exact.add("b", ["sunny"])
    exact.add("c", ["other"])
    assert bits_equal(exact.get_scores(["windy", "windy"]), BM25Okapi(exact.corpus()).get_scores(["windy", "windy"]))


def test_fixes_half_document_frequency():
    # rank_bm25 #39: a term in exactly N/2 documents has idf 0.
    corpus = [
        "This text contains keyword1 and Keyword2",
        "That is a text that contains keyword1 and term1",
        "Page contains no keywords but contains term1 and term2",
        "This text contains no keywords",
    ]
    toks = [d.split() for d in corpus]
    plain = BM25Okapi(toks)
    fixed = BM25Okapi(toks, fixes=True)
    q = "This is a question about keyword1 & term1".split()
    assert plain.get_scores(q)[0] == 0.0
    assert fixed.get_scores(q)[0] > 0.0
    # Terms that are not at exactly 50% df keep their idf bits.
    n = len(toks)
    for term, idf in plain.idf.items():
        df = sum(term in doc for doc in toks)
        if df != n / 2:
            assert fixed.idf[term] == idf
        elif plain.epsilon * plain.average_idf > 0:
            assert fixed.idf[term] == plain.epsilon * plain.average_idf
    assert fixed.average_idf == plain.average_idf


# ----- live index -----------------------------------------------------------

def test_live_matches_rebuild_after_add_delete_update():
    live = LiveIndex(k1=1.2, b=0.6, epsilon=0.3)
    live.add("a", ["red", "fox", "fox"])
    live.add("b", ["blue", "sky"])
    live.add("c", ["red", "sky", "red"])
    live.delete("b")
    live.update("a", ["red", "moon"])
    live.add("d", "red fox", metadata={"lang": "en", "year": 2020})
    ref = _static_like(live)
    for q in (["red"], ["fox", "fox"], ["sky", "missing"], []):
        assert bits_equal(live.get_scores(q), ref.get_scores(q))
    assert live.idf() == ref.idf
    assert live.doc_freqs() == ref.doc_freqs
    assert live.doc_len() == ref.doc_len
    assert live.avgdl() == ref.avgdl
    assert [h["id"] for h in live.search(["red", "fox"], k=3)][0] == "d"
    hits = live.search(["red"], k=10, where={"lang": "en", "year": {"$gte": 2020}})
    assert [h["id"] for h in hits] == ["d"]
    assert live.search(["red"], k=5, where=lambda m: m.get("year") == 2020)[0]["id"] == "d"
    assert live.delete("missing") is False
    with pytest.raises(ValueError):
        live.add("d", "again")


@pytest.mark.parametrize("variant,compat", [("okapi", "0.2.2"), ("l", "0.2.2"), ("l", "47aa3dd"), ("plus", "47aa3dd")])
def test_live_variants_match_static(variant, compat):
    live = LiveIndex(variant=variant, compat=compat, fixes=True)
    live.add("1", ["aa", "bb", "aa"])
    live.add("2", ["bb"])
    live.add("3", ["cc", "aa"])
    live.delete("2")
    live.update("1", ["bb", "cc", "cc"])
    ref = _static_like(live)
    assert bits_equal(live.get_scores(["aa", "bb", "aa"]), ref.get_scores(["aa", "bb", "aa"]))
    assert bits_equal(live.get_scores(["aa", "aa"]), ref.get_scores(["aa"]))  # fixes dedupes both


def test_live_save_load_and_thread_safety(tmp_path: Path):
    live = LiveIndex(tokenizer=Tokenizer(lowercase=True))
    live.add("a", "Hello World")
    live.add("b", "hello there")
    path = tmp_path / "live.rbm25"
    live.save(path)
    loaded = LiveIndex.load(path)
    assert bits_equal(loaded.get_scores("hello"), live.get_scores("hello"))
    loaded.add("c", "Hello again")
    assert "c" in loaded.ids()

    # Concurrent adds and searches. The lock must not drop or duplicate ids,
    # and the final scores must match a static rebuild.
    box = LiveIndex()
    errors = []

    def writer(prefix, n):
        try:
            for i in range(n):
                box.add(f"{prefix}{i}", f"token{i % 7} shared term{prefix}")
                if i % 5 == 4:
                    box.delete(f"{prefix}{i}")
        except Exception as exc:  # noqa: BLE001 - the test reports it
            errors.append(exc)

    def reader():
        try:
            for _ in range(30):
                box.search("shared token0", k=3)
                box.get_scores(["shared"])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(p, 20)) for p in ("x", "y", "z")]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    ref = _static_like(box)
    assert bits_equal(box.get_scores(["shared", "token0"]), ref.get_scores(["shared", "token0"]))
    assert len(box.ids()) == len(set(box.ids()))


def test_live_filter_does_not_reenter():
    live = LiveIndex()
    live.add("a", ["a"], metadata={"n": 1})

    def bad(_meta):
        live.add("b", ["b"])
        return True

    with pytest.raises(RuntimeError):
        live.search(["a"], where=bad)


# ----- tokenizers and batch -------------------------------------------------

def test_tokenizer_does_not_change_default_classes():
    # No tokenizer: the compat class still scores the tokens it is given.
    a = BM25Okapi([["Hello", "World"]])
    assert a.get_scores(["Hello"])[0] != 0.0
    tok = Tokenizer(lowercase=True, stopwords="english", stemmer="english", cjk=True)
    assert tok("The running dogs") == ["run", "dog"]
    # Overlapping CJK bigrams, plus a Latin token in the same string.
    assert tok("汉语 english") == ["汉语", "english"]
    assert tok("汉") == ["汉"]
    many = tok.tokenize_many(["The cats", "A dog"])
    assert list(many) == [["cat"], ["dog"]]
    restored = pickle.loads(pickle.dumps(tok))
    assert restored("The running dogs") == tok("The running dogs")
    # Stemmer is opt-in. Building a compat class without one is unchanged.
    plain = BM25Okapi([["running"]])
    stemmed = BM25Okapi(["running"], tokenizer=Tokenizer(lowercase=True, stemmer="english"))
    assert plain.get_scores(["running"])[0] != 0.0
    assert stemmed.get_scores(["running"])[0] == 0.0
    assert stemmed.get_scores(["run"])[0] != 0.0


def test_batch_matches_single_and_runs():
    docs = [[f"t{i % 17}", "common", f"u{i}"] for i in range(40)]
    a = BM25Okapi(docs)
    queries = [["common", "t1"], ["missing"], ["u3", "common", "u3"], []]
    many = a.get_scores_many(queries)
    assert many.shape == (4, 40)
    for i, q in enumerate(queries):
        assert bits_equal(many[i], a.get_scores(q))
    tops = a.top_n_many(queries, n=3)
    for q, (ids, scores) in zip(queries, tops):
        one_ids, one_scores = a.top_n(q, n=3)
        assert ids.tolist() == one_ids.tolist()
        assert bits_equal(scores, one_scores)
    live = LiveIndex()
    for i, doc in enumerate(docs):
        live.add(str(i), doc)
    live_many = live.get_scores_many(queries)
    for i, q in enumerate(queries):
        assert bits_equal(live_many[i], live.get_scores(q))


# ----- fusion ---------------------------------------------------------------

def test_fuse_rrf_and_weighted():
    bm25 = [("a", 3.0), ("b", 1.0), ("c", 0.5)]
    vector = [("b", 0.9), ("d", 0.2)]
    assert dict(fuse([bm25, vector])) == dict(fuse([bm25, vector], method="rrf", k=60))
    rrf = fuse([bm25, vector], method="rrf", k=60)
    scores = dict(rrf)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)
    assert rrf[0][0] == "b"
    # ids-only lists use rank, not a score
    assert dict(fuse([["a", "b"], ["b"]]))["b"] == pytest.approx(1 / 62 + 1 / 61)
    weighted = dict(fuse([bm25, [("b", 1.0), ("a", 0.0)]], method="weighted", weights=[1.0, 1.0]))
    # bm25 min-max: a=1, b=(1-0.5)/(3-0.5)=0.2, c=0. Second list: b=1, a=0.
    assert weighted["a"] == pytest.approx(1.0 + 0.0)
    assert weighted["b"] == pytest.approx((1.0 - 0.5) / (3.0 - 0.5) + 1.0)
    with pytest.raises(ValueError):
        fuse([bm25], method="nope")
    with pytest.raises(TypeError):
        fuse([["a", "b"]], method="weighted")


# ----- integrations and MCP cache ------------------------------------------

def test_langchain_retriever(tmp_path: Path):
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from rapidbm25.integrations.langchain import BM25Retriever

    retriever = BM25Retriever.from_texts(["windy London", "sunny Paris"], metadatas=[{"city": "london"}, {"city": "paris"}], k=1)
    docs = retriever.invoke("london wind")
    assert docs[0].page_content == "windy London"
    retriever.save(tmp_path / "lc")
    loaded = BM25Retriever.load(tmp_path / "lc")
    again = loaded.invoke("london wind")
    assert again[0].page_content == "windy London"
    via_docs = BM25Retriever.from_documents([Document(page_content="alpha beta", metadata={"k": 1}, id="doc-a")], k=1)
    assert via_docs.invoke("alpha")[0].id == "doc-a"


def test_llama_index_retriever(tmp_path: Path):
    pytest.importorskip("llama_index.core")
    from rapidbm25.integrations.llama_index import BM25Retriever

    retriever = BM25Retriever.from_texts(["parse_header timeout", "unrelated note", "another page"], k=1)
    nodes = retriever.retrieve("parse_header")
    assert nodes[0].node.text == "parse_header timeout"
    assert nodes[0].score > 0
    retriever.save(tmp_path / "li")
    loaded = BM25Retriever.load(tmp_path / "li")
    assert loaded.retrieve("parse_header")[0].node.text == "parse_header timeout"


def test_mcp_folder_cache_reindexes_changed_files(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAPIDBM25_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("RAPIDBM25_ROOTS", str(tmp_path))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.py").write_text("def parse_header(x):\n    return x\n")
    (docs / "b.md").write_text("windy weather in London today\n")
    from rapidbm25.mcp_server import call_tool

    first = call_tool("bm25_index_path", {"index": "repo", "path": str(docs)})
    assert first["chunks"] == 2 and first["files_reindexed"] == 2 and first["from_cache"] is False
    second = call_tool("bm25_index_path", {"index": "repo", "path": str(docs)})
    assert second["chunks"] == 2 and second["files_reindexed"] == 0 and second["from_cache"] is True
    (docs / "b.md").write_text("a unique token zzzyyy lives here\n")
    third = call_tool("bm25_index_path", {"index": "repo", "path": str(docs)})
    assert third["files_reindexed"] == 1
    hit = call_tool("bm25_search", {"index": "repo", "query": "zzzyyy", "k": 1})
    assert hit["results"][0]["id"].startswith("b.md")
    (docs / "a.py").unlink()
    fourth = call_tool("bm25_index_path", {"index": "repo", "path": str(docs)})
    assert fourth["files_reindexed"] == 1
    ids = [h["id"] for h in call_tool("bm25_search", {"index": "repo", "query": "parse_header", "k": 5})["results"]]
    assert ids == []


def test_path_index_matches_textindex(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAPIDBM25_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("RAPIDBM25_ROOTS", str(tmp_path))
    (tmp_path / "a.txt").write_text("red fox and blue sky\n")
    (tmp_path / "b.txt").write_text("blue bird\n")
    from rapidbm25.textindex import TextIndex, chunk_path, index_folder

    ids, texts = chunk_path(str(tmp_path), "*.txt", 40, 10)
    fresh = TextIndex(ids, texts)
    cached = index_folder(str(tmp_path), "*.txt", 40, 10, use_cache=False)
    assert cached.search("blue", k=2) == fresh.search("blue", k=2)
