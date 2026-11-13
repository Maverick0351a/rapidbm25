"""Regression tests for the independent review of 5bac40a.

Each test fails on that revision and passes after the corresponding fix.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from rapidbm25 import BM25Okapi, LiveIndex, Tokenizer


def bits_equal(a, b) -> bool:
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def _roots(monkeypatch, path: Path):
    monkeypatch.setenv("RAPIDBM25_ROOTS", str(path))
    monkeypatch.setenv("RAPIDBM25_CACHE", str(path / "cache"))


def test_symlink_and_hardlink_do_not_escape_roots(tmp_path: Path, monkeypatch):
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret = secret_dir / "hidden.txt"
    secret.write_text("super-secret-token ALPHAOMEGA\n")
    root = tmp_path / "docs"
    root.mkdir()
    (root / "ok.txt").write_text("ordinary public note\n")
    _roots(monkeypatch, root)
    from rapidbm25.textindex import index_folder

    try:
        (root / "leak.txt").symlink_to(secret)
    except OSError:
        pytest.skip("symlinks are not available")
    ix = index_folder(str(root), "*")
    assert ix.search("ALPHAOMEGA", k=5) == []
    assert any(h["id"].startswith("ok.txt") for h in ix.search("ordinary", k=5))

    link = root / "via-hardlink.txt"
    try:
        os.link(secret, link)
    except OSError:
        return
    ix = index_folder(str(root), "*", use_cache=False)
    assert ix.search("ALPHAOMEGA", k=5) == []
    assert link.stat().st_nlink > 1


def test_native_load_defaults_to_a_copy(tmp_path: Path):
    import rapidbm25

    path = tmp_path / "idx.rbm25"
    BM25Okapi([["hello", "world"]]).save(path)
    loaded = rapidbm25._native.Index.load(str(path))
    assert loaded.mapped is False


def test_hostile_header_raises_value_error(tmp_path: Path):
    path = tmp_path / "bad.rbm25"
    # Magic, then a header length that overflows 16 + len on a 64-bit host.
    path.write_bytes(b"RBM25\x01\n\x00" + (2**64 - 1).to_bytes(8, "little"))
    with pytest.raises(ValueError, match="too large"):
        BM25Okapi.load(path)

    header = {
        "format": 1, "kind": "static", "endian": "little", "variant": "okapi",
        "fixes": False, "k1": "3ff8000000000000", "b": "3fe8000000000000",
        "epsilon": "3fd0000000000000", "delta": "3fe0000000000000",
        "avgdl": "0000000000000000", "average_idf": "0000000000000000",
        "sections": {
            # u32::MAX + 1 documents still fit (ids 0..=u32::MAX). One more does not.
            "doc_len": ["0000000000000020", 2**32 + 1],
            "idf": ["0000000000000020", 1],
            "offsets": ["0000000000000020", 2],
            "post_doc": ["0000000000000020", 0],
            "post_tf": ["0000000000000020", 0],
            "norm": ["0000000000000020", 2**32 + 1],
            "doc_terms_off": ["0000000000000020", 2**32 + 2],
            "doc_terms_tid": ["0000000000000020", 0],
            "doc_terms_tf": ["0000000000000020", 0],
            "terms": ["0000000000000020", 0],
        },
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    huge = tmp_path / "huge.rbm25"
    huge.write_bytes(b"RBM25\x01\n\x00" + len(raw).to_bytes(8, "little") + raw)
    with pytest.raises(ValueError, match="does not fit in u32"):
        BM25Okapi.load(huge)


def test_default_load_survives_truncation_and_mmap_is_opt_in(tmp_path: Path):
    path = tmp_path / "idx.rbm25"
    src = BM25Okapi([["hello", "world"], ["other", "text"]])
    src.save(path)
    copied = BM25Okapi.load(path)
    assert copied.mapped is False
    before = copied.get_scores(["hello"])
    os.truncate(path, 0)
    assert bits_equal(copied.get_scores(["hello"]), before)

    src.save(path)
    mapped = BM25Okapi.load(path, mmap=True)
    assert mapped.mapped is True
    assert bits_equal(mapped.get_scores(["hello"]), src.get_scores(["hello"]))


def _patch_header_field(path: Path, old: bytes, new: bytes):
    raw = bytearray(path.read_bytes())
    assert old in raw
    raw = raw.replace(old, new, 1)
    path.write_bytes(raw)


def test_misaligned_or_corrupt_index_raises_value_error(tmp_path: Path):
    path = tmp_path / "idx.rbm25"
    BM25Okapi([["hello", "world"], ["other"]]).save(path)
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[8:16], "little")
    header = json.loads(raw[16:16 + header_len])
    post = header["sections"]["post_doc"]
    # Shift the posting column by one byte. Fixed-width hex keeps the header size.
    shifted = format((int(post[0], 16) + 1) & ((1 << 64) - 1), "016x")
    assert len(shifted) == len(post[0])
    bad = tmp_path / "shifted.rbm25"
    bad.write_bytes(raw.replace(post[0].encode(), shifted.encode(), 1))
    with pytest.raises(ValueError):
        BM25Okapi.load(bad)
    with pytest.raises(ValueError):
        BM25Okapi.load(bad, mmap=True)

    # A posting doc id past n_docs used to panic inside get_scores.
    off = int(post[0], 16)
    mutated = bytearray(raw)
    mutated[off:off + 4] = (100_000).to_bytes(4, "little")
    wild = tmp_path / "wild.rbm25"
    wild.write_bytes(mutated)
    with pytest.raises(ValueError):
        BM25Okapi.load(wild)

    _patch_header_field(path, b'"endian":"little"', b'"endian":"big"   ')
    # same length so the section offsets still point at the columns
    assert len(b'"endian":"little"') == len(b'"endian":"big"   ')
    with pytest.raises(ValueError):
        BM25Okapi.load(path)


def test_live_index_rejects_huge_document_count(tmp_path: Path):
    path = tmp_path / "live.rbm25"
    ix = LiveIndex(fixes=False)
    ix.add("a", ["hello"])
    ix.save(path)
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[8:16], "little")
    header = json.loads(raw[16:16 + header_len])
    off = int(header["sections"]["docs"][0], 16)
    mutated = bytearray(raw)
    mutated[off:off + 8] = (2**64 - 1).to_bytes(8, "little")
    path.write_bytes(mutated)
    with pytest.raises(ValueError):
        LiveIndex.load(path)


def test_retriever_skips_zero_score_doc_with_no_shared_term():
    pytest.importorskip("langchain_core")
    from rapidbm25.integrations.langchain import BM25Retriever

    # First document does not contain the query. Both Okapi scores are 0.
    hit = BM25Retriever.from_texts(["sunny Paris", "windy London"], k=1).invoke("windy")
    assert hit[0].page_content == "windy London"
    assert hit[0].metadata["bm25_score"] == 0.0


def test_llama_index_retriever_skips_disjoint_doc():
    pytest.importorskip("llama_index.core")
    from rapidbm25.integrations.llama_index import BM25Retriever

    nodes = BM25Retriever.from_texts(["sunny Paris", "windy London"], k=1).retrieve("windy")
    assert nodes[0].node.text == "windy London"


def test_cache_detects_same_size_same_mtime_rewrite(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "a.py"
    target.write_text("def parse_header(x):\n    return x\n")
    monkeypatch.setenv("RAPIDBM25_ROOTS", str(docs))
    monkeypatch.setenv("RAPIDBM25_CACHE", str(tmp_path / "cache"))
    from rapidbm25.textindex import index_folder

    first = index_folder(str(docs))
    assert first.search("parse_header", k=1)
    st = target.stat()
    target.write_text("def parse_headXX(x):\n    return x\n")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert target.stat().st_mtime_ns == st.st_mtime_ns
    assert target.stat().st_size == st.st_size
    second = index_folder(str(docs))
    assert second.last_stats["files_reindexed"] == 1
    assert second.search("parse_header", k=3) == []
    assert second.search("parse_headXX", k=1)


def test_dotfiles_skipped_unless_glob_asks_and_cache_is_private(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "visible.txt").write_text("public token VISIBLE\n")
    (docs / ".env").write_text("API_KEY=SHOULDNOTINDEX\n")
    cache = tmp_path / "cache"
    _roots(monkeypatch, docs)
    monkeypatch.setenv("RAPIDBM25_CACHE", str(cache))
    from rapidbm25.textindex import index_folder

    ix = index_folder(str(docs), "*")
    assert ix.search("SHOULDNOTINDEX", k=3) == []
    assert ix.search("VISIBLE", k=1)
    asked = index_folder(str(docs), ".env", use_cache=False)
    assert asked.search("SHOULDNOTINDEX", k=1)

    if os.name == "nt":
        return
    key_dirs = [p for p in cache.iterdir() if p.is_dir()]
    assert key_dirs
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert stat.S_IMODE(key_dirs[0].stat().st_mode) == 0o700
    assert stat.S_IMODE((key_dirs[0] / "manifest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((key_dirs[0] / "index.rbm25").stat().st_mode) == 0o600


def test_filters_do_not_deadlock_and_reject_string_in_and_bool_int():
    first = LiveIndex(fixes=False)
    second = LiveIndex(fixes=False)
    first.add("a", ["hello"], metadata={"tag": "red", "flag": True})
    second.add("b", ["hello"], metadata={"tag": "blue"})
    import threading
    barrier = threading.Barrier(2)
    errors = []

    def run(index, other):
        try:
            def pred(_meta):
                barrier.wait(timeout=2)
                other.search(["hello"], k=1, where=lambda _m: True)
                return True
            index.search(["hello"], where=pred)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(first, second)),
        threading.Thread(target=run, args=(second, first)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []

    with pytest.raises(ValueError):
        first.search(["hello"], where={"tag": {"$in": "red"}})
    assert first.search(["hello"], where={"flag": 1}) == []
    assert first.search(["hello"], where={"flag": True})[0]["id"] == "a"
    assert first.search(["hello"], where={"tag": {"$in": ["red"]}})[0]["id"] == "a"


def test_get_batch_scores_matches_rank_bm25_on_bad_ids():
    rank_bm25 = pytest.importorskip("rank_bm25")
    docs = [["a", "b"], ["c", "d"], ["e"]]
    ref = rank_bm25.BM25Okapi(docs)
    ours = BM25Okapi(docs)
    for ids in ([1.9, True], [True], [1.0]):
        with pytest.raises(IndexError):
            ref.get_batch_scores(["a"], ids)
        with pytest.raises(IndexError):
            ours.get_batch_scores(["a"], ids)
    assert ours.get_batch_scores(["a"], [-1]) == ref.get_batch_scores(["a"], [-1])
    assert ours.get_batch_scores(["a"], []) == []
    assert ours.get_batch_scores(["a"], [1, 0]) == ref.get_batch_scores(["a"], [1, 0])


def test_search_terms_finds_zero_score_match_past_the_window():
    docs = ["common"] * 60 + ["common windy"] * 60
    live = LiveIndex()
    for i, text in enumerate(docs):
        live.add(str(i), text)
    assert float(live.idf()["windy"]) == 0.0
    hits = live.search_terms("windy", k=1)
    assert hits and hits[0]["text"] == "common windy"
    assert len(live.search_terms("windy", k=3)) == 3
    pytest.importorskip("langchain_core")
    from rapidbm25.integrations.langchain import BM25Retriever
    found = BM25Retriever.from_texts(docs, k=1).invoke("windy")
    assert found[0].page_content == "common windy"


def test_save_does_not_follow_temp_symlink(tmp_path: Path):
    victim = tmp_path / "victim"
    victim.write_text("VICTIM-DATA-KEEP")
    dest = tmp_path / "index.rbm25"
    try:
        dest.with_suffix(".rbm25.tmp").symlink_to(victim)
    except OSError:
        pytest.skip("symlinks are not available")
    # with_suffix replaces the extension: index.rbm25 -> index.rbm25.tmp is
    # Path.with_extension("rbm25.tmp") == index.rbm25.tmp. Match the writer.
    planted = dest.with_name("index.rbm25.tmp")
    if not planted.exists():
        planted.unlink(missing_ok=True)
        planted.symlink_to(victim)
    live = LiveIndex(fixes=False)
    live.add("a", ["hello"])
    live.save(dest)
    assert victim.read_text() == "VICTIM-DATA-KEEP"
    assert dest.is_file() and not dest.is_symlink()
    assert LiveIndex.load(dest).ids() == ["a"]


def test_halfwidth_katakana_is_bigrammed_and_package_is_typed():
    tok = Tokenizer(cjk=True, lowercase=False)
    assert tok("ｶﾀｶﾅ") == ["ｶﾀ", "ﾀｶ", "ｶﾅ"]
    import rapidbm25
    marker = Path(rapidbm25.__file__).parent / "py.typed"
    assert marker.is_file()
