#!/usr/bin/env python3
"""Fuzz equivalence: rapidbm25 vs the reference implementation, bit for bit.

The reference module is loaded from a local, gitignored checkout (--upstream);
it is never vendored into this repo.

Checks per case: get_scores (bitwise, NaN-aware), get_batch_scores, idf dict,
doc_len, avgdl, average_idf, doc_freqs, and top_n vs a lexsort reference
(score desc, doc id asc) computed from the reference scores.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time

import numpy as np

import rapidbm25


def load_ref(path):
    spec = importlib.util.spec_from_file_location("_ref_bm25", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def dict_bits_equal(a: dict, b: dict) -> bool:
    if list(a.keys()) != list(b.keys()):
        return False
    return bits_equal(list(a.values()), list(b.values()))


def ref_topn(scores, n):
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:n]


def gen_case(rng: random.Random):
    vocab_n = rng.choice([1, 2, 5, 20, 200, 2000])
    vocab = [f"t{i}" for i in range(vocab_n)] + ["Ünï", "a b", ""]
    n_docs = rng.choice([1, 2, 3, 10, 50, 300, 1000])
    zipf = rng.random() < 0.6
    corpus = []
    for _ in range(n_docs):
        L = rng.choice([0, 1, 2, rng.randint(0, 30), rng.randint(0, 300)]) if rng.random() < 0.95 else 0
        if zipf:
            doc = [vocab[min(int(rng.paretovariate(1.1)) - 1, len(vocab) - 1)] for _ in range(L)]
        else:
            doc = [rng.choice(vocab) for _ in range(L)]
        corpus.append(doc)
    if all(len(d) == 0 for d in corpus):
        corpus[0] = [vocab[0]]
    if rng.random() < 0.2:  # duplicated docs => score ties
        corpus += [list(corpus[0])] * rng.randint(1, 5)
    variant = rng.choice(["BM25Okapi", "BM25L", "BM25Plus"])
    kw = {}
    if rng.random() < 0.5:
        kw["k1"] = rng.choice([0.0, 0.5, 1.2, 1.5, 2.0, 3.7])
        kw["b"] = rng.choice([0.0, 0.3, 0.75, 1.0])
    if variant == "BM25Okapi" and rng.random() < 0.3:
        kw["epsilon"] = rng.choice([0.0, 0.1, 0.25, 1.0])
    if variant != "BM25Okapi" and rng.random() < 0.3:
        kw["delta"] = rng.choice([0.0, 0.5, 1, 2.5])
    queries = []
    for _ in range(rng.randint(1, 4)):
        ql = rng.randint(0, 12)
        q = [rng.choice(vocab) if rng.random() < 0.85 else f"oov{rng.randint(0, 9)}" for _ in range(ql)]
        if q and rng.random() < 0.3:
            q.append(q[0])  # repeated term
        queries.append(q)
    return variant, corpus, kw, queries


COMPAT = None  # set from --compat; passed to BM25L / BM25Plus only


def ours(variant, corpus, kw):
    extra = {"compat": COMPAT} if COMPAT and variant != "BM25Okapi" else {}
    return getattr(rapidbm25, variant)(corpus, **kw, **extra)


def run_case(ref, variant, corpus, kw, queries, rng):
    problems = []
    try:
        r = getattr(ref, variant)(corpus, **kw)
        ref_err = None
    except Exception as e:  # reference raised; we must raise the same type
        ref_err = type(e).__name__
    try:
        c = ours(variant, corpus, kw)
        our_err = None
    except Exception as e:
        our_err = type(e).__name__
    if ref_err or our_err:
        if ref_err != our_err:
            problems.append(f"constructor error mismatch ref={ref_err} ours={our_err}")
        return problems, 0.0
    if c.corpus_size != r.corpus_size or c.doc_len != r.doc_len:
        problems.append("corpus_size/doc_len")
    if not bits_equal([c.avgdl], [r.avgdl]):
        problems.append("avgdl")
    if variant == "BM25Okapi" and not bits_equal([c.average_idf], [r.average_idf]):
        problems.append("average_idf")
    if not dict_bits_equal(c.idf, r.idf):
        problems.append("idf")
    if c.doc_freqs != r.doc_freqs or [list(d) for d in c.doc_freqs] != [list(d) for d in r.doc_freqs]:
        problems.append("doc_freqs")
    max_abs = 0.0
    n = r.corpus_size
    for q in queries:
        with np.errstate(all="ignore"):
            rs = np.asarray(r.get_scores(q), dtype=np.float64)
        cs = c.get_scores(q)
        if not bits_equal(cs, rs):
            with np.errstate(all="ignore"):
                d = np.nanmax(np.abs(cs - rs)) if np.isfinite(rs).any() else float("nan")
            max_abs = max(max_abs, float(d) if d == d else 0.0)
            problems.append(f"get_scores q={q[:4]}")
        ids = [rng.randrange(n) for _ in range(rng.randint(0, 5))]
        with np.errstate(all="ignore"):
            rb = r.get_batch_scores(q, ids)
        if not bits_equal(c.get_batch_scores(q, ids), rb):
            problems.append("get_batch_scores")
        k = rng.choice([1, 3, 10, n + 5])
        if not np.isnan(rs).any():
            tid, tsc = c.top_n(q, k)
            if not np.array_equal(tid, ref_topn(rs, k)) or not bits_equal(tsc, rs[ref_topn(rs, k)]):
                problems.append("top_n")
    return problems, max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="path to reference rank_bm25.py (local, gitignored)")
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--out", default="")
    ap.add_argument("--compat", default="0.2.2", help="rapidbm25 compat mode matching the reference: 0.2.2 or 47aa3dd")
    a = ap.parse_args()
    global COMPAT
    COMPAT = a.compat
    ref = load_ref(a.upstream)
    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    fails, max_abs, counts, dense = [], 0.0, {}, 0
    for i in range(a.cases):
        variant, corpus, kw, queries = gen_case(rng)
        counts[variant] = counts.get(variant, 0) + 1
        try:
            dense += int(ours(variant, corpus, kw)._index.dense_fallback)
        except Exception:
            pass
        probs, m = run_case(ref, variant, corpus, kw, queries, rng)
        max_abs = max(max_abs, m)
        if probs and len(fails) < 20:
            fails.append({"case": i, "variant": variant, "kw": kw, "problems": probs[:5]})
        elif probs:
            fails.append({"case": i})
    res = {
        "n_cases": a.cases, "seed": a.seed, "compare": "bitwise (float64 bit patterns, NaN-aware)",
        "per_variant": counts, "dense_fallback_cases": dense,
        "n_failing_cases": len(fails), "max_abs": max_abs, "fail_samples": fails[:20],
        "ok": not fails, "elapsed_s": round(time.perf_counter() - t0, 3),
        "rapidbm25_version": rapidbm25.__version__, "compat": a.compat,
    }
    s = json.dumps(res, indent=2)
    print(s)
    if a.out:
        open(a.out, "w").write(s + "\n")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
