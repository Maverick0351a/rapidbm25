#!/usr/bin/env python3
"""Measure 0.2 save/load, incremental add/delete, and batch search.

Prints one JSON object and a markdown table. Numbers are from this process;
do not copy them from anywhere else.

    python bench/bench_v02.py
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from rapidbm25 import BM25Okapi, LiveIndex


def _docs(n, vocab, length, seed=7):
    rng = np.random.default_rng(seed)
    vocab_ids = rng.integers(0, vocab, size=(n, length))
    return [[f"t{int(t)}" for t in row] for row in vocab_ids]


def _queries(n, vocab, length, seed=11):
    rng = np.random.default_rng(seed)
    vocab_ids = rng.integers(0, vocab, size=(n, length))
    return [[f"t{int(t)}" for t in row] for row in vocab_ids]


def median(samples):
    return statistics.median(samples)


def bench_once(fn, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=8000)
    ap.add_argument("--doc-len", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=64)
    ap.add_argument("--query-len", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--delete", type=int, default=800)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    docs = _docs(args.docs, args.vocab, args.doc_len)
    queries = _queries(args.queries, args.vocab, args.query_len)
    scratch = Path("/tmp/rapidbm25-bench-v02.rbm25")

    build_t, _ = bench_once(lambda: BM25Okapi(docs), repeats=args.repeats)
    index = BM25Okapi(docs)

    def _save():
        index.save(scratch)

    save_t, _ = bench_once(_save, repeats=args.repeats)

    def _load_mmap():
        loaded = BM25Okapi.load(scratch, mmap=True)
        # touch a score so a mapped index is actually used
        loaded.get_scores(queries[0])
        assert loaded.mapped

    def _load_copy():
        loaded = BM25Okapi.load(scratch, mmap=False)
        loaded.get_scores(queries[0])
        assert not loaded.mapped

    # The saved file is mapped by _load_mmap; drop those objects before the next save.
    load_mmap_t, _ = bench_once(_load_mmap, repeats=args.repeats)
    load_copy_t, _ = bench_once(_load_copy, repeats=args.repeats)

    def _seq():
        for q in queries:
            index.get_scores(q)

    def _batch():
        index.get_scores_many(queries)

    seq_t, _ = bench_once(_seq, repeats=args.repeats)
    batch_t, _ = bench_once(_batch, repeats=args.repeats)
    # correctness spot-check, not timed
    many = index.get_scores_many(queries[:4])
    for i, q in enumerate(queries[:4]):
        if not np.array_equal(many[i].view(np.uint64), index.get_scores(q).view(np.uint64)):
            raise SystemExit("batch scores diverged")

    from rapidbm25 import BM25Plus

    def batch_case(label, cls, n, vocab, length, nq, ql):
        d = _docs(n, vocab, length, seed=21)
        q = _queries(nq, vocab, ql, seed=22)
        ix = cls(d)
        got = ix.get_scores_many(q[:2])
        for i, qq in enumerate(q[:2]):
            if not np.array_equal(got[i].view(np.uint64), ix.get_scores(qq).view(np.uint64)):
                raise SystemExit(f"batch scores diverged: {label}")
        s, _ = bench_once(lambda: [ix.get_scores(qq) for qq in q], repeats=args.repeats)
        b, _ = bench_once(lambda: ix.get_scores_many(q), repeats=args.repeats)
        return {"label": label, "docs": n, "vocab": vocab, "doc_len": length, "queries": nq,
                "query_len": ql, "sequential_s": s, "batch_s": b, "speedup": s / b}

    def _add_all():
        live = LiveIndex()
        for i, doc in enumerate(docs):
            live.add(str(i), doc)
        return live

    # add is one shot per repeat; include construction
    add_samples = []
    lives = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        lives.append(_add_all())
        add_samples.append(time.perf_counter() - t0)
    add_t = median(add_samples)
    live = lives[-1]
    lives.clear()

    ids = [str(i) for i in range(args.delete)]

    del_samples = []
    for _ in range(args.repeats):
        live = _add_all()
        t0 = time.perf_counter()
        for doc_id in ids:
            live.delete(doc_id)
        del_samples.append(time.perf_counter() - t0)
    del_t = median(del_samples)

    remaining = docs[args.delete:]

    def _rebuild():
        BM25Okapi(remaining)

    rebuild_t, _ = bench_once(_rebuild, repeats=args.repeats)

    extra = _docs(200, args.vocab, args.doc_len, seed=99)

    def _add_200():
        live_ix = LiveIndex()
        for i, doc in enumerate(docs):
            live_ix.add(str(i), doc)
        t0 = time.perf_counter()
        for i, doc in enumerate(extra):
            live_ix.add(f"e{i}", doc)
        return time.perf_counter() - t0

    _add_200()  # warmup
    add_200_t = median([_add_200() for _ in range(args.repeats)])

    def _rebuild_plus():
        BM25Okapi(docs + extra)

    rebuild_plus_t, _ = bench_once(_rebuild_plus, repeats=args.repeats)
    file_bytes = scratch.stat().st_size
    batch_cases = [
        batch_case("okapi short postings", BM25Okapi, args.docs, args.vocab, args.doc_len, args.queries, args.query_len),
        batch_case("okapi long postings", BM25Okapi, 30000, 200, 40, 64, 8),
        batch_case("plus touches every doc", BM25Plus, 30000, 4000, 32, 48, 8),
    ]

    rows = [
        ("static build", args.docs, build_t),
        ("save", args.docs, save_t),
        ("load mmap + one query", args.docs, load_mmap_t),
        ("load copy + one query", args.docs, load_copy_t),
        (f"get_scores × {args.queries} sequential", args.docs, seq_t),
        (f"get_scores_many × {args.queries}", args.docs, batch_t),
        ("LiveIndex add", args.docs, add_t),
        ("LiveIndex delete", args.delete, del_t),
        ("static rebuild of docs left after delete", args.docs - args.delete, rebuild_t),
        (f"LiveIndex add 200 onto an existing {args.docs}", 200, add_200_t),
        (f"static rebuild of those {args.docs + 200}", args.docs + 200, rebuild_plus_t),
    ]
    machine = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }
    payload = {
        "machine": machine,
        "corpus": {"docs": args.docs, "doc_len": args.doc_len, "vocab": args.vocab,
                   "queries": args.queries, "query_len": args.query_len, "repeats": args.repeats},
        "file_bytes": file_bytes,
        "median_seconds": {name: t for name, _, t in rows},
        "batch_speedup_vs_sequential": seq_t / batch_t if batch_t else None,
        "batch_cases": batch_cases,
    }
    print(json.dumps(payload, indent=2))
    print()
    print("| operation | n | median seconds |")
    print("|---|---:|---:|")
    for name, n, t in rows:
        print(f"| {name} | {n} | {t:.4f} |")
    print()
    print("| batch workload | sequential s | get_scores_many s | speedup |")
    print("|---|---:|---:|---:|")
    for case in batch_cases:
        print(f"| {case['label']} | {case['sequential_s']:.6f} | {case['batch_s']:.6f} | {case['speedup']:.2f}× |")
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
