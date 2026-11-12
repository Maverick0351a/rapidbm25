#!/usr/bin/env python3
"""One fresh process: build an index, time query batches, print one JSON line.

impl: baseline   reference rank_bm25 (--upstream path to rank_bm25.py, or the installed package)
      rapid      rapidbm25, get_scores + lexsort top-10 (no top-n fix)
      rapid_topn rapidbm25, Rust partial-select top_n(q, 10)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")  # same tokenizer as the workload


def tokenize(t):
    return [m.group(0).lower() for m in TOKEN_RE.finditer(t)]


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def load_file_module(name, path, extra_path=None):
    if extra_path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["baseline", "rapid", "rapid_topn"])
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--upstream", default="")
    ap.add_argument("--repeat", type=int, default=1, help="batches per process (median reported)")
    a = ap.parse_args()
    wl = Path(a.workloads)
    corpus = [r["tokens"] for r in jl(wl / "corpus.jsonl")]
    queries = [r["tokens"] for r in jl(wl / "queries_bench.jsonl")]
    raw = [r["text"] for r in jl(wl / "raw_queries_bench.jsonl")]

    if a.impl == "baseline":
        if a.upstream:
            cls = load_file_module("_ref", a.upstream).BM25Okapi
        else:
            from rank_bm25 import BM25Okapi as cls
    else:
        import rapidbm25
        cls = rapidbm25.BM25Okapi

    t0 = time.perf_counter()
    bm = cls(corpus)
    index_s = time.perf_counter() - t0
    n = bm.corpus_size
    ar = np.arange(n)

    def scores_batch():
        for q in queries:
            bm.get_scores(q)

    if a.impl == "rapid_topn":
        def e2e_batch():
            for t in raw:
                bm.top_n(tokenize(t), 10)
    else:
        def e2e_batch():
            for t in raw:
                s = bm.get_scores(tokenize(t))
                np.lexsort((ar, -s))[:10]

    def timed(fn):
        ts = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    rec = {"impl": a.impl, "index_s": index_s, "scores_batch_s": timed(scores_batch),
           "e2e_batch_s": timed(e2e_batch), "n_docs": n, "n_queries": len(queries), "repeat": a.repeat}
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
