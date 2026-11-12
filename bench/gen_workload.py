#!/usr/bin/env python3
"""Build a code/docs retrieval workload for bench/bench_interleaved.py.

Chunks text files under --roots (default: this Python's stdlib) into 200-token
windows with 40-token overlap, then samples 200 held-out queries from them.
Writes corpus.jsonl, queries_bench.jsonl and raw_queries_bench.jsonl to --out.

    python bench/gen_workload.py --out /tmp/bm25-workload
    python bench/bench_interleaved.py --workloads /tmp/bm25-workload --out bench.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sysconfig
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")  # same tokenizer as bench_once.py
STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "for",
        "with", "as", "by", "at", "from", "this", "that", "it", "if", "else", "not", "no", "yes", "we",
        "you", "they", "i", "self", "none", "true", "false", "return", "import", "def", "class", "pass"}
SUFFIXES = {".py", ".md", ".rst", ".txt", ".c", ".h"}
SIZE, OVERLAP = 200, 40


def tokenize(text):
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def windows(tokens):
    if len(tokens) < 32:
        if tokens:
            yield tokens
        return
    i = 0
    while i < len(tokens):
        j = min(len(tokens), i + SIZE)
        yield tokens[i:j]
        if j >= len(tokens):
            break
        i += SIZE - OVERLAP


def chunks(roots, limit):
    out = []
    for root in roots:
        for p in sorted(Path(root).rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SUFFIXES or p.stat().st_size > 2_000_000:
                continue
            for w in windows(tokenize(p.read_text(encoding="utf-8", errors="ignore"))):
                out.append({"id": len(out), "tokens": w})
                if len(out) >= limit:
                    return out
    return out


def query(tokens, rng, k):
    cands = [t for t in tokens if t not in STOP and len(t) >= 3] or [t for t in tokens if len(t) >= 2] or ["empty"]
    k = max(3, min(k, len(cands)))
    start = rng.randrange(0, max(1, len(cands) - k + 1))
    q = cands[start:start + k]
    if rng.random() < 0.3 and len(cands) > k:
        q += [e for e in rng.sample(cands, 2) if e not in q]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[sysconfig.get_paths()["stdlib"]])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=16000)
    ap.add_argument("--queries", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    docs = chunks(a.roots, a.docs)
    ids = list(range(len(docs)))
    random.Random(20261111).shuffle(ids)
    pool = ids[len(ids) // 2:]  # held-out half: bench queries never come from tuning data
    rng, seen, rows = random.Random(202), set(), []
    lengths = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    while len(rows) < a.queries:
        q = query(docs[rng.choice(pool)]["tokens"], rng, rng.choice(lengths))
        if tuple(q) not in seen:
            seen.add(tuple(q))
            rows.append(q)
    w = lambda name, recs: (a.out / name).write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs))
    w("corpus.jsonl", docs)
    w("queries_bench.jsonl", [{"id": i, "tokens": q} for i, q in enumerate(rows)])
    w("raw_queries_bench.jsonl", [{"id": i, "text": " ".join(q)} for i, q in enumerate(rows)])
    lens = [len(d["tokens"]) for d in docs]
    print(json.dumps({"docs": len(docs), "mean_doc_len": round(sum(lens) / len(lens), 1),
                      "vocab": len({t for d in docs for t in d["tokens"]}), "queries": len(rows)}))


if __name__ == "__main__":
    main()
