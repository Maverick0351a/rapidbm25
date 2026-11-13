#!/usr/bin/env python3
"""Compare rapidbm25 with rank_bm25 and write JSON plus markdown tables.

Synthetic corpora use a fixed seed (500, 4,000, and 8,000 documents). Each
batch time is the median of several runs. Per-query p50 and p99 are one extra
pass. rank_bm25 has no ``top_n``; that column times ``get_top_n``.

Index memory is the VmRSS the constructor keeps, measured in a fresh process
per library, because Rust allocations are outside ``tracemalloc``. The
rapidbm25 index is also saved once and its file size is recorded.

``--quality`` scores nDCG@10 and recall@100 on a local BEIR dataset, or on
every dataset directory under a path. ``bm25s`` is included when installed.
Nothing is downloaded.

    python bench/bench.py --out /tmp/rapidbm25-bench
    python bench/bench.py --sizes '' --quality /path/to/beir-datasets --out /tmp/rapidbm25-bench
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import re
import select
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# get_scores speedup (rank_bm25 / rapidbm25) required at GATE_DOCS documents.
# Ratios, not absolute times: CI runners are noisy. 200× is far below the
# README's measured ~1,470×, so a slow runner still passes and a real
# regression still fails.
GET_SCORES_SPEEDUP_FLOOR = 200.0
GATE_DOCS = 4000

# Absolute nDCG@10 gap allowed between rapidbm25 and rank_bm25. Both sides use
# the same tokenization and the same tie break, so a larger gap is a real split.
NDCG10_PARITY = 1e-6

SEED = 0
DOC_LEN = 64
VOCAB = 8000
QUERY_LEN = 8
# Long enough that get_scores at 4k docs is tens of milliseconds on the fast side.
# A shorter batch makes the ratio swing with scheduler noise and can fail the floor.
N_QUERIES = 2048
TOP_N = 10
RECALL_K = 100
REPEATS = 5
WARMUP = 1

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def make_tokens(n, length, vocab, seed):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=(n, length))
    return [[f"t{int(t)}" for t in row] for row in ids]


def parse_sizes(text):
    text = text.strip()
    if not text:
        return []
    sizes = []
    for part in text.split(","):
        try:
            n = int(part.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not a document count: {part.strip()!r}") from exc
        if n <= 0:
            raise argparse.ArgumentTypeError("sizes must be positive")
        sizes.append(n)
    return sizes


def once(fn):
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    t0 = time.perf_counter()
    try:
        value = fn()
        dt = time.perf_counter() - t0
    finally:
        if enabled:
            gc.enable()
    return dt, value


def median_pair(fns, repeats, warmup):
    names = list(fns)
    for _ in range(warmup):
        for name in names:
            fns[name]()
    samples = {name: [] for name in names}
    for i in range(repeats):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            dt, _value = once(fns[name])
            samples[name].append(dt)
    return {name: statistics.median(samples[name]) for name in names}


def speedup(rank_s, rapid_s):
    if rapid_s <= 0 or rank_s < 0:
        return None
    return rank_s / rapid_s


def p99(samples):
    if len(samples) < 2:
        return samples[0]
    # n=100 inclusive quantiles: index 98 is the 99th percentile.
    return statistics.quantiles(samples, n=100, method="inclusive")[98]


def query_latency(fn, queries):
    fn(queries[0])
    samples = []
    for query in queries:
        t0 = time.perf_counter()
        fn(query)
        samples.append(time.perf_counter() - t0)
    return {"p50_s": statistics.median(samples), "p99_s": p99(samples)}


def same_scores(rank_ix, rapid_ix, query):
    a = np.asarray(rank_ix.get_scores(query), dtype=np.float64)
    b = np.asarray(rapid_ix.get_scores(query), dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def time_corpus(rank_cls, rapid_cls, docs, queries, repeats, warmup):
    def build_rank():
        return rank_cls(docs)

    def build_rapid():
        return rapid_cls(docs)

    built = median_pair({"rank": build_rank, "rapid": build_rapid}, repeats, warmup)
    rank_ix = rank_cls(docs)
    rapid_ix = rapid_cls(docs)
    # One query, not the whole set: the test suite owns bit-exact coverage.
    if not same_scores(rank_ix, rapid_ix, queries[0]):
        raise SystemExit("get_scores diverged from rank_bm25; not timing a mismatch")

    def rank_scores():
        for q in queries:
            rank_ix.get_scores(q)

    def rapid_scores():
        for q in queries:
            rapid_ix.get_scores(q)

    def rank_top():
        for q in queries:
            rank_ix.get_top_n(q, docs, n=TOP_N)

    def rapid_top():
        for q in queries:
            rapid_ix.top_n(q, TOP_N)

    scored = median_pair({"rank": rank_scores, "rapid": rapid_scores}, repeats, warmup)
    topped = median_pair({"rank": rank_top, "rapid": rapid_top}, repeats, warmup)
    latency = {
        "rank_bm25": {
            "get_scores": query_latency(rank_ix.get_scores, queries),
            "top_n": query_latency(lambda q: rank_ix.get_top_n(q, docs, n=TOP_N), queries),
        },
        "rapidbm25": {
            "get_scores": query_latency(rapid_ix.get_scores, queries),
            "top_n": query_latency(lambda q: rapid_ix.top_n(q, TOP_N), queries),
        },
    }
    rank = {"build_s": built["rank"], "get_scores_s": scored["rank"], "top_n_s": topped["rank"]}
    rapid = {"build_s": built["rapid"], "get_scores_s": scored["rapid"], "top_n_s": topped["rapid"]}
    ratios = {
        "build": speedup(rank["build_s"], rapid["build_s"]),
        "get_scores": speedup(rank["get_scores_s"], rapid["get_scores_s"]),
        "top_n": speedup(rank["top_n_s"], rapid["top_n_s"]),
    }
    return rank, rapid, ratios, latency


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir_tokens(path, n_queries):
    corpus_path = path / "corpus.jsonl"
    queries_path = path / "queries.jsonl"
    if not corpus_path.is_file() or not queries_path.is_file():
        raise SystemExit(
            f"{path} has no corpus.jsonl and queries.jsonl. "
            "Pass a local BEIR directory (for example scifact). This script does not download it."
        )
    docs = []
    for row in read_jsonl(corpus_path):
        title = row.get("title") or ""
        text = row.get("text") or ""
        docs.append(tokenize(f"{title} {text}".strip()))
    keyed = []
    for i, row in enumerate(read_jsonl(queries_path)):
        keyed.append((str(row.get("_id", i)), tokenize(row.get("text") or "")))
    keyed.sort()
    queries = [tokens for _id, tokens in keyed[:n_queries]]
    if not docs or not queries:
        raise SystemExit(f"{path} has no documents or queries")
    return docs, queries


def dataset_dirs(path):
    if (path / "corpus.jsonl").is_file():
        return [path]
    found = [p for p in sorted(path.iterdir()) if p.is_dir() and (p / "corpus.jsonl").is_file()]
    if not found:
        raise SystemExit(
            f"{path} is not a BEIR dataset and does not contain any. "
            "Expected corpus.jsonl, queries.jsonl, and qrels/test.tsv. This script does not download them."
        )
    return found


def load_qrels(path):
    qrels_file = path / "qrels" / "test.tsv"
    if not qrels_file.is_file():
        qrels_file = path / "qrels" / "dev.tsv"
    if not qrels_file.is_file():
        raise SystemExit(f"{path} has no qrels/test.tsv. This script does not download datasets.")
    qrels = {}
    for line in qrels_file.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            rel = float(parts[2])
        except ValueError:
            continue
        qrels.setdefault(parts[0], {})[parts[1]] = rel
    return qrels


def load_quality_dataset(path):
    if not (path / "corpus.jsonl").is_file() or not (path / "queries.jsonl").is_file():
        raise SystemExit(f"{path} is missing corpus.jsonl or queries.jsonl")
    doc_ids = []
    tokens = []
    raw = []
    for row in read_jsonl(path / "corpus.jsonl"):
        title = row.get("title") or ""
        text = row.get("text") or ""
        body = f"{title} {text}".strip()
        doc_ids.append(str(row.get("_id", len(doc_ids))))
        raw.append(body)
        tokens.append(tokenize(body))
    queries = []
    for i, row in enumerate(read_jsonl(path / "queries.jsonl")):
        text = row.get("text") or ""
        queries.append((str(row.get("_id", i)), tokenize(text), text))
    qrels = load_qrels(path)
    kept = [q for q in queries if qrels.get(q[0])]
    if not doc_ids or not kept:
        raise SystemExit(f"{path} has no documents or no queries in common with qrels")
    return {"name": path.name, "doc_ids": doc_ids, "tokens": tokens, "raw": raw, "queries": kept, "qrels": qrels}


def top_doc_ids(scores, doc_ids, k):
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.shape[0])
    k = min(k, n)
    if k <= 0:
        return []
    # Higher score, then lower index. argpartition's top-k keeps an arbitrary
    # subset of the documents tied at the cutoff, so a relevant zero would
    # depend on NumPy's partition.
    order = np.lexsort((np.arange(n), -scores))[:k]
    return [doc_ids[int(i)] for i in order]


def ndcg_at(ranked, rels, k):
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        rel = rels.get(doc_id, 0.0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    ideal = sorted((rel for rel in rels.values() if rel > 0), reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal))
    if idcg == 0:
        return None
    return dcg / idcg


def recall_at(ranked, rels, k):
    relevant = {doc_id for doc_id, rel in rels.items() if rel > 0}
    if not relevant:
        return None
    hit = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hit / len(relevant)


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def metrics_from_scores(get_scores, dataset):
    ndcgs = []
    recalls = []
    for qid, tokens, _raw in dataset["queries"]:
        ranked = top_doc_ids(get_scores(tokens), dataset["doc_ids"], RECALL_K)
        rels = dataset["qrels"][qid]
        ndcgs.append(ndcg_at(ranked, rels, TOP_N))
        recalls.append(recall_at(ranked, rels, RECALL_K))
    return {"ndcg@10": mean(ndcgs), "recall@100": mean(recalls), "queries": len(dataset["queries"])}


def metrics_bm25s(dataset):
    try:
        import bm25s
    except ImportError:
        return {"skipped": True, "reason": "bm25s is not installed"}
    try:
        tokenized = bm25s.tokenize(dataset["raw"], stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokenized, show_progress=False)
        query_tokens = bm25s.tokenize([q[2] for q in dataset["queries"]], stopwords="en", show_progress=False)
        k = min(RECALL_K, len(dataset["doc_ids"]))
        found = retriever.retrieve(query_tokens, k=k, show_progress=False)
        documents = found.documents
        ndcgs = []
        recalls = []
        for row, (qid, _tokens, _raw) in zip(documents, dataset["queries"]):
            ranked = [dataset["doc_ids"][int(i)] for i in row if int(i) >= 0]
            rels = dataset["qrels"][qid]
            ndcgs.append(ndcg_at(ranked, rels, TOP_N))
            recalls.append(recall_at(ranked, rels, RECALL_K))
        return {
            "skipped": False,
            "ndcg@10": mean(ndcgs),
            "recall@100": mean(recalls),
            "queries": len(dataset["queries"]),
        }
    except Exception as exc:
        return {"skipped": True, "reason": f"bm25s failed: {exc}"}


def score_quality(rank_cls, rapid_cls, dataset):
    print(f"quality {dataset['name']} docs={len(dataset['doc_ids'])} queries={len(dataset['queries'])}", file=sys.stderr)
    rank_ix = rank_cls(dataset["tokens"])
    rapid_ix = rapid_cls(dataset["tokens"])
    return {
        "dataset": dataset["name"],
        "docs": len(dataset["doc_ids"]),
        "rank_bm25": metrics_from_scores(rank_ix.get_scores, dataset),
        "rapidbm25": metrics_from_scores(rapid_ix.get_scores, dataset),
        "bm25s": metrics_bm25s(dataset),
    }


def current_rss_bytes():
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def read_corpus_file(path):
    docs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            # Space-separated tokens. JSON would spike RSS above the index itself.
            docs.append(line.split(" ") if line else [])
    return docs


def write_corpus_file(path, docs):
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(" ".join(doc) + "\n")


def run_rss_worker(kind, path, n, doc_len, vocab, seed):
    # Synthetic corpora are rebuilt from the seed so a JSON or text parse does
    # not set the high-water mark above the index build.
    if kind == "rank":
        from rank_bm25 import BM25Okapi as cls
    elif kind == "rapid":
        import rapidbm25
        cls = rapidbm25.BM25Okapi
    else:
        raise SystemExit(f"unknown rss worker {kind}")
    if n:
        docs = make_tokens(n, doc_len, vocab, seed)
    else:
        docs = read_corpus_file(path)
    gc.collect()
    before = current_rss_bytes()
    index = cls(docs)
    gc.collect()
    after = current_rss_bytes()
    added = None if before is None or after is None else max(0, after - before)
    payload = {"index_rss_bytes": added}
    if before is None:
        payload["reason"] = "VmRSS is unavailable"
    del index
    print(json.dumps(payload))


def measure_rss(kind, corpus_path=None, spec=None):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--rss-worker", kind]
    if spec:
        cmd += ["--rss-n", str(spec["n"]), "--rss-doc-len", str(spec["doc_len"]),
                "--rss-vocab", str(spec["vocab"]), "--rss-seed", str(spec["seed"])]
    else:
        cmd += ["--rss-corpus", str(corpus_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "rss worker failed").strip()
        return {"index_rss_bytes": None, "error": detail[-500:]}
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), "")
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"index_rss_bytes": None, "error": proc.stdout.strip()[-500:] or "rss worker returned no JSON"}


def measure_memory(rapid_mod, docs, label, spec=None):
    print(f"memory {label} docs={len(docs)}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = None
        if spec is None:
            corpus_path = Path(tmp) / "corpus.txt"
            write_corpus_file(corpus_path, docs)
        rank = measure_rss("rank", corpus_path, spec)
        rapid = measure_rss("rapid", corpus_path, spec)
        index_path = Path(tmp) / "index.rbm25"
        index_bytes = None
        index_error = None
        try:
            rapid_mod.BM25Okapi(docs).save(index_path)
            index_bytes = index_path.stat().st_size
        except Exception as exc:
            index_error = str(exc)
    out = {
        "label": label,
        "docs": len(docs),
        "note": "added by index is the VmRSS the constructor keeps after the corpus has been collected",
        "rank_bm25_index_rss_bytes": rank.get("index_rss_bytes"),
        "rapidbm25_index_rss_bytes": rapid.get("index_rss_bytes"),
        "rapidbm25_index_bytes": index_bytes,
    }
    if rank.get("error") or rank.get("reason"):
        out["rank_bm25_rss_note"] = rank.get("error") or rank.get("reason")
    if rapid.get("error") or rapid.get("reason"):
        out["rapidbm25_rss_note"] = rapid.get("error") or rapid.get("reason")
    if index_error:
        out["rapidbm25_index_error"] = index_error
    return out


def _mcp_send(proc, payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def _mcp_read(proc, timeout):
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError("MCP server timed out")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed stdout")
    return json.loads(line)


def _mcp_tool_text(message):
    result = message.get("result") if isinstance(message, dict) else None
    if not isinstance(result, dict):
        return ""
    content = result.get("content") or []
    if not content or not isinstance(content[0], dict):
        return ""
    return str(content[0].get("text") or "")


def mcp_smoke():
    print("mcp smoke", file=sys.stderr)
    from rapidbm25.mcp_server import _MAX_DOC_BYTES

    oversize = {"skipped": False, "passed": False, "limit_bytes": _MAX_DOC_BYTES}
    proc = None
    try:
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "rapidbm25.mcp_server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        _mcp_send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "bench", "version": "0"}},
        })
        first = _mcp_read(proc, 30)
        first_s = time.perf_counter() - t0
        first_ok = "result" in first and "error" not in first
        _mcp_send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "bm25_index_texts", "arguments": {
                "index": "oversize", "documents": ["x" * (_MAX_DOC_BYTES + 1)],
            }},
        })
        rejected = _mcp_read(proc, 30)
        rejected_text = _mcp_tool_text(rejected)
        rejected_error = isinstance(rejected, dict) and bool((rejected.get("result") or {}).get("isError"))
        oversize["is_error"] = rejected_error
        oversize["message"] = rejected_text[:200]
        proc.stdin.write("{not-json\n")
        proc.stdin.flush()
        bad = _mcp_read(proc, 30)
        code = bad.get("error", {}).get("code") if isinstance(bad, dict) else None
        _mcp_send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        again = _mcp_read(proc, 30)
        alive = isinstance(again, dict) and "result" in again and "error" not in again
        oversize["passed"] = bool(rejected_error and str(_MAX_DOC_BYTES) in rejected_text and alive)
        return {
            "first_response_s": first_s if first_ok else None,
            "first_response_ok": first_ok,
            "malformed_json": {"error_code": code, "passed": code == -32700 and alive},
            "server_kept_answering": alive,
            "oversize": oversize,
        }
    except Exception as exc:
        oversize["error"] = str(exc)
        return {
            "first_response_s": None,
            "first_response_ok": False,
            "malformed_json": {"passed": False, "error": str(exc)},
            "server_kept_answering": False,
            "oversize": oversize,
            "error": str(exc),
        }
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def evaluate_speed_gate(rows, enforce):
    row = next((r for r in rows if r["corpus"] == "synthetic" and r["docs"] == GATE_DOCS), None)
    ratio = None if row is None else row["speedup"]["get_scores"]
    passed = None if ratio is None else ratio >= GET_SCORES_SPEEDUP_FLOOR
    return {
        "metric": "get_scores",
        "docs": GATE_DOCS,
        "floor": GET_SCORES_SPEEDUP_FLOOR,
        "speedup": ratio,
        "enforced": bool(enforce and ratio is not None),
        "passed": passed,
    }


def evaluate_quality_gate(rows, enforce):
    datasets = []
    comparable = []
    for row in rows:
        rapid = row["rapidbm25"].get("ndcg@10")
        rank = row["rank_bm25"].get("ndcg@10")
        if rapid is None or rank is None:
            datasets.append({"name": row["dataset"], "abs_diff": None, "passed": None})
            continue
        diff = abs(rapid - rank)
        ok = diff <= NDCG10_PARITY
        datasets.append({"name": row["dataset"], "abs_diff": diff, "passed": ok})
        comparable.append(ok)
    passed = None if not comparable else all(comparable)
    return {
        "metric": "ndcg@10",
        "tolerance": NDCG10_PARITY,
        "enforced": bool(enforce and comparable),
        "passed": passed,
        "datasets": datasets,
    }


def fmt_s(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_x(ratio):
    if ratio is None:
        return "n/a"
    if ratio >= 100:
        return f"{ratio:.0f}×"
    if ratio >= 10:
        return f"{ratio:.1f}×"
    return f"{ratio:.2f}×"


def fmt_bytes(n):
    if n is None:
        return "n/a"
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024 ** 2:.2f} MiB"


def fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def markdown(payload):
    p = payload["params"]
    lines = [
        "### rapidbm25 vs rank_bm25",
        "",
        (
            f"Median of {p['repeats']} runs after {p['warmup']} warmup. "
            f"Synthetic seed {p['seed']}, doc length {p['doc_len']}, vocab {p['vocab']}, "
            f"query length {p['query_len']}, {p['queries']} queries, top {p['top_n']}. "
            "rank_bm25's top-n column is `get_top_n`. "
            "p50 and p99 are one pass, one query at a time."
        ),
        "",
        "#### Speed",
        "",
        "| corpus | docs | rank build | rapid build | × | rank get_scores | rapid get_scores | × | rank top_n | rapid top_n | × |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        rank = row["rank_bm25"]
        rapid = row["rapidbm25"]
        ratio = row["speedup"]
        lines.append(
            "| {corpus} | {docs} | {rb} | {ab} | {xb} | {rs} | {as_} | {xs} | {rt} | {at} | {xt} |".format(
                corpus=row["corpus"],
                docs=row["docs"],
                rb=fmt_s(rank["build_s"]),
                ab=fmt_s(rapid["build_s"]),
                xb=fmt_x(ratio["build"]),
                rs=fmt_s(rank["get_scores_s"]),
                as_=fmt_s(rapid["get_scores_s"]),
                xs=fmt_x(ratio["get_scores"]),
                rt=fmt_s(rank["top_n_s"]),
                at=fmt_s(rapid["top_n_s"]),
                xt=fmt_x(ratio["top_n"]),
            )
        )
    lines.extend([
        "",
        "#### Query latency",
        "",
        "| corpus | docs | impl | get_scores p50 | get_scores p99 | top_n p50 | top_n p99 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in payload["rows"]:
        for impl in ("rank_bm25", "rapidbm25"):
            lat = row["latency_s"][impl]
            lines.append(
                f"| {row['corpus']} | {row['docs']} | {impl} | "
                f"{fmt_s(lat['get_scores']['p50_s'])} | {fmt_s(lat['get_scores']['p99_s'])} | "
                f"{fmt_s(lat['top_n']['p50_s'])} | {fmt_s(lat['top_n']['p99_s'])} |"
            )
    memory = payload.get("memory")
    lines.extend(["", "#### Memory and index size", ""])
    if not memory:
        lines.append("Not measured (no corpus in this run).")
    else:
        lines.extend([
            f"Corpus: {memory['label']} ({memory['docs']} docs). {memory['note']}.",
            "",
            "| impl | added by index | saved index |",
            "|---|---:|---:|",
            f"| rank_bm25 | {fmt_bytes(memory.get('rank_bm25_index_rss_bytes'))} | |",
            f"| rapidbm25 | {fmt_bytes(memory.get('rapidbm25_index_rss_bytes'))} | {fmt_bytes(memory.get('rapidbm25_index_bytes'))} |",
        ])
    lines.extend([
        "",
        "#### Retrieval quality",
        "",
        "nDCG@10 and recall@100. rapidbm25 and rank_bm25 share a tokenizer; ties break toward the lower document id. bm25s uses its own tokenizer.",
        "",
        "| dataset | impl | nDCG@10 | recall@100 |",
        "|---|---|---:|---:|",
    ])
    if not payload["quality"]:
        lines.append("| | | not run | |")
    for row in payload["quality"]:
        for impl in ("rank_bm25", "rapidbm25", "bm25s"):
            cell = row[impl]
            if cell.get("skipped"):
                reason = str(cell.get("reason", "")).replace("|", "/")
                lines.append(f"| {row['dataset']} | {impl} | skipped | {reason} |")
            else:
                lines.append(
                    f"| {row['dataset']} | {impl} | {fmt_metric(cell.get('ndcg@10'))} | {fmt_metric(cell.get('recall@100'))} |"
                )
    mcp = payload["mcp"]
    lines.extend([
        "",
        "#### MCP smoke",
        "",
        "| check | result |",
        "|---|---|",
        f"| first successful response | {fmt_s(mcp.get('first_response_s'))} |",
        "| malformed JSON line | "
        + ("parse error, server still answered" if mcp.get("malformed_json", {}).get("passed") else "failed")
        + " |",
        "| oversized document | "
        + (
            "error, server still answered"
            if mcp.get("oversize", {}).get("passed")
            else "skipped: " + str(mcp.get("oversize", {}).get("reason", ""))
            if mcp.get("oversize", {}).get("skipped")
            else "failed"
        )
        + " |",
        "",
    ])
    gate = payload["gate"]
    if gate["speedup"] is None:
        lines.append(
            f"Speed gate: no synthetic {gate['docs']}-doc `get_scores` row, so the {gate['floor']:.0f}× floor was not applied."
        )
    else:
        result = "pass" if gate["passed"] else "fail"
        lines.append(
            f"Speed gate: `get_scores` at {gate['docs']} docs is {fmt_x(gate['speedup'])} "
            f"(floor {gate['floor']:.0f}×). {result}."
        )
    qgate = payload["quality_gate"]
    if not qgate["datasets"]:
        lines.append("nDCG gate: not run.")
    for item in qgate["datasets"]:
        if item["abs_diff"] is None:
            lines.append(f"nDCG gate: {item['name']} had no comparable nDCG@10.")
        else:
            result = "pass" if item["passed"] else "fail"
            lines.append(
                f"nDCG gate: {item['name']} abs diff {item['abs_diff']:.3e} "
                f"(tolerance {qgate['tolerance']:.0e}). {result}."
            )
    lines.append("")
    return "\n".join(lines)


def package_version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="500,4000,8000", help="comma-separated document counts; empty skips synthetic")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--queries", type=int, default=N_QUERIES)
    ap.add_argument("--beir", type=Path, default=None, help="time one local BEIR directory; not downloaded")
    ap.add_argument("--quality", type=Path, default=None, help="score one BEIR dataset or a directory of them; not downloaded")
    ap.add_argument("--out", type=Path, default=Path("/tmp/rapidbm25-bench"))
    ap.add_argument("--no-gate", action="store_true", help="report gate results without failing")
    ap.add_argument("--rss-worker", choices=("rank", "rapid"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rss-corpus", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rss-n", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--rss-doc-len", type=int, default=DOC_LEN, help=argparse.SUPPRESS)
    ap.add_argument("--rss-vocab", type=int, default=VOCAB, help=argparse.SUPPRESS)
    ap.add_argument("--rss-seed", type=int, default=SEED, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.rss_worker:
        if not args.rss_n and args.rss_corpus is None:
            ap.error("--rss-worker requires --rss-corpus or --rss-n")
        run_rss_worker(args.rss_worker, args.rss_corpus, args.rss_n, args.rss_doc_len, args.rss_vocab, args.rss_seed)
        return
    if args.repeats < 1 or args.warmup < 0 or args.queries < 1:
        ap.error("--repeats and --queries must be >= 1 and --warmup must be >= 0")
    try:
        sizes = parse_sizes(args.sizes)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    if not sizes and args.beir is None and args.quality is None:
        ap.error("pass --sizes, --beir, or --quality")

    try:
        import rapidbm25
        from rank_bm25 import BM25Okapi as RankBM25
    except ImportError as exc:
        raise SystemExit(f"install rapidbm25 and rank-bm25==0.2.2 before benchmarking ({exc})") from exc

    rank_version = package_version("rank-bm25") or "unknown"
    if rank_version != "0.2.2":
        print(f"warning: rank-bm25 {rank_version} is installed; CI pins 0.2.2", file=sys.stderr)

    rows = []
    mem_docs = None
    mem_label = None
    mem_spec = None
    if sizes:
        n_max = max(sizes)
        corpus = make_tokens(n_max, DOC_LEN, VOCAB, SEED)
        queries = make_tokens(args.queries, QUERY_LEN, VOCAB, SEED + 1)
        mem_n = GATE_DOCS if GATE_DOCS in sizes else n_max
        mem_docs = corpus[:mem_n]
        mem_label = f"synthetic {mem_n}"
        mem_spec = {"n": mem_n, "doc_len": DOC_LEN, "vocab": VOCAB, "seed": SEED}
        for n in sizes:
            print(f"timing synthetic n={n}", file=sys.stderr)
            rank, rapid, ratios, latency = time_corpus(
                RankBM25, rapidbm25.BM25Okapi, corpus[:n], queries, args.repeats, args.warmup,
            )
            rows.append({
                "corpus": "synthetic",
                "docs": n,
                "queries": len(queries),
                "rank_bm25": rank,
                "rapidbm25": rapid,
                "speedup": ratios,
                "latency_s": latency,
            })

    if args.beir is not None:
        print(f"timing beir {args.beir}", file=sys.stderr)
        docs, queries = load_beir_tokens(args.beir, args.queries)
        rank, rapid, ratios, latency = time_corpus(
            RankBM25, rapidbm25.BM25Okapi, docs, queries, args.repeats, args.warmup,
        )
        rows.append({
            "corpus": f"beir:{args.beir.name}",
            "docs": len(docs),
            "queries": len(queries),
            "rank_bm25": rank,
            "rapidbm25": rapid,
            "speedup": ratios,
            "latency_s": latency,
        })

    quality = []
    if args.quality is not None:
        for path in dataset_dirs(args.quality):
            dataset = load_quality_dataset(path)
            quality.append(score_quality(RankBM25, rapidbm25.BM25Okapi, dataset))
            if mem_docs is None:
                mem_docs = dataset["tokens"]
                mem_label = dataset["name"]

    memory = measure_memory(rapidbm25, mem_docs, mem_label, mem_spec) if mem_docs is not None else None
    mcp = mcp_smoke()
    enforce = not args.no_gate
    gate = evaluate_speed_gate(rows, enforce)
    quality_gate = evaluate_quality_gate(quality, enforce)
    payload = {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rank_bm25": rank_version,
            "rapidbm25": rapidbm25.__version__,
            "bm25s": package_version("bm25s"),
        },
        "params": {
            "seed": SEED,
            "doc_len": DOC_LEN,
            "vocab": VOCAB,
            "query_len": QUERY_LEN,
            "queries": args.queries,
            "top_n": TOP_N,
            "recall_k": RECALL_K,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "sizes": sizes,
            "beir": None if args.beir is None else str(args.beir),
            "quality": None if args.quality is None else str(args.quality),
        },
        "note": "rank_bm25 top_n_s is get_top_n; rapidbm25 top_n_s is top_n. Only the speed floor and nDCG parity fail the process.",
        "rows": rows,
        "memory": memory,
        "quality": quality,
        "mcp": mcp,
        "gate": gate,
        "quality_gate": quality_gate,
    }
    text = markdown(payload)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out / "bench.md").write_text(text, encoding="utf-8")
    print(text, end="")
    failures = []
    if gate["enforced"] and gate["passed"] is False:
        failures.append(
            f"get_scores speedup at {GATE_DOCS} docs is {gate['speedup']:.1f}×, "
            f"below the floor of {GET_SCORES_SPEEDUP_FLOOR:.0f}×"
        )
    if quality_gate["enforced"] and quality_gate["passed"] is False:
        bad = [d for d in quality_gate["datasets"] if d["passed"] is False]
        detail = ", ".join(f"{d['name']} abs diff {d['abs_diff']:.3e}" for d in bad)
        failures.append(f"nDCG@10 parity exceeded {NDCG10_PARITY:.0e}: {detail}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
