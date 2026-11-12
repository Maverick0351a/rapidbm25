"""Contract adapter for rank_bm25-compatible modules (used by tests/replay_contract.py).

`cases()` lists the calls whose outputs we observe on the original; `run()`
executes one call against a module (original or new build) and returns a JSON
value in which every float is a float.hex() string, so replay is exact.
"""
from __future__ import annotations

import random

import numpy as np

SMALL = [d.split(" ") for d in ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]]


def _hexify(v):
    if isinstance(v, (float, np.floating)):
        return float(v).hex()
    if isinstance(v, np.ndarray):
        return [_hexify(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _hexify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_hexify(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def cases():
    out = []
    rng = random.Random(1729)
    vocab = [f"w{i}" for i in range(40)]
    corpora = {
        "small": SMALL,
        "with_empty_docs": [["a", "b", "a"], [], ["b", "c"], ["a"], []],
        "single_doc": [["only", "doc", "only"]],
        "dupes": [["x", "y"], ["x", "y"], ["z"], ["x"]],
        "random40": [[rng.choice(vocab) for _ in range(rng.randint(0, 25))] for _ in range(40)],
    }
    queries = [["windy", "London"], ["a", "a", "b"], ["missing"], [], ["w1", "w2", "w3", "w1"], ["x"], ["only"]]
    params = {
        "BM25Okapi": [{}, {"k1": 1.2, "b": 0.9, "epsilon": 0.1}, {"k1": 0.0, "b": 1.0}],
        "BM25L": [{}, {"k1": 2.0, "b": 0.3, "delta": 1.0}, {"b": 1.0}],
        "BM25Plus": [{}, {"k1": 0.9, "b": 0.6, "delta": 0.5}, {"k1": 0.0, "b": 1.0, "delta": 0}],
    }
    for cname, corpus in corpora.items():
        for cls, plist in params.items():
            for i, kw in enumerate(plist):
                base = {"class": cls, "corpus_name": cname, "corpus": corpus, "init": kw}
                out.append({**base, "name": f"{cls}/{cname}/p{i}/attrs", "method": "attrs", "args": []})
                for q in queries:
                    out.append({**base, "name": f"{cls}/{cname}/p{i}/get_scores/{'+'.join(q) or 'EMPTY'}",
                                "method": "get_scores", "args": [q]})
                n = len(corpus)
                out.append({**base, "name": f"{cls}/{cname}/p{i}/get_batch_scores",
                            "method": "get_batch_scores", "args": [queries[1], list(range(n))[::-1][:3]]})
    out.append({"class": "BM25Okapi", "corpus_name": "empty", "corpus": [], "init": {},
                "name": "BM25Okapi/empty/raises", "method": "attrs", "args": []})
    out.append({"class": "BM25Okapi", "corpus_name": "small", "corpus": SMALL, "init": {},
                "name": "BM25Okapi/small/get_top_n", "method": "get_top_n",
                "args": [["windy", "London"], [" ".join(d) for d in SMALL], 1]})
    return out


def run(mod, call):
    try:
        with np.errstate(all="ignore"):
            obj = getattr(mod, call["class"])(call["corpus"], **call["init"])
            m = call["method"]
            if m == "attrs":
                v = {"corpus_size": obj.corpus_size, "avgdl": obj.avgdl, "doc_len": list(obj.doc_len),
                     "idf": dict(obj.idf), "doc_freqs": [dict(d) for d in obj.doc_freqs]}
                if hasattr(obj, "average_idf"):
                    v["average_idf"] = obj.average_idf
            else:
                v = getattr(obj, m)(*call["args"])
        return {"ok": _hexify(v)}
    except Exception as e:
        return {"raises": type(e).__name__}
