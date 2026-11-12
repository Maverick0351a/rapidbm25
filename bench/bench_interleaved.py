#!/usr/bin/env python3
"""Interleaved benchmark: every round runs each impl once, in a fresh process,
in a rotated order (ABCD, BCDA, ...), so slow drift on a shared VM hits every
impl about equally. Writes a summary JSON with medians, MAD, and speedups."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mad(xs):
    m = statistics.median(xs)
    return statistics.median([abs(x - m) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--upstream", default="", help="path to rank_bm25.py (default: installed rank_bm25)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20, help="batches per candidate process")
    ap.add_argument("--impls", default="baseline,rapid,rapid_topn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    impls = [i for i in a.impls.split(",") if i]
    recs = []
    t0 = time.time()
    for r in range(a.rounds):
        order = impls[r % len(impls):] + impls[: r % len(impls)]
        for impl in order:
            cmd = [sys.executable, str(HERE / "bench_once.py"), "--impl", impl, "--workloads", a.workloads,
                   "--upstream", a.upstream, "--repeat", "1" if impl == "baseline" else str(a.repeat)]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
            rec = json.loads(out)
            rec["round"] = r
            recs.append(rec)
            print(json.dumps(rec), flush=True)
    summ = {}
    for impl in impls:
        rs = [x for x in recs if x["impl"] == impl]
        summ[impl] = {k: {"median": statistics.median([x[k] for x in rs]), "mad": mad([x[k] for x in rs]),
                          "n": len(rs)} for k in ("index_s", "scores_batch_s", "e2e_batch_s")}
    base = summ.get("baseline")
    if base:
        for impl, s in summ.items():
            s["speedup_vs_baseline"] = {k: base[k]["median"] / s[k]["median"]
                                        for k in ("scores_batch_s", "e2e_batch_s", "index_s")}
    if "rapid" in summ and "rapid_topn" in summ:
        summ["topn_fix_e2e_gain"] = summ["rapid"]["e2e_batch_s"]["median"] / summ["rapid_topn"]["e2e_batch_s"]["median"]
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    try:
        load = open("/proc/loadavg").read().split()[:3]
    except Exception:
        load = []
    result = {"rounds": a.rounds, "repeat_per_candidate_process": a.repeat, "order": "rotated per round",
              "elapsed_s": round(time.time() - t0, 1),
              "machine": {"cpu": cpu, "nproc": __import__("os").cpu_count(), "python": platform.python_version(),
                          "platform": platform.platform(), "loadavg_at_end": load},
              "summary": summ, "records": recs}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
