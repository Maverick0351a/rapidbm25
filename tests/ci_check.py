#!/usr/bin/env python3
"""Per-platform equivalence check, run by CI after installing a built wheel.

1. Finds the installed reference rank_bm25 and checks it is 0.2.2 (file sha256).
2. Short fuzz: rapidbm25 vs that reference, bit for bit, on this platform.
3. Contract replay: the 407 vectors recorded on Linux x86_64. Strict by default;
   set RAPIDBM25_REPLAY_STRICT=0 to report a mismatch without failing (it would
   mean this platform's libm differs from glibc, see the contract's open questions).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REF_SHA256 = "2f28cc795415c01e9f3db5a8ed019774f9cba747b272c9c304271589b8081ac6"  # rank_bm25.py in the PyPI 0.2.2 wheel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    tool = here.parent
    import numpy
    import rank_bm25
    import rapidbm25

    ref = Path(rank_bm25.__file__)
    sha = hashlib.sha256(ref.read_bytes()).hexdigest()
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "numpy": numpy.__version__, "rapidbm25": rapidbm25.__version__, "native": rapidbm25._native.__file__,
            "reference": str(ref), "reference_sha256_ok": sha == REF_SHA256}
    print(json.dumps(info, indent=2))
    if sha != REF_SHA256:
        sys.exit(f"reference rank_bm25 is not 0.2.2 (sha256 {sha})")

    fz = subprocess.run([sys.executable, str(here / "fuzz_equivalence.py"), "--upstream", str(ref),
                         "--cases", str(a.cases)], capture_output=True, text=True)
    print(fz.stdout[-3000:], fz.stderr[-2000:])
    if fz.returncode != 0:
        sys.exit("fuzz equivalence FAILED")

    for name in ("bm25.contract.json", "bm25-47aa3dd.contract.json"):
        rp = subprocess.run([sys.executable, str(here / "replay_contract.py"), str(tool / "contract" / name)],
                            capture_output=True, text=True)
        print(name, rp.stdout[-1500:], rp.stderr[-2000:])
        if rp.returncode != 0:
            if os.environ.get("RAPIDBM25_REPLAY_STRICT", "1") != "0":
                sys.exit(f"contract replay FAILED: {name}")
            print(f"WARNING: {name} replay mismatch on this platform (non-strict mode)")
    print("ci_check: OK")


if __name__ == "__main__":
    main()
