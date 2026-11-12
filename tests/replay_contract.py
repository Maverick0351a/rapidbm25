#!/usr/bin/env python3
"""Replay a behavior contract's observed vectors against rapidbm25.

Each vector in contract/*.contract.json was recorded by running the original
rank_bm25 on a fixture. This script rebuilds the same call against rapidbm25
(adding any `build_init_extra` keyword arguments, e.g. compat="47aa3dd") and
compares the result exactly: every float is serialized with float.hex().

    python tests/replay_contract.py contract/bm25.contract.json
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("_bm25_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_call(contract: dict, vector: dict) -> dict:
    call = dict(vector["call"])
    if "corpus_name" in call:
        call["corpus"] = contract["fixtures"][call["corpus_name"]]
    extra = contract.get("build_init_extra", {}).get(call.get("class"), {})
    if extra:
        call["init"] = {**call.get("init", {}), **extra}
    return call


def replay(contract_path: Path, adapter_path: Path, build: str = "rapidbm25") -> dict:
    contract = json.loads(contract_path.read_text())
    adapter = load_adapter(adapter_path)
    mod = importlib.import_module(build)
    t0 = time.perf_counter()
    fails = []
    for v in contract["vectors"]:
        got = adapter.run(mod, build_call(contract, v))
        if got != v["expect"]:
            fails.append({"name": v["name"], "expect": str(v["expect"])[:200], "got": str(got)[:200]})
    n = len(contract["vectors"])
    return {"contract": contract_path.name, "build": build, "vectors": n, "passed": n - len(fails),
            "failed": len(fails), "compare": "exact JSON (floats as hex)", "fail_samples": fails[:10],
            "ok": n > 0 and not fails, "elapsed_s": round(time.perf_counter() - t0, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("contract", type=Path)
    ap.add_argument("--adapter", type=Path, default=ROOT / "contract" / "adapter.py")
    ap.add_argument("--build", default="rapidbm25", help="module to test (default: rapidbm25)")
    a = ap.parse_args()
    res = replay(a.contract, a.adapter, a.build)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
