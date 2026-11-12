#!/usr/bin/env python3
"""Fail unless every wheel's native module carries cargo-auditable's dependency
section (.dep-v0 on ELF/PE; the same section name inside __DATA on Mach-O)."""
import sys
import zipfile
from pathlib import Path

bad = 0
for whl in sys.argv[1:]:
    with zipfile.ZipFile(whl) as z:
        mods = [n for n in z.namelist() if "/_native" in n and n.endswith((".so", ".pyd"))]
        ok = bool(mods) and all(b".dep-v0" in z.read(n) for n in mods)
    print(f"{'ok ' if ok else 'BAD'} {Path(whl).name} {mods}")
    bad += not ok
sys.exit(1 if bad else 0)
