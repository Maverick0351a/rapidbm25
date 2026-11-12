#!/usr/bin/env python3
"""Search a folder with BM25: python bm25_search.py --path DIR --query "terms" [-k 10] [--glob '*.py'] [--json]"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=".")
    ap.add_argument("--query", required=True)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--glob", default="*")
    ap.add_argument("--chunk-lines", type=int, default=40)
    ap.add_argument("--variant", default="okapi", choices=["okapi", "l", "plus"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        from rapidbm25.textindex import TextIndex, chunk_path
    except ImportError:
        sys.exit("rapidbm25 is not installed: pip install rapidbm25")
    os.environ.setdefault("RAPIDBM25_ROOTS", os.path.abspath(a.path))
    ids, texts = chunk_path(a.path, a.glob, a.chunk_lines, max(0, a.chunk_lines // 4))
    if not texts:
        sys.exit("no text files found")
    hits = TextIndex(ids, texts, a.variant).search(a.query, a.k)
    if a.json:
        print(json.dumps({"chunks_indexed": len(texts), "results": hits}, indent=2))
        return
    for h in hits:
        first = next((l.strip() for l in h["text"].splitlines() if l.strip()), "")[:100]
        print(f"{h['rank']:>3}  {h['score']:>9.4f}  {h['id']}  {first}")
    if not hits:
        print("no matches")


if __name__ == "__main__":
    main()
