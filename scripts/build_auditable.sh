#!/usr/bin/env bash
# Build the rapidbm25 wheel with cargo-auditable's dependency manifest embedded
# (ELF section .dep-v0), the same way `cargo auditable build` does it: run the
# cargo-auditable binary as RUSTC_WORKSPACE_WRAPPER underneath maturin.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HERE/target/wheels}"
CA="${CARGO_AUDITABLE:-$(command -v cargo-auditable || true)}"  # cargo install cargo-auditable
MATURIN="${MATURIN:-maturin}"
PY="${PYTHON:-python}"
[ -n "$CA" ] && [ -x "$CA" ] || { echo "cargo-auditable not found at $CA" >&2; exit 2; }
export RUSTC_WORKSPACE_WRAPPER="$CA"
export CARGO_AUDITABLE_ORIG_ARGS='{"offline":false,"locked":false,"frozen":false,"config":[]}'
cd "$HERE"
cargo clean -q -p rapidbm25 --release 2>/dev/null || true  # force the wrapper to run on our crate
"$MATURIN" build --release --locked -i "$PY" -o "$OUT"
WHL="$(ls -t "$OUT"/rapidbm25-*.whl | head -1)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"$PY" -m zipfile -e "$WHL" "$TMP" >/dev/null
SO="$(find "$TMP" -name '_native*.so' | head -1)"
if objdump -h "$SO" | grep -q '\.dep-v0'; then
  echo "auditable: .dep-v0 present in $(basename "$SO")"
else
  echo "auditable: .dep-v0 MISSING" >&2; exit 3
fi
echo "$WHL"
