#!/usr/bin/env bash
# Build the native engines for THIS machine (-march=native / -C target-cpu=native).
# The Python layer also builds these automatically on first use; this is the manual path.
set -e
cd "$(dirname "$0")"
echo "building C++  -> engine_core.so"
g++ -O3 -march=native -funroll-loops -shared -fPIC engine_core.cpp -o engine_core.so
if command -v rustc >/dev/null 2>&1; then
  echo "building Rust -> engine_core_rs.so"
  rustc -C opt-level=3 -C target-cpu=native --crate-type=cdylib engine_core.rs -o engine_core_rs.so
else
  echo "rustc not found — skipping Rust engine (install via https://rustup.rs to enable)"
fi
echo "done."
