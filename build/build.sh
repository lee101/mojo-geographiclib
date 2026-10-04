#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$repo_dir/dist/libmojo-geographiclib.so"
mkdir -p "$repo_dir/dist"

# src/capi.mojo is the single entry point; -I src resolves the ported.* module
# that holds the kernels.  On success only the warning headlines are shown,
# minus two classes that are noise for this port: upstream's docstrings are
# kept verbatim and this toolchain warns about the ones that are not one-line
# summaries, and the port pre-initialises its "real" variables to NaN exactly
# where upstream declares them, which the compiler reports as a dead store.
# On failure the full diagnostic is printed.
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
if mojo build --emit shared-lib --fp-mode contract=off -I "$repo_dir/src" \
       "$repo_dir/src/capi.mojo" -o "$out" >"$log" 2>&1 && [ -f "$out" ]; then
  grep -E "error|warning" "$log" \
    | grep -v -e 'doc string summary' -e 'was never used' || true
  echo "built $out"
else
  cat "$log" >&2
  exit 1
fi
