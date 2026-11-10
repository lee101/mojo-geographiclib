"""Compare worker counts and thresholds across builds, one process per cell.

    pixi run bash -c 'python bench/parallel_builds.py <BASE_COMMIT>'
    pixi run bash -c 'python bench/parallel_ab.py'

This is the script behind the numbers in the README's *parallel pass*
section.  It measures the same batch calls on several builds and prints ms.

It exists in this shape because every simpler version was wrong, and each way
is easy to repeat:

* **Two libraries in one process.**  Pointing `_lib.LIB` at library A, then
  at library B, and alternating gave "serial" numbers 10x the truth: the
  second library's first call builds its own worker pool while the first is
  still alive.  So every cell here is a fresh `python -c` process.
* **A reference tree that rebuilds itself.**  `_lib.build()` recompiles when
  any `src/*.mojo` is newer than the `.so`.  A reference tree whose `src` is
  symlinked to the working tree therefore silently compiled *this* tree into
  the "old" library.  `parallel_builds.py` copies the sources and backdates
  them, and every process here asserts the loaded `.so` hashes to what the
  driver asked for.
* **A reference tree with this tree's Python package.**  It set
  `MAX_WORKERS = 4`, so the old build's 32-worker cap was never exercised.
  `parallel_builds.py` restores that file from the base commit.

Trees (override with the environment variables):

    PAR_SERIAL_TREE  a build whose kernels never split (the serial reference)
    PAR_BEFORE_TREE  the build before this pass (cap 32, one 8192 threshold)
    new              this tree
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREES = {
    "serial": os.environ.get("PAR_SERIAL_TREE", "/tmp/b_serial"),
    "before": os.environ.get("PAR_BEFORE_TREE", "/tmp/b_before"),
    "new": ROOT,
}
SIZES = (4096, 8192, 16384, 20480, 24576, 65536, 100_000)

CHILD = r"""
import hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from mojogeographiclib import _lib
assert _lib.LIB == sys.argv[4], (_lib.LIB, sys.argv[4])
got = hashlib.sha256(open(_lib.LIB, "rb").read()).hexdigest()
assert got == sys.argv[5], f"loaded {got}, wanted {sys.argv[5]}"
from mojogeographiclib.geodesic import Geodesic
from mojogeographiclib.geodesicline import GeodesicLine
case, n, reps = sys.argv[2], int(sys.argv[3]), 11
rng = np.random.default_rng(7)
lat = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, 300000))))
lon = np.ascontiguousarray(rng.uniform(-180, 180, 300000))
lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, 300000))))
lon2 = np.ascontiguousarray(rng.uniform(-180, 180, 300000))
azi = np.ascontiguousarray(rng.uniform(-180, 180, 300000))
s = np.ascontiguousarray(rng.uniform(-2e7, 2e7, 300000))
g = Geodesic.WGS84
line = GeodesicLine(g, 37.0, -122.0, 51.0, g.ALL)
g.InversePairs(lat[:16384], lon[:16384], lat2[:16384], lon2[:16384], g.ALL)
a, b, c, e, f, ss = (v[:n] for v in (lat, lon, lat2, lon2, azi, s))
if case == "InversePairs":
    fn = lambda: g.InversePairs(a, b, c, e, g.ALL)
elif case == "DirectMany":
    fn = lambda: g.DirectMany(a, b, f, ss, g.ALL)
else:
    fn = lambda: line.Positions(ss, line.caps)
fn()
best = float("inf")
for _ in range(reps):
    t0 = time.perf_counter()
    fn()
    best = min(best, time.perf_counter() - t0)
print(json.dumps({"ms": best * 1e3, "workers": g._workers}))
"""


def tree_of(name):
    return TREES[name]


def lib_hash(tree):
    so = os.path.join(tree, "dist", "libmojo-geographiclib.so")
    with open(so, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def measure(name, case, n):
    tree = tree_of(name)
    so = os.path.join(tree, "dist", "libmojo-geographiclib.so")
    env = dict(os.environ, PYTHONPATH="",
               MOJOGEOGRAPHICLIB_LIB=so,
               MOJOGEOGRAPHICLIB_MOJO=os.path.join(
                   ROOT, ".pixi/envs/default/bin/mojo"))
    out = subprocess.run(
        [sys.executable, "-c", CHILD, os.path.join(tree, "python"), case,
         str(n), so, lib_hash(tree)],
        capture_output=True, text=True, env=env, cwd="/tmp")
    if out.returncode != 0:
        sys.exit(f"{name} {case} {n} failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


def main():
    for name, tree in TREES.items():
        if not os.path.isdir(os.path.join(tree, "dist")):
            sys.exit(f"missing tree {name}: {tree}"
                     f"\nrun: python bench/parallel_builds.py <BASE_COMMIT>")
    hashes = {k: lib_hash(v) for k, v in TREES.items()}
    if len(set(hashes.values())) != len(hashes):
        sys.exit(f"reference trees are not distinct builds: {hashes}")
    for k, v in hashes.items():
        print(f"{k:8s} {v[:16]}  {TREES[k]}")
    print()
    print("| case | n | serial ms | before ms | new ms | new vs serial "
          "| new vs before |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in ("InversePairs", "DirectMany", "Positions"):
        for n in SIZES:
            got = {k: [] for k in TREES}
            # Two rounds, opposite order, so box drift lands on every side.
            for order in (("serial", "before", "new"),
                          ("new", "before", "serial")):
                for k in order:
                    got[k].append(measure(k, case, n)["ms"])
            m = {k: statistics.median(v) for k, v in got.items()}
            print(f"| {case} | {n} | {m['serial']:.3f} | {m['before']:.3f} | "
                  f"{m['new']:.3f} | {m['new'] / m['serial']:.2f}x | "
                  f"{m['new'] / m['before']:.2f}x |")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
