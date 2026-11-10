"""Interleaved `InversePairs` A/B: this tree against the pre-SIMD tree.

    pixi run bash -c 'flock /tmp/mojo-bench.lock python bench/inverse_ab.py'

The SIMD pass claims InversePairs at 100k comes out at 31.6-34.2 ms (new)
against 32.4-40.1 ms (old), minima of 25 runs per process.  That measurement
took the machine-wide lock; this script is the same comparison in one process,
two rounds (new,old then old,new) of REPEATS each, min per round, 100k points,
ALL mask, the bench's own input generator.

"Old" is 43354b2^: its src/ and its Python package, built into /tmp by
bench/parallel_builds.py's `make` helper is NOT reusable here (it is tied to
the parallel pass), so the tree is exported with `git archive` and built.  The
old package is imported under its own name, and both sides share the one
`libmojo-geographiclib.so` from the working tree only if the kernels did not
change -- parallel_builds.py confirms they did not (the .so hash is the same
before and after here), which is checked below.
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "43354b2^"
TREE = os.path.join(tempfile.gettempdir(), "b_pre_simd")
REPEATS = 25


def export():
    if os.path.isdir(TREE):
        return
    os.makedirs(TREE)
    blob = subprocess.run(["git", "-C", ROOT, "archive", BASE],
                          capture_output=True, check=True).stdout
    path = os.path.join(tempfile.gettempdir(), "pre_simd.tar")
    with open(path, "wb") as fh:
        fh.write(blob)
    with tarfile.open(path) as tf:
        tf.extractall(TREE, filter="data")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def arrs(n, seed=0):
    rng = np.random.default_rng(seed)
    return (np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n)))),
            np.ascontiguousarray(rng.uniform(-180, 180, n)),
            np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n)))),
            np.ascontiguousarray(rng.uniform(-180, 180, n)))


def run(geod, data, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        geod.InversePairs(*data, geod.ALL)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    export()
    old_pkg = os.path.join(TREE, "python")
    sys.path.insert(0, old_pkg)
    og = load("oldmojo.geodesic", os.path.join(old_pkg, "mojogeographiclib",
                                               "geodesic.py"))
    sys.path.pop(0)
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import mojogeographiclib.geodesic as ng  # noqa: E402

    o, n = og.Geodesic.WGS84, ng.Geodesic.WGS84
    d = arrs(100_000)
    a = o.InversePairs(*d, o.ALL)
    b = n.InversePairs(*d, n.ALL)
    for k in b:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    print("pre-SIMD and shipped InversePairs output is bit-identical at 100k")

    rounds = {"old": [], "new": []}
    for order in (("new", "old"), ("old", "new")):
        for side in order:
            rounds[side].append(run(n if side == "new" else o, d, REPEATS))
    for k, v in rounds.items():
        print(f"{k:4s} {', '.join(f'{x:.2f}' for x in v)} ms  "
              f"median {statistics.median(v):.2f}  min {min(v):.2f}")
    print(json.dumps({k: min(v) for k, v in rounds.items()}))


if __name__ == "__main__":
    main()
