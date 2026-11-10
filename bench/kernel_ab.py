"""Serial kernel A/B: the unrolled `_C1f` against the loop it replaced.

    pixi run bash -c 'flock /tmp/mojo-bench.lock python bench/kernel_ab.py'

Rebuilds the README's unroll table: one ctypes call into `mg_inverse` and
`mg_direct` with a preallocated output block, no `_result`, at 100k points.
"Before" is `c201134^` and "after" is `c201134`; `git diff` between
those two commits touches only src/ported.mojo (the unroll) and the docs.

Both libraries are built into /tmp and loaded one per child process, with the
sha256 of the loaded .so asserted inside the child, the same guard
bench/parallel_ab.py uses -- two libraries alternating in one process gave
wrong "serial" numbers in the parallel pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOJO = os.path.join(ROOT, ".pixi/envs/default/bin/mojo")
BASE = "c201134^"
TREES = {"before": "/tmp/k_before", "after": "/tmp/k_after"}
PACKAGE_DONOR = ROOT

CHILD = r"""
import hashlib, json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from mojogeographiclib import _lib
assert _lib.LIB == sys.argv[2], (_lib.LIB, sys.argv[2])
assert hashlib.sha256(open(_lib.LIB, "rb").read()).hexdigest() == sys.argv[3]
from mojogeographiclib.geodesic import Geodesic
g = Geodesic.WGS84
n = 100_000
rng = np.random.default_rng(0)
lat1 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
lon1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
lon2 = np.ascontiguousarray(rng.uniform(-180, 180, n))
azi1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
s12 = np.ascontiguousarray(rng.uniform(-2e7, 2e7, n))
out = np.empty(_lib.INV_N * n)
o = _lib.addr(out)
inv = _lib.fn("mg_inverse"); d = _lib.fn("mg_direct")
addr = _lib.addr
P = [addr(v) for v in (lat1, lon1, lat2, lon2, azi1, s12)]
a, f, blk = g.a, g.f, g._addr
def kern():
    inv(a, f, blk, P[0], P[1], P[2], P[3], g.ALL, o, n, 1)
    d(a, f, blk, P[0], P[1], P[4], P[5], g.ALL, o, n, 1)
kern()
best = float("inf")
for _ in range(5):
    t0 = time.perf_counter(); kern(); best = min(best, time.perf_counter() - t0)
print(json.dumps({"ms": best * 1e3}))
"""


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build(name, commit):
    tree = TREES[name]
    if os.path.isdir(tree):
        shutil.rmtree(tree)
    os.makedirs(os.path.join(tree, "src"))
    os.makedirs(os.path.join(tree, "dist"))
    for f in ("capi.mojo", "ported.mojo"):
        with open(os.path.join(tree, "src", f), "wb") as fh:
            fh.write(subprocess.run(["git", "-C", ROOT, "show",
                                     f"{commit}:src/{f}"],
                                    capture_output=True, check=True).stdout)
    so = os.path.join(tree, "dist", "libmojo-geographiclib.so")
    cmd = [MOJO, "build", "--emit", "shared-lib", "--fp-mode", "contract=off",
           "-I", os.path.join(tree, "src"),
           os.path.join(tree, "src", "capi.mojo"), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(so):
        sys.exit((r.stderr or r.stdout)[-3000:])
    shutil.copytree(os.path.join(PACKAGE_DONOR, "python", "mojogeographiclib"),
                    os.path.join(tree, "python", "mojogeographiclib"))
    return so


def measure(name):
    tree, so = TREES[name], os.path.join(TREES[name], "dist",
                                         "libmojo-geographiclib.so")
    env = dict(os.environ, PYTHONPATH="", MOJOGEOGRAPHICLIB_LIB=so)
    r = subprocess.run([sys.executable, "-c", CHILD,
                        os.path.join(tree, "python"), so, sha(so)],
                       capture_output=True, text=True, env=env, cwd="/tmp")
    if r.returncode != 0:
        sys.exit(r.stderr[-3000:])
    return json.loads(r.stdout)["ms"]


def main():
    before_so = build("before", BASE)
    after_so = build("after", "c201134")
    if sha(before_so) == sha(after_so):
        sys.exit("the two builds are identical; the A/B would be void")
    print(f"before {sha(before_so)[:16]}  after {sha(after_so)[:16]}")
    rows = {k: [] for k in TREES}
    for order in (("before", "after"), ("after", "before"),
                  ("before", "after")):
        for k in order:
            rows[k].append(measure(k))
    for k, v in rows.items():
        print(f"{k:6s} " + ", ".join(f"{x:.1f}" for x in v) +
              f"  (ms; median {statistics.median(v):.1f})")


if __name__ == "__main__":
    main()
