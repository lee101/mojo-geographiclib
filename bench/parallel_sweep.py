"""The measurements the parallel-pass thresholds in src/capi.mojo come from.

    pixi run bash -c 'flock /tmp/mojo-bench.lock python bench/parallel_sweep.py'

Three curves, all on fixed inputs:

1. worker curve  -- kernel-only time at 1, 2, 4, 8, 16 and 32 workers at 100k
   points, for every batch entry point.  MG_MAX_WORKERS is 4; this prints what
   the wider counts would do, which is the evidence behind the README's
   statement that the cap is where the gain stops.
2. size curve    -- user-visible InversePairs time, serial vs 4 workers, from
   64 to 200k points.
3. kernel size curve -- the same sizes, kernel-only and per point, so the
   crossover can be read without the Python glue or the result dict in it.

Kernel-only means one ctypes call into the exported function with
preallocated, already-warm buffers: no `_run`, no `_result`, no result dict,
no output copy.  The public call is what a caller actually pays, and it is
what `bench/parallel_ab.py` measures across builds -- prefer that script for
any decision, because a single process cannot fairly hold two libraries (see
its docstring).
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojogeographiclib import _lib  # noqa: E402
from mojogeographiclib.geodesic import Geodesic  # noqa: E402
from mojogeographiclib.geodesicline import GeodesicLine  # noqa: E402

g = Geodesic.WGS84
line = GeodesicLine(g, 37.0, -122.0, 51.0, g.ALL)


def data(n, seed=0):
    rng = np.random.default_rng(seed)
    return (np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n)))),
            np.ascontiguousarray(rng.uniform(-180, 180, n)),
            np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n)))),
            np.ascontiguousarray(rng.uniform(-180, 180, n)),
            np.ascontiguousarray(rng.uniform(-180, 180, n)),
            np.ascontiguousarray(rng.uniform(-2e7, 2e7, n)))


def best(fn, reps):
    b = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def geod_kernel(name, d, workers, reps):
    """One raw mg_inverse / mg_direct / mg_arcdirect call, no glue.

    `workers` is passed explicitly, so this measures the kernel at that count
    or at MG_MAX_WORKERS, whichever is smaller -- the clamp is the point of
    the worker curve.
    """
    n = d[0].size
    out = np.empty((_lib.INV_N if name == "mg_inverse" else _lib.DIR_N) * n,
                   dtype=np.float64)
    fn = _lib.fn(name)
    a, f, blk = g.a, g.f, g._addr
    add = _lib.addr
    o = add(out)
    x1, x2, x3, x4 = (add(v) for v in d)

    def call():
        fn(a, f, blk, x1, x2, x3, x4, g.ALL, o, n, workers)

    call()
    return best(call, reps)


def line_kernel(d, workers, reps, arcmode=False):
    """One raw mg_line_position / mg_line_arcposition call, no glue."""
    n = d.size
    out = np.empty(_lib.DIR_N * n, dtype=np.float64)
    fn = _lib.fn("mg_line_arcposition" if arcmode else "mg_line_position")
    o, s = _lib.addr(out), _lib.addr(d)
    a, f, blk, st = g.a, g.f, g._addr, line._addr
    caps = line.caps

    def call():
        fn(a, f, blk, st, s, caps, o, n, workers)

    call()
    return best(call, reps)


N = 100_000
lat1, lon1, lat2, lon2, azi1, s12 = data(N)
a12 = np.ascontiguousarray(np.linspace(-180.0, 180.0, N))
inv_d = (lat1, lon1, lat2, lon2)
dir_d = (lat1, lon1, azi1, s12)
arc_d = (lat1, lon1, azi1, a12)

print("machine:", platform.processor(), os.cpu_count(), "threads")
print("default workers on this Geodesic:", g._workers)
print()

print(f"## worker curve, kernel only, {N} points, ms and speedup vs 1 worker")
print(f"{'workers':>8} {'inverse':>16} {'direct':>16} {'arcdirect':>16} {'positions':>16}")
serial = None
for w in (1, 2, 4, 8, 16, 32):
    t = (geod_kernel("mg_inverse", inv_d, w, 5),
         geod_kernel("mg_direct", dir_d, w, 5),
         geod_kernel("mg_arcdirect", arc_d, w, 5),
         line_kernel(s12, w, 5))
    if serial is None:
        serial = t
    print(f"{w:8d} " + " ".join(f"{x * 1e3:9.2f} ({s / x:4.1f}x)"
                                 for x, s in zip(t, serial)))
print()

big = data(200_000, seed=1)
print("## size curve, public InversePairs (STANDARD mask)")
print(f"{'n':>8} {'serial ms':>12} {'4 workers ms':>14} {'speedup':>8}")
sizes = (64, 256, 512, 1000, 2048, 4096, 8192, 16384, 32768, 65536,
         100_000, 200_000)
for n in sizes:
    sl = tuple(v[:n] for v in big[:4])
    g._workers = 1
    s = best(lambda: g.InversePairs(*sl, g.STANDARD), 7)
    g._workers = 4
    p = best(lambda: g.InversePairs(*sl, g.STANDARD), 7)
    print(f"{n:8d} {s * 1e3:12.3f} {p * 1e3:14.3f} {s / p:7.2f}x")
print()

print("## kernel size curve, serial vs 4 workers, us per point")
print(f"{'n':>8} {'inv 1w':>9} {'inv 4w':>9} {'sp':>6} "
      f"{'dir 1w':>9} {'dir 4w':>9} {'sp':>6} {'pos 1w':>9} {'pos 4w':>9} {'sp':>6}")
for n in (256, 1000, 2048, 4096, 8192, 16384, 65536, 100_000):
    id_ = tuple(v[:n] for v in (lat1, lon1, lat2, lon2))
    dd = tuple(v[:n] for v in (lat1, lon1, azi1, s12))
    pd = s12[:n]
    a1 = geod_kernel("mg_inverse", id_, 1, 5)
    a4 = geod_kernel("mg_inverse", id_, 4, 5)
    b1 = geod_kernel("mg_direct", dd, 1, 5)
    b4 = geod_kernel("mg_direct", dd, 4, 5)
    c1 = line_kernel(pd, 1, 5)
    c4 = line_kernel(pd, 4, 5)
    print(f"{n:8d} {a1 * 1e6 / n:9.4f} {a4 * 1e6 / n:9.4f} {a1 / a4:5.2f}x "
          f"{b1 * 1e6 / n:9.4f} {b4 * 1e6 / n:9.4f} {b1 / b4:5.2f}x "
          f"{c1 * 1e6 / n:9.4f} {c4 * 1e6 / n:9.4f} {c1 / c4:5.2f}x")
