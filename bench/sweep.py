"""The two measurements the batch thresholds in src/capi.mojo are set from.

    pixi run bash -c 'flock /tmp/mojo-bench.lock python bench/sweep.py'

MG_MAX_WORKERS (64) is where the worker curve stops paying; MG_MIN_PARALLEL
(8192) is where a parallel launch stops costing more than it saves.  Both
are quoted in the README, so they are reproducible here.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojogeographiclib.geodesic import Geodesic
from mojogeographiclib.geodesicline import GeodesicLine

g = Geodesic.WGS84
line = GeodesicLine(g, 37.0, -122.0, 51.0, g.ALL)
DEFAULT_WORKERS = g._workers


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


N = 100_000
lat1, lon1, lat2, lon2, azi1, s12 = data(N)

print(f"worker sweep, {N} points")
print(f"{'workers':>8} {'inv ALL':>10} {'inv STD':>10} {'direct':>10} {'positions':>10}")
for w in (1, 2, 4, 8, 16, 24, 32, 48, 64, 72):
    g._workers = w
    a = best(lambda: g.InversePairs(lat1, lon1, lat2, lon2, g.ALL), 5) * 1e3
    b = best(lambda: g.InversePairs(lat1, lon1, lat2, lon2, g.STANDARD), 5) * 1e3
    c = best(lambda: g.DirectMany(lat1, lon1, azi1, s12, g.ALL), 5) * 1e3
    d = best(lambda: line.Positions(s12, line.caps), 5) * 1e3
    print(f"{w:8d} {a:10.2f} {b:10.2f} {c:10.2f} {d:10.2f}")
g._workers = DEFAULT_WORKERS

print("\nsize sweep, us per point (inverse, STANDARD mask)")
print(f"{'n':>8} {'serial':>10} {'parallel':>10} {'speedup':>8}")
big = data(200_000)
for n in (64, 256, 1024, 4096, 8191, 8192, 16384, 65536, 200_000):
    g._workers = 1
    s = best(lambda: g.InversePairs(big[0][:n], big[1][:n], big[2][:n],
                                    big[3][:n], g.STANDARD), 7) * 1e6 / n
    g._workers = DEFAULT_WORKERS
    p = best(lambda: g.InversePairs(big[0][:n], big[1][:n], big[2][:n],
                                    big[3][:n], g.STANDARD), 7) * 1e6 / n
    print(f"{n:8d} {s:10.4f} {p:10.4f} {s / p:8.2f}x")
