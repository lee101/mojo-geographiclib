"""Where the time goes, per code path.  Not part of the benchmark table.

    pixi run python bench/profile.py

`bench/bench.py` compares against upstream; this one is for finding out which
part of this port to work on next.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojogeographiclib import _lib
from mojogeographiclib.geodesic import Geodesic
from mojogeographiclib.geodesiccapability import GeodesicCapability
from mojogeographiclib.geodesicline import GeodesicLine
from mojogeographiclib.geomath import Math, ang_diff, ang_normalize, atan2d, lat_fix

REPS = 20_000


def us(fn, reps=REPS, rounds=5):
    best = math.inf
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6 / reps


g = Geodesic.WGS84
line = GeodesicLine(g, 37.0, -122.0, 51.0, g.ALL)
om = Geodesic.ALL
W = _lib.io("profile", 64)
inv = _lib.fn("mg_inverse")
pos = _lib.fn("mg_line_position")
a, f, blk, lstate = g.a, g.f, g._addr, line._addr

print("single-point path")
print(f"  mg_inverse, n=0 (pure FFI)     {us(lambda: inv(a, f, blk, W.addr, W.addr + 8, W.addr + 16, W.addr + 24, om, W.addr + 32, 0, 1)):7.3f} us")
print(f"  mg_inverse, n=1                {us(lambda: inv(a, f, blk, W.addr, W.addr + 8, W.addr + 16, W.addr + 24, om, W.addr + 32, 1, 1)):7.3f} us")
print(f"  mg_line_position, n=0          {us(lambda: pos(a, f, blk, lstate, W.addr, om, W.addr + 8, 0, 1)):7.3f} us")
print(f"  mg_line_position, n=1          {us(lambda: pos(a, f, blk, lstate, W.addr, om, W.addr + 8, 1, 1)):7.3f} us")
print(f"  struct.pack_into <4d           {us(lambda: _lib.pack('<4d', W.mem, 0, 1.0, 2.0, 3.0, 4.0)):7.3f} us")
print(f"  struct.unpack_from <10d        {us(lambda: _lib.unpack('<10d', W.mem, 32)):7.3f} us")
print(f"  Math.LatFix                    {us(lambda: Math.LatFix(1.0)):7.3f} us")
print(f"  Math.AngNormalize              {us(lambda: Math.AngNormalize(1.0)):7.3f} us")
print(f"  Math.atan2d                    {us(lambda: Math.atan2d(1.0, 2.0)):7.3f} us")
print(f"  Geodesic.Inverse               {us(lambda: g.Inverse(1.0, 2.0, 3.0, 4.0, om)):7.3f} us")
print(f"  GeodesicLine.Position          {us(lambda: line.Position(1.0e6)):7.3f} us")

N = 100_000
rng = np.random.default_rng(0)
lat1 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, N))))
lon1 = np.ascontiguousarray(rng.uniform(-180, 180, N))
lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, N))))
lon2 = np.ascontiguousarray(rng.uniform(-180, 180, N))
azi1 = np.ascontiguousarray(rng.uniform(-180, 180, N))
s12 = np.ascontiguousarray(rng.uniform(-2e7, 2e7, N))


def ms(fn, reps=7):
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


out = g._run("mg_inverse", (lat1, lon1, lat2, lon2), om, 10)
salp1, calp1, salp2, calp2 = out[2], out[3], out[4], out[5]
workers = g._workers

print(f"\nbatch path, {N} points")
print(f"  InversePairs, all parallel     {ms(lambda: g.InversePairs(lat1, lon1, lat2, lon2, om)):7.2f} ms")
g._workers = 1
print(f"  InversePairs, serial kernel    {ms(lambda: g.InversePairs(lat1, lon1, lat2, lon2, om)):7.2f} ms")
g._workers = workers
print(f"  _run (kernel + copy + slice)   {ms(lambda: g._run('mg_inverse', (lat1, lon1, lat2, lon2), om, 10)):7.2f} ms")
print(f"  _result (numpy post)           {ms(lambda: g._result(om, out, (lat1, lon1, lat2), lon2, lon1, lat1, lat2, True)):7.2f} ms")
print(f"    lat_fix x2                   {ms(lambda: (lat_fix(lat1), lat_fix(lat2))):7.2f} ms")
print(f"    ang_diff                     {ms(lambda: ang_diff(lon1, lon2)):7.2f} ms")
print(f"    atan2d x2                    {ms(lambda: (atan2d(salp1, calp1), atan2d(salp2, calp2))):7.2f} ms")
print(f"    np.arctan2                   {ms(lambda: np.arctan2(salp1, calp1)):7.2f} ms")
print(f"  DirectMany, all parallel       {ms(lambda: g.DirectMany(lat1, lon1, azi1, s12, om)):7.2f} ms")
print(f"  Positions, all parallel        {ms(lambda: line.Positions(s12, line.caps)):7.2f} ms")
