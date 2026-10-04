"""mojo-geographiclib against the upstream geographiclib package.

    pixi run bench

The upstream package has no batch API, so every "upstream" number is a plain
Python loop over the same inputs; that is the honest comparison for a drop-in
replacement, and it is why the batch rows are so lopsided.  The scalar rows loop
on both sides and are the fair test of the single-point path.
"""

from __future__ import annotations

import math
import os
import platform
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import geographiclib.geodesic as ug  # noqa: E402
import geographiclib.geodesicline as ugl  # noqa: E402

from mojogeographiclib.geodesic import Geodesic  # noqa: E402
from mojogeographiclib.geodesicline import GeodesicLine  # noqa: E402

REPEAT = 3


def timeit(fn) -> float:
    best = math.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    lat1 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon2 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    azi1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    s12 = np.ascontiguousarray(rng.uniform(-2e7, 2e7, n))
    return lat1, lon1, lat2, lon2, azi1, s12


CASES = []


def case(name, note=""):
    def deco(fn):
        CASES.append((name, note, fn))
        return fn
    return deco


N_BIG = 100_000
N_LOOP = 20_000
BIG = data(N_BIG)
LOOP = data(N_LOOP, seed=1)

OURS = Geodesic.WGS84
THEIRS = ug.Geodesic.WGS84


@case("Inverse, 20k scalar calls (Python loop)", "scalar API, STANDARD mask")
def _():
    lat1, lon1, lat2, lon2 = LOOP[0], LOOP[1], LOOP[2], LOOP[3]
    o = OURS.ALL

    def ours():
        inv = OURS.Inverse
        for i in range(len(lat1)):
            inv(lat1[i], lon1[i], lat2[i], lon2[i], o)

    def theirs():
        inv = THEIRS.Inverse
        for i in range(len(lat1)):
            inv(lat1[i], lon1[i], lat2[i], lon2[i], o)

    return ours, theirs


@case("Inverse, 100k points, all outputs", "batch API vs upstream loop")
def _():
    lat1, lon1, lat2, lon2 = BIG[0], BIG[1], BIG[2], BIG[3]
    o = OURS.ALL

    def ours():
        OURS.InversePairs(lat1, lon1, lat2, lon2, o)

    def theirs():
        inv = THEIRS.Inverse
        for i in range(N_BIG):
            inv(lat1[i], lon1[i], lat2[i], lon2[i], o)

    return ours, theirs


@case("Inverse, 100k points, STANDARD mask", "batch API vs upstream loop")
def _():
    lat1, lon1, lat2, lon2 = BIG[0], BIG[1], BIG[2], BIG[3]
    o = OURS.STANDARD

    def ours():
        OURS.InversePairs(lat1, lon1, lat2, lon2, o)

    def theirs():
        inv = THEIRS.Inverse
        for i in range(N_BIG):
            inv(lat1[i], lon1[i], lat2[i], lon2[i], o)

    return ours, theirs


@case("Direct, 100k points, all outputs", "batch API vs upstream loop")
def _():
    lat1, lon1, azi1, s12 = BIG[0], BIG[1], BIG[4], BIG[5]
    o = OURS.ALL

    def ours():
        OURS.DirectMany(lat1, lon1, azi1, s12, o)

    def theirs():
        d = THEIRS.Direct
        for i in range(N_BIG):
            d(lat1[i], lon1[i], azi1[i], s12[i], o)

    return ours, theirs


@case("ArcDirect, 100k points, all outputs", "batch API vs upstream loop")
def _():
    lat1, lon1, azi1 = BIG[0], BIG[1], BIG[4]
    a12 = np.ascontiguousarray(np.linspace(-180.0, 180.0, N_BIG))
    o = OURS.ALL

    def ours():
        OURS.ArcDirectMany(lat1, lon1, azi1, a12, o)

    def theirs():
        d = THEIRS.ArcDirect
        for i in range(N_BIG):
            d(lat1[i], lon1[i], azi1[i], a12[i], o)

    return ours, theirs


@case("GeodesicLine.Position, 20k calls", "scalar API, one line")
def _():
    line_ours = GeodesicLine(OURS, 37.0, -122.0, 51.0, OURS.ALL)
    line_theirs = ugl.GeodesicLine(THEIRS, 37.0, -122.0, 51.0,
                                               THEIRS.ALL)
    d = LOOP[5]

    def ours():
        p = line_ours.Position
        for i in range(len(d)):
            p(d[i])

    def theirs():
        p = line_theirs.Position
        for i in range(len(d)):
            p(d[i])

    return ours, theirs


@case("GeodesicLine.Positions, 100k", "batch API vs upstream loop")
def _():
    line_ours = GeodesicLine(OURS, 37.0, -122.0, 51.0, OURS.ALL)
    line_theirs = ugl.GeodesicLine(THEIRS, 37.0, -122.0, 51.0,
                                               THEIRS.ALL)
    d = BIG[5]

    def ours():
        line_ours.Positions(d, line_ours.caps)

    def theirs():
        p = line_theirs.Position
        for i in range(N_BIG):
            p(d[i])

    return ours, theirs


@case("PolygonArea, 20k vertices (AddPoints)", "batch API vs upstream loop")
def _():
    rng = np.random.default_rng(7)
    lat = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, N_LOOP))))
    lon = np.ascontiguousarray(rng.uniform(-180, 180, N_LOOP))

    def ours():
        p = OURS.Polygon()
        p.AddPoints(lat, lon)
        p.Compute()

    def theirs():
        p = THEIRS.Polygon()
        for i in range(N_LOOP):
            p.AddPoint(lat[i], lon[i])
        p.Compute()

    return ours, theirs


@case("PolygonArea, 20k vertices (AddPoint)", "scalar API on both sides")
def _():
    rng = np.random.default_rng(7)
    lat = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, N_LOOP))))
    lon = np.ascontiguousarray(rng.uniform(-180, 180, N_LOOP))

    def ours():
        p = OURS.Polygon()
        add = p.AddPoint
        for i in range(N_LOOP):
            add(lat[i], lon[i])
        p.Compute()

    def theirs():
        p = THEIRS.Polygon()
        add = p.AddPoint
        for i in range(N_LOOP):
            add(lat[i], lon[i])
        p.Compute()

    return ours, theirs


def machine() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            model = re.search(r"^model name\s*:\s*(.+)$", fh.read(), re.M)
        cpu = model.group(1).strip() if model else platform.processor()
    except OSError:
        cpu = platform.processor()
    return f"{cpu}, {os.cpu_count()} threads, {platform.system()} {platform.machine()}"


def main() -> None:
    print(f"# mojo-geographiclib benchmarks")
    print()
    print(f"machine: {machine()}")
    print(f"python: {platform.python_version()}, numpy: {np.__version__}")
    print(f"timing: best of {REPEAT}")
    print()
    print("| case | note | mojo-geographiclib | upstream geographiclib | speedup |")
    print("| --- | --- | ---: | ---: | ---: |")
    for name, note, build in CASES:
        ours, theirs = build()
        ours()  # warm the shared library out of the measurement
        a = timeit(ours)
        b = timeit(theirs)
        print(f"| {name} | {note} | {a * 1e3:.2f} ms | {b * 1e3:.2f} ms | "
              f"{b / a:.1f}x |")


if __name__ == "__main__":
    main()
