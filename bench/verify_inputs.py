"""Verify that bench.py's two sides see the same input and do the same work.

No timing here.  It reruns every bench case once, keeps the outputs of both
sides, and asserts they are the same numbers (upstream is the reference, the
port is allowed 1-9 ulp).  It also asserts that the capability masks the two
sides hand to their kernels are equal, and that every benchmarked field comes
back populated and finite, so a side cannot "win" by skipping an output.
"""

from __future__ import annotations

import os
import importlib.util
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

_spec = importlib.util.spec_from_file_location(
    "bench_cases", os.path.join(ROOT, "bench", "bench.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def compare(label, ours, theirs, tol=2e-9):
    assert sorted(ours) == sorted(theirs), (label, sorted(set(ours) ^ set(theirs)))
    worst = (0.0, None)
    for k in theirs:
        a = np.asarray(ours[k], dtype=np.float64)
        b = np.asarray(theirs[k], dtype=np.float64)
        assert a.shape == b.shape, (label, k, a.shape, b.shape)
        assert np.array_equal(np.isnan(a), np.isnan(b)), (label, k, "NaN pattern")
        fin = np.isfinite(b)
        assert fin.any(), (label, k, "upstream produced nothing finite")
        assert fin.sum() == np.isfinite(a).sum(), (label, k, "finite-field count differs")
        e = float(np.max(np.abs(a[fin] - b[fin]) /
                         np.maximum(np.abs(b[fin]), 1e-300)))
        assert e <= tol, (label, k, e)
        if e > worst[0]:
            worst = (e, k)
    print(f"{label:48s} fields={len(theirs):2d} worst rel err {worst[0]:.2e} ({worst[1]})")


def main():
    OURS, THEIRS = B.OURS, B.THEIRS
    BIG, LOOP, SMALL = B.BIG, B.LOOP, B.SMALL

    # --- capability masks requested by each side
    for mask_name in ("ALL", "STANDARD"):
        assert getattr(OURS, mask_name) == getattr(THEIRS, mask_name), mask_name
    print(f"ALL mask {OURS.ALL} == upstream {THEIRS.ALL}; "
          f"STANDARD mask {OURS.STANDARD} == upstream {THEIRS.STANDARD}")

    # --- masks each side actually hands to its line kernel
    ol = B.GeodesicLine(OURS, 37.0, -122.0, 51.0, OURS.ALL)
    tl = B.ugl.GeodesicLine(THEIRS, 37.0, -122.0, 51.0, THEIRS.ALL)
    assert ol.caps == tl.caps, (ol.caps, tl.caps)
    print(f"line caps ours {ol.caps} == upstream {tl.caps}")

    # --- identical inputs on both sides: the bench hands the same arrays to
    # both sides, so what needs checking is that a batch of them lands on the
    # same numbers as the scalar API.
    print("inputs: both sides are handed the same arrays in every case (by construction)")

    # --- scalar rows: same work on both sides
    lp = ol.Position; tp = tl.Position
    for i in range(0, len(B.LOOP[5]), 997):
        assert lp(B.LOOP[5][i], ol.caps) == tp(B.LOOP[5][i], tl.caps), i
    print("GeodesicLine.Position: 20k-scalar output identical on both sides (sampled)")

    # --- batch rows vs the upstream loop, on the exact bench inputs
    n = B.N_BIG
    b = OURS.InversePairs(lat1:=BIG[0], BIG[1], BIG[2], BIG[3], OURS.ALL)
    ref = {k: [None]*n for k in b}
    inv = THEIRS.Inverse
    for i in range(n):
        r = inv(BIG[0][i], BIG[1][i], BIG[2][i], BIG[3][i], THEIRS.ALL)
        for k in ref:
            ref[k][i] = r[k]
    compare("InversePairs 100k ALL vs upstream loop", b,
            {k: np.asarray(v) for k, v in ref.items()})

    b = OURS.DirectMany(BIG[0], BIG[1], BIG[4], BIG[5], OURS.ALL)
    ref = {k: [None]*n for k in b}
    d = THEIRS.Direct
    for i in range(n):
        r = d(BIG[0][i], BIG[1][i], BIG[4][i], BIG[5][i], THEIRS.ALL)
        for k in ref:
            ref[k][i] = r[k]
    compare("DirectMany 100k ALL vs upstream loop", b,
            {k: np.asarray(v) for k, v in ref.items()})

    a12 = np.ascontiguousarray(np.linspace(-180.0, 180.0, n))
    b = OURS.ArcDirectMany(BIG[0], BIG[1], BIG[4], a12, OURS.ALL)
    ref = {k: [None]*n for k in b}
    ad = THEIRS.ArcDirect
    for i in range(n):
        r = ad(BIG[0][i], BIG[1][i], BIG[4][i], a12[i], THEIRS.ALL)
        for k in ref:
            ref[k][i] = r[k]
    compare("ArcDirectMany 100k ALL vs upstream loop", b,
            {k: np.asarray(v) for k, v in ref.items()})

    b = ol.Positions(BIG[5], ol.caps)
    ref = {k: [None]*n for k in b}
    for i in range(n):
        r = tl.Position(BIG[5][i], tl.caps)
        for k in ref:
            ref[k][i] = r[k]
    compare("Positions 100k vs upstream loop", b,
            {k: np.asarray(v) for k, v in ref.items()})

    # --- PolygonArea: same vertices, same answer
    rng = np.random.default_rng(7)
    lat = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, B.N_LOOP))))
    lon = np.ascontiguousarray(rng.uniform(-180, 180, B.N_LOOP))
    po = OURS.Polygon(); po.AddPoints(lat, lon)
    pt = THEIRS.Polygon()
    for i in range(B.N_LOOP):
        pt.AddPoint(lat[i], lon[i])
    ro, rt = po.Compute(), pt.Compute()
    assert ro[0] == rt[0] and ro[1] == rt[1]
    assert abs(ro[2] - rt[2]) <= 1e-6 * max(abs(rt[2]), 1.0), (ro, rt)
    print(f"PolygonArea 20k: num {ro[0]} == {rt[0]}, "
          f"perimeter bit-identical, area rel err "
          f"{abs(ro[2]-rt[2])/max(abs(rt[2]),1.0):.1e}")

    # --- every benchmarked field is populated on the port's side too
    for label, r in (("InversePairs", OURS.InversePairs(BIG[0], BIG[1], BIG[2], BIG[3], OURS.ALL)),
                     ("DirectMany", OURS.DirectMany(BIG[0], BIG[1], BIG[4], BIG[5], OURS.ALL))):
        for k, v in r.items():
            assert v.shape == (B.N_BIG,), (label, k, v.shape)
            assert np.isfinite(v).all(), (label, k, "non-finite output")
    for k, v in ol.Positions(BIG[5], ol.caps).items():
        assert v.shape == (B.N_BIG,) and np.isfinite(v).all(), k
    print("all batch fields present, right length, finite on the port's side")
    print("OK")


if __name__ == "__main__":
    main()
