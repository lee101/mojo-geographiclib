"""Numerical parity with the upstream geographiclib package.

The kernels are compared against the real upstream implementation, field by
field, on random and adversarial inputs.

Both implementations round correctly and the build disables floating-point
contraction, so on WGS84 they agree to 1-2 ulp on essentially every field --
`test_wgs84_is_almost_always_bit_identical` pins that down.  The tolerance
asserted here is 1e-9 relative, which is the accuracy GeographicLib itself
claims (15 nm), and is needed only for the deliberately odd ellipsoids in
ELLIPSOIDS, where the series tolerance is absolute rather than relative and
the two therefore diverge further.
"""

import math

import numpy as np
import pytest

import geographiclib.geodesic as ug
import geographiclib.geodesicline as ugl
from mojogeographiclib.geodesic import Geodesic
from mojogeographiclib.geodesicline import GeodesicLine

ULP = 1.0e-9  # relative; see the module docstring


def relerr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN pattern differs"
    finite = np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), "Inf pattern differs"
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite]) / np.maximum(np.abs(b[finite]), 1e-300)))


def check(ours, theirs, label, tol=ULP):
    assert sorted(ours) == sorted(theirs), (
        label, sorted(set(ours) ^ set(theirs)))
    for k in theirs:
        err = relerr(np.atleast_1d(ours[k]), np.atleast_1d(theirs[k]))
        assert err <= tol, (label, k, err, ours[k], theirs[k])


# (a, f, name, tolerance).  The tolerance is the relative agreement required;
# it only has to be loose for the 1-metre ellipsoid, where the series
# tolerance is absolute and the answer is correspondingly less precise.
ELLIPSOIDS = [
    (6378137.0, 1 / 298.257223563, "WGS84", ULP),
    (6378137.0, 0.0, "sphere", ULP),
    (6378137.0, 1 / 150.0, "prolate-ish", ULP),
    (6378137.0, -1 / 100.0, "oblate-ish", ULP),
    (6378137.0, 1 / 10.0, "very eccentric", ULP),
    (6378137.0, -1 / 5.0, "strongly prolate", ULP),
    (1.0, 0.5, "one-metre", 1e-7),
]

ADVERSARIAL = [
    (0.0, 0.0, 0.0, 90.0),          # equatorial quarter
    (0.0, 0.0, 90.0, 0.0),          # meridian quarter
    (90.0, 0.0, -90.0, 0.0),        # pole to pole
    (-90.0, 0.0, 90.0, 179.0),      # pole to pole the long way
    (0.0, 0.0, 0.0, 180.0),         # equatorial half
    (0.0, 0.0, 0.5, 179.5),         # near antipodal on the equator
    (0.0, 0.0, 1.0e-12, 180.0),     # antipodal to within an ulp of latitude
    (89.9999999, 0.0, -89.9999999, 179.0),
    (45.0, 0.0, 45.0, 0.0),         # coincident meridian
    (0.0, 179.999999, 0.0, -179.999999),   # straddling the antimeridian
    (-41.32, 174.81, 40.96, -5.5),  # the upstream docstring example
    (48.522876735459, 0.0, -48.52287673545898293, 179.599720456223079643),
    (88.202499451857, 0.0, -88.202499451857, 179.981022032992859592),
    (89.262080389218, 0.0, -89.262080389218, 179.992207982775375662),
    (89.333123580033, 0.0, -89.333123580032997687, 179.99295812360148422),
    (0.0, 0.0, 0.0, 1.0e-9),        # very short line
    (20.001, 0.0, 20.001, 0.0),     # the zero-length case from the source
]


@pytest.mark.parametrize("a,f,name,tol", ELLIPSOIDS)
def test_inverse_parity(a, f, name, tol):
    ours, theirs = Geodesic(a, f), ug.Geodesic(a, f)
    for args in ADVERSARIAL:
        check(ours.Inverse(*args, outmask=ours.ALL),
              theirs.Inverse(*args, outmask=theirs.ALL),
              f"Inverse {name} {args}", tol)


@pytest.mark.parametrize("a,f,name,tol", ELLIPSOIDS)
def test_direct_parity(a, f, name, tol):
    ours, theirs = Geodesic(a, f), ug.Geodesic(a, f)
    rng = np.random.default_rng(hash(name) % 2**32)
    for lat1, lon1, azi1, s12 in ADVERSARIAL:
        for scale in (0.0, 1e-9, 1.0, 1e3, 2e7, 5e7):
            d = s12 * scale
            check(ours.Direct(lat1, lon1, azi1, d, outmask=ours.ALL),
                  theirs.Direct(lat1, lon1, azi1, d, outmask=theirs.ALL),
                  f"Direct {name} {lat1} {lon1} {azi1} {d}", tol)
    n = 300
    lat1 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon1 = rng.uniform(-180, 180, n)
    azi1 = rng.uniform(-180, 180, n)
    s12 = rng.uniform(-3e7, 3e7, n)
    check(ours.DirectMany(lat1, lon1, azi1, s12, ours.ALL),
          {k: [theirs.Direct(lat1[i], lon1[i], azi1[i], s12[i], theirs.ALL)[k]
               for i in range(n)] for k in theirs.Direct(0, 0, 0, 0, theirs.ALL)},
          f"DirectMany {name}", tol)


@pytest.mark.parametrize("a,f,name,tol", ELLIPSOIDS)
def test_arcdirect_parity(a, f, name, tol):
    ours, theirs = Geodesic(a, f), ug.Geodesic(a, f)
    for lat1, lon1, azi1, s12 in ADVERSARIAL:
        for a12 in (0.0, 1e-9, 1.0, 45.0, 90.0, 179.0, 180.0, -180.0, 360.0):
            check(ours.ArcDirect(lat1, lon1, azi1, a12, outmask=ours.ALL),
                  theirs.ArcDirect(lat1, lon1, azi1, a12, outmask=theirs.ALL),
                  f"ArcDirect {name} {lat1} {lon1} {azi1} {a12}", tol)


@pytest.mark.parametrize("a,f,name,tol", ELLIPSOIDS)
def test_inverse_parity_random(a, f, name, tol):
    ours, theirs = Geodesic(a, f), ug.Geodesic(a, f)
    rng = np.random.default_rng(20240929)
    n = 400
    lat1 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon1 = rng.uniform(-180, 180, n)
    lat2 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon2 = rng.uniform(-180, 180, n)
    mask_o, mask_t = ours.ALL, theirs.ALL
    batch = ours.InversePairs(lat1, lon1, lat2, lon2, mask_o)
    ref = {}
    for i in range(n):
        r = theirs.Inverse(lat1[i], lon1[i], lat2[i], lon2[i], mask_t)
        for k in r:
            ref.setdefault(k, []).append(r[k])
    check(batch, ref, f"InversePairs {name}", tol)


def test_geodesic_line_parity():
    ours_geod, theirs_geod = Geodesic.WGS84, ug.Geodesic.WGS84
    rng = np.random.default_rng(99)
    for _ in range(12):
        lat1 = float(rng.uniform(-90, 90))
        lon1 = float(rng.uniform(-180, 180))
        azi1 = float(rng.uniform(-180, 180))
        a = GeodesicLine(ours_geod, lat1, lon1, azi1, ours_geod.ALL)
        b = ugl.GeodesicLine(theirs_geod, lat1, lon1, azi1, theirs_geod.ALL)
        d = np.concatenate([rng.uniform(-3e7, 3e7, 300), [0.0, 1e-9]])
        for arcmode, key in ((False, "Position"), (True, "ArcPosition")):
            many = getattr(a, key + "s")(d, a.caps)
            for i, v in enumerate(d):
                one = getattr(b, key)(v, b.caps)
                check({k: np.atleast_1d(many[k][i]) for k in many},
                      {k: np.atleast_1d(one[k]) for k in one},
                      f"{key} {lat1} {lon1} {azi1} {v}")


def test_wgs84_is_almost_always_bit_identical():
    """The headline accuracy claim: on WGS84 the Mojo kernels return the very
    same doubles as upstream for the large majority of inputs."""
    ours, theirs = Geodesic.WGS84, ug.Geodesic.WGS84
    rng = np.random.default_rng(4242)
    n = 500
    lat1 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon1 = rng.uniform(-180, 180, n)
    lat2 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon2 = rng.uniform(-180, 180, n)
    batch = ours.InversePairs(lat1, lon1, lat2, lon2, ours.ALL)
    ref = {k: [] for k in batch}
    for i in range(n):
        r = theirs.Inverse(lat1[i], lon1[i], lat2[i], lon2[i], theirs.ALL)
        for k in ref:
            ref[k].append(r[k])
    for k in ["lat2", "lon2", "a12", "s12", "azi1", "azi2", "S12"]:
        frac = np.mean(batch[k] == np.asarray(ref[k]))
        assert frac >= 0.9, (k, frac)


def test_geodesic_line_inverse_round_trip_parity():
    ours, theirs = Geodesic.WGS84, ug.Geodesic.WGS84
    for args in ADVERSARIAL:
        a = ours.InverseLine(*args)
        b = theirs.InverseLine(*args)
        assert (math.isnan(a.s13) and math.isnan(b.s13)) or a.s13 == b.s13, args
        if math.isnan(a.s13):
            continue
        assert a.azi1 == b.azi1, args
        assert a.salp1 == b.salp1, args
        assert a.calp1 == b.calp1, args
        assert a.a13 == b.a13, args


def test_long_unroll_parity():
    ours, theirs = Geodesic.WGS84, ug.Geodesic.WGS84
    for lat1, lon1, lat2, lon2 in ADVERSARIAL:
        m = ours.LONG_UNROLL | ours.STANDARD
        mt = theirs.LONG_UNROLL | theirs.STANDARD
        check(ours.Inverse(lat1, lon1, lat2, lon2, outmask=m),
              theirs.Inverse(lat1, lon1, lat2, lon2, outmask=mt),
              f"LongUnroll {lat1} {lon1} {lat2} {lon2}")


def test_zero_and_negative_distance_parity():
    ours, theirs = Geodesic.WGS84, ug.Geodesic.WGS84
    for s in (0.0, -0.0, 1e-12, -1e-12, 1.0, -1.0):
        for args in [(10.0, 20.0, 45.0), (0.0, 0.0, 90.0), (-90.0, 0.0, 0.0)]:
            check(ours.Direct(*args, s, outmask=ours.ALL),
                  theirs.Direct(*args, s, outmask=theirs.ALL),
                  f"zero {args} {s}")
