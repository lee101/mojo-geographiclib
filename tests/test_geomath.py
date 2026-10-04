"""Parity of the scalar Math mirror and its array forms with upstream."""

import math

import numpy as np
import pytest

import geographiclib.geomath as ug
from mojogeographiclib import geomath as mg

SCALARS = [
    0.0, 1.0, -1.0, 0.5, -0.5, 45.0, 90.0, -90.0, 180.0, -180.0, 359.9, -359.9,
    360.0, 720.0, 1e-12, -1e-12, 1e-200, 1e200, 1.0 / 16.0, 1.0 / 32.0,
    89.99999999999999, 90.00000000000001, 1e16, -1e16, float("inf"),
    float("-inf"), float("nan"), 2.5, 3.5, 15.0, 135.0, 225.0, 315.0,
]


def same(x, y):
    if isinstance(x, float) and x != x:
        return isinstance(y, float) and y != y
    return x == y and math.copysign(1.0, x) == math.copysign(1.0, y)


@pytest.mark.parametrize("x", SCALARS)
def test_ang_round(x):
    assert same(mg.Math.AngRound(x), ug.Math.AngRound(x))


@pytest.mark.parametrize("x", SCALARS)
def test_ang_normalize(x):
    assert same(mg.Math.AngNormalize(x), ug.Math.AngNormalize(x))


@pytest.mark.parametrize("x", SCALARS)
def test_lat_fix(x):
    assert same(mg.Math.LatFix(x), ug.Math.LatFix(x))


@pytest.mark.parametrize("x", SCALARS)
def test_remainder(x):
    assert same(mg.Math.remainder(x, 360.0), ug.Math.remainder(x, 360.0))


@pytest.mark.parametrize("x", SCALARS)
def test_sincosd(x):
    s1, c1 = mg.Math.sincosd(x)
    s2, c2 = ug.Math.sincosd(x)
    assert same(s1, s2)
    assert same(c1, c2)


@pytest.mark.parametrize("x", [v for v in SCALARS if math.isfinite(v)])
def test_sincosde(x):
    for t in (0.0, 1e-15, 0.5, -0.5, 1e-8):
        s1, c1 = mg.Math.sincosde(x, t)
        s2, c2 = ug.Math.sincosde(x, t)
        assert same(s1, s2), (x, t)
        assert same(c1, c2), (x, t)


@pytest.mark.parametrize("x", SCALARS)
def test_atan2d(x):
    for y in SCALARS:
        assert same(mg.Math.atan2d(x, y), ug.Math.atan2d(x, y)), (x, y)


@pytest.mark.parametrize("x", SCALARS)
@pytest.mark.parametrize("y", SCALARS)
def test_ang_diff(x, y):
    d1, t1 = mg.Math.AngDiff(x, y)
    d2, t2 = ug.Math.AngDiff(x, y)
    assert same(d1, d2), (x, y)
    assert same(t1, t2), (x, y)


def test_norm_and_sum():
    for x, y in [(3.0, 4.0), (1e-200, 1e-200), (1e200, 1e200), (-1.0, 2.0)]:
        assert mg.Math.norm(x, y) == ug.Math.norm(x, y)
        assert mg.Math.sum(x, y) == ug.Math.sum(x, y)


def test_cbrt():
    for x in SCALARS:
        assert same(mg.Math.cbrt(x), ug.Math.cbrt(x)), x


def test_polyval():
    p = [1.0, 2.0, 3.0, 4.0]
    for n, s in [(0, 0), (1, 0), (3, 0), (-1, 2), (2, 1)]:
        assert same(mg.Math.polyval(n, p, s, 0.5), ug.Math.polyval(n, p, s, 0.5))


# ------------------------------------------------------------ array forms
def test_array_forms_match_the_scalar_mirror():
    rng = np.random.default_rng(11)
    x = np.concatenate([
        rng.uniform(-400, 400, 4000),
        rng.uniform(-1e6, 1e6, 1000),
        np.array(SCALARS),
    ])
    y = np.concatenate([rng.uniform(-400, 400, x.size - len(SCALARS)),
                        np.array(SCALARS)])

    for fn, want in [(mg.ang_normalize, [ug.Math.AngNormalize(v) for v in x]),
                     (mg.lat_fix, [ug.Math.LatFix(v) for v in x])]:
        got = fn(x)
        for g, w in zip(got, want):
            assert same(float(g), w), (fn.__name__, g, w)

    got_d, got_t = mg.ang_diff(x, y)
    for i in range(x.size):
        d, t = ug.Math.AngDiff(float(x[i]), float(y[i]))
        assert same(float(got_d[i]), d), i
        assert same(float(got_t[i]), t), i

    for i in range(x.size):
        want = ug.Math.atan2d(float(x[i]), float(y[i]))
        assert same(float(mg.atan2d(x, y)[i]), want), i


def test_array_atan2d_all_quadrants():
    v = np.array([1.0, -1.0, 1e-300, -1e-300, 1e300, -1e300])
    for x in v:
        for y in v:
            assert same(float(mg.atan2d(x, y)), ug.Math.atan2d(float(x), float(y)))
