"""PolygonArea parity with upstream, including the accumulator's long sums."""

import math

import numpy as np
import pytest

import geographiclib.geodesic as ug
from mojogeographiclib.geodesic import Geodesic

OURS = Geodesic.WGS84
THEIRS = ug.Geodesic.WGS84

POLYGONS = {
    "octagon": [(0, 0), (0, 90), (45, 45), (90, 0), (90, -45), (45, -45),
                (0, -90), (-45, -45), (-90, 0), (-45, 45)],
    "antimeridian": [(0, 170), (0, -170), (10, -175), (-10, 175)],
    "polar": [(89.9, 0), (89.9, 120), (-89.9, -120), (-89.9, 60)],
    "small": [(10, 10), (10.001, 10), (10.001, 10.001), (10, 10.001)],
    "self-intersecting": [(0, 0), (0, 90), (90, 45), (90, -45), (0, 90),
                          (0, 0)],
    "equator": [(0, -170), (0, -90), (0, 0), (0, 90), (0, 170)],
    "hemisphere": [(0, 0), (0, 90), (0, 180), (0, -90)],
}


def build(pts, polyline=False):
    a, b = OURS.Polygon(polyline), THEIRS.Polygon(polyline)
    for lat, lon in pts:
        a.AddPoint(lat, lon)
        b.AddPoint(lat, lon)
    return a, b


@pytest.mark.parametrize("name", sorted(POLYGONS))
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sign", [False, True])
def test_compute_parity(name, reverse, sign):
    a, b = build(POLYGONS[name])
    c1 = a.Compute(reverse, sign)
    c2 = b.Compute(reverse, sign)
    assert c1[0] == c2[0]
    assert c1[1] == pytest.approx(c2[1], rel=1e-12, abs=1e-6)
    assert c1[2] == pytest.approx(c2[2], rel=1e-12, abs=1e-6)


@pytest.mark.parametrize("name", sorted(POLYGONS))
def test_polyline_parity(name):
    a, b = build(POLYGONS[name], polyline=True)
    c1, c2 = a.Compute(), b.Compute()
    assert c1[0] == c2[0]
    assert c1[1] == pytest.approx(c2[1], rel=1e-12, abs=1e-6)
    assert math.isnan(c1[2]) and math.isnan(c2[2])


@pytest.mark.parametrize("name", sorted(POLYGONS))
def test_test_point_parity(name):
    a, b = build(POLYGONS[name])
    for lat, lon in [(0.0, 0.0), (12.5, -77.25), (-89.0, 179.0), (89.0, -179.0)]:
        for reverse in (False, True):
            for sign in (False, True):
                c1, c2 = a.TestPoint(lat, lon, reverse, sign), b.TestPoint(lat, lon, reverse, sign)
                assert c1[0] == c2[0]
                assert c1[1] == pytest.approx(c2[1], rel=1e-12, abs=1e-6)
                assert c1[2] == pytest.approx(c2[2], rel=1e-12, abs=1e-6)


@pytest.mark.parametrize("name", sorted(POLYGONS))
def test_test_edge_parity(name):
    a, b = build(POLYGONS[name])
    for azi, s in [(45.0, 1.0e6), (0.0, 0.0), (-120.0, -5.0e5), (179.0, 2.0e7)]:
        for reverse in (False, True):
            for sign in (False, True):
                c1, c2 = a.TestEdge(azi, s, reverse, sign), b.TestEdge(azi, s, reverse, sign)
                assert c1[0] == c2[0]
                assert c1[1] == pytest.approx(c2[1], rel=1e-12, abs=1e-6)
                assert c1[2] == pytest.approx(c2[2], rel=1e-12, abs=1e-6)


def test_add_edge_parity():
    a, b = OURS.Polygon(), THEIRS.Polygon()
    for lat, lon in [(0.0, 0.0)]:
        a.AddPoint(lat, lon)
        b.AddPoint(lat, lon)
    for azi, s in [(45.0, 1.0e6), (135.0, 2.5e6), (-90.0, 1.0e5)]:
        a.AddEdge(azi, s)
        b.AddEdge(azi, s)
        assert a.lat1 == pytest.approx(b.lat1)
        assert a.lon1 == pytest.approx(b.lon1)
    c1, c2 = a.Compute(), b.Compute()
    assert c1[0] == c2[0]
    assert c1[1] == pytest.approx(c2[1], rel=1e-12, abs=1e-6)
    assert c1[2] == pytest.approx(c2[2], rel=1e-12, abs=1e-6)


def test_add_edge_without_a_start_is_a_no_op():
    a, b = OURS.Polygon(), THEIRS.Polygon()
    a.AddEdge(45.0, 1.0e6)
    b.AddEdge(45.0, 1.0e6)
    assert a.num == b.num == 0
    assert a.Compute() == b.Compute()


def test_clear_parity():
    a, b = build(POLYGONS["octagon"])
    assert a.num > 0
    a.Clear()
    b.Clear()
    assert a.num == 0
    assert math.isnan(a.lat1) and math.isnan(a.lon1)
    assert a.Compute() == b.Compute()


def same3(c1, c2):
    def eq(x, y):
        return (math.isnan(x) and math.isnan(y)) or x == y
    return c1[0] == c2[0] and eq(c1[1], c2[1]) and eq(c1[2], c2[2])


def test_empty_polygon_answers():
    a, b = OURS.Polygon(), THEIRS.Polygon()
    assert same3(a.Compute(), b.Compute())
    assert same3(a.TestPoint(1.0, 2.0), b.TestPoint(1.0, 2.0))
    assert same3(a.TestEdge(1.0, 2.0), b.TestEdge(1.0, 2.0))
    a.AddPoint(10.0, 20.0)
    b.AddPoint(10.0, 20.0)
    assert same3(a.Compute(), b.Compute())
    assert same3(a.TestPoint(1.0, 2.0), b.TestPoint(1.0, 2.0))
    assert same3(a.TestEdge(1.0, 2.0), b.TestEdge(1.0, 2.0))


def test_state_survives_recompute():
    a, b = build(POLYGONS["octagon"])
    first = a.Compute()
    assert a.Compute() == first
    assert b.Compute() == first
    assert a.num == b.num
    a.AddPoint(0.0, 180.0)
    b.AddPoint(0.0, 180.0)
    assert a.Compute()[1] == pytest.approx(b.Compute()[1], rel=1e-12)
    a.AddPoint(0.0, 180.0)
    b.AddPoint(0.0, 180.0)
    assert a.Compute()[2] == pytest.approx(b.Compute()[2], rel=1e-12)


def test_long_polygon_uses_the_accumulator():
    """A 20k-vertex ring: the Shewchuk accumulator is what keeps the sum
    exact, and it is where a naive port would drift."""
    n = 20000
    lat = np.linspace(-80.0, 80.0, n)
    lon = np.linspace(0.0, 360.0, n, endpoint=False)
    a, b = OURS.Polygon(), THEIRS.Polygon()
    a.AddPoints(lat, lon)
    for la, lo in zip(lat, lon):
        b.AddPoint(la, lo)
    c1, c2 = a.Compute(), b.Compute()
    assert c1[0] == c2[0] == n
    assert c1[1] == pytest.approx(c2[1], rel=1e-13, abs=1e-6)
    assert c1[2] == pytest.approx(c2[2], rel=1e-13, abs=1e-2)


def test_area0_is_the_whole_ellipsoid():
    a = OURS.Polygon()
    assert a.area0 == pytest.approx(4 * math.pi * OURS._c2(), rel=0)
    assert a.area0 == pytest.approx(5.100656217240886e14, rel=1e-12)
