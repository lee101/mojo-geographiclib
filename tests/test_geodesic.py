"""API shape and published values for Geodesic / GeodesicLine."""

import math

import numpy as np
import pytest

from geographiclib.geodesic import Geodesic as UpGeodesic
from geographiclib.geodesiccapability import GeodesicCapability

from mojogeographiclib.constants import Constants
from mojogeographiclib.geodesic import Geodesic
from mojogeographiclib.geodesicline import GeodesicLine
from mojogeographiclib.polygonarea import PolygonArea


def test_wgs84_constants_match_upstream():
    assert Constants.WGS84_a == 6378137.0
    assert Constants.WGS84_f == 1 / 298.257223563
    assert Geodesic.WGS84.a == UpGeodesic.WGS84.a
    assert Geodesic.WGS84.f == UpGeodesic.WGS84.f


def test_capability_constants_match_upstream():
    for name in ["CAP_NONE", "CAP_C1", "CAP_C1p", "CAP_C2", "CAP_C3", "CAP_C4",
                 "CAP_ALL", "CAP_MASK", "OUT_ALL", "OUT_MASK", "EMPTY",
                 "LATITUDE", "LONGITUDE", "AZIMUTH", "DISTANCE", "STANDARD",
                 "DISTANCE_IN", "REDUCEDLENGTH", "GEODESICSCALE", "AREA",
                 "LONG_UNROLL", "ALL"]:
        assert getattr(Geodesic, name) == getattr(GeodesicCapability, name), name


def test_order_constants_match_upstream():
    for name in ["GEOGRAPHICLIB_GEODESIC_ORDER", "nA1_", "nC1_", "nC1p_", "nA2_",
                 "nC2_", "nA3_", "nA3x_", "nC3_", "nC3x_", "nC4_", "nC4x_",
                 "maxit1_", "maxit2_", "tiny_", "tol0_", "tol1_", "tol2_",
                 "tolb_", "xthresh_"]:
        assert getattr(Geodesic, name) == getattr(UpGeodesic, name), name


# ------------------------------------------- published GeographicLib values
# The two numbers below are the ones printed by the reference GeodSolve for
# "0 0 0 90" and "0 0 90 0" on WGS84; they are quoted to 6 decimal places in
# the GeographicLib documentation.
def test_quarter_equator():
    r = Geodesic.WGS84.Inverse(0, 0, 0, 90)
    assert r["s12"] == pytest.approx(10018754.171, abs=1e-3)
    assert r["a12"] == pytest.approx(90.30276808388787, rel=1e-14)
    assert r["azi1"] == 90.0
    assert r["azi2"] == 90.0


def test_quarter_meridian():
    r = Geodesic.WGS84.Inverse(0, 0, 90, 0)
    assert r["s12"] == pytest.approx(10001965.729, abs=1e-3)
    assert r["a12"] == 90.0
    assert r["azi1"] == 0.0


def test_inverse_documented_example():
    # The worked example in the upstream class docstring.
    r = Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)
    assert r["a12"] == pytest.approx(179.6197069334283, rel=1e-14)
    assert r["s12"] == pytest.approx(19959679.26735382, rel=1e-14)
    assert r["azi1"] == pytest.approx(161.06766998615882, abs=1e-9)
    assert r["azi2"] == pytest.approx(18.825195123248392, abs=1e-9)
    assert r["lat1"] == -41.32 and r["lon1"] == 174.81
    assert r["lat2"] == 40.96 and r["lon2"] == -5.5


def test_direct_inverts_inverse():
    r = Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50, Geodesic.ALL)
    d = Geodesic.WGS84.Direct(r["lat1"], r["lon1"], r["azi1"], r["s12"],
                              Geodesic.ALL)
    assert d["lat2"] == pytest.approx(r["lat2"], abs=1e-9)
    assert abs((d["lon2"] - r["lon2"] + 180) % 360 - 180) < 1e-9
    assert d["azi2"] == pytest.approx(r["azi2"], abs=1e-9)
    assert d["a12"] == pytest.approx(r["a12"], rel=1e-14)


def test_arc_direct_inverts_arc():
    r = Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50, Geodesic.ALL)
    d = Geodesic.WGS84.ArcDirect(r["lat1"], r["lon1"], r["azi1"], r["a12"],
                                 Geodesic.ALL)
    assert d["lat2"] == pytest.approx(r["lat2"], abs=1e-9)
    assert d["s12"] == pytest.approx(r["s12"], rel=1e-14)


# ------------------------------------------------------------------ errors
def test_bad_equatorial_radius():
    with pytest.raises(ValueError, match="Equatorial radius"):
        Geodesic(0.0, 1 / 298.257223563)
    with pytest.raises(ValueError, match="Equatorial radius"):
        Geodesic(-1.0, 0.0)
    with pytest.raises(ValueError, match="Equatorial radius"):
        Geodesic(float("nan"), 0.0)


def test_bad_polar_semi_axis():
    with pytest.raises(ValueError, match="Polar semi-axis"):
        Geodesic(6378137.0, 2.0)


# ------------------------------------------------------------------ latfix
def test_latfix_replaces_out_of_range():
    assert math.isnan(Geodesic.WGS84.Inverse(95, 0, 0, 0)["lat1"])
    assert not math.isnan(Geodesic.WGS84.Inverse(90, 0, 0, 0)["lat1"])


def test_nan_input_gives_nan_output():
    r = Geodesic.WGS84.Inverse(float("nan"), 0, 0, 0, Geodesic.ALL)
    assert math.isnan(r["a12"])


# ------------------------------------------------------------------- lines
def test_line_attributes():
    line = Geodesic.WGS84.Line(10.0, 20.0, 30.0)
    assert line.a == 6378137.0
    assert line.f == 1 / 298.257223563
    assert line.lat1 == 10.0 and line.lon1 == 20.0
    assert line.azi1 == 30.0
    assert line.salp1 == pytest.approx(math.sin(math.radians(30.0)))
    assert line.calp1 == pytest.approx(math.cos(math.radians(30.0)))
    assert line.caps == (GeodesicCapability.STANDARD
                         | GeodesicCapability.DISTANCE_IN
                         | Geodesic.LATITUDE | Geodesic.AZIMUTH
                         | Geodesic.LONG_UNROLL)
    assert math.isnan(line.s13) and math.isnan(line.a13)


def test_set_distance_then_set_arc_round_trips():
    line = Geodesic.WGS84.Line(10.0, 20.0, 30.0)
    line.SetDistance(1000000.0)
    assert line.s13 == 1000000.0
    a13 = line.a13
    line.SetArc(a13)
    assert line.a13 == a13
    assert line.s13 == pytest.approx(1000000.0, rel=1e-12)


def test_line_constructors_agree_with_upstream():
    ours = Geodesic.WGS84
    theirs = UpGeodesic.WGS84
    for maker in ["Line", "DirectLine", "ArcDirectLine", "InverseLine"]:
        args = (10.0, 20.0, 45.0) if maker == "Line" else (
            (10.0, 20.0, 30.0, 1.0e6) if maker == "DirectLine" else
            (10.0, 20.0, 30.0, 1.0e6) if maker == "ArcDirectLine" else
            (10.0, 20.0, 30.0, 40.0))
        if maker in ("DirectLine", "ArcDirectLine"):
            a = getattr(ours, maker)(*args)
            b = getattr(theirs, maker)(*args)
        else:
            a = getattr(ours, maker)(*args)
            b = getattr(theirs, maker)(*args)
        assert isinstance(a, GeodesicLine)
        assert a.Position(5.0e5) == b.Position(5.0e5), maker
        assert a.ArcPosition(5.0) == b.ArcPosition(5.0), maker


def test_line_requires_distance_in_for_position():
    line = Geodesic.WGS84.Line(10.0, 20.0, 30.0, Geodesic.CAP_NONE)
    r = line.Position(1000.0, Geodesic.DISTANCE)
    assert math.isnan(r["a12"])
    assert "lat2" not in r


# ---------------------------------------------------------------- polygons
def test_polygon_constructor():
    p = Geodesic.WGS84.Polygon()
    assert isinstance(p, PolygonArea)
    assert p.polyline is False
    assert p.earth is Geodesic.WGS84
    assert p.area0 == pytest.approx(5.100656217240886e14, rel=1e-12)
    assert p.num == 0
    assert math.isnan(p.lat1) and math.isnan(p.lon1)


def test_polygon_polyline_flag():
    assert Geodesic.WGS84.Polygon(True).polyline is True


# ------------------------------------------------------- batch == scalar
def test_batch_entry_points_match_the_scalar_ones():
    rng = np.random.default_rng(5)
    n = 200
    lat1 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon1 = rng.uniform(-180, 180, n)
    lat2 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon2 = rng.uniform(-180, 180, n)
    az1 = rng.uniform(-180, 180, n)
    s12 = rng.uniform(-1e7, 1e7, n)

    pairs = Geodesic.WGS84.InversePairs(lat1, lon1, lat2, lon2, Geodesic.ALL)
    for i in range(n):
        one = Geodesic.WGS84.Inverse(lat1[i], lon1[i], lat2[i], lon2[i],
                                     Geodesic.ALL)
        for k in one:
            assert pairs[k][i] == one[k], ("InversePairs", i, k)

    many = Geodesic.WGS84.DirectMany(lat1, lon1, az1, s12, Geodesic.ALL)
    for i in range(n):
        one = Geodesic.WGS84.Direct(lat1[i], lon1[i], az1[i], s12[i],
                                    Geodesic.ALL)
        for k in one:
            assert many[k][i] == one[k], ("DirectMany", i, k)

    a12 = rng.uniform(-180, 180, n)
    many = Geodesic.WGS84.ArcDirectMany(lat1, lon1, az1, a12, Geodesic.ALL)
    for i in range(n):
        one = Geodesic.WGS84.ArcDirect(lat1[i], lon1[i], az1[i], a12[i],
                                       Geodesic.ALL)
        for k in one:
            assert many[k][i] == one[k], ("ArcDirectMany", i, k)


def test_batch_line_positions_match_the_scalar_ones():
    rng = np.random.default_rng(6)
    line = Geodesic.WGS84.Line(10.0, 20.0, 45.0, Geodesic.ALL)
    d = np.concatenate([rng.uniform(-2e7, 2e7, 500), [0.0]])
    many = line.Positions(d, line.caps)
    for i, v in enumerate(d):
        one = line.Position(v, line.caps)
        for k in one:
            assert many[k][i] == one[k], ("Positions", i, k)
    many = line.ArcPositions(d, line.caps)
    for i, v in enumerate(d):
        one = line.ArcPosition(v, line.caps)
        for k in one:
            assert many[k][i] == one[k], ("ArcPositions", i, k)



# --------------------------------------------------------------- parallel
# Above 8192 points the batch kernels split the batch across workers, each
# with its own Clenshaw scratch from the Geodesic block pool.  Every point is
# independent, so the parallel answer has to be the serial answer bit for
# bit; that is the whole contract.  The sizes are deliberately not multiples
# of the worker count, so the last chunk is short and the tail is covered.
PARALLEL_THRESHOLD = 8192


@pytest.mark.parametrize("n", [PARALLEL_THRESHOLD, PARALLEL_THRESHOLD + 1,
                                3 * PARALLEL_THRESHOLD + 997, 20011])
def test_parallel_batch_is_bit_identical_to_the_serial_batch(n):
    g = Geodesic.WGS84
    if g._workers < 2:
        pytest.skip("single-core machine; the batch kernels stay serial")
    rng = np.random.default_rng(n)
    lat1 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon2 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    azi1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    s12 = np.ascontiguousarray(rng.uniform(-2e7, 2e7, n))
    a12 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    line = g.Line(37.0, -122.0, 51.0, g.ALL)

    def run():
        return (g.InversePairs(lat1, lon1, lat2, lon2, g.ALL),
                g.DirectMany(lat1, lon1, azi1, s12, g.ALL),
                g.ArcDirectMany(lat1, lon1, azi1, a12, g.ALL),
                line.Positions(s12, line.caps),
                line.ArcPositions(a12, line.caps))

    parallel = run()
    saved = g._workers
    try:
        g._workers = 1
        serial = run()
    finally:
        g._workers = saved
    for par, ser in zip(parallel, serial):
        assert sorted(par) == sorted(ser)
        for k in ser:
            assert np.array_equal(par[k], ser[k]), (k, n)


def test_batches_below_the_threshold_match_the_scalar_path():
    # One point short of the threshold the kernel must take the serial
    # branch, and must still agree with the single-point API exactly.
    n = PARALLEL_THRESHOLD - 1
    g = Geodesic.WGS84
    rng = np.random.default_rng(11)
    lat1 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon2 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    batch = g.InversePairs(lat1, lon1, lat2, lon2, g.STANDARD)
    for i in (0, 1, n // 2, n - 2, n - 1):
        one = g.Inverse(lat1[i], lon1[i], lat2[i], lon2[i], g.STANDARD)
        for k in one:
            assert batch[k][i] == one[k], (k, i)


def test_add_points_matches_a_loop():
    rng = np.random.default_rng(7)
    n = 500
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    a = Geodesic.WGS84.Polygon()
    b = Geodesic.WGS84.Polygon()
    a.AddPoints(lat, lon)
    for la, lo in zip(lat, lon):
        b.AddPoint(la, lo)
    assert a.num == b.num == n
    assert a.Compute() == b.Compute()
