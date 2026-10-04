"""Define the :class:`~mojogeographiclib.polygonarea.PolygonArea` class

The constructor initializes a empty polygon.  The available methods are

  * :meth:`~mojogeographiclib.polygonarea.PolygonArea.Clear` reset the
    polygon
  * :meth:`~mojogeographiclib.polygonarea.PolygonArea.AddPoint` add a vertex
    to the polygon
  * :meth:`~mojogeographiclib.polygonarea.PolygonArea.AddEdge` add an edge
    to the polygon
  * :meth:`~mojogeographiclib.polygonarea.PolygonArea.Compute` compute the
    properties of the polygon
  * :meth:`~mojogeographiclib.polygonarea.PolygonArea.TestPoint` compute the
    properties of the polygon with a tentative additional vertex
  * :meth:`~mojogeographiclib.polygonarea.PolygonArea.TestEdge` compute the
    properties of the polygon with a tentative additional edge

The public attributes for this class are

  * :attr:`~mojogeographiclib.polygonarea.PolygonArea.earth`
    :attr:`~mojogeographiclib.polygonarea.PolygonArea.polyline`
  * :attr:`~mojogeographiclib.polygonarea.PolygonArea.area0`
  * :attr:`~mojogeographiclib.polygonarea.PolygonArea.num`
  * :attr:`~mojogeographiclib.polygonarea.PolygonArea.lat1`
  * :attr:`~mojogeographiclib.polygonarea.PolygonArea.lon1`

"""
# polygonarea.py
#
# This is a rather literal translation of the GeographicLib::PolygonArea class
# to python; the numerics live in src/ported.mojo and are called through
# ctypes.  See the documentation for the C++ class for more information at
#
#    https://geographiclib.sourceforge.io/C++/doc/annotated.html
#
# The algorithms are derived in
#
#    Charles F. F. Karney,
#    Algorithms for geodesics, J. Geodesy 87, 43-55 (2013),
#    https://doi.org/10.1007/s00190-012-0578-z
#    Addenda: https://geographiclib.sourceforge.io/geod-addenda.html
#
# Copyright (c) Charles Karney (2011-2022) <karney@alum.mit.edu> and licensed
# under the MIT/X11 License.  For more information, see
# https://geographiclib.sourceforge.io/
######################################################################

import math
import numpy as np
from mojogeographiclib import _lib
from mojogeographiclib.geodesiccapability import GeodesicCapability

__all__ = ["PolygonArea"]

# Slots of the state block written by src/capi.mojo.
_NUM, _CROSS, _F = 0, 1, 2


class PolygonArea:
  """Area of a geodesic polygon"""

  def __init__(self, earth, polyline = False):
    """Construct a PolygonArea object

    :param earth: a :class:`~mojogeographiclib.geodesic.Geodesic` object
    :param polyline: if true, treat object as a polyline instead of a polygon

    Initially the polygon has no vertices.
    """

    self.earth = earth
    """The geodesic object (readonly)"""
    self.polyline = polyline
    """Is this a polyline? (readonly)"""
    self.area0 = 4 * math.pi * earth._c2()
    """The total area of the ellipsoid in meter^2 (readonly)"""
    self._geod_addr = earth._addr
    # num, crossings, then lat0, lon0, lat1, lon1 and the two accumulators.
    self._state = np.zeros(_lib.PA_BLOCK, dtype=np.int64)
    self._state[_F + 4:] = 0.0
    self._addr = _lib.addr(self._state)
    self._out = np.zeros(2, dtype=np.float64)
    _lib.fn("mg_polygon_clear")(earth.a, earth.f, earth._addr, self._addr)

  @property
  def num(self):
    """The current number of points in the polygon (readonly)"""
    return int(self._state[_NUM])

  @property
  def lat1(self):
    """The current latitude in degrees (readonly)"""
    return float(self._state.view(np.float64)[_F + 2])

  @property
  def lon1(self):
    """The current longitude in degrees (readonly)"""
    return float(self._state.view(np.float64)[_F + 3])

  def Clear(self):
    """Reset to empty polygon."""
    _lib.fn("mg_polygon_clear")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr,
    )

  def AddPoint(self, lat, lon):
    """Add the next vertex to the polygon

    :param lat: the latitude of the point in degrees
    :param lon: the longitude of the point in degrees

    This adds an edge from the current vertex to the new vertex.
    """

    _lib.fn("mg_polygon_addpoint")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr, lat, lon,
    )

  def AddPoints(self, lat, lon):
    """Add the next vertices to the polygon

    Not in upstream; a drop-in for looping over :meth:`AddPoint` except that
    the whole vertex list goes through one kernel call.

    :param lat: the latitudes of the points in degrees
    :param lon: the longitudes of the points in degrees

    """

    x = _lib.f64(lat)
    y = _lib.f64(lon)
    _lib.fn("mg_polygon_addpoints")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr, _lib.addr(x), _lib.addr(y), x.size,
    )

  def AddEdge(self, azi, s):
    """Add the next edge to the polygon

    :param azi: the azimuth at the current the point in degrees
    :param s: the length of the edge in meters

    This specifies the new vertex in terms of the edge from the current
    vertex.
    """

    _lib.fn("mg_polygon_addedge")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr, azi, s,
    )

  def _result(self, num, out):
    return (int(num), float(out[0]), float(out[1]))

  def Compute(self, reverse = False, sign = True):
    """Compute the properties of the polygon

    :param reverse: if true then clockwise (instead of
      counter-clockwise) traversal counts as a positive area
    :param sign: if true then return a signed result for the area if the
      polygon is traversed in the "wrong" direction instead of returning
      the area for the rest of the earth
    :return: a tuple of number, perimeter (meters), area (meters^2)

    Arbitrarily complex polygons are allowed.  In the case of
    self-intersecting polygons the area is accumulated "algebraically",
    e.g., the areas of the 2 loops in a figure-8 polygon will partially
    cancel.

    If the object is a polygon (and not a polyline), the perimeter
    includes the length of a final edge connecting the current point to
    the initial point.  If the object is a polyline, then area is nan.

    More points can be added to the polygon after this call.

    """

    out = _lib.f64(self._out)
    num = _lib.fn("mg_polygon_compute")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr, int(reverse), int(sign), int(self.polyline),
      _lib.addr(out),
    )
    return self._result(num, out)

  def TestPoint(self, lat, lon, reverse = False, sign = True):
    """Compute the properties for a tentative additional vertex

    :param lat: the latitude of the point in degrees
    :param lon: the longitude of the point in degrees
    :param reverse: if true then clockwise (instead of
      counter-clockwise) traversal counts as a positive area
    :param sign: if true then return a signed result for the area if the
      polygon is traversed in the "wrong" direction instead of returning
      the area for the rest of the earth
    :return: a tuple of number, perimeter (meters), area (meters^2)

    """

    out = _lib.f64(self._out)
    num = _lib.fn("mg_polygon_testpoint")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr, lat, lon, int(reverse), int(sign),
      int(self.polyline), _lib.addr(out),
    )
    return self._result(num, out)

  def TestEdge(self, azi, s, reverse = False, sign = True):
    """Compute the properties for a tentative additional edge

    :param azi: the azimuth at the current the point in degrees
    :param s: the length of the edge in meters
    :param reverse: if true then clockwise (instead of
      counter-clockwise) traversal counts as a positive area
    :param sign: if true then return a signed result for the area if the
      polygon is traversed in the "wrong" direction instead of returning
      the area for the rest of the earth
    :return: a tuple of number, perimeter (meters), area (meters^2)

    """

    out = _lib.f64(self._out)
    num = _lib.fn("mg_polygon_testedge")(
      self.earth.a, self.earth.f, self._geod_addr,
      self._addr, azi, s, int(reverse), int(sign),
      int(self.polyline), _lib.addr(out),
    )
    return self._result(num, out)
