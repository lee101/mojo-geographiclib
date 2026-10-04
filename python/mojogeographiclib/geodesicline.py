"""Define the :class:`~mojogeographiclib.geodesicline.GeodesicLine` class

The constructor defines the starting point of the line.  Points on the
line are given by

  * :meth:`~mojogeographiclib.geodesicline.GeodesicLine.Position` position
    given in terms of distance
  * :meth:`~mojogeographiclib.geodesicline.GeodesicLine.ArcPosition` position
    given in terms of spherical arc length

A reference point 3 can be defined with

  * :meth:`~mojogeographiclib.geodesicline.GeodesicLine.SetDistance` set
    position of 3 in terms of the distance from the starting point
  * :meth:`~mojogeographiclib.geodesicline.GeodesicLine.SetArc` set
    position of 3 in terms of the spherical arc length from the starting point

The object can also be constructed by

  * :meth:`Geodesic.Line <mojogeographiclib.geodesic.Geodesic.Line>`
  * :meth:`Geodesic.DirectLine <mojogeographiclib.geodesic.Geodesic.DirectLine>`
  * :meth:`Geodesic.ArcDirectLine
    <mojogeographiclib.geodesic.Geodesic.ArcDirectLine>`
  * :meth:`Geodesic.InverseLine <mojogeographiclib.geodesic.Geodesic.InverseLine>`

The public attributes for this class are

  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.a`
    :attr:`~mojogeographiclib.geodesicline.GeodesicLine.f`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.caps`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.lat1`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.lon1`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.azi1`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.salp1`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.calp1`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.s13`
  * :attr:`~mojogeographiclib.geodesicline.GeodesicLine.a13`

"""
# geodesicline.py
#
# This is a rather literal translation of the GeographicLib::GeodesicLine class
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
from mojogeographiclib.geomath import Math, ang_normalize, atan2d
from mojogeographiclib.geodesiccapability import GeodesicCapability

__all__ = ["GeodesicLine"]


class GeodesicLine:
  """Points on a geodesic path"""

  def __init__(self, geod, lat1, lon1, azi1,
               caps = GeodesicCapability.STANDARD |
               GeodesicCapability.DISTANCE_IN,
               salp1 = math.nan, calp1 = math.nan):
    """Construct a GeodesicLine object

    :param geod: a :class:`~mojogeographiclib.geodesic.Geodesic` object
    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param azi1: azimuth at the first point in degrees
    :param caps: the :ref:`capabilities <outmask>`

    This creates an object allowing points along a geodesic starting at
    (*lat1*, *lon1*), with azimuth *azi1* to be found.  The default
    value of *caps* is STANDARD | DISTANCE_IN.  The optional parameters
    *salp1* and *calp1* should not be supplied; they are part of the
    private interface.

    """

    self._geod = geod
    # Scalars and Clenshaw scratch for this line; owned here and reloaded on
    # every call so a Position costs one kernel call, not a line rebuild.
    self._state = np.zeros(_lib.LINE_BLOCK, dtype=np.float64)
    self._addr = _lib.addr(self._state)
    self.a = geod.a
    """The equatorial radius in meters (readonly)"""
    self.f = geod.f
    """The flattening (readonly)"""
    self.caps = caps | GeodesicCapability.LATITUDE | GeodesicCapability.AZIMUTH | \
                GeodesicCapability.LONG_UNROLL
    """the capabilities (readonly)"""

    # Guard against underflow in salp0
    self.lat1 = Math.LatFix(lat1)
    """the latitude of the first point in degrees (readonly)"""
    self.lon1 = lon1
    """the longitude of the first point in degrees (readonly)"""
    if math.isnan(salp1) or math.isnan(calp1):
      self.azi1 = Math.AngNormalize(azi1)
      self.salp1, self.calp1 = Math.sincosd(Math.AngRound(azi1))
    else:
      self.azi1 = azi1
      """the azimuth at the first point in degrees (readonly)"""
      self.salp1 = salp1
      """the sine of the azimuth at the first point (readonly)"""
      self.calp1 = calp1
      """the cosine of the azimuth at the first point (readonly)"""

    self.s13 = math.nan
    """the distance between point 1 and point 3 in meters (readonly)"""
    self.a13 = math.nan
    """the arc length between point 1 and point 3 in degrees (readonly)"""

    self._fn = {
      False: _lib.fn("mg_line_position"),
      True: _lib.fn("mg_line_arcposition"),
    }
    _lib.fn("mg_line_new")(
      self.a, self.f, geod._addr,
      self.lat1, self.lon1, self.azi1, self.salp1, self.calp1, self.caps,
      self._addr,
    )

  def _positions(self, s12_a12, outmask, arcmode, vector):
    fn = self._fn[arcmode]
    workers = self._geod._workers
    if vector:
      d = _lib.f64(s12_a12)
      nout = d.size
      buf, addr = _lib.buf("line.out", nout * _lib.DIR_N)
      fn(self.a, self.f, self._geod._addr, self._addr,
         _lib.addr(d), outmask, addr, nout, workers)
      out = buf.copy()
      a12, lat2, lon2, azi2, s12, m12, M12, M21, S12 = [
        out[k * nout:(k + 1) * nout] for k in range(_lib.DIR_N)
      ]
    else:
      # One buffer holds the single input and the whole result, so the call
      # is one struct pack and one struct unpack, with no numpy element
      # access on either side of it.
      w = _lib.io("line.io", 1 + _lib.DIR_N)
      _lib.pack("<d", w.mem, 0, s12_a12)
      a = w.addr
      fn(self.a, self.f, self._geod._addr, self._addr, a, outmask, a + 8, 1,
         workers)
      a12, lat2, lon2, azi2, s12, m12, M12, M21, S12 = \
        _lib.unpack("<9d", w.mem, 8)
    outmask &= GeodesicCapability.OUT_MASK  # upstream does the same
    if vector:
      result = {
        'lat1': np.full(nout, self.lat1),
        'lon1': np.full(nout, self.lon1) if outmask & GeodesicCapability.LONG_UNROLL
                else ang_normalize(np.full(nout, self.lon1)),
        'azi1': np.full(nout, self.azi1),
        'a12' if arcmode else 's12': d,
      }
    else:
      result = {'lat1': self.lat1,
                'lon1': self.lon1 if outmask & GeodesicCapability.LONG_UNROLL else
                Math.AngNormalize(self.lon1),
                'azi1': self.azi1,
                'a12' if arcmode else 's12': s12_a12}
    if arcmode:
      if outmask & GeodesicCapability.DISTANCE: result['s12'] = s12
    else:
      result['s12'] = s12
    result['a12'] = a12
    if outmask & GeodesicCapability.LATITUDE: result['lat2'] = lat2
    if outmask & GeodesicCapability.LONGITUDE: result['lon2'] = lon2
    if outmask & GeodesicCapability.AZIMUTH: result['azi2'] = azi2
    if outmask & GeodesicCapability.REDUCEDLENGTH: result['m12'] = m12
    if outmask & GeodesicCapability.GEODESICSCALE:
      result['M12'] = M12; result['M21'] = M21
    if outmask & GeodesicCapability.AREA: result['S12'] = S12
    return result

  def Position(self, s12, outmask = GeodesicCapability.STANDARD):
    """Find the position on the line given *s12*

    :param s12: the distance from the first point to the second in
      meters
    :param outmask: the :ref:`output mask <outmask>`
    :return: a :ref:`dict`

    The default value of *outmask* is STANDARD, i.e., the *lat1*,
    *lon1*, *azi1*, *lat2*, *lon2*, *azi2*, *s12*, *a12* entries are
    returned.  The :class:`~mojogeographiclib.geodesicline.GeodesicLine`
    object must have been constructed with the DISTANCE_IN capability.

    """

    return self._positions(s12, outmask, False, False)

  def ArcPosition(self, a12, outmask = GeodesicCapability.STANDARD):
    """Find the position on the line given *a12*

    :param a12: spherical arc length from the first point to the second
      in degrees
    :param outmask: the :ref:`output mask <outmask>`
    :return: a :ref:`dict`

    The default value of *outmask* is STANDARD, i.e., the *lat1*,
    *lon1*, *azi1*, *lat2*, *lon2*, *azi2*, *s12*, *a12* entries are
    returned.

    """

    return self._positions(a12, outmask, True, False)

  def Positions(self, s12, outmask = GeodesicCapability.STANDARD):
    """Find the positions on the line for an array of distances.

    Not in upstream; a drop-in for looping over :meth:`Position` except that
    every entry of the returned dict is a float64 array.

    """

    return self._positions(s12, outmask, False, True)

  def ArcPositions(self, a12, outmask = GeodesicCapability.STANDARD):
    """Find the positions on the line for an array of arc lengths.

    Not in upstream; a drop-in for looping over :meth:`ArcPosition`.

    """

    return self._positions(a12, outmask, True, True)

  def SetDistance(self, s13):
    """Specify the position of point 3 in terms of distance

    :param s13: distance from point 1 to point 3 in meters

    """

    self.s13 = s13
    self.a13 = _lib.fn("mg_line_set_distance")(
      self.a, self.f, self._geod._addr, self._addr, s13
    )

  def SetArc(self, a13):
    """Specify the position of point 3 in terms of arc length

    :param a13: spherical arc length from point 1 to point 3 in degrees

    """

    self.a13 = a13
    self.s13 = _lib.fn("mg_line_set_arc")(
      self.a, self.f, self._geod._addr, self._addr, a13
    )
