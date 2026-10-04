"""Define the :class:`~mojogeographiclib.geodesic.Geodesic` class

The ellipsoid parameters are defined by the constructor.  The direct and
inverse geodesic problems are solved by

  * :meth:`~mojogeographiclib.geodesic.Geodesic.Inverse` Solve the inverse
    geodesic problem
  * :meth:`~mojogeographiclib.geodesic.Geodesic.Direct` Solve the direct
    geodesic problem
  * :meth:`~mojogeographiclib.geodesic.Geodesic.ArcDirect` Solve the direct
    geodesic problem in terms of spherical arc length

:class:`~mojogeographiclib.geodesicline.GeodesicLine` objects can be created
with

  * :meth:`~mojogeographiclib.geodesic.Geodesic.Line`
  * :meth:`~mojogeographiclib.geodesic.Geodesic.DirectLine`
  * :meth:`~mojogeographiclib.geodesic.Geodesic.ArcDirectLine`
  * :meth:`~mojogeographiclib.geodesic.Geodesic.InverseLine`

:class:`~mojogeographiclib.polygonarea.PolygonArea` objects can be created
with

  * :meth:`~mojogeographiclib.geodesic.Geodesic.Polygon`

The public attributes for this class are

  * :attr:`~mojogeographiclib.geodesic.Geodesic.a`
    :attr:`~mojogeographiclib.geodesic.Geodesic.f`

*outmask* and *caps* bit masks are

  * :const:`~mojogeographiclib.geodesic.Geodesic.EMPTY`
  * :const:`~mojogeographiclib.geodesic.Geodesic.LATITUDE`
  * :const:`~mojogeographiclib.geodesic.Geodesic.LONGITUDE`
  * :const:`~mojogeographiclib.geodesic.Geodesic.AZIMUTH`
  * :const:`~mojogeographiclib.geodesic.Geodesic.DISTANCE`
  * :const:`~mojogeographiclib.geodesic.Geodesic.STANDARD`
  * :const:`~mojogeographiclib.geodesic.Geodesic.DISTANCE_IN`
  * :const:`~mojogeographiclib.geodesic.Geodesic.REDUCEDLENGTH`
  * :const:`~mojogeographiclib.geodesic.Geodesic.GEODESICSCALE`
  * :const:`~mojogeographiclib.geodesic.Geodesic.AREA`
  * :const:`~mojogeographiclib.geodesic.Geodesic.ALL`
  * :const:`~mojogeographiclib.geodesic.Geodesic.LONG_UNROLL`

::Example:

    >>> from mojogeographiclib.geodesic import Geodesic
    >>> # The geodesic inverse problem
    ... Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)
    {'lat1': -41.32,
     'a12': 179.6197069334283,
     's12': 19959679.26735382,
     'lat2': 40.96,
     'azi2': 18.825195123248392,
     'azi1': 161.06766998615882,
     'lon1': 174.81,
     'lon2': -5.5}

"""
# geodesic.py
#
# This is a rather literal translation of the GeographicLib::Geodesic class to
# python; the numerics live in src/ported.mojo and are called through ctypes.
# See the documentation for the C++ class for more information at
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
import os
import sys

import numpy as np
from mojogeographiclib import _lib
from mojogeographiclib.geomath import Math, ang_diff, ang_normalize, atan2d, lat_fix
from mojogeographiclib.constants import Constants
from mojogeographiclib.geodesiccapability import GeodesicCapability

__all__ = ["Geodesic"]


def _singleton(value, n):
  """Broadcast a Python scalar across a length-n result dict."""
  return np.full(n, float(value), dtype=np.float64)


class Geodesic:
  """Solve geodesic problems"""

  WGS84: "Geodesic"
  GEOGRAPHICLIB_GEODESIC_ORDER = 6
  nA1_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nC1_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nC1p_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nA2_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nC2_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nA3_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nA3x_ = nA3_
  nC3_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nC3x_ = (nC3_ * (nC3_ - 1)) // 2
  nC4_ = GEOGRAPHICLIB_GEODESIC_ORDER
  nC4x_ = (nC4_ * (nC4_ + 1)) // 2
  maxit1_ = 20
  maxit2_ = maxit1_ + 53 + 10

  tiny_ = math.sqrt(sys.float_info.min)
  tol0_ = sys.float_info.epsilon
  tol1_ = 200 * tol0_
  tol2_ = math.sqrt(tol0_)
  tolb_ = tol0_
  xthresh_ = 1000 * tol2_

  CAP_NONE = GeodesicCapability.CAP_NONE
  CAP_C1   = GeodesicCapability.CAP_C1
  CAP_C1p  = GeodesicCapability.CAP_C1p
  CAP_C2   = GeodesicCapability.CAP_C2
  CAP_C3   = GeodesicCapability.CAP_C3
  CAP_C4   = GeodesicCapability.CAP_C4
  CAP_ALL  = GeodesicCapability.CAP_ALL
  CAP_MASK = GeodesicCapability.CAP_MASK
  OUT_ALL  = GeodesicCapability.OUT_ALL
  OUT_MASK = GeodesicCapability.OUT_MASK

  def __init__(self, a, f):
    """Construct a Geodesic object

    :param a: the equatorial radius of the ellipsoid in meters
    :param f: the flattening of the ellipsoid

    An exception is thrown if *a* or the polar semi-axis *b* = *a* (1 -
    *f*) is not a finite positive quantity.

    """

    self.a = float(a)
    """The equatorial radius in meters (readonly)"""
    self.f = float(f)
    """The flattening (readonly)"""
    # The kernel entry points are resolved once: a `getattr` on the shared
    # library costs about as much as one of the Math calls in the result
    # dict, and a single-point call does not have that to spare.
    self._fn = {n: _lib.fn(n) for n in
                ("mg_inverse", "mg_direct", "mg_arcdirect")}
    self._init = _lib.fn("mg_geodesic_init")
    self._c2_fn = _lib.fn("mg_geodesic_c2")
    # Series coefficients and Clenshaw scratch; owned here, borrowed by every
    # kernel call made through this object.  The pool holds one block per
    # batch worker: the coefficient tables are identical in all of them, but
    # the Clenshaw scratch is written per point, so a worker that shared
    # block 0 with another worker would corrupt it.  Slot 0 is the one every
    # single-point call uses, and the serial batch path.
    self._workers = min(os.cpu_count() or 1, _lib.MAX_WORKERS)
    self._block = np.zeros(
      _lib.GEOD_BLOCK * (1 + self._workers), dtype=np.float64)
    # ctypes' .ctypes.data builds a new object on every access, so the
    # address of the block is taken once and reused for every kernel call.
    self._addr = _lib.addr(self._block)
    rc = self._init(self.a, self.f, self._addr)
    if rc == 1:
      raise ValueError("Equatorial radius is not positive")
    if rc == 2:
      raise ValueError("Polar semi-axis is not positive")
    pool = self._block.reshape(1 + self._workers, _lib.GEOD_BLOCK)
    pool[1:] = pool[0]

  def _c2(self):
    return self._c2_fn(self.a, self.f, self._addr)

  def _run(self, name, inputs, outmask, nout):
    """Call a batch kernel over `inputs` (4 float64 arrays) and split the
    structure-of-arrays result into a list of fields.

    A batch result can be held on to, so it is copied out of the scratch
    buffer; the caller has to own what it gets back.
    """
    n = len(inputs[0])
    b, addr = _lib.buf(name + ".out", n * nout)
    self._fn[name](
      self.a, self.f, self._addr,
      _lib.addr(inputs[0]), _lib.addr(inputs[1]),
      _lib.addr(inputs[2]), _lib.addr(inputs[3]),
      outmask, addr, n, self._workers,
    )
    out = b.copy()
    return [out[k * n:(k + 1) * n] for k in range(nout)]

  def _run1(self, name, outmask, nout, *scalars):
    """Call a kernel once, for four scalars, and return `nout` floats.

    The four inputs and all the outputs share one thread-local buffer, so a
    single-point call is four `struct` writes and one ctypes call -- no numpy
    element access, no per-call address lookup, no per-call attribute lookup
    on the shared library.
    """
    fn = self._fn[name]
    w = _lib.io(name, 4 + nout)
    a = w.addr
    _lib.pack("<4d", w.mem, 0, *scalars)
    fn(self.a, self.f, self._addr, a, a + 8, a + 16, a + 24, outmask, a + 32, 1,
       self._workers)
    return _lib.unpack("<%dd" % nout, w.mem, 32)

  def _result(self, outmask, out, orig, lon2, lon1, lat1, lat2, vector):
    """Assemble the result dict exactly as upstream does, from either scalars
    (n == 1) or whole arrays."""
    outmask &= Geodesic.OUT_MASK  # upstream: outmask &= Geodesic.OUT_MASK
    a12, s12, salp1, calp1, salp2, calp2, m12, M12, M21, S12 = out
    if vector:
      result = {
        'lat1': lat_fix(lat1),
        'lon1': lon1 if outmask & Geodesic.LONG_UNROLL else ang_normalize(lon1),
        'lat2': lat_fix(lat2),
        'lon2': lon2,
      }
      azi = atan2d
    else:
      # upstream echoes the caller's own objects, so an int in gives an int
      # out; keep that.
      result = {'lat1': Math.LatFix(orig[0]),
                'lon1': orig[1] if outmask & Geodesic.LONG_UNROLL else
                Math.AngNormalize(orig[1]),
                'lat2': Math.LatFix(orig[2]),
                'lon2': lon2}
      azi = Math.atan2d
    result['a12'] = a12
    if outmask & Geodesic.DISTANCE: result['s12'] = s12
    if outmask & Geodesic.AZIMUTH:
      result['azi1'] = azi(salp1, calp1)
      result['azi2'] = azi(salp2, calp2)
    if outmask & Geodesic.REDUCEDLENGTH: result['m12'] = m12
    if outmask & Geodesic.GEODESICSCALE:
      result['M12'] = M12; result['M21'] = M21
    if outmask & Geodesic.AREA: result['S12'] = S12
    return result

  def _inv_fields(self, lat1, lon1, lat2, lon2, outmask, vector):
    """Run the inverse kernel; returns the inputs as the kernel saw them and
    the raw _GenInverse fields (a12, s12, salp1, calp1, salp2, calp2, m12,
    M12, M21, S12).  Upstream's line constructors use this to read the
    azimuth back out of the sine and cosine."""
    if not vector:
      out = self._run1("mg_inverse", outmask, len(_lib.INV_FIELDS),
                       lat1, lon1, lat2, lon2)
      return lat1, lon1, lat2, lon2, out
    l1 = _lib.f64(lat1); o1 = _lib.f64(lon1)
    l2 = _lib.f64(lat2); o2 = _lib.f64(lon2)
    out = self._run("mg_inverse", (l1, o1, l2, o2), outmask, len(_lib.INV_FIELDS))
    return l1, o1, l2, o2, out

  def _gen_inverse(self, lat1, lon1, lat2, lon2, outmask, vector):
    """Run the inverse kernel over matching-length inputs."""
    if not vector:
      out = self._run1("mg_inverse", outmask, len(_lib.INV_FIELDS),
                       lat1, lon1, lat2, lon2)
      if outmask & Geodesic.OUT_MASK & Geodesic.LONG_UNROLL:
        lon12, e = Math.AngDiff(lon1, lon2)
        lon2 = (lon1 + lon12) + e
      else:
        lon2 = Math.AngNormalize(lon2)
      return self._result(outmask, out, (lat1, lon1, lat2), lon2,
                          None, None, None, False)
    l1 = _lib.f64(lat1); o1 = _lib.f64(lon1)
    l2 = _lib.f64(lat2); o2 = _lib.f64(lon2)
    out = self._run("mg_inverse", (l1, o1, l2, o2), outmask, len(_lib.INV_FIELDS))
    if outmask & Geodesic.OUT_MASK & Geodesic.LONG_UNROLL:
      lon12, e = ang_diff(o1, o2)
      lon2 = (o1 + lon12) + e
    else:
      lon2 = ang_normalize(o2)
    return self._result(outmask, out, (lat1, lon1, lat2), lon2, o1, l1, l2, True)

  def Inverse(self, lat1, lon1, lat2, lon2,
              outmask = GeodesicCapability.STANDARD):
    """Solve the inverse geodesic problem

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param lat2: latitude of the second point in degrees
    :param lon2: longitude of the second point in degrees
    :param outmask: the :ref:`output mask <outmask>`
    :return: a :ref:`dict`

    Compute geodesic between (*lat1*, *lon1*) and (*lat2*, *lon2*).
    The default value of *outmask* is STANDARD, i.e., the *lat1*,
    *lon1*, *azi1*, *lat2*, *lon2*, *azi2*, *s12*, *a12* entries are
    returned.

    """

    return self._gen_inverse(lat1, lon1, lat2, lon2, outmask, False)

  def InversePairs(self, lat1, lon1, lat2, lon2,
                   outmask = GeodesicCapability.STANDARD):
    """Solve the inverse geodesic problem for many pairs at once.

    Not in upstream: a drop-in for looping over :meth:`Inverse`, except that
    every input and every entry of the returned dict is a float64 array of the
    same length.

    :param lat1: latitudes of the first points in degrees
    :param lon1: longitudes of the first points in degrees
    :param lat2: latitudes of the second points in degrees
    :param lon2: longitudes of the second points in degrees
    :param outmask: the :ref:`output mask <outmask>`
    :return: a :ref:`dict` of float64 arrays

    """

    return self._gen_inverse(lat1, lon1, lat2, lon2, outmask, True)

  def _gen_direct(self, lat1, lon1, azi1, arcmode, s12_a12, outmask, vector):
    name = "mg_arcdirect" if arcmode else "mg_direct"
    if vector:
      l1 = _lib.f64(lat1); o1 = _lib.f64(lon1)
      a1 = _lib.f64(azi1); d = _lib.f64(s12_a12)
      out = self._run(name, (l1, o1, a1, d), outmask, _lib.DIR_N)
    else:
      out = self._run1(name, outmask, _lib.DIR_N,
                       lat1, lon1, azi1, s12_a12)
    a12, lat2, lon2, azi2, s, m12, M12, M21, S12 = out
    if vector:
      result = {
        'lat1': lat_fix(l1),
        'lon1': o1 if outmask & Geodesic.LONG_UNROLL else ang_normalize(o1),
        'azi1': ang_normalize(a1),
      }
    else:
      result = {'lat1': Math.LatFix(lat1),
                'lon1': lon1 if outmask & Geodesic.LONG_UNROLL else
                Math.AngNormalize(lon1),
                'azi1': Math.AngNormalize(azi1)}
    if arcmode:
      result['a12'] = a12
      if outmask & Geodesic.DISTANCE: result['s12'] = s
    else:
      result['s12'] = s
      result['a12'] = a12
    if outmask & Geodesic.LATITUDE: result['lat2'] = lat2
    if outmask & Geodesic.LONGITUDE: result['lon2'] = lon2
    if outmask & Geodesic.AZIMUTH: result['azi2'] = azi2
    if outmask & Geodesic.REDUCEDLENGTH: result['m12'] = m12
    if outmask & Geodesic.GEODESICSCALE:
      result['M12'] = M12; result['M21'] = M21
    if outmask & Geodesic.AREA: result['S12'] = S12
    return result

  def Direct(self, lat1, lon1, azi1, s12,
             outmask = GeodesicCapability.STANDARD):
    """Solve the direct geodesic problem

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param azi1: azimuth at the first point in degrees
    :param s12: the distance from the first point to the second in
      meters
    :param outmask: the :ref:`output mask <outmask>`
    :return: a :ref:`dict`

    Compute geodesic starting at (*lat1*, *lon1*) with azimuth *azi1*
    and length *s12*.  The default value of *outmask* is STANDARD, i.e.,
    the *lat1*, *lon1*, *azi1*, *lat2*, *lon2*, *azi2*, *s12*, *a12*
    entries are returned.

    """

    return self._gen_direct(lat1, lon1, azi1, False, s12, outmask, False)

  def DirectMany(self, lat1, lon1, azi1, s12,
                 outmask = GeodesicCapability.STANDARD):
    """Solve the direct geodesic problem for many starts at once.

    Not in upstream; see :meth:`InversePairs`.

    """

    return self._gen_direct(lat1, lon1, azi1, False, s12, outmask, True)

  def ArcDirect(self, lat1, lon1, azi1, a12,
                outmask = GeodesicCapability.STANDARD):
    """Solve the direct geodesic problem in terms of spherical arc length

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param azi1: azimuth at the first point in degrees
    :param a12: spherical arc length from the first point to the second
      in degrees
    :param outmask: the :ref:`output mask <outmask>`
    :return: a :ref:`dict`

    Compute geodesic starting at (*lat1*, *lon1*) with azimuth *azi1*
    and arc length *a12*.  The default value of *outmask* is STANDARD,
    i.e., the *lat1*, *lon1*, *azi1*, *lat2*, *lon2*, *azi2*, *s12*,
    *a12* entries are returned.

    """

    return self._gen_direct(lat1, lon1, azi1, True, a12, outmask, False)

  def ArcDirectMany(self, lat1, lon1, azi1, a12,
                    outmask = GeodesicCapability.STANDARD):
    """Solve the arc-length direct problem for many starts at once.

    Not in upstream; see :meth:`InversePairs`.

    """

    return self._gen_direct(lat1, lon1, azi1, True, a12, outmask, True)

  def Line(self, lat1, lon1, azi1,
           caps = GeodesicCapability.STANDARD |
           GeodesicCapability.DISTANCE_IN):
    """Return a GeodesicLine object

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param azi1: azimuth at the first point in degrees
    :param caps: the :ref:`capabilities <outmask>`
    :return: a :class:`~mojogeographiclib.geodesicline.GeodesicLine`

    This allows points along a geodesic starting at (*lat1*, *lon1*),
    with azimuth *azi1* to be found.  The default value of *caps* is
    STANDARD | DISTANCE_IN, allowing direct geodesic problem to be
    solved.

    """

    from mojogeographiclib.geodesicline import GeodesicLine
    return GeodesicLine(self, lat1, lon1, azi1, caps)

  def DirectLine(self, lat1, lon1, azi1, s12,
                 caps = GeodesicCapability.STANDARD |
                 GeodesicCapability.DISTANCE_IN):
    """Define a GeodesicLine object in terms of the direct geodesic
    problem specified in terms of spherical arc length

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param azi1: azimuth at the first point in degrees
    :param s12: the distance from the first point to the second in
      meters
    :param caps: the :ref:`capabilities <outmask>`
    :return: a :class:`~mojogeographiclib.geodesicline.GeodesicLine`

    This function sets point 3 of the GeodesicLine to correspond to
    point 2 of the direct geodesic problem.  The default value of *caps*
    is STANDARD | DISTANCE_IN, allowing direct geodesic problem to be
    solved.

    """

    return self._gen_direct_line(lat1, lon1, azi1, False, s12, caps)

  def ArcDirectLine(self, lat1, lon1, azi1, a12,
                    caps = GeodesicCapability.STANDARD |
                    GeodesicCapability.DISTANCE_IN):
    """Define a GeodesicLine object in terms of the direct geodesic
    problem specified in terms of spherical arc length

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param azi1: azimuth at the first point in degrees
    :param a12: spherical arc length from the first point to the second
      in degrees
    :param caps: the :ref:`capabilities <outmask>`
    :return: a :class:`~mojogeographiclib.geodesicline.GeodesicLine`

    This function sets point 3 of the GeodesicLine to correspond to
    point 2 of the direct geodesic problem.  The default value of *caps*
    is STANDARD | DISTANCE_IN, allowing direct geodesic problem to be
    solved.

    """

    return self._gen_direct_line(lat1, lon1, azi1, True, a12, caps)

  def _gen_direct_line(self, lat1, lon1, azi1, arcmode, s12_a12, caps):
    # Automatically supply DISTANCE_IN if necessary
    if not arcmode: caps |= GeodesicCapability.DISTANCE_IN
    line = self.Line(lat1, lon1, azi1, caps)
    if arcmode:
      line.SetArc(s12_a12)
    else:
      line.SetDistance(s12_a12)
    return line

  def InverseLine(self, lat1, lon1, lat2, lon2,
                  caps = GeodesicCapability.STANDARD |
                  GeodesicCapability.DISTANCE_IN):
    """Define a GeodesicLine object in terms of the invese geodesic problem

    :param lat1: latitude of the first point in degrees
    :param lon1: longitude of the first point in degrees
    :param lat2: latitude of the second point in degrees
    :param lon2: longitude of the second point in degrees
    :param caps: the :ref:`capabilities <outmask>`
    :return: a :class:`~mojogeographiclib.geodesicline.GeodesicLine`

    This function sets point 3 of the GeodesicLine to correspond to
    point 2 of the inverse geodesic problem.  The default value of *caps*
    is STANDARD | DISTANCE_IN, allowing direct geodesic problem to be
    solved.

    """

    from mojogeographiclib.geodesicline import GeodesicLine
    # outmask 0 leaves the raw _GenInverse fields untouched, so salp1/calp1
    # are still available to recover the azimuth, as upstream does.
    _, _, _, _, out = self._inv_fields(
      lat1, lon1, lat2, lon2, GeodesicCapability.EMPTY, False)
    a12 = out[0]
    salp1 = out[2]
    calp1 = out[3]
    azi1 = Math.atan2d(salp1, calp1)
    if caps & (GeodesicCapability.OUT_MASK & GeodesicCapability.DISTANCE_IN):
      caps |= GeodesicCapability.DISTANCE
    line = GeodesicLine(self, lat1, lon1, azi1, caps, salp1, calp1)
    line.SetArc(a12)
    return line

  def Polygon(self, polyline = False):
    """Return a PolygonArea object

    :param polyline: if True then the object describes a polyline
      instead of a polygon
    :return: a :class:`~mojogeographiclib.polygonarea.PolygonArea`

    """

    from mojogeographiclib.polygonarea import PolygonArea
    return PolygonArea(self, polyline)

  EMPTY         = GeodesicCapability.EMPTY
  """No capabilities, no output."""
  LATITUDE      = GeodesicCapability.LATITUDE
  """Calculate latitude *lat2*."""
  LONGITUDE     = GeodesicCapability.LONGITUDE
  """Calculate longitude *lon2*."""
  AZIMUTH       = GeodesicCapability.AZIMUTH
  """Calculate azimuths *azi1* and *azi2*."""
  DISTANCE      = GeodesicCapability.DISTANCE
  """Calculate distance *s12*."""
  STANDARD      = GeodesicCapability.STANDARD
  """All of the above."""
  DISTANCE_IN   = GeodesicCapability.DISTANCE_IN
  """Allow distance *s12* to be used as input in the direct geodesic
  problem."""
  REDUCEDLENGTH = GeodesicCapability.REDUCEDLENGTH
  """Calculate reduced length *m12*."""
  GEODESICSCALE = GeodesicCapability.GEODESICSCALE
  """Calculate geodesic scales *M12* and *M21*."""
  AREA          = GeodesicCapability.AREA
  """Calculate area *S12*."""
  ALL           = GeodesicCapability.ALL
  """All of the above."""
  LONG_UNROLL   = GeodesicCapability.LONG_UNROLL
  """Unroll longitudes, rather than reducing them to the range
  [-180d,180d].

  """



Geodesic.WGS84 = Geodesic(Constants.WGS84_a, Constants.WGS84_f)
"""Instantiation for the WGS84 ellipsoid"""
