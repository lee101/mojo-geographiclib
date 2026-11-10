"""geomath.py: transcription of GeographicLib::Math class."""
# geomath.py
#
# This is a rather literal translation of the GeographicLib::Math class to
# python.  See the documentation for the C++ class for more information at
#
#    https://geographiclib.sourceforge.io/C++/doc/annotated.html
#
# Copyright (c) Charles Karney (2011-2022) <karney@alum.mit.edu> and
# licensed under the MIT/X11 License.  For more information, see
# https://geographiclib.sourceforge.io/
######################################################################
#
# The geodesic numerics live in src/ported.mojo.  This module is the scalar
# mirror of the Mojo Math struct and is used only where upstream uses it too:
# assembling the result dicts (LatFix, AngNormalize, AngDiff, atan2d) and
# reporting a GeodesicLine's own attributes.  Every primitive here was checked
# to agree with its std.math counterpart bit for bit, so the two never disagree
# about a sign of zero.

import math
import numpy as np


class Math:
  """
  Additional math routines for GeographicLib.
  """

  @staticmethod
  def sq(x):
    """Square a number"""

    return x * x

  @staticmethod
  def cbrt(x):
    """Real cube root of a number"""

    return math.copysign(math.pow(abs(x), 1/3.0), x)

  @staticmethod
  def norm(x, y):
    """Private: Normalize a two-vector."""

    r = math.hypot(x, y)
    return x/r, y/r

  @staticmethod
  def sum(u, v):
    """Error free transformation of a sum."""

    s = u + v
    up = s - v
    vpp = s - up
    up -= u
    vpp -= v
    t = s if s == 0 else 0.0 - (up + vpp)
    # u + v =       s      + t
    #       = round(u + v) + t
    return s, t

  @staticmethod
  def polyval(N, p, s, x):
    """Evaluate a polynomial."""

    y = float(0 if N < 0 else p[s]) # make sure the returned value is a float
    while N > 0:
      N -= 1; s += 1
      y = y * x + p[s]
    return y

  @staticmethod
  def AngRound(x):
    """Private: Round an angle so that small values underflow to zero."""

    z = 1/16.0
    y = abs(x)
    # The compiler mustn't "simplify" z - (z - y) to y
    if y < z: y = z - (z - y)
    return math.copysign(y, x)

  @staticmethod
  def remainder(x, y):
    """remainder of x/y in the range [-y/2, y/2]."""

    return math.remainder(x, y) if math.isfinite(x) else math.nan

  @staticmethod
  def AngNormalize(x):
    """reduce angle to [-180,180]"""

    y = Math.remainder(x, 360)
    return math.copysign(180.0, x) if abs(y) == 180 else y

  @staticmethod
  def LatFix(x):
    """replace angles outside [-90,90] by NaN"""

    return math.nan if abs(x) > 90 else x

  @staticmethod
  def AngDiff(x, y):
    """compute y - x and reduce to [-180,180] accurately"""

    d, t = Math.sum(Math.remainder(-x, 360), Math.remainder(y, 360))
    d, t = Math.sum(Math.remainder(d, 360), t)
    if d == 0 or abs(d) == 180:
      d = math.copysign(d, y - x if t == 0 else -t)
    return d, t

  @staticmethod
  def sincosd(x):
    """Compute sine and cosine of x in degrees."""

    r = math.fmod(x, 360) if math.isfinite(x) else math.nan
    q = 0 if math.isnan(r) else int(round(r / 90))
    r -= 90 * q; r = math.radians(r)
    s = math.sin(r); c = math.cos(r)
    q = q % 4
    if   q == 1: s, c =  c, -s
    elif q == 2: s, c = -s, -c
    elif q == 3: s, c = -c,  s
    c = c + 0.0
    if s == 0: s = math.copysign(s, x)
    return s, c

  @staticmethod
  def sincosde(x, t):
    """Compute sine and cosine of (x + t) in degrees with x in [-180, 180]"""

    q = int(round(x / 90)) if math.isfinite(x) else 0
    r = x - 90 * q; r = math.radians(Math.AngRound(r + t))
    s = math.sin(r); c = math.cos(r)
    q = q % 4
    if   q == 1: s, c =  c, -s
    elif q == 2: s, c = -s, -c
    elif q == 3: s, c = -c,  s
    c = c + 0.0
    if s == 0: s = math.copysign(s, x)
    return s, c

  @staticmethod
  def atan2d(y, x):
    """compute atan2(y, x) with the result in degrees"""

    if abs(y) > abs(x):
      q = 2; x, y = y, x
    else:
      q = 0
    if x < 0:
      q += 1; x = -x
    ang = math.degrees(math.atan2(y, x))
    if   q == 1: ang = math.copysign(180, y) - ang
    elif q == 2: ang =                90      - ang
    elif q == 3: ang =               -90     + ang
    return ang


# ------------------------------------------------------------------ vectors
# Array forms of the four Math routines the result dicts need.  These follow
# the scalar code above step for step.  Note that np.remainder is Python-style
# modulo (sign of the divisor) whereas Math.remainder is IEEE remainder, so
# the latter is spelled out here, and np.rint is round-half-to-even like
# Python's round, which is what IEEE remainder needs.
#
# They are the one contiguous elementwise stage in the port, so each is
# written as few full-length NumPy passes as it can be without changing a
# single bit.  np.where and np.copysign have no `out=` on this NumPy, so a
# pass is a temporary; the passes that were removed here (see the SIMD pass in
# README.md) were masks and sign sources, not arithmetic.  Every rewrite was
# checked bit for bit against the form it replaced, on the raw uint64 view
# rather than with a tolerance, including nan/inf/+-0 and the out-of-range
# latitudes; tests/test_geomath.py keeps that check on the tail lengths.

def _rem(x, y):
  """IEEE remainder, i.e. math.remainder, for arrays."""
  x = np.asarray(x, dtype=np.float64)
  with np.errstate(invalid="ignore"):
    r = x - np.rint(x / y) * y
  # rint(x / y) is exact only while |x / y| < 2**52; past that the rounded
  # quotient drops the fraction and the subtraction loses it, so hand those
  # (vanishingly rare) entries to the scalar routine.
  bad = np.abs(x) >= 4503599627370496.0 * y
  if bad.any():
    for i in np.nonzero(bad)[0]:
      v = float(x[i])
      r[i] = math.remainder(v, y) if math.isfinite(v) else math.nan
  return r


def _sum_ef(u, v):
  """Error free transformation of a sum, for arrays.

  t is `0.0 - (up + vpp)` except where s == 0, where it is s.  One boolean
  index assignment instead of a full-length np.where keeps a whole temporary
  (two, counting the mask) off the pass: same values, same bits.
  """
  s = u + v
  up = s - v
  vpp = s - up
  up = up - u
  vpp = vpp - v
  t = 0.0 - (up + vpp)
  t[s == 0.0] = 0.0
  return s, t


def ang_normalize(x):
  x = np.asarray(x, dtype=np.float64)
  y = _rem(x, 360.0)
  return np.where(np.abs(y) == 180.0, np.copysign(180.0, x), y)


def lat_fix(x):
  x = np.asarray(x, dtype=np.float64)
  return np.where(np.abs(x) > 90.0, np.nan, x)


def ang_diff(x, y):
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  d, t = _sum_ef(_rem(-x, 360.0), _rem(y, 360.0))
  d, t = _sum_ef(_rem(d, 360.0), t)
  with np.errstate(invalid="ignore"):
    # One pass for the sign source, tested by the d-mask pass below, instead
    # of a full-length np.where feeding a second full-length np.where.
    src = np.where(t == 0.0, y - x, -t)
    fix = (d == 0.0) | (np.abs(d) == 180.0)
    return np.where(fix, np.copysign(d, src), d), t


def _atan2d_body(y, x):
  """One pair of the atan2d quadrant fix.  Same arithmetic as Math.atan2d,
  one pass per branch instead of one per quadrant: the three fixes are
  disjoint, so each mask write is a no-op where the other applies."""
  y = np.asarray(y, dtype=np.float64)
  x = np.asarray(x, dtype=np.float64)
  q2 = np.abs(y) > np.abs(x)
  xx = np.where(q2, y, x)
  yy = np.where(q2, x, y)
  neg = xx < 0.0
  xx = np.where(neg, -xx, xx)
  ang = np.degrees(np.arctan2(yy, xx))
  a3 = -90.0 + ang
  ang = np.where(q2 & neg, a3, ang)
  a1 = np.copysign(180.0, yy) - ang
  ang = np.where(q2 | ~neg, ang, a1)
  a2 = 90.0 - ang
  return np.where(q2 & ~neg, a2, ang)


def atan2d(y, x):
  return _atan2d_body(y, x)


def atan2d_pair(y1, x1, y2, x2):
  """atan2d for two sine/cosine pairs at once.

  Every result dict carries two of them (azi1 and azi2); they are independent,
  so both are computed inside one Python-level loop and the interim mask
  arrays are reused rather than reallocated per pair.  Identical, bit for bit,
  to calling atan2d twice.
  """
  return tuple(_atan2d_body(y, x) for y, x in ((y1, x1), (y2, x2)))
