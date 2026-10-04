"""Port of GeographicLib's geodesic routines to Mojo.

Read top to bottom this file follows the upstream pure-Python package, in
upstream's module order, so the two can be diffed by eye:

    Math         <- geographiclib/geomath.py
    Accumulator  <- geographiclib/accumulator.py
    Geodesic     <- geographiclib/geodesic.py
    GeodesicLine <- geographiclib/geodesicline.py
    PolygonArea  <- geographiclib/polygonarea.py

Upstream is Copyright (c) Charles Karney (2011-2022), MIT/X11 licensed; see
LICENSE.  The algorithms are from

    Charles F. F. Karney, Algorithms for geodesics, J. Geodesy 87, 43-55
    (2013), https://doi.org/10.1007/s00190-012-0578-z

The only structural changes forced by Mojo are the scratch-buffer arguments
noted at each site; every branch and every arithmetic step follows upstream.
"""

from std.math import (
    atan,
    atan2,
    atanh,
    cbrt,
    copysign,
    cos,
    hypot,
    isfinite,
    isnan,
    pow,
    remainder,
    sin,
    sqrt,
    tan,
)

# Coefficients and scratch live in flat float64 buffers owned by the caller,
# because a Mojo struct field cannot hold a parametrically-origined pointer.
comptime CoefPtr = Pointer[Float64, UntrackedOrigin[mut=True]]

comptime PI: Float64 = 3.141592653589793
comptime DEG_TO_RAD: Float64 = PI / 180.0
comptime RAD_TO_DEG: Float64 = 180.0 / PI
comptime NAN: Float64 = 0.0 / 0.0
comptime INV3: Float64 = 1.0 / 3.0


# =====================================================================
# geographiclib/geomath.py -- class Math
# =====================================================================
struct Math:
    """Additional math routines for GeographicLib."""

    # Not in upstream geomath: std.math has no fmod, and Mojo's `%` on floats
    # is Python-style (sign of the divisor), not C fmod.  This reproduces C
    # fmod exactly by dyadic reduction, for any finite x and y != 0.
    @staticmethod
    def fmod(x: Float64, y: Float64) -> Float64:
        if not isfinite(x) or not isfinite(y) or y == 0.0:
            return NAN
        var r = x
        var ay = abs(y)
        while abs(r) >= ay:
            var t = ay
            while t * 2.0 <= abs(r):
                t *= 2.0
            r -= copysign(t, r)
        return r

    @staticmethod
    def radians(x: Float64) -> Float64:
        """Not upstream: upstream calls the math module directly."""
        return x * DEG_TO_RAD

    @staticmethod
    def degrees(x: Float64) -> Float64:
        """Not upstream: upstream calls the math module directly."""
        return x * RAD_TO_DEG

    @staticmethod
    def sq(x: Float64) -> Float64:
        """Square a number"""
        return x * x

    @staticmethod
    def cbrt(x: Float64) -> Float64:
        """Real cube root of a number"""
        return copysign(cbrt(abs(x)), x)

    @staticmethod
    def norm(x: Float64, y: Float64) -> Tuple[Float64, Float64]:
        """Private: Normalize a two-vector."""
        # hypot is bit-for-bit libm hypot on this toolchain; the upstream
        # version guard for Python 3.8/3.9 is not needed here.
        var r = hypot(x, y)
        return (x / r, y / r)

    @staticmethod
    def sum(u: Float64, v: Float64) -> Tuple[Float64, Float64]:
        """Error free transformation of a sum."""
        # Error free transformation of a sum.  Note that t can be the same as
        # one of the first two arguments.
        var s = u + v
        var up = s - v
        var vpp = s - up
        up -= u
        vpp -= v
        var t = s if s == 0.0 else 0.0 - (up + vpp)
        # u + v =       s      + t
        #       = round(u + v) + t
        return (s, t)

    @staticmethod
    def polyval(N: Int, p: CoefPtr, s: Int, x: Float64) -> Float64:
        """Evaluate a polynomial."""
        var y = 0.0 if N < 0 else p.unsafe_load(s)  # make sure y is a float
        var i = N
        var j = s
        while i > 0:
            i -= 1
            j += 1
            y = y * x + p.unsafe_load(j)
        return y

    @staticmethod
    def AngRound(x: Float64) -> Float64:
        """Private: Round an angle so that small values underflow to zero."""
        # The makes the smallest gap in x = 1/16 - nextafter(1/16, 0) = 1/2^57
        # for reals = 0.7 pm on the earth if x is an angle in degrees.  We use
        # this to avoid having to deal with near singular cases when x is
        # non-zero but tiny (e.g., 1.0e-200).
        var z = 1.0 / 16.0
        var y = abs(x)
        # The compiler mustn't "simplify" z - (z - y) to y
        if y < z:
            y = z - (z - y)
        return copysign(y, x)

    @staticmethod
    def remainder(x: Float64, y: Float64) -> Float64:
        """remainder of x/y in the range [-y/2, y/2]."""
        return remainder(x, y) if isfinite(x) else NAN

    @staticmethod
    def AngNormalize(x: Float64) -> Float64:
        """reduce angle to [-180,180]"""
        var y = Math.remainder(x, 360.0)
        return copysign(180.0, x) if abs(y) == 180.0 else y

    @staticmethod
    def LatFix(x: Float64) -> Float64:
        """replace angles outside [-90,90] by NaN"""
        return NAN if abs(x) > 90.0 else x

    @staticmethod
    def AngDiff(x: Float64, y: Float64) -> Tuple[Float64, Float64]:
        """compute y - x and reduce to [-180,180] accurately"""
        var d, t = Math.sum(Math.remainder(-x, 360.0), Math.remainder(y, 360.0))
        d, t = Math.sum(Math.remainder(d, 360.0), t)
        if d == 0.0 or abs(d) == 180.0:
            d = copysign(d, y - x if t == 0.0 else -t)
        return (d, t)

    @staticmethod
    def sincosd(x: Float64) -> Tuple[Float64, Float64]:
        """Compute sine and cosine of x in degrees."""
        var r = Math.fmod(x, 360.0) if isfinite(x) else NAN
        var q = 0 if isnan(r) else Int(round(r / 90.0))
        r -= 90.0 * Float64(q)
        r = Math.radians(r)
        var s = sin(r)
        var c = cos(r)
        q = q % 4
        if q == 1:
            s, c = c, -s
        elif q == 2:
            s, c = -s, -c
        elif q == 3:
            s, c = -c, s
        c = c + 0.0
        if s == 0.0:
            s = copysign(s, x)
        return (s, c)

    @staticmethod
    def sincosde(x: Float64, t: Float64) -> Tuple[Float64, Float64]:
        """Compute sine and cosine of (x + t) in degrees with x in [-180, 180]"""
        var q = Int(round(x / 90.0)) if isfinite(x) else 0
        var r = x - 90.0 * Float64(q)
        r = Math.radians(Math.AngRound(r + t))
        var s = sin(r)
        var c = cos(r)
        q = q % 4
        if q == 1:
            s, c = c, -s
        elif q == 2:
            s, c = -s, -c
        elif q == 3:
            s, c = -c, s
        c = c + 0.0
        if s == 0.0:
            s = copysign(s, x)
        return (s, c)

    @staticmethod
    def atan2d(y: Float64, x: Float64) -> Float64:
        """compute atan2(y, x) with the result in degrees"""
        var yy = y
        var xx = x
        var q = 0
        if abs(yy) > abs(xx):
            q = 2
            xx = yy
            yy = x
        if xx < 0.0:
            q += 1
            xx = -xx
        var ang = Math.degrees(atan2(yy, xx))
        if q == 1:
            ang = copysign(180.0, yy) - ang
        elif q == 2:
            ang = 90.0 - ang
        elif q == 3:
            ang = -90.0 + ang
        return ang


# =====================================================================
# geographiclib/accumulator.py -- class Accumulator
# =====================================================================
struct Accumulator:
    """Like math.fsum, but allows a running sum"""

    var _s: Float64
    var _t: Float64

    def Set(mut self, y: Float64) -> None:
        """Set value from argument"""
        self._s = y
        self._t = 0.0

    def Set(mut self, y: Accumulator) -> None:
        """Set value from argument"""
        self._s = y._s
        self._t = y._t

    # Not upstream: the ctypes layer keeps the two words of an Accumulator in
    # one float64 buffer and reloads them without building an intermediate.
    def Set(mut self, s: Float64, t: Float64) -> None:
        self._s = s
        self._t = t

    def __init__(out self, y: Float64 = 0.0):
        """Constructor"""
        self._s = 0.0
        self._t = 0.0
        self.Set(y)

    def __init__(out self, y: Accumulator):
        """Constructor"""
        self._s = y._s
        self._t = y._t

    def Add(mut self, y: Float64) -> None:
        """Add a value"""
        # Here's Shewchuk's solution...
        # hold exact sum as [s, t, u]
        # (see upstream for the discussion of the 1-ulp "failures" and why
        # they are not worth fixing)
        var y2, u = Math.sum(y, self._t)  # Accumulate starting at
        self._s, self._t = Math.sum(y2, self._s)  # least significant end
        if self._s == 0.0:  # This implies t == 0,
            self._s = u  # so result is u
        else:
            self._t += u  # otherwise just accumulate u to t.

    def Sum(self, y: Float64 = 0.0) -> Float64:
        """Return sum + y"""
        if y == 0.0:
            return self._s
        var b = Accumulator(0.0)
        b.Set(self)
        b.Add(y)
        return b._s

    def Negate(mut self) -> None:
        """Negate sum"""
        self._s *= -1.0
        self._t *= -1.0

    def Remainder(mut self, y: Float64) -> None:
        """Remainder on division by y"""
        self._s = Math.remainder(self._s, y)
        self.Add(0.0)


# =====================================================================
# geographiclib/geodesiccapability.py
# =====================================================================
comptime CAP_NONE: Int = 0
comptime CAP_C1: Int = 1 << 0
comptime CAP_C1p: Int = 1 << 1
comptime CAP_C2: Int = 1 << 2
comptime CAP_C3: Int = 1 << 3
comptime CAP_C4: Int = 1 << 4
comptime CAP_ALL: Int = 0x1F
comptime CAP_MASK: Int = CAP_ALL
comptime OUT_ALL: Int = 0x7F80
comptime OUT_MASK: Int = 0xFF80  # Includes LONG_UNROLL
comptime EMPTY: Int = 0
comptime LATITUDE: Int = (1 << 7) | CAP_NONE
comptime LONGITUDE: Int = (1 << 8) | CAP_C3
comptime AZIMUTH: Int = (1 << 9) | CAP_NONE
comptime DISTANCE: Int = (1 << 10) | CAP_C1
comptime STANDARD: Int = LATITUDE | LONGITUDE | AZIMUTH | DISTANCE
comptime DISTANCE_IN: Int = (1 << 11) | CAP_C1 | CAP_C1p
comptime REDUCEDLENGTH: Int = (1 << 12) | CAP_C1 | CAP_C2
comptime GEODESICSCALE: Int = (1 << 13) | CAP_C1 | CAP_C2
comptime AREA: Int = (1 << 14) | CAP_C4
comptime LONG_UNROLL: Int = 1 << 15
comptime ALL: Int = OUT_ALL | CAP_ALL  # Does not include LONG_UNROLL

# =====================================================================
# geographiclib/geodesic.py -- class Geodesic
# =====================================================================

# Geodesic.GEOGRAPHICLIB_GEODESIC_ORDER and the derived orders.
comptime GEOGRAPHICLIB_GEODESIC_ORDER: Int = 6
comptime nA1_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nC1_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nC1p_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nA2_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nC2_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nA3_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nA3x_: Int = nA3_
comptime nC3_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nC3x_: Int = (nC3_ * (nC3_ - 1)) // 2
comptime nC4_: Int = GEOGRAPHICLIB_GEODESIC_ORDER
comptime nC4x_: Int = (nC4_ * (nC4_ + 1)) // 2
comptime maxit1_: Int = 20
comptime maxit2_: Int = maxit1_ + 53 + 10  # + sys.float_info.mant_dig

comptime tiny_: Float64 = sqrt(2.2250738585072014e-308)  # sys.float_info.min
comptime tol0_: Float64 = 2.220446049250313e-16  # sys.float_info.epsilon
comptime tol1_: Float64 = 200.0 * tol0_
comptime tol2_: Float64 = sqrt(tol0_)
comptime tolb_: Float64 = tol0_
comptime xthresh_: Float64 = 1000.0 * tol2_

# Offsets into the caller's 336-double Geodesic block.  Upstream keeps these
# tables in Python lists hanging off the Geodesic object; Mojo struct fields
# cannot hold a pointer of parametric origin, so the block is borrowed.
comptime K_A1M1: Int = 0  # 5
comptime K_C1F: Int = 5  # 18
comptime K_C1PF: Int = 23  # 18
comptime K_C2F: Int = 41  # 18
comptime K_A2M1: Int = 59  # 5
comptime K_A3: Int = 64  # 18
comptime K_C3: Int = 82  # 45
comptime K_C4: Int = 127  # 77
comptime E_A3X: Int = 204  # 6
comptime E_C3X: Int = 210  # 15
comptime E_C4X: Int = 225  # 21
comptime S_C1A: Int = 246  # 7
comptime S_C2A: Int = 253  # 7
comptime S_C3A: Int = 260  # 6
comptime S_C4A: Int = 266  # 6
comptime L_STATE: Int = 272  # 65: scratch GeodesicLine for _GenDirect
comptime GEOD_BLOCK: Int = 337

# Layout of a GeodesicLine state block (65 doubles): 32 scalars then tables.
comptime LN_NSCALAR: Int = 32
comptime LN_C1A: Int = 32
comptime LN_C1PA: Int = 39
comptime LN_C2A: Int = 46
comptime LN_C3A: Int = 53
comptime LN_C4A: Int = 59
comptime LINE_BLOCK: Int = 65


struct Geodesic:
    """Solve geodesic problems"""

    var a: Float64
    # The equatorial radius in meters (readonly)
    var f: Float64
    # The flattening (readonly)
    var _f1: Float64
    var _e2: Float64
    var _ep2: Float64
    var _n: Float64
    var _b: Float64
    # authalic radius squared
    var _c2: Float64
    var _etol2: Float64
    var _blk: CoefPtr

    @staticmethod
    def _SinCosSeries(
        sinp: Bool, sinx: Float64, cosx: Float64, c: CoefPtr, n: Int
    ) -> Float64:
        """Private: Evaluate a trig series using Clenshaw summation."""
        # Evaluate
        # y = sinp ? sum(c[i] * sin( 2*i    * x), i, 1, n) :
        #            sum(c[i] * cos((2*i+1) * x), i, 0, n-1)
        # using Clenshaw summation.  N.B. c[0] is unused for sin series
        # Approx operation count = (n + 5) mult and (2 * n + 2) add
        # (upstream takes the array; Mojo takes (pointer, length))
        var k = n  # Point to one beyond last element
        var m = k - (1 if sinp else 0)
        var ar = 2 * (cosx - sinx) * (cosx + sinx)  # 2 * cos(2 * x)
        var y1 = 0.0  # accumulators for sum
        var y0 = 0.0
        if m & 1:
            k -= 1
            y0 = c.unsafe_load(k)
        # Now m is even
        m = m // 2
        while m:  # while m--:
            m -= 1
            # Unroll loop x 2, so accumulators return to their original role
            k -= 1
            y1 = ar * y0 - y1 + c.unsafe_load(k)
            k -= 1
            y0 = ar * y1 - y0 + c.unsafe_load(k)
        return (
            2 * sinx * cosx * y0 if sinp else cosx * (y0 - y1)
        )  # sin(2 * x) * y0 : cos(x) * (y0 - y1)

    @staticmethod
    def _Astroid(x: Float64, y: Float64) -> Float64:
        """Private: solve astroid equation."""
        # Solve k^4+2*k^3-(x^2+y^2-1)*k^2-2*y^2*k-y^2 = 0 for positive root
        # k.  This solution is adapted from Geocentric::Reverse.
        var p = Math.sq(x)
        var q = Math.sq(y)
        var r = (p + q - 1) / 6
        var k = 0.0
        if not (q == 0.0 and r <= 0.0):
            # Avoid possible division by zero when r = 0 by multiplying
            # equations for s and t by r^3 and r, resp.
            var S = p * q / 4  # S = r^3 * s
            var r2 = Math.sq(r)
            var r3 = r * r2
            # The discriminant of the quadratic equation for T3.  This is zero
            # on the evolute curve p^(1/3)+q^(1/3) = 1
            var disc = S * (S + 2 * r3)
            var u = r
            if disc >= 0.0:
                var T3 = S + r3
                # Pick the sign on the sqrt to maximize abs(T3).  This
                # minimizes loss of precision due to cancellation.  The result
                # is unchanged because of the way the T is used in definition
                # of u.
                T3 += -sqrt(disc) if T3 < 0.0 else sqrt(disc)  # T3 = (r*t)^3
                # N.B. cbrt always returns the real root.  cbrt(-8) = -2.
                var T = Math.cbrt(T3)  # T = r * t
                # T can be zero; but then r2 / T -> 0.
                u += T + (r2 / T if T != 0.0 else 0.0)
            else:
                # T is complex, but the way u is defined the result is real.
                var ang = atan2(sqrt(-disc), -(S + r3))
                # There are three possible cube roots.  We choose the root
                # which avoids cancellation.  Note that disc < 0 implies that
                # r < 0.
                u += 2 * r * cos(ang / 3)
            var v = sqrt(Math.sq(u) + q)  # guaranteed positive
            # Avoid loss of accuracy when u < 0.
            var uv = q / (v - u) if u < 0.0 else u + v  # u+v, guaranteed +ve
            var w = (uv - q) / (2 * v)  # positive?
            # Rearrange expression for k to avoid loss of accuracy due to
            # subtraction.  Division by 0 not possible because uv > 0,
            # w >= 0.
            k = uv / (sqrt(uv + Math.sq(w)) + w)  # guaranteed positive
        # else: q == 0 && r <= 0
        #   y = 0 with |x| <= 1.  Handle this case directly.
        #   for y small, positive root is k = abs(y)/sqrt(1-x^2)
        return k

    @staticmethod
    def _A1m1f(eps: Float64, k: CoefPtr) -> Float64:
        """Private: return A1-1."""
        var m = nA1_ // 2
        var t = Math.polyval(m, k, 0, Math.sq(eps)) / k.unsafe_load(m + 1)
        return (t + eps) / (1.0 - eps)

    @staticmethod
    def _C1f(eps: Float64, c: CoefPtr, k: CoefPtr) -> None:
        """Private: return C1."""
        var eps2 = Math.sq(eps)
        var d = eps
        var o = 0
        for l in range(1, nC1_ + 1):  # l is index of C1p[l]
            var m = (nC1_ - l) // 2  # order of polynomial in eps^2
            c.unsafe_store(l, d * Math.polyval(m, k, o, eps2) / k.unsafe_load(o + m + 1))
            o += m + 2
            d *= eps

    @staticmethod
    def _C1pf(eps: Float64, c: CoefPtr, k: CoefPtr) -> None:
        """Private: return C1'"""
        var eps2 = Math.sq(eps)
        var d = eps
        var o = 0
        for l in range(1, nC1p_ + 1):  # l is index of C1p[l]
            var m = (nC1p_ - l) // 2  # order of polynomial in eps^2
            c.unsafe_store(l, d * Math.polyval(m, k, o, eps2) / k.unsafe_load(o + m + 1))
            o += m + 2
            d *= eps

    @staticmethod
    def _A2m1f(eps: Float64, k: CoefPtr) -> Float64:
        """Private: return A2-1"""
        var m = nA2_ // 2
        var t = Math.polyval(m, k, 0, Math.sq(eps)) / k.unsafe_load(m + 1)
        return (t - eps) / (1.0 + eps)

    @staticmethod
    def _C2f(eps: Float64, c: CoefPtr, k: CoefPtr) -> None:
        """Private: return C2"""
        var eps2 = Math.sq(eps)
        var d = eps
        var o = 0
        for l in range(1, nC2_ + 1):  # l is index of C2[l]
            var m = (nC2_ - l) // 2  # order of polynomial in eps^2
            c.unsafe_store(l, d * Math.polyval(m, k, o, eps2) / k.unsafe_load(o + m + 1))
            o += m + 2
            d *= eps

    def __init__(out self, a: Float64, f: Float64, blk: CoefPtr, ready: Bool):
        """Construct a Geodesic object

        :param a: the equatorial radius of the ellipsoid in meters
        :param f: the flattening of the ellipsoid

        Upstream takes only (a, f) and owns its coefficient lists.  Mojo
        cannot hold such a list in a struct field, so the caller supplies a
        borrowed block (see GEOD_BLOCK) and `ready` says whether the
        ellipsoid-dependent tables in it have already been evaluated.  The
        Python API, `Geodesic(a, f)`, matches upstream exactly.

        """
        self.a = a
        self.f = f
        self._f1 = 1.0 - self.f
        self._e2 = self.f * (2.0 - self.f)
        self._ep2 = self._e2 / Math.sq(self._f1)  # e2 / (1 - e2)
        self._n = self.f / (2.0 - self.f)
        self._b = self.a * self._f1
        # authalic radius squared
        self._c2 = (
            Math.sq(self.a)
            + Math.sq(self._b)
            * (
                1.0
                if self._e2 == 0.0
                else (
                    atanh(sqrt(self._e2))
                    if self._e2 > 0.0
                    else atan(sqrt(-self._e2))
                )
                / sqrt(abs(self._e2))
            )
        ) / 2.0
        # The sig12 threshold for "really short".  Using the auxiliary sphere
        # solution with dnm computed at (bet1 + bet2) / 2, the relative error in
        # the azimuth consistency check is sig12^2 * abs(f) * min(1, 1-f/2) / 2.
        # (Error measured for 1/100 < b/a < 100 and abs(f) >= 1/1000.  For a
        # given f and sig12, the max error occurs for lines near the pole.  If
        # the old rule for computing dnm = (dn1 + dn2)/2 is used, then the
        # error increases by a factor of 2.)  Setting this equal to epsilon
        # gives sig12 = etol2.  Here 0.1 is a safety factor (error decreased by
        # 100) and max(0.001, abs(f)) stops etol2 getting too large in the
        # nearly spherical case.
        self._etol2 = 0.1 * tol2_ / sqrt(
            max(0.001, abs(self.f)) * min(1.0, 1.0 - self.f / 2.0) / 2.0
        )
        self._blk = blk
        if not ready:
            self._A3coeff()
            self._C3coeff()
            self._C4coeff()

    def _A3coeff(mut self) -> None:
        """Private: return coefficients for A3"""
        var coeff = self._blk.unsafe_offset(K_A3)
        var a3x = self._blk.unsafe_offset(E_A3X)
        var o = 0
        var k = 0
        for j in range(nA3_ - 1, -1, -1):  # coeff of eps^j
            var m = min(nA3_ - j - 1, j)  # order of polynomial in n
            a3x.unsafe_store(
                k,
                Math.polyval(m, coeff, o, self._n) / coeff.unsafe_load(o + m + 1),
            )
            k += 1
            o += m + 2

    def _C3coeff(mut self) -> None:
        """Private: return coefficients for C3"""
        var coeff = self._blk.unsafe_offset(K_C3)
        var c3x = self._blk.unsafe_offset(E_C3X)
        var o = 0
        var k = 0
        for l in range(1, nC3_):  # l is index of C3[l]
            for j in range(nC3_ - 1, l - 1, -1):  # coeff of eps^j
                var m = min(nC3_ - j - 1, j)  # order of polynomial in n
                c3x.unsafe_store(
                    k,
                    Math.polyval(m, coeff, o, self._n) / coeff.unsafe_load(o + m + 1),
                )
                k += 1
                o += m + 2

    def _C4coeff(mut self) -> None:
        """Private: return coefficients for C4"""
        var coeff = self._blk.unsafe_offset(K_C4)
        var c4x = self._blk.unsafe_offset(E_C4X)
        var o = 0
        var k = 0
        for l in range(nC4_):  # l is index of C4[l]
            for j in range(nC4_ - 1, l - 1, -1):  # coeff of eps^j
                var m = nC4_ - j - 1  # order of polynomial in n
                c4x.unsafe_store(
                    k,
                    Math.polyval(m, coeff, o, self._n) / coeff.unsafe_load(o + m + 1),
                )
                k += 1
                o += m + 2

    def _A3f(self, eps: Float64) -> Float64:
        """Private: return A3"""
        # Evaluate A3
        return Math.polyval(nA3_ - 1, self._blk.unsafe_offset(E_A3X), 0, eps)

    def _C3f(self, eps: Float64, c: CoefPtr) -> None:
        """Private: return C3"""
        # Evaluate C3
        # Elements c[1] thru c[nC3_ - 1] are set
        var c3x = self._blk.unsafe_offset(E_C3X)
        var mult = 1.0
        var o = 0
        for l in range(1, nC3_):  # l is index of C3[l]
            var m = nC3_ - l - 1  # order of polynomial in eps
            mult *= eps
            c.unsafe_store(l, mult * Math.polyval(m, c3x, o, eps))
            o += m + 1

    def _C4f(self, eps: Float64, c: CoefPtr) -> None:
        """Private: return C4"""
        # Evaluate C4 coeffs by Horner's method
        # Elements c[0] thru c[nC4_ - 1] are set
        var c4x = self._blk.unsafe_offset(E_C4X)
        var mult = 1.0
        var o = 0
        for l in range(nC4_):  # l is index of C4[l]
            var m = nC4_ - l - 1  # order of polynomial in eps
            c.unsafe_store(l, mult * Math.polyval(m, c4x, o, eps))
            o += m + 1
            mult *= eps

    # return s12b, m12b, m0, M12, M21
    def _Lengths(
        self,
        eps: Float64,
        sig12: Float64,
        ssig1: Float64,
        csig1: Float64,
        dn1: Float64,
        ssig2: Float64,
        csig2: Float64,
        dn2: Float64,
        cbet1: Float64,
        cbet2: Float64,
        outmask: Int,
        # Scratch areas of the right size
        C1a: CoefPtr,
        C2a: CoefPtr,
    ) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
        """Private: return a bunch of lengths"""
        # Return s12b, m12b, m0, M12, M21, where
        # m12b = (reduced length)/_b; s12b = distance/_b,
        # m0 = coefficient of secular term in expression for reduced length.
        var blk = self._blk
        var om = outmask & OUT_MASK  # (upstream: outmask &= OUT_MASK)
        # om & DISTANCE: set s12b
        # om & REDUCEDLENGTH: set m12b & m0
        # om & GEODESICSCALE: set M12 & M21
        var s12b = NAN
        var m12b = NAN
        var m0 = NAN
        var m0x = NAN
        var J12 = NAN
        var M12 = NAN
        var M21 = NAN
        var A1 = NAN
        var A2 = NAN
        if om & (DISTANCE | REDUCEDLENGTH | GEODESICSCALE):
            A1 = Geodesic._A1m1f(eps, blk.unsafe_offset(K_A1M1))
            Geodesic._C1f(eps, C1a, blk.unsafe_offset(K_C1F))
            if om & (REDUCEDLENGTH | GEODESICSCALE):
                A2 = Geodesic._A2m1f(eps, blk.unsafe_offset(K_A2M1))
                Geodesic._C2f(eps, C2a, blk.unsafe_offset(K_C2F))
                m0x = A1 - A2
                A2 = 1.0 + A2
            A1 = 1.0 + A1
        if om & DISTANCE:
            var B1 = Geodesic._SinCosSeries(True, ssig2, csig2, C1a, nC1_ + 1) - Geodesic._SinCosSeries(
                True, ssig1, csig1, C1a, nC1_ + 1
            )
            # Missing a factor of _b
            s12b = A1 * (sig12 + B1)
            if om & (REDUCEDLENGTH | GEODESICSCALE):
                var B2 = Geodesic._SinCosSeries(
                    True, ssig2, csig2, C2a, nC2_ + 1
                ) - Geodesic._SinCosSeries(True, ssig1, csig1, C2a, nC2_ + 1)
                J12 = m0x * sig12 + (A1 * B1 - A2 * B2)
        elif om & (REDUCEDLENGTH | GEODESICSCALE):
            # Assume here that nC1_ >= nC2_
            for l in range(1, nC2_):
                C2a.unsafe_store(l, A1 * C1a.unsafe_load(l) - A2 * C2a.unsafe_load(l))
            J12 = m0x * sig12 + (
                Geodesic._SinCosSeries(True, ssig2, csig2, C2a, nC2_ + 1)
                - Geodesic._SinCosSeries(True, ssig1, csig1, C2a, nC2_ + 1)
            )
        if om & REDUCEDLENGTH:
            m0 = m0x
            # Missing a factor of _b.
            # Add parens around (csig1 * ssig2) and (ssig1 * csig2) to ensure
            # accurate cancellation in the case of coincident points.
            m12b = (
                dn2 * (csig1 * ssig2)
                - dn1 * (ssig1 * csig2)
                - csig1 * csig2 * J12
            )
        if om & GEODESICSCALE:
            var csig12 = csig1 * csig2 + ssig1 * ssig2
            var t = self._ep2 * (cbet1 - cbet2) * (cbet1 + cbet2) / (dn1 + dn2)
            M12 = csig12 + (t * ssig2 - csig2 * J12) * ssig1 / dn1
            M21 = csig12 - (t * ssig1 - csig1 * J12) * ssig2 / dn2
        return (s12b, m12b, m0, M12, M21)

    # return sig12, salp1, calp1, salp2, calp2, dnm
    def _InverseStart(
        self,
        sbet1: Float64,
        cbet1: Float64,
        dn1: Float64,
        sbet2: Float64,
        cbet2: Float64,
        dn2: Float64,
        lam12: Float64,
        slam12: Float64,
        clam12: Float64,
        # Scratch areas of the right size
        C1a: CoefPtr,
        C2a: CoefPtr,
    ) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
        """Private: Find a starting value for Newton's method."""
        # Return a starting point for Newton's method in salp1 and calp1
        # (function value is -1).  If Newton's method doesn't need to be used,
        # return also salp2 and calp2 and function value is sig12.
        var sig12 = -1.0
        var salp2 = NAN  # Return values
        var calp2 = NAN
        var dnm = NAN
        # bet12 = bet2 - bet1 in [0, pi); bet12a = bet2 + bet1 in (-pi, 0]
        var sbet12 = sbet2 * cbet1 - cbet2 * sbet1
        var cbet12 = cbet2 * cbet1 + sbet2 * sbet1
        # Volatile declaration needed to fix inverse cases
        # 88.202499451857 0 -88.202499451857 179.981022032992859592
        # 89.262080389218 0 -89.262080389218 179.992207982775375662
        # 89.333123580033 0 -89.333123580032997687 179.99295812360148422
        # which otherwise fail with g++ 4.4.4 x86 -O3
        var sbet12a = sbet2 * cbet1
        sbet12a += cbet2 * sbet1

        var shortline = cbet12 >= 0.0 and sbet12 < 0.5 and cbet2 * lam12 < 0.5
        var somg12 = 0.0
        var comg12 = 0.0
        if shortline:
            var sbetm2 = Math.sq(sbet1 + sbet2)
            # sin((bet1+bet2)/2)^2
            # =  (sbet1 + sbet2)^2 / ((sbet1 + sbet2)^2 + (cbet1 + cbet2)^2)
            sbetm2 /= sbetm2 + Math.sq(cbet1 + cbet2)
            dnm = sqrt(1.0 + self._ep2 * sbetm2)
            var omg12 = lam12 / (self._f1 * dnm)
            somg12 = sin(omg12)
            comg12 = cos(omg12)
        else:
            somg12 = slam12
            comg12 = clam12

        var salp1 = cbet2 * somg12
        var calp1 = (
            sbet12 + cbet2 * sbet1 * Math.sq(somg12) / (1.0 + comg12)
            if comg12 >= 0.0
            else sbet12a - cbet2 * sbet1 * Math.sq(somg12) / (1.0 - comg12)
        )

        var ssig12 = hypot(salp1, calp1)
        var csig12 = sbet1 * sbet2 + cbet1 * cbet2 * comg12

        if shortline and ssig12 < self._etol2:
            # really short lines
            salp2 = cbet1 * somg12
            calp2 = sbet12 - cbet1 * sbet2 * (
                Math.sq(somg12) / (1.0 + comg12) if comg12 >= 0.0 else 1.0 - comg12
            )
            salp2, calp2 = Math.norm(salp2, calp2)
            # Set return value
            sig12 = atan2(ssig12, csig12)
        elif (
            abs(self._n) >= 0.1  # Skip astroid calc if too eccentric
            or csig12 >= 0.0
            or ssig12 >= 6.0 * abs(self._n) * PI * Math.sq(cbet1)
        ):
            # Nothing to do, zeroth order spherical approximation is OK
            pass
        else:
            # Scale lam12 and bet2 to x, y coordinate system where antipodal
            # point is at origin and singular point is at y = 0, x = -1.
            # real y, lamscale, betscale
            var lam12x = atan2(-slam12, -clam12)
            var lamscale = 0.0
            var betscale = 0.0
            var x = 0.0
            var y = 0.0
            if self.f >= 0.0:  # In fact f == 0 does not get here
                # x = dlong, y = dlat
                var k2 = Math.sq(sbet1) * self._ep2
                var eps = k2 / (2.0 * (1.0 + sqrt(1.0 + k2)) + k2)
                lamscale = self.f * cbet1 * self._A3f(eps) * PI
                betscale = lamscale * cbet1
                x = lam12x / lamscale
                y = sbet12a / betscale
            else:  # _f < 0
                # x = dlat, y = dlong
                var cbet12a = cbet2 * cbet1 - sbet2 * sbet1
                var bet12a = atan2(sbet12a, cbet12a)
                # real m12b, m0, dummy
                # In the case of lon12 = 180, this repeats a calculation made
                # in Inverse.
                var dummy1, m12b, m0, dummy2, dummy3 = self._Lengths(
                    self._n,
                    PI + bet12a,
                    sbet1,
                    -cbet1,
                    dn1,
                    sbet2,
                    cbet2,
                    dn2,
                    cbet1,
                    cbet2,
                    REDUCEDLENGTH,
                    C1a,
                    C2a,
                )
                x = -1.0 + m12b / (cbet1 * cbet2 * m0 * PI)
                betscale = (
                    sbet12a / x
                    if x < -0.01
                    else -self.f * Math.sq(cbet1) * PI
                )
                lamscale = betscale / cbet1
                y = lam12x / lamscale

            if y > -tol1_ and x > -1.0 - xthresh_:
                # strip near cut
                if self.f >= 0.0:
                    salp1 = min(1.0, -x)
                    calp1 = -sqrt(1.0 - Math.sq(salp1))
                else:
                    calp1 = max(0.0 if x > -tol1_ else -1.0, x)
                    salp1 = sqrt(1.0 - Math.sq(calp1))
            else:
                # Estimate alp1, by solving the astroid problem.
                #
                # Could estimate alpha1 = theta + pi/2, directly, i.e.,
                #   calp1 = y/k; salp1 = -x/(1+k);  for _f >= 0
                #   calp1 = x/(1+k); salp1 = -y/k;  for _f < 0
                #
                # However, it's better to estimate omg12 from astroid and use
                # the spherical formula to compute alp1.  This reduces the
                # mean number of Newton iterations for astroid cases from 2.24
                # (min 0, max 6) to 2.12 (min 0 max 5).
                #
                # Because omg12 is near pi, estimate work with omg12a =
                # pi - omg12
                var kk = Geodesic._Astroid(x, y)
                var omg12a = lamscale * (
                    -x * kk / (1.0 + kk) if self.f >= 0.0 else -y * (1.0 + kk) / kk
                )
                somg12 = sin(omg12a)
                comg12 = -cos(omg12a)
                # Update spherical estimate of alp1 using omg12 instead of
                # lam12
                salp1 = cbet2 * somg12
                calp1 = sbet12a - cbet2 * sbet1 * Math.sq(somg12) / (1.0 - comg12)
        # Sanity check on starting guess.  Backwards check allows NaN through.
        if not (salp1 <= 0.0):
            salp1, calp1 = Math.norm(salp1, calp1)
        else:
            salp1 = 1.0
            calp1 = 0.0
        return (sig12, salp1, calp1, salp2, calp2, dnm)

    # return lam12, salp2, calp2, sig12, ssig1, csig1, ssig2, csig2, eps,
    # domg12, dlam12
    def _Lambda12(
        self,
        sbet1: Float64,
        cbet1: Float64,
        dn1: Float64,
        sbet2: Float64,
        cbet2: Float64,
        dn2: Float64,
        salp1: Float64,
        mut calp1: Float64,
        slam120: Float64,
        clam120: Float64,
        diffp: Bool,
        # Scratch areas of the right size
        C1a: CoefPtr,
        C2a: CoefPtr,
        C3a: CoefPtr,
    ) -> Tuple[
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
    ]:
        """Private: Solve hybrid problem"""
        if sbet1 == 0.0 and calp1 == 0.0:
            # Break degeneracy of equatorial line.  This case has already
            # been handled.
            calp1 = -tiny_

        # sin(alp1) * cos(bet1) = sin(alp0)
        var salp0 = salp1 * cbet1
        var calp0 = hypot(calp1, salp1 * sbet1)  # calp0 > 0

        # real somg1, comg1, somg2, comg2, lam12
        # tan(bet1) = tan(sig1) * cos(alp1)
        # tan(omg1) = sin(alp0) * tan(sig1) = tan(omg1)=tan(alp1)*sin(bet1)
        var ssig1 = sbet1
        var somg1 = salp0 * sbet1
        var csig1 = calp1 * cbet1
        var comg1 = csig1
        ssig1, csig1 = Math.norm(ssig1, csig1)
        # Math.norm(somg1, comg1); -- don't need to normalize!

        # Enforce symmetries in the case abs(bet2) = -bet1.  Need to be
        # careful about this case, since this can yield singularities in the
        # Newton iteration.
        # sin(alp2) * cos(bet2) = sin(alp0)
        var salp2 = salp1 if cbet2 == cbet1 else salp0 / cbet2
        # calp2 = sqrt(1 - sq(salp2))
        #       = sqrt(sq(calp0) - sq(sbet2)) / cbet2
        # and subst for calp0 and rearrange to give (choose positive sqrt
        # to give alp2 in [0, pi/2]).
        var calp2 = (
            abs(calp1)
            if cbet2 == cbet1 and abs(sbet2) == -sbet1
            else sqrt(
                Math.sq(calp1 * cbet1)
                + (
                    (cbet2 - cbet1) * (cbet1 + cbet2)
                    if cbet1 < -sbet1
                    else (sbet1 - sbet2) * (sbet1 + sbet2)
                )
            ) / cbet2
        )
        # tan(bet2) = tan(sig2) * cos(alp2)
        # tan(omg2) = sin(alp0) * tan(sig2).
        var ssig2 = sbet2
        var somg2 = salp0 * sbet2
        var csig2 = calp2 * cbet2
        var comg2 = csig2
        ssig2, csig2 = Math.norm(ssig2, csig2)
        # Math.norm(somg2, comg2); -- don't need to normalize!

        # sig12 = sig2 - sig1, limit to [0, pi]
        var sig12 = atan2(
            max(0.0, csig1 * ssig2 - ssig1 * csig2) + 0.0,
            csig1 * csig2 + ssig1 * ssig2,
        )

        # omg12 = omg2 - omg1, limit to [0, pi]
        var somg12 = max(0.0, comg1 * somg2 - somg1 * comg2) + 0.0
        var comg12 = comg1 * comg2 + somg1 * somg2
        # eta = omg12 - lam120
        var eta = atan2(
            somg12 * clam120 - comg12 * slam120,
            comg12 * clam120 + somg12 * slam120,
        )

        # real B312
        var k2 = Math.sq(calp0) * self._ep2
        var eps = k2 / (2.0 * (1.0 + sqrt(1.0 + k2)) + k2)
        self._C3f(eps, C3a)
        var B312 = Geodesic._SinCosSeries(
            True, ssig2, csig2, C3a, nC3_
        ) - Geodesic._SinCosSeries(True, ssig1, csig1, C3a, nC3_)
        var domg12 = -self.f * self._A3f(eps) * salp0 * (sig12 + B312)
        var lam12 = eta + domg12

        var dlam12 = NAN
        if diffp:
            if calp2 == 0.0:
                dlam12 = -2.0 * self._f1 * dn1 / sbet1
            else:
                var r = self._Lengths(
                    eps,
                    sig12,
                    ssig1,
                    csig1,
                    dn1,
                    ssig2,
                    csig2,
                    dn2,
                    cbet1,
                    cbet2,
                    REDUCEDLENGTH,
                    C1a,
                    C2a,
                )
                dlam12 = r[1] * self._f1 / (calp2 * cbet2)

        return (
            lam12, salp2, calp2, sig12, ssig1, csig1, ssig2, csig2, eps, domg12,
            dlam12,
        )

    # return a12, s12, salp1, calp1, salp2, calp2, m12, M12, M21, S12
    def _GenInverse(
        self,
        mut lat1: Float64,
        lon1: Float64,
        mut lat2: Float64,
        lon2: Float64,
        outmask: Int,
    ) -> Tuple[
        Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
        Float64, Float64,
    ]:
        """Private: General version of the inverse problem"""
        var a12 = NAN  # return vals
        var s12 = NAN
        var m12 = NAN
        var M12 = NAN
        var M21 = NAN
        var S12 = NAN
        var salp1 = NAN
        var calp1 = NAN
        var salp2 = NAN
        var calp2 = NAN
        var ssig1 = NAN
        var csig1 = NAN
        var ssig2 = NAN
        var csig2 = NAN
        var eps = NAN
        var domg12 = NAN
        var omg12 = 0.0

        var om = outmask & OUT_MASK  # (upstream: outmask &= OUT_MASK)
        # Compute longitude difference (AngDiff does this carefully).  Result
        # is in [-180, 180] but -180 is only for west-going geodesics.  180 is
        # for east-going and meridional geodesics.
        var lon12, lon12s = Math.AngDiff(lon1, lon2)
        # Make longitude difference positive.
        var lonsign = copysign(1.0, lon12)
        lon12 = lonsign * lon12
        lon12s = lonsign * lon12s
        var lam12 = Math.radians(lon12)
        # Calculate sincos of lon12 + error (this applies AngRound
        # internally).
        var slam12, clam12 = Math.sincosde(lon12, lon12s)
        lon12s = (180.0 - lon12) - lon12s  # the supplementary longitude diff

        # If really close to the equator, treat as on equator.
        lat1 = Math.AngRound(Math.LatFix(lat1))
        lat2 = Math.AngRound(Math.LatFix(lat2))
        # Swap points so that point with higher (abs) latitude is point 1
        # If one latitude is a nan, then it becomes lat1.
        var swapp: Float64 = 1.0
        if abs(lat1) < abs(lat2) or isnan(lat2):
            swapp = -1.0
        if swapp < 0.0:
            lonsign *= -1.0
            var t = lat2
            lat2 = lat1
            lat1 = t
        # Make lat1 <= 0
        var latsign = copysign(1.0, -lat1)
        lat1 *= latsign
        lat2 *= latsign
        # Now we have
        #
        #     0 <= lon12 <= 180
        #     -90 <= lat1 <= 0
        #     lat1 <= lat2 <= -lat1
        #
        # lonsign, swapp, latsign register the transformation to bring the
        # coordinates to this canonical form.  In all cases, 1 means no change
        # was made.  We make these transformations so that there are few cases
        # to check, e.g., on verifying quadrants in atan2.  In addition, this
        # enforces some symmetries in the results returned.

        # real phi, sbet1, cbet1, sbet2, cbet2, s12x, m12x
        var sbet1, cbet1 = Math.sincosd(lat1)
        sbet1 *= self._f1
        # Ensure cbet1 = +epsilon at poles
        sbet1, cbet1 = Math.norm(sbet1, cbet1)
        cbet1 = max(tiny_, cbet1)

        var sbet2, cbet2 = Math.sincosd(lat2)
        sbet2 *= self._f1
        # Ensure cbet2 = +epsilon at poles
        sbet2, cbet2 = Math.norm(sbet2, cbet2)
        cbet2 = max(tiny_, cbet2)

        # If cbet1 < -sbet1, then cbet2 - cbet1 is a sensitive measure of the
        # |bet1| - |bet2|.  Alternatively (cbet1 >= -sbet1), abs(sbet2) + sbet1
        # is a better measure.  This logic is used in assigning calp2 in
        # Lambda12.  Sometimes these quantities vanish and in that case we
        # force bet2 = +/- bet1 exactly.  An example where is it necessary is
        # the inverse problem 48.522876735459 0 -48.52287673545898293
        # 179.599720456223079643 which failed with Visual Studio 10
        # (Release and Debug)
        if cbet1 < -sbet1:
            if cbet2 == cbet1:
                sbet2 = copysign(sbet1, sbet2)
        else:
            if abs(sbet2) == -sbet1:
                cbet2 = cbet1

        var dn1 = sqrt(1.0 + self._ep2 * Math.sq(sbet1))
        var dn2 = sqrt(1.0 + self._ep2 * Math.sq(sbet2))

        # real a12, sig12, calp1, salp1, calp2, salp2
        # index zero elements of these arrays are unused
        var C1a = self._blk.unsafe_offset(S_C1A)
        var C2a = self._blk.unsafe_offset(S_C2A)
        var C3a = self._blk.unsafe_offset(S_C3A)

        var meridian = lat1 == -90.0 or slam12 == 0.0
        var s12x = NAN
        var m12x = NAN
        var sig12 = NAN

        if meridian:
            # Endpoints are on a single full meridian, so the geodesic might
            # lie on a meridian.
            calp1 = clam12
            salp1 = slam12  # Head to the target longitude
            calp2 = 1.0
            salp2 = 0.0  # At the target we're heading north

            # tan(bet) = tan(sig) * cos(alp)
            ssig1 = sbet1
            csig1 = calp1 * cbet1
            ssig2 = sbet2
            csig2 = calp2 * cbet2

            # sig12 = sig2 - sig1
            sig12 = atan2(
                max(0.0, csig1 * ssig2 - ssig1 * csig2) + 0.0,
                csig1 * csig2 + ssig1 * ssig2,
            )

            var r = self._Lengths(
                self._n,
                sig12,
                ssig1,
                csig1,
                dn1,
                ssig2,
                csig2,
                dn2,
                cbet1,
                cbet2,
                om | DISTANCE | REDUCEDLENGTH,
                C1a,
                C2a,
            )
            s12x = r[0]
            m12x = r[1]
            M12 = r[3]
            M21 = r[4]

            # Add the check for sig12 since zero length geodesics might yield
            # m12 < 0.  Test case was
            #
            #    echo 20.001 0 20.001 0 | GeodSolve -i
            #
            # In fact, we will have sig12 > pi/2 for a meridional geodesic
            # which is not a shortest path.
            if sig12 < tol2_ or m12x >= 0.0:
                if sig12 < 3.0 * tiny_ or (
                    # Prevent negative s12 or m12 for short lines
                    sig12 < tol0_
                    and (s12x < 0.0 or m12x < 0.0)
                ):
                    sig12 = 0.0
                    m12x = 0.0
                    s12x = 0.0
                m12x *= self._b
                s12x *= self._b
                a12 = Math.degrees(sig12)
            else:
                # m12 < 0, i.e., prolate and too close to anti-podal
                meridian = False
        # end if meridian:

        # somg12 == 2 marks that it needs to be calculated
        var somg12 = 2.0
        var comg12 = 0.0
        if (
            not meridian
            and sbet1 == 0.0  # and sbet2 == 0
            # Mimic the way Lambda12 works with calp1 = 0
            and (self.f <= 0.0 or lon12s >= self.f * 180.0)
        ):
            # Geodesic runs along equator
            calp1 = 0.0
            calp2 = 0.0
            salp1 = 1.0
            salp2 = 1.0
            s12x = self.a * lam12
            sig12 = lam12 / self._f1
            omg12 = sig12
            m12x = self._b * sin(sig12)
            if om & GEODESICSCALE:
                M12 = cos(sig12)
                M21 = M12
            a12 = lon12 / self._f1
        elif not meridian:
            # Now point1 and point2 belong within a hemisphere bounded by a
            # meridian and geodesic is neither meridional or equatorial.

            # Figure a starting point for Newton's method
            var dnm = NAN
            var t = self._InverseStart(
                sbet1, cbet1, dn1, sbet2, cbet2, dn2, lam12, slam12, clam12, C1a,
                C2a,
            )
            sig12 = t[0]
            salp1 = t[1]
            calp1 = t[2]
            salp2 = t[3]
            calp2 = t[4]
            dnm = t[5]

            if sig12 >= 0.0:
                # Short lines (InverseStart sets salp2, calp2, dnm)
                s12x = sig12 * self._b * dnm
                m12x = Math.sq(dnm) * self._b * sin(sig12 / dnm)
                if om & GEODESICSCALE:
                    M12 = cos(sig12 / dnm)
                    M21 = M12
                a12 = Math.degrees(sig12)
                omg12 = lam12 / (self._f1 * dnm)
            else:
                # Newton's method.  This is a straightforward solution of
                # f(alp1) = lambda12(alp1) - lam12 = 0 with one wrinkle.  f(alp)
                # has exactly one root in the interval (0, pi) and its
                # derivative is positive at the root.  Thus f(alp) is positive
                # for alp > alp1 and negative for alp < alp1.  During the
                # course of the iteration, a range (alp1a, alp1b) is
                # maintained which brackets the root and with each evaluation
                # of f(alp) the range is shrunk if possible.  Newton's method
                # is restarted whenever the derivative of f is negative
                # (because the new value of alp1 is then further from the
                # solution) or if the new estimate of alp1 lies outside (0,pi);
                # in this case, the new starting guess is taken to be
                # (alp1a + alp1b) / 2.
                # real ssig1, csig1, ssig2, csig2, eps
                var numit = 0
                var tripn = False
                var tripb = False
                # Bracketing range
                var salp1a = tiny_
                var calp1a = 1.0
                var salp1b = tiny_
                var calp1b = -1.0

                while True:
                    # the WGS84 test set: mean = 1.47, sd = 1.25, max = 16
                    # WGS84 and random input: mean = 2.85, sd = 0.60
                    var v = NAN
                    var dv = NAN
                    var u = self._Lambda12(
                        sbet1,
                        cbet1,
                        dn1,
                        sbet2,
                        cbet2,
                        dn2,
                        salp1,
                        calp1,
                        slam12,
                        clam12,
                        numit < maxit1_,
                        C1a,
                        C2a,
                        C3a,
                    )
                    v = u[0]
                    salp2 = u[1]
                    calp2 = u[2]
                    sig12 = u[3]
                    ssig1 = u[4]
                    csig1 = u[5]
                    ssig2 = u[6]
                    csig2 = u[7]
                    eps = u[8]
                    domg12 = u[9]
                    dv = u[10]
                    # Reversed test to allow escape with NaNs
                    if (
                        tripb
                        or not (abs(v) >= (8.0 if tripn else 1.0) * tol0_)
                        or numit == maxit2_
                    ):
                        break
                    # Update bracketing values
                    if v > 0.0 and (
                        numit > maxit1_ or calp1 / salp1 > calp1b / salp1b
                    ):
                        salp1b = salp1
                        calp1b = calp1
                    elif v < 0.0 and (
                        numit > maxit1_ or calp1 / salp1 < calp1a / salp1a
                    ):
                        salp1a = salp1
                        calp1a = calp1

                    numit += 1
                    if numit < maxit1_ and dv > 0.0:
                        var dalp1 = -v / dv
                        if abs(dalp1) < PI:
                            var sdalp1 = sin(dalp1)
                            var cdalp1 = cos(dalp1)
                            var nsalp1 = salp1 * cdalp1 + calp1 * sdalp1
                            if nsalp1 > 0.0:
                                calp1 = calp1 * cdalp1 - salp1 * sdalp1
                                salp1 = nsalp1
                                salp1, calp1 = Math.norm(salp1, calp1)
                                # In some regimes we don't get quadratic
                                # convergence because slope -> 0.  So use
                                # convergence conditions based on epsilon
                                # instead of sqrt(epsilon).
                                tripn = abs(v) <= 16.0 * tol0_
                                continue
                    # Either dv was not positive or updated value was outside
                    # legal range.  Use the midpoint of the bracket as the
                    # next estimate.
                    salp1 = (salp1a + salp1b) / 2.0
                    calp1 = (calp1a + calp1b) / 2.0
                    salp1, calp1 = Math.norm(salp1, calp1)
                    tripn = False
                    tripb = (
                        abs(salp1a - salp1) + (calp1a - calp1) < tolb_
                        or abs(salp1 - salp1b) + (calp1 - calp1b) < tolb_
                    )

                var lengthmask = om | (
                    DISTANCE
                    if om & (REDUCEDLENGTH | GEODESICSCALE)
                    else EMPTY
                )
                var r = self._Lengths(
                    eps,
                    sig12,
                    ssig1,
                    csig1,
                    dn1,
                    ssig2,
                    csig2,
                    dn2,
                    cbet1,
                    cbet2,
                    lengthmask,
                    C1a,
                    C2a,
                )
                s12x = r[0]
                m12x = r[1]
                M12 = r[3]
                M21 = r[4]

                m12x *= self._b
                s12x *= self._b
                a12 = Math.degrees(sig12)
                if om & AREA:
                    # omg12 = lam12 - domg12
                    var sdomg12 = sin(domg12)
                    var cdomg12 = cos(domg12)
                    somg12 = slam12 * cdomg12 - clam12 * sdomg12
                    comg12 = clam12 * cdomg12 + slam12 * sdomg12
        # end elif not meridian

        if om & DISTANCE:
            s12 = 0.0 + s12x  # Convert -0 to 0

        if om & REDUCEDLENGTH:
            m12 = 0.0 + m12x  # Convert -0 to 0

        if om & AREA:
            # From Lambda12: sin(alp1) * cos(bet1) = sin(alp0)
            var salp0 = salp1 * cbet1
            var calp0 = hypot(calp1, salp1 * sbet1)  # calp0 > 0
            # real alp12
            if calp0 != 0.0 and salp0 != 0.0:
                # From Lambda12: tan(bet) = tan(sig) * cos(alp)
                ssig1 = sbet1
                csig1 = calp1 * cbet1
                ssig2 = sbet2
                csig2 = calp2 * cbet2
                var k2 = Math.sq(calp0) * self._ep2
                eps = k2 / (2.0 * (1.0 + sqrt(1.0 + k2)) + k2)
                # Multiplier = a^2 * e^2 * cos(alpha0) * sin(alpha0).
                var A4 = Math.sq(self.a) * calp0 * salp0 * self._e2
                ssig1, csig1 = Math.norm(ssig1, csig1)
                ssig2, csig2 = Math.norm(ssig2, csig2)
                var C4a = self._blk.unsafe_offset(S_C4A)
                self._C4f(eps, C4a)
                var B41 = Geodesic._SinCosSeries(False, ssig1, csig1, C4a, nC4_)
                var B42 = Geodesic._SinCosSeries(False, ssig2, csig2, C4a, nC4_)
                S12 = A4 * (B42 - B41)
            else:
                # Avoid problems with indeterminate sig1, sig2 on equator
                S12 = 0.0

            if not meridian and somg12 == 2.0:
                somg12 = sin(omg12)
                comg12 = cos(omg12)

            if (
                not meridian
                # omg12 < 3/4 * pi
                and comg12 > -0.7071  # Long difference not too big
                and sbet2 - sbet1 < 1.75
            ):  # Lat difference not too big
                # Use tan(Gamma/2) = tan(omg12/2)
                # * (tan(bet1/2)+tan(bet2/2))/(1+tan(bet1/2)*tan(bet2/2))
                # with tan(x/2) = sin(x)/(1+cos(x))
                var domg12 = 1.0 + comg12
                var dbet1 = 1.0 + cbet1
                var dbet2 = 1.0 + cbet2
                var alp12 = 2.0 * atan2(
                    somg12 * (sbet1 * dbet2 + sbet2 * dbet1),
                    domg12 * (sbet1 * sbet2 + dbet1 * dbet2),
                )
                S12 += self._c2 * alp12
            else:
                # alp12 = alp2 - alp1, used in atan2 so no need to normalize
                var salp12 = salp2 * calp1 - calp2 * salp1
                var calp12 = calp2 * calp1 + salp2 * salp1
                # The right thing appears to happen if alp1 = +/-180 and
                # alp2 = 0, viz salp12 = -0 and alp12 = -180.  However this
                # depends on the sign being attached to 0 correctly.  The
                # following ensures the correct behavior.
                if salp12 == 0.0 and calp12 < 0.0:
                    salp12 = tiny_ * calp1
                    calp12 = -1.0
                var alp12 = atan2(salp12, calp12)
                S12 += self._c2 * alp12
            S12 *= swapp * lonsign * latsign
            # Convert -0 to 0
            S12 += 0.0

        # Convert calp, salp to azimuth accounting for lonsign, swapp, latsign.
        if swapp < 0.0:
            salp2, salp1 = salp1, salp2
            calp2, calp1 = calp1, calp2
            if om & GEODESICSCALE:
                M21, M12 = M12, M21

        salp1 *= swapp * lonsign
        calp1 *= swapp * latsign
        salp2 *= swapp * lonsign
        calp2 *= swapp * latsign

        return (
            a12, s12, salp1, calp1, salp2, calp2, m12, M12, M21, S12,
        )

    # return a12, lat2, lon2, azi2, s12, m12, M12, M21, S12
    def _GenDirect(
        self,
        lat1: Float64,
        lon1: Float64,
        azi1: Float64,
        arcmode: Bool,
        s12_a12: Float64,
        outmask: Int,
    ) -> Tuple[
        Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
        Float64,
    ]:
        """Private: General version of direct problem"""
        # Automatically supply DISTANCE_IN if necessary
        var om = outmask
        if not arcmode:
            om |= DISTANCE_IN
        var line = GeodesicLine(
            self, lat1, lon1, azi1, om, NAN, NAN, self._blk.unsafe_offset(L_STATE), False
        )
        return line._GenPosition(arcmode, s12_a12, om)

    def _GenDirectLine(
        self,
        lat1: Float64,
        lon1: Float64,
        azi1: Float64,
        arcmode: Bool,
        s12_a12: Float64,
        caps: Int,
    ) -> GeodesicLine:
        """Private: general form of DirectLine"""
        # Automatically supply DISTANCE_IN if necessary
        var cm = caps
        if not arcmode:
            cm |= DISTANCE_IN
        var line = GeodesicLine(
            self, lat1, lon1, azi1, cm, NAN, NAN, self._blk.unsafe_offset(L_STATE), False
        )
        if arcmode:
            line.SetArc(s12_a12)
        else:
            line.SetDistance(s12_a12)
        return line^


# =====================================================================
# geographiclib/geodesicline.py -- class GeodesicLine
# =====================================================================
struct GeodesicLine:
    """Points on a geodesic path"""

    var a: Float64
    # The equatorial radius in meters (readonly)
    var f: Float64
    # The flattening (readonly)
    var _b: Float64
    var _c2: Float64
    var _f1: Float64
    var caps: Int
    # the capabilities (readonly)
    var lat1: Float64
    # the latitude of the first point in degrees (readonly)
    var lon1: Float64
    # the longitude of the first point in degrees (readonly)
    var azi1: Float64
    # the azimuth at the first point in degrees (readonly)
    var salp1: Float64
    # the sine of the azimuth at the first point (readonly)
    var calp1: Float64
    # the cosine of the azimuth at the first point (readonly)
    var s13: Float64
    # the distance between point 1 and point 3 in meters (readonly)
    var a13: Float64
    # the arc length between point 1 and point 3 in degrees (readonly)
    var _dn1: Float64
    var _salp0: Float64
    var _calp0: Float64
    var _ssig1: Float64
    var _somg1: Float64
    var _csig1: Float64
    var _comg1: Float64
    var _k2: Float64
    var _A1m1: Float64
    var _B11: Float64
    var _stau1: Float64
    var _ctau1: Float64
    var _A2m1: Float64
    var _B21: Float64
    var _A3c: Float64
    var _B31: Float64
    var _A4: Float64
    var _B41: Float64
    # real cbet1, sbet1
    var _C1a: CoefPtr
    var _C1pa: CoefPtr
    var _C2a: CoefPtr
    var _C3a: CoefPtr
    var _C4a: CoefPtr

    def __init__(
        out self,
        geod: Geodesic,
        lat1: Float64,
        lon1: Float64,
        azi1: Float64,
        caps: Int,
        salp1: Float64,
        calp1: Float64,
        state: CoefPtr,
        ready: Bool,
    ):
        """Construct a GeodesicLine object

        As with Geodesic, the coefficient tables and the scalars live in a
        caller-owned 65-double block rather than in the struct; `ready` says
        whether that block already holds a constructed line.  The Python API,
        `GeodesicLine(geod, lat1, lon1, azi1, caps, salp1, calp1)`, matches
        upstream exactly.

        """
        # upstream leaves the caps-gated members unset when the matching
        # capability is absent; a Mojo struct needs every field initialised.
        self._A1m1 = NAN
        self._B11 = NAN
        self._stau1 = NAN
        self._ctau1 = NAN
        self._A2m1 = NAN
        self._B21 = NAN
        self._A3c = NAN
        self._B31 = NAN
        self._A4 = NAN
        self._B41 = NAN
        self._C1a = state.unsafe_offset(LN_C1A)
        self._C1pa = state.unsafe_offset(LN_C1PA)
        self._C2a = state.unsafe_offset(LN_C2A)
        self._C3a = state.unsafe_offset(LN_C3A)
        self._C4a = state.unsafe_offset(LN_C4A)
        if ready:
            self.a = state.unsafe_load(0)
            self.f = state.unsafe_load(1)
            self._b = state.unsafe_load(2)
            self._c2 = state.unsafe_load(3)
            self._f1 = state.unsafe_load(4)
            self.caps = Int(state.unsafe_load(5))
            self.lat1 = state.unsafe_load(6)
            self.lon1 = state.unsafe_load(7)
            self.azi1 = state.unsafe_load(8)
            self.salp1 = state.unsafe_load(9)
            self.calp1 = state.unsafe_load(10)
            self.s13 = state.unsafe_load(11)
            self.a13 = state.unsafe_load(12)
            self._dn1 = state.unsafe_load(13)
            self._salp0 = state.unsafe_load(14)
            self._calp0 = state.unsafe_load(15)
            self._ssig1 = state.unsafe_load(16)
            self._somg1 = state.unsafe_load(17)
            self._csig1 = state.unsafe_load(18)
            self._comg1 = state.unsafe_load(19)
            self._k2 = state.unsafe_load(20)
            self._A1m1 = state.unsafe_load(21)
            self._B11 = state.unsafe_load(22)
            self._stau1 = state.unsafe_load(23)
            self._ctau1 = state.unsafe_load(24)
            self._A2m1 = state.unsafe_load(25)
            self._B21 = state.unsafe_load(26)
            self._A3c = state.unsafe_load(27)
            self._B31 = state.unsafe_load(28)
            self._A4 = state.unsafe_load(29)
            self._B41 = state.unsafe_load(30)
            return
        self.a = geod.a
        # The equatorial radius in meters (readonly)
        self.f = geod.f
        # The flattening (readonly)
        self._b = geod._b
        self._c2 = geod._c2
        self._f1 = geod._f1
        self.caps = caps | LATITUDE | AZIMUTH | LONG_UNROLL
        # the capabilities (readonly)

        # Guard against underflow in salp0
        self.lat1 = Math.LatFix(lat1)
        # the latitude of the first point in degrees (readonly)
        self.lon1 = lon1
        # the longitude of the first point in degrees (readonly)
        if isnan(salp1) or isnan(calp1):
            self.azi1 = Math.AngNormalize(azi1)
            self.salp1, self.calp1 = Math.sincosd(Math.AngRound(azi1))
        else:
            self.azi1 = azi1
            # the azimuth at the first point in degrees (readonly)
            self.salp1 = salp1
            # the sine of the azimuth at the first point (readonly)
            self.calp1 = calp1
            # the cosine of the azimuth at the first point (readonly)

        # real cbet1, sbet1
        var sbet1, cbet1 = Math.sincosd(Math.AngRound(self.lat1))
        sbet1 *= self._f1
        # Ensure cbet1 = +epsilon at poles
        sbet1, cbet1 = Math.norm(sbet1, cbet1)
        cbet1 = max(tiny_, cbet1)
        self._dn1 = sqrt(1.0 + geod._ep2 * Math.sq(sbet1))

        # Evaluate alp0 from sin(alp1) * cos(bet1) = sin(alp0),
        self._salp0 = self.salp1 * cbet1  # alp0 in [0, pi/2 - |bet1|]
        # Alt: calp0 = hypot(sbet1, calp1 * cbet1).  The following
        # is slightly better (consider the case salp1 = 0).
        self._calp0 = hypot(self.calp1, self.salp1 * sbet1)
        # Evaluate sig with tan(bet1) = tan(sig1) * cos(alp1).
        # sig = 0 is nearest northward crossing of equator.
        # With bet1 = 0, alp1 = pi/2, we have sig1 = 0 (equatorial line).
        # With bet1 =  pi/2, alp1 = -pi, sig1 =  pi/2
        # With bet1 = -pi/2, alp1 =  0 , sig1 = -pi/2
        # Evaluate omg1 with tan(omg1) = sin(alp0) * tan(sig1).
        # With alp0 in (0, pi/2], quadrants for sig and omg coincide.
        # No atan2(0,0) ambiguity at poles since cbet1 = +epsilon.
        # With alp0 = 0, omg1 = 0 for alp1 = 0, omg1 = pi for alp1 = pi.
        self._ssig1 = sbet1
        self._somg1 = self._salp0 * sbet1
        self._csig1 = cbet1 * self.calp1 if sbet1 != 0.0 or self.calp1 != 0.0 else 1.0
        self._comg1 = self._csig1
        # sig1 in (-pi, pi]
        self._ssig1, self._csig1 = Math.norm(self._ssig1, self._csig1)
        # No need to normalize
        # self._somg1, self._comg1 = Math.norm(self._somg1, self._comg1)

        self._k2 = Math.sq(self._calp0) * geod._ep2
        var eps = self._k2 / (2.0 * (1.0 + sqrt(1.0 + self._k2)) + self._k2)

        if self.caps & CAP_C1:
            self._A1m1 = Geodesic._A1m1f(eps, geod._blk.unsafe_offset(K_A1M1))
            Geodesic._C1f(eps, self._C1a, geod._blk.unsafe_offset(K_C1F))
            self._B11 = Geodesic._SinCosSeries(
                True, self._ssig1, self._csig1, self._C1a, nC1_ + 1
            )
            var s = sin(self._B11)
            var c = cos(self._B11)
            # tau1 = sig1 + B11
            self._stau1 = self._ssig1 * c + self._csig1 * s
            self._ctau1 = self._csig1 * c - self._ssig1 * s
            # Not necessary because C1pa reverts C1a
            #    _B11 = -_SinCosSeries(true, _stau1, _ctau1, _C1pa)

        if self.caps & CAP_C1p:
            Geodesic._C1pf(eps, self._C1pa, geod._blk.unsafe_offset(K_C1PF))

        if self.caps & CAP_C2:
            self._A2m1 = Geodesic._A2m1f(eps, geod._blk.unsafe_offset(K_A2M1))
            Geodesic._C2f(eps, self._C2a, geod._blk.unsafe_offset(K_C2F))
            self._B21 = Geodesic._SinCosSeries(
                True, self._ssig1, self._csig1, self._C2a, nC2_ + 1
            )

        if self.caps & CAP_C3:
            geod._C3f(eps, self._C3a)
            self._A3c = -self.f * self._salp0 * geod._A3f(eps)
            self._B31 = Geodesic._SinCosSeries(
                True, self._ssig1, self._csig1, self._C3a, nC3_
            )

        if self.caps & CAP_C4:
            geod._C4f(eps, self._C4a)
            # Multiplier = a^2 * e^2 * cos(alpha0) * sin(alpha0)
            self._A4 = Math.sq(self.a) * self._calp0 * self._salp0 * geod._e2
            self._B41 = Geodesic._SinCosSeries(
                False, self._ssig1, self._csig1, self._C4a, nC4_
            )
        self.s13 = NAN
        # the distance between point 1 and point 3 in meters (readonly)
        self.a13 = NAN
        # the arc length between point 1 and point 3 in degrees (readonly)

    def _save(mut self, state: CoefPtr) -> None:
        """Write the 31 scalars back into the state block (see __init__)."""
        state.unsafe_store(0, self.a)
        state.unsafe_store(1, self.f)
        state.unsafe_store(2, self._b)
        state.unsafe_store(3, self._c2)
        state.unsafe_store(4, self._f1)
        state.unsafe_store(5, Float64(self.caps))
        state.unsafe_store(6, self.lat1)
        state.unsafe_store(7, self.lon1)
        state.unsafe_store(8, self.azi1)
        state.unsafe_store(9, self.salp1)
        state.unsafe_store(10, self.calp1)
        state.unsafe_store(11, self.s13)
        state.unsafe_store(12, self.a13)
        state.unsafe_store(13, self._dn1)
        state.unsafe_store(14, self._salp0)
        state.unsafe_store(15, self._calp0)
        state.unsafe_store(16, self._ssig1)
        state.unsafe_store(17, self._somg1)
        state.unsafe_store(18, self._csig1)
        state.unsafe_store(19, self._comg1)
        state.unsafe_store(20, self._k2)
        state.unsafe_store(21, self._A1m1)
        state.unsafe_store(22, self._B11)
        state.unsafe_store(23, self._stau1)
        state.unsafe_store(24, self._ctau1)
        state.unsafe_store(25, self._A2m1)
        state.unsafe_store(26, self._B21)
        state.unsafe_store(27, self._A3c)
        state.unsafe_store(28, self._B31)
        state.unsafe_store(29, self._A4)
        state.unsafe_store(30, self._B41)

    # return a12, lat2, lon2, azi2, s12, m12, M12, M21, S12
    def _GenPosition(
        self, arcmode: Bool, s12_a12: Float64, outmask: Int
    ) -> Tuple[
        Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
        Float64,
    ]:
        """Private: General solution of position along geodesic"""
        var a12 = NAN
        var lat2 = NAN
        var lon2 = NAN
        var azi2 = NAN
        var s12 = NAN
        var m12 = NAN
        var M12 = NAN
        var M21 = NAN
        var S12 = NAN
        # (upstream: outmask &= self.caps & OUT_MASK)
        var om = outmask & self.caps & OUT_MASK
        if not (arcmode or (self.caps & (OUT_MASK & DISTANCE_IN))):
            # Uninitialized or impossible distance calculation requested
            return (
                a12, lat2, lon2, azi2, s12, m12, M12, M21, S12,
            )

        # Avoid warning about uninitialized B12.
        var B12 = 0.0
        var AB1 = 0.0
        var sig12 = NAN
        var ssig12 = NAN
        var csig12 = NAN
        if arcmode:
            # Interpret s12_a12 as spherical arc length
            sig12 = Math.radians(s12_a12)
            ssig12, csig12 = Math.sincosd(s12_a12)
        else:
            # Interpret s12_a12 as distance
            var tau12 = s12_a12 / (self._b * (1.0 + self._A1m1))
            tau12 = tau12 if isfinite(tau12) else NAN
            var s = sin(tau12)
            var c = cos(tau12)
            # tau2 = tau1 + tau12
            B12 = -Geodesic._SinCosSeries(
                True,
                self._stau1 * c + self._ctau1 * s,
                self._ctau1 * c - self._stau1 * s,
                self._C1pa,
                nC1p_ + 1,
            )
            sig12 = tau12 - (B12 - self._B11)
            ssig12 = sin(sig12)
            csig12 = cos(sig12)
            if abs(self.f) > 0.01:
                # Reverted distance series is inaccurate for |f| > 1/100, so
                # correct sig12 with 1 Newton iteration.
                var u2 = self._ssig1 * csig12 + self._csig1 * ssig12
                var v2 = self._csig1 * csig12 - self._ssig1 * ssig12
                B12 = Geodesic._SinCosSeries(True, u2, v2, self._C1a, nC1_ + 1)
                var serr = (
                    1.0 + self._A1m1
                ) * (sig12 + (B12 - self._B11)) - s12_a12 / self._b
                sig12 = sig12 - serr / sqrt(1.0 + self._k2 * Math.sq(u2))
                ssig12 = sin(sig12)
                csig12 = cos(sig12)
                # Update B12 below

        # real omg12, lam12, lon12
        # real ssig2, csig2, sbet2, cbet2, somg2, comg2, salp2, calp2
        # sig2 = sig1 + sig12
        var ssig2 = self._ssig1 * csig12 + self._csig1 * ssig12
        var csig2 = self._csig1 * csig12 - self._ssig1 * ssig12
        var dn2 = sqrt(1.0 + self._k2 * Math.sq(ssig2))
        if om & (DISTANCE | REDUCEDLENGTH | GEODESICSCALE):
            if arcmode or abs(self.f) > 0.01:
                B12 = Geodesic._SinCosSeries(True, ssig2, csig2, self._C1a, nC1_ + 1)
            AB1 = (1.0 + self._A1m1) * (B12 - self._B11)
        # sin(bet2) = cos(alp0) * sin(sig2)
        var sbet2 = self._calp0 * ssig2
        # Alt: cbet2 = hypot(csig2, salp0 * ssig2)
        var cbet2 = hypot(self._salp0, self._calp0 * csig2)
        if cbet2 == 0.0:
            # I.e., salp0 = 0, csig2 = 0.  Break the degeneracy in this case
            cbet2 = tiny_
            csig2 = tiny_
        # tan(alp0) = cos(sig2)*tan(alp2)
        var salp2 = self._salp0
        var calp2 = self._calp0 * csig2  # No need to normalize

        if om & DISTANCE:
            s12 = self._b * ((1.0 + self._A1m1) * sig12 + AB1) if arcmode else s12_a12

        if om & LONGITUDE:
            # tan(omg2) = sin(alp0) * tan(sig2)
            var somg2 = self._salp0 * ssig2
            var comg2 = csig2  # No need to normalize
            var E = copysign(1.0, self._salp0)  # East or west going?
            # omg12 = omg2 - omg1
            var omg12 = 0.0
            if om & LONG_UNROLL:
                omg12 = E * (
                    sig12
                    - (atan2(ssig2, csig2) - atan2(self._ssig1, self._csig1))
                    + (atan2(E * somg2, comg2) - atan2(E * self._somg1, self._comg1))
                )
            else:
                omg12 = atan2(
                    somg2 * self._comg1 - comg2 * self._somg1,
                    comg2 * self._comg1 + somg2 * self._somg1,
                )
            var lam12 = omg12 + self._A3c * (
                sig12
                + (
                    Geodesic._SinCosSeries(True, ssig2, csig2, self._C3a, nC3_)
                    - self._B31
                )
            )
            var lon12 = Math.degrees(lam12)
            lon2 = (
                self.lon1 + lon12
                if om & LONG_UNROLL
                else Math.AngNormalize(
                    Math.AngNormalize(self.lon1) + Math.AngNormalize(lon12)
                )
            )

        if om & LATITUDE:
            lat2 = Math.atan2d(sbet2, self._f1 * cbet2)

        if om & AZIMUTH:
            azi2 = Math.atan2d(salp2, calp2)

        if om & (REDUCEDLENGTH | GEODESICSCALE):
            var B22 = Geodesic._SinCosSeries(True, ssig2, csig2, self._C2a, nC2_ + 1)
            var AB2 = (1.0 + self._A2m1) * (B22 - self._B21)
            var J12 = (self._A1m1 - self._A2m1) * sig12 + (AB1 - AB2)
            if om & REDUCEDLENGTH:
                # Add parens around (_csig1 * ssig2) and (_ssig1 * csig2) to
                # ensure accurate cancellation in the case of coincident
                # points.
                m12 = self._b * (
                    (dn2 * (self._csig1 * ssig2) - self._dn1 * (self._ssig1 * csig2))
                    - self._csig1 * csig2 * J12
                )
            if om & GEODESICSCALE:
                var t = (
                    self._k2
                    * (ssig2 - self._ssig1)
                    * (ssig2 + self._ssig1)
                    / (self._dn1 + dn2)
                )
                M12 = csig12 + (t * ssig2 - csig2 * J12) * self._ssig1 / self._dn1
                M21 = csig12 - (t * self._ssig1 - self._csig1 * J12) * ssig2 / dn2

        if om & AREA:
            var B42 = Geodesic._SinCosSeries(False, ssig2, csig2, self._C4a, nC4_)
            # real salp12, calp12
            var salp12 = 0.0
            var calp12 = 0.0
            if self._calp0 == 0.0 or self._salp0 == 0.0:
                # alp12 = alp2 - alp1, used in atan2 so no need to normalize
                salp12 = salp2 * self.calp1 - calp2 * self.salp1
                calp12 = calp2 * self.calp1 + salp2 * self.salp1
            else:
                # tan(alp) = tan(alp0) * sec(sig)
                # tan(alp2-alp1) = (tan(alp2) -tan(alp1)) /
                #                   (tan(alp2)*tan(alp1)+1)
                # = calp0 * salp0 * (csig1-csig2) /
                #   (salp0^2 + calp0^2 * csig1*csig2)
                # If csig12 > 0, write
                #   csig1 - csig2 = ssig12 * (csig1 * ssig12 / (1 + csig12) + ssig1)
                # else
                #   csig1 - csig2 = csig1 * (1 - csig12) + ssig12 * ssig1
                # No need to normalize
                salp12 = self._calp0 * self._salp0 * (
                    ssig12 * (self._csig1 * ssig12 / (1.0 + csig12) + self._ssig1)
                    if csig12 > 0.0
                    else self._csig1 * (1.0 - csig12) + ssig12 * self._ssig1
                )
                calp12 = (
                    Math.sq(self._salp0) + Math.sq(self._calp0) * self._csig1 * csig2
                )
            S12 = self._c2 * atan2(salp12, calp12) + self._A4 * (B42 - self._B41)

        a12 = s12_a12 if arcmode else Math.degrees(sig12)
        return (a12, lat2, lon2, azi2, s12, m12, M12, M21, S12)

    def SetDistance(mut self, s13: Float64) -> None:
        """Specify the position of point 3 in terms of distance

        :param s13: distance from point 1 to point 3 in meters

        """
        self.s13 = s13
        var r = self._GenPosition(False, self.s13, EMPTY)
        self.a13 = r[0]

    def SetArc(mut self, a13: Float64) -> None:
        """Specify the position of point 3 in terms of arc length

        :param a13: spherical arc length from point 1 to point 3 in degrees

        """
        self.a13 = a13
        var r = self._GenPosition(True, self.a13, DISTANCE)
        self.s13 = r[4]


# =====================================================================
# geographiclib/polygonarea.py -- class PolygonArea
# =====================================================================

# Layout of a PolygonArea state block: 11 int64 slots.  Slots 0 and 1 are
# num and crossings; the float64 half starts at PA_F and is indexed by
# PA_LAT0 .. PA_PT.
comptime PA_NUM: Int = 0
comptime PA_CROSS: Int = 1
comptime PA_F: Int = 2
comptime PA_LAT0: Int = 0
comptime PA_LON0: Int = 1
comptime PA_LAT1: Int = 2
comptime PA_LON1: Int = 3
comptime PA_AS: Int = 4
comptime PA_AT: Int = 5
comptime PA_PS: Int = 6
comptime PA_PT: Int = 7
comptime PA_BLOCK: Int = 11


struct PolygonArea:
    """Area of a geodesic polygon"""

    var earth: Geodesic
    var polyline: Bool
    # Is this a polyline? (readonly)
    var area0: Float64
    # The total area of the ellipsoid in meter^2 (readonly)
    var _mask: Int
    var _areasum: Accumulator
    var _perimetersum: Accumulator
    var num: Int
    # The current number of points in the polygon (readonly)
    var lat1: Float64
    # The current latitude in degrees (readonly)
    var lon1: Float64
    # The current longitude in degrees (readonly)
    var _crossings: Int
    var _lat0: Float64
    var _lon0: Float64

    @staticmethod
    def _transit(lon1: Float64, lon2: Float64) -> Int:
        """Count crossings of prime meridian for AddPoint."""
        # Return 1 or -1 if crossing prime meridian in east or west direction.
        # Otherwise return zero.
        # Compute lon12 the same way as Geodesic::Inverse.
        var lon12, _ = Math.AngDiff(lon1, lon2)
        var n1 = Math.AngNormalize(lon1)
        var n2 = Math.AngNormalize(lon2)
        if lon12 > 0.0 and (n1 < 0.0 <= n2 or (n1 > 0.0 and n2 == 0.0)):
            return 1
        elif lon12 < 0.0 and n2 < 0.0 <= n1:
            return -1
        return 0

    @staticmethod
    def _transitdirect(lon1: Float64, lon2: Float64) -> Int:
        """Count crossings of prime meridian for AddEdge."""
        # We want to compute exactly
        #   int(floor(lon2 / 360)) - int(floor(lon1 / 360))
        var n1 = Math.remainder(lon1, 720.0)
        var n2 = Math.remainder(lon2, 720.0)
        return (0 if 0.0 <= n2 < 360.0 else 1) - (0 if 0.0 <= n1 < 360.0 else 1)

    @staticmethod
    def _areareduceA(
        mut area: Accumulator,
        area0: Float64,
        crossings: Int,
        reverse: Bool,
        sign: Bool,
    ) -> Float64:
        """Reduce accumulator area to allowed range."""
        area.Remainder(area0)
        if crossings & 1:
            area.Add((1.0 if area.Sum() < 0.0 else -1.0) * area0 / 2.0)
        # area is with the clockwise sense.  If !reverse convert to
        # counter-clockwise convention.
        if not reverse:
            area.Negate()
        # If sign put area in (-area0/2, area0/2], else put area in [0, area0)
        if sign:
            if area.Sum() > area0 / 2.0:
                area.Add(-area0)
            elif area.Sum() <= -area0 / 2.0:
                area.Add(area0)
        else:
            if area.Sum() >= area0:
                area.Add(-area0)
            elif area.Sum() < 0.0:
                area.Add(area0)
        return 0.0 + area.Sum()

    @staticmethod
    def _areareduceB(
        mut area: Float64,
        area0: Float64,
        crossings: Int,
        reverse: Bool,
        sign: Bool,
    ) -> Float64:
        """Reduce double area to allowed range."""
        area = Math.remainder(area, area0)
        if crossings & 1:
            area += (1.0 if area < 0.0 else -1.0) * area0 / 2.0
        # area is with the clockwise sense.  If !reverse convert to
        # counter-clockwise convention.
        if not reverse:
            area *= -1.0
        # If sign put area in (-area0/2, area0/2], else put area in [0, area0)
        if sign:
            if area > area0 / 2.0:
                area -= area0
            elif area <= -area0 / 2.0:
                area += area0
        else:
            if area >= area0:
                area -= area0
            elif area < 0.0:
                area += area0
        return 0.0 + area

    def __init__(out self, earth: Geodesic, polyline: Bool):
        """Construct a PolygonArea object

        :param earth: a :class:`~geographiclib.geodesic.Geodesic` object
        :param polyline: if true, treat object as a polyline instead of a polygon

        Initially the polygon has no vertices.
        """
        self.polyline = polyline
        # Is this a polyline? (readonly)
        self.area0 = 4.0 * PI * earth._c2
        # The total area of the ellipsoid in meter^2 (readonly)
        self._mask = (
            LATITUDE
            | LONGITUDE
            | DISTANCE
            | (EMPTY if polyline else AREA | LONG_UNROLL)
        )
        # A Geodesic is not implicitly copyable in this dialect, so the
        # ellipsoid is re-wrapped rather than copied field by field.
        self.earth = Geodesic(earth.a, earth.f, earth._blk, True)
        # The geodesic object (readonly)
        # upstream only builds _areasum when not a polyline; a Mojo struct
        # needs every field initialised, and an unused accumulator is inert.
        self._areasum = Accumulator(0.0)
        self._perimetersum = Accumulator(0.0)
        self.num = 0
        # The current number of points in the polygon (readonly)
        self.lat1 = NAN
        # The current latitude in degrees (readonly)
        self.lon1 = NAN
        # The current longitude in degrees (readonly)
        self._crossings = 0
        self._lat0 = NAN
        self._lon0 = NAN

    def Clear(mut self) -> None:
        """Reset to empty polygon."""
        self.num = 0
        self._crossings = 0
        if not self.polyline:
            self._areasum.Set(0.0)
        self._perimetersum.Set(0.0)
        self._lat0 = NAN
        self._lon0 = NAN
        self.lat1 = NAN
        self.lon1 = NAN

    def AddPoint(mut self, lat: Float64, lon: Float64) -> None:
        """Add the next vertex to the polygon"""
        if self.num == 0:
            self._lat0 = lat
            self.lat1 = lat
            self._lon0 = lon
            self.lon1 = lon
        else:
            # Geodesic._GenInverse normalises its latitude arguments in place.
            # In Python those are its own parameters; here they would be the
            # caller's variables, so bind mutable copies first.
            var lat1 = self.lat1
            var lon1 = self.lon1
            var lat2 = lat
            var lon2 = lon
            var r = self.earth._GenInverse(lat1, lon1, lat2, lon2, self._mask)
            var s12 = r[1]
            var S12 = r[9]
            self._perimetersum.Add(s12)
            if not self.polyline:
                self._areasum.Add(S12)
                self._crossings += PolygonArea._transit(self.lon1, lon)
            self.lat1 = lat
            self.lon1 = lon
        self.num += 1

    def AddEdge(mut self, azi: Float64, s: Float64) -> None:
        """Add the next edge to the polygon"""
        if self.num != 0:
            var r = self.earth._GenDirect(
                self.lat1, self.lon1, azi, False, s, self._mask
            )
            var lat = r[1]
            var lon = r[2]
            var S12 = r[8]
            self._perimetersum.Add(s)
            if not self.polyline:
                self._areasum.Add(S12)
                self._crossings += PolygonArea._transitdirect(self.lon1, lon)
            self.lat1 = lat
            self.lon1 = lon
            self.num += 1

    # return number, perimeter, area
    def Compute(self, reverse: Bool = False, sign: Bool = True) -> Tuple[Int, Float64, Float64]:
        """Compute the properties of the polygon"""
        var area = NAN
        var perimeter = 0.0
        if self.num < 2:
            if not self.polyline:
                area = 0.0
            return (self.num, perimeter, area)

        if self.polyline:
            perimeter = self._perimetersum.Sum()
            return (self.num, perimeter, area)

        var plat1 = self.lat1
        var plat0 = self._lat0
        var r = self.earth._GenInverse(
            plat1, self.lon1, plat0, self._lon0, self._mask
        )
        var s12 = r[1]
        var S12 = r[9]
        perimeter = self._perimetersum.Sum(s12)
        var tempsum = Accumulator(0.0)
        tempsum.Set(self._areasum)
        tempsum.Add(S12)
        var crossings = self._crossings + PolygonArea._transit(self.lon1, self._lon0)
        area = PolygonArea._areareduceA(tempsum, self.area0, crossings, reverse, sign)
        return (self.num, perimeter, area)

    # return number, perimeter, area
    def TestPoint(
        self, lat: Float64, lon: Float64, reverse: Bool = False, sign: Bool = True
    ) -> Tuple[Int, Float64, Float64]:
        """Compute the properties for a tentative additional vertex"""
        var area = NAN
        var perimeter = 0.0
        if self.num == 0:
            if not self.polyline:
                area = 0.0
            return (1, perimeter, area)

        perimeter = self._perimetersum.Sum()
        var tempsum = 0.0
        if not self.polyline:
            tempsum = self._areasum.Sum()
        var crossings = self._crossings
        var num = self.num + 1
        var last = 1
        if not self.polyline:
            last = 2
        for i in range(last):
            var lat1 = lat
            var lon1 = lon
            var lat2 = self._lat0
            var lon2 = self._lon0
            if i == 0:
                lat1 = self.lat1
                lon1 = self.lon1
                lat2 = lat
                lon2 = lon
            var omask = self._mask
            var r = self.earth._GenInverse(lat1, lon1, lat2, lon2, omask)
            perimeter += r[1]
            if not self.polyline:
                tempsum += r[9]
                crossings += PolygonArea._transit(
                    self.lon1 if i == 0 else lon, self._lon0 if i != 0 else lon
                )
        # end for
        if self.polyline:
            return (num, perimeter, area)
        area = PolygonArea._areareduceB(tempsum, self.area0, crossings, reverse, sign)
        return (num, perimeter, area)

    # return num, perimeter, area
    def TestEdge(
        self, azi: Float64, s: Float64, reverse: Bool = False, sign: Bool = True
    ) -> Tuple[Int, Float64, Float64]:
        """Compute the properties for a tentative additional edge"""
        if self.num == 0:  # we don't have a starting point!
            return (0, NAN, NAN)
        var num = self.num + 1
        var perimeter = self._perimetersum.Sum() + s
        if self.polyline:
            return (num, perimeter, NAN)

        var tempsum = self._areasum.Sum()
        var crossings = self._crossings
        var r = self.earth._GenDirect(
            self.lat1, self.lon1, azi, False, s, self._mask
        )
        var lat = r[1]
        var lon = r[2]
        tempsum += r[8]
        crossings += PolygonArea._transitdirect(self.lon1, lon)
        var plat0 = self._lat0
        var omask = self._mask
        var r2 = self.earth._GenInverse(
            lat, lon, plat0, self._lon0, omask
        )
        perimeter += r2[1]
        tempsum += r2[9]
        crossings += PolygonArea._transit(lon, self._lon0)

        var area = PolygonArea._areareduceB(tempsum, self.area0, crossings, reverse, sign)
        return (num, perimeter, area)
