"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric.  Callers own all
memory, including scratch, so nothing here allocates and nothing here can leak.

Two caller-owned layouts are used, both float64 and both sized by constants in
ported.mojo:

  Geodesic block (GEOD_BLOCK = 337 doubles)
      [0..203]  the static series coefficient tables (identical for every
                ellipsoid, written once by mg_geodesic_init)
      [204..245] A3x, C3x, C4x for this (a, f)
      [246..271] C1a, C2a, C3a, C4a scratch
      [272..336] scratch GeodesicLine state used by _GenDirect

  GeodesicLine state (LINE_BLOCK = 65 doubles)
      [0..31]    scalars, written by GeodesicLine._save
      [32..64]   C1a, C1pa, C2a, C3a, C4a

  PolygonArea state (PA_BLOCK = 11 int64 slots)
      [0] num, [1] crossings, then 8 float64 from slot 2

Batch entry points take parallel float64 input arrays and write a
structure-of-arrays output block, field-major: out[field * n + i].
"""

from std.math import isfinite
from max.algorithm import parallelize
from ported import (
    K_A1M1,
    K_A2M1,
    K_A3,
    K_C1F,
    K_C1PF,
    K_C2F,
    K_C3,
    K_C4,
    GEOD_BLOCK,
    PA_AS,
    PA_AT,
    PA_CROSS,
    PA_F,
    PA_LAT0,
    PA_LAT1,
    PA_LON0,
    PA_LON1,
    PA_NUM,
    PA_PS,
    PA_PT,
    Accumulator,
    Geodesic,
    GeodesicLine,
    NAN,
    PolygonArea,
)

comptime CP = Pointer[Float64, UntrackedOrigin[mut=True]]
comptime IP = Pointer[Int, UntrackedOrigin[mut=True]]


def p(a: Int) -> CP:
    """Borrow a caller-owned float64 buffer."""
    return CP(unsafe_from_address=a)


def ip(a: Int) -> IP:
    """Borrow a caller-owned int64 buffer."""
    return IP(unsafe_from_address=a)

comptime INV_N: Int = 10
comptime DIR_N: Int = 9

# Batch parallelism.  Every point is independent, so the only thing the
# workers cannot share is the Clenshaw scratch that lives in the Geodesic
# block; the Python side hands over a pool of MG_MAX_WORKERS + 1 identical
# blocks and worker j uses slot j.  `bench/sweep.py` re-measures both
# numbers.
#
# MG_MAX_WORKERS: the inverse kernel over 100k points took 210 ms on one
# worker, 56 at 8, 46 at 16, 38.5 at 24 and 37.9 at 64 -- flat from 24 up, so
# the cap is set where the curve stops paying rather than at the core count.
#
# MG_MIN_PARALLEL: a parallel launch costs several milliseconds, and it only
# repays itself once the batch is big enough to hide that.  4096 points came
# out at 1.00x, 8191 at 0.99x, 8192 at 1.65x, 200k at 3.99x.
comptime MG_MAX_WORKERS: Int = 32
comptime MG_MIN_PARALLEL: Int = 8192
comptime MG_MIN_PER_WORKER: Int = 1024


def nworkers(n: Int, workers: Int) -> Int:
    """How many workers a batch of n points should use."""
    if workers <= 1 or n < MG_MIN_PARALLEL:
        return 1
    return min(min(MG_MAX_WORKERS, workers), n // MG_MIN_PER_WORKER)

# ---------------------------------------------------------------- ellipsoid
def setcoeff(blk: CP):
    """Write the static series tables; identical for every ellipsoid."""
    var a1m1 = blk.unsafe_offset(K_A1M1)
    a1m1.unsafe_store(0, 1.0)
    a1m1.unsafe_store(1, 4.0)
    a1m1.unsafe_store(2, 64.0)
    a1m1.unsafe_store(3, 0.0)
    a1m1.unsafe_store(4, 256.0)
    var a2m1 = blk.unsafe_offset(K_A2M1)
    a2m1.unsafe_store(0, -11.0)
    a2m1.unsafe_store(1, -28.0)
    a2m1.unsafe_store(2, -192.0)
    a2m1.unsafe_store(3, 0.0)
    a2m1.unsafe_store(4, 256.0)
    var c1f = blk.unsafe_offset(K_C1F)
    var i = 0
    c1f.unsafe_store(i, -1.0)
    c1f.unsafe_store(i + 1, 6.0)
    c1f.unsafe_store(i + 2, -16.0)
    c1f.unsafe_store(i + 3, 32.0)
    i += 4
    c1f.unsafe_store(i, -9.0)
    c1f.unsafe_store(i + 1, 64.0)
    c1f.unsafe_store(i + 2, -128.0)
    c1f.unsafe_store(i + 3, 2048.0)
    i += 4
    c1f.unsafe_store(i, 9.0)
    c1f.unsafe_store(i + 1, -16.0)
    c1f.unsafe_store(i + 2, 768.0)
    i += 3
    c1f.unsafe_store(i, 3.0)
    c1f.unsafe_store(i + 1, -5.0)
    c1f.unsafe_store(i + 2, 512.0)
    i += 3
    c1f.unsafe_store(i, -7.0)
    c1f.unsafe_store(i + 1, 1280.0)
    i += 2
    c1f.unsafe_store(i, -7.0)
    c1f.unsafe_store(i + 1, 2048.0)
    var c1pf = blk.unsafe_offset(K_C1PF)
    i = 0
    c1pf.unsafe_store(i, 205.0)
    c1pf.unsafe_store(i + 1, -432.0)
    c1pf.unsafe_store(i + 2, 768.0)
    c1pf.unsafe_store(i + 3, 1536.0)
    i += 4
    c1pf.unsafe_store(i, 4005.0)
    c1pf.unsafe_store(i + 1, -4736.0)
    c1pf.unsafe_store(i + 2, 3840.0)
    c1pf.unsafe_store(i + 3, 12288.0)
    i += 4
    c1pf.unsafe_store(i, -225.0)
    c1pf.unsafe_store(i + 1, 116.0)
    c1pf.unsafe_store(i + 2, 384.0)
    i += 3
    c1pf.unsafe_store(i, -7173.0)
    c1pf.unsafe_store(i + 1, 2695.0)
    c1pf.unsafe_store(i + 2, 7680.0)
    i += 3
    c1pf.unsafe_store(i, 3467.0)
    c1pf.unsafe_store(i + 1, 7680.0)
    i += 2
    c1pf.unsafe_store(i, 38081.0)
    c1pf.unsafe_store(i + 1, 61440.0)
    var c2f = blk.unsafe_offset(K_C2F)
    i = 0
    c2f.unsafe_store(i, 1.0)
    c2f.unsafe_store(i + 1, 2.0)
    c2f.unsafe_store(i + 2, 16.0)
    c2f.unsafe_store(i + 3, 32.0)
    i += 4
    c2f.unsafe_store(i, 35.0)
    c2f.unsafe_store(i + 1, 64.0)
    c2f.unsafe_store(i + 2, 384.0)
    c2f.unsafe_store(i + 3, 2048.0)
    i += 4
    c2f.unsafe_store(i, 15.0)
    c2f.unsafe_store(i + 1, 80.0)
    c2f.unsafe_store(i + 2, 768.0)
    i += 3
    c2f.unsafe_store(i, 7.0)
    c2f.unsafe_store(i + 1, 35.0)
    c2f.unsafe_store(i + 2, 512.0)
    i += 3
    c2f.unsafe_store(i, 63.0)
    c2f.unsafe_store(i + 1, 1280.0)
    i += 2
    c2f.unsafe_store(i, 77.0)
    c2f.unsafe_store(i + 1, 2048.0)
    var a3 = blk.unsafe_offset(K_A3)
    i = 0
    a3.unsafe_store(i, -3.0)
    a3.unsafe_store(i + 1, 128.0)
    i += 2
    a3.unsafe_store(i, -2.0)
    a3.unsafe_store(i + 1, -3.0)
    a3.unsafe_store(i + 2, 64.0)
    i += 3
    a3.unsafe_store(i, -1.0)
    a3.unsafe_store(i + 1, -3.0)
    a3.unsafe_store(i + 2, -1.0)
    a3.unsafe_store(i + 3, 16.0)
    i += 4
    a3.unsafe_store(i, 3.0)
    a3.unsafe_store(i + 1, -1.0)
    a3.unsafe_store(i + 2, -2.0)
    a3.unsafe_store(i + 3, 8.0)
    i += 4
    a3.unsafe_store(i, 1.0)
    a3.unsafe_store(i + 1, -1.0)
    a3.unsafe_store(i + 2, 2.0)
    i += 3
    a3.unsafe_store(i, 1.0)
    a3.unsafe_store(i + 1, 1.0)
    var c3 = blk.unsafe_offset(K_C3)
    i = 0
    c3.unsafe_store(i, 3.0)
    c3.unsafe_store(i + 1, 128.0)
    i += 2
    c3.unsafe_store(i, 2.0)
    c3.unsafe_store(i + 1, 5.0)
    c3.unsafe_store(i + 2, 128.0)
    i += 3
    c3.unsafe_store(i, -1.0)
    c3.unsafe_store(i + 1, 3.0)
    c3.unsafe_store(i + 2, 3.0)
    c3.unsafe_store(i + 3, 64.0)
    i += 4
    c3.unsafe_store(i, -1.0)
    c3.unsafe_store(i + 1, 0.0)
    c3.unsafe_store(i + 2, 1.0)
    c3.unsafe_store(i + 3, 8.0)
    i += 4
    c3.unsafe_store(i, -1.0)
    c3.unsafe_store(i + 1, 1.0)
    c3.unsafe_store(i + 2, 4.0)
    i += 3
    c3.unsafe_store(i, 5.0)
    c3.unsafe_store(i + 1, 256.0)
    i += 2
    c3.unsafe_store(i, 1.0)
    c3.unsafe_store(i + 1, 3.0)
    c3.unsafe_store(i + 2, 128.0)
    i += 3
    c3.unsafe_store(i, -3.0)
    c3.unsafe_store(i + 1, -2.0)
    c3.unsafe_store(i + 2, 3.0)
    c3.unsafe_store(i + 3, 64.0)
    i += 4
    c3.unsafe_store(i, 1.0)
    c3.unsafe_store(i + 1, -3.0)
    c3.unsafe_store(i + 2, 2.0)
    c3.unsafe_store(i + 3, 32.0)
    i += 4
    c3.unsafe_store(i, 7.0)
    c3.unsafe_store(i + 1, 512.0)
    i += 2
    c3.unsafe_store(i, -10.0)
    c3.unsafe_store(i + 1, 9.0)
    c3.unsafe_store(i + 2, 384.0)
    i += 3
    c3.unsafe_store(i, 5.0)
    c3.unsafe_store(i + 1, -9.0)
    c3.unsafe_store(i + 2, 5.0)
    c3.unsafe_store(i + 3, 192.0)
    i += 4
    c3.unsafe_store(i, 7.0)
    c3.unsafe_store(i + 1, 512.0)
    i += 2
    c3.unsafe_store(i, -14.0)
    c3.unsafe_store(i + 1, 7.0)
    c3.unsafe_store(i + 2, 512.0)
    i += 3
    c3.unsafe_store(i, 21.0)
    c3.unsafe_store(i + 1, 2560.0)
    var c4 = blk.unsafe_offset(K_C4)
    i = 0
    c4.unsafe_store(i, 97.0)
    c4.unsafe_store(i + 1, 15015.0)
    i += 2
    c4.unsafe_store(i, 1088.0)
    c4.unsafe_store(i + 1, 156.0)
    c4.unsafe_store(i + 2, 45045.0)
    i += 3
    c4.unsafe_store(i, -224.0)
    c4.unsafe_store(i + 1, -4784.0)
    c4.unsafe_store(i + 2, 1573.0)
    c4.unsafe_store(i + 3, 45045.0)
    i += 4
    c4.unsafe_store(i, -10656.0)
    c4.unsafe_store(i + 1, 14144.0)
    c4.unsafe_store(i + 2, -4576.0)
    c4.unsafe_store(i + 3, -858.0)
    c4.unsafe_store(i + 4, 45045.0)
    i += 5
    c4.unsafe_store(i, 64.0)
    c4.unsafe_store(i + 1, 624.0)
    c4.unsafe_store(i + 2, -4576.0)
    c4.unsafe_store(i + 3, 6864.0)
    c4.unsafe_store(i + 4, -3003.0)
    c4.unsafe_store(i + 5, 15015.0)
    i += 6
    c4.unsafe_store(i, 100.0)
    c4.unsafe_store(i + 1, 208.0)
    c4.unsafe_store(i + 2, 572.0)
    c4.unsafe_store(i + 3, 3432.0)
    c4.unsafe_store(i + 4, -12012.0)
    c4.unsafe_store(i + 5, 30030.0)
    c4.unsafe_store(i + 6, 45045.0)
    i += 7
    c4.unsafe_store(i, 1.0)
    c4.unsafe_store(i + 1, 9009.0)
    i += 2
    c4.unsafe_store(i, -2944.0)
    c4.unsafe_store(i + 1, 468.0)
    c4.unsafe_store(i + 2, 135135.0)
    i += 3
    c4.unsafe_store(i, 5792.0)
    c4.unsafe_store(i + 1, 1040.0)
    c4.unsafe_store(i + 2, -1287.0)
    c4.unsafe_store(i + 3, 135135.0)
    i += 4
    c4.unsafe_store(i, 5952.0)
    c4.unsafe_store(i + 1, -11648.0)
    c4.unsafe_store(i + 2, 9152.0)
    c4.unsafe_store(i + 3, -2574.0)
    c4.unsafe_store(i + 4, 135135.0)
    i += 5
    c4.unsafe_store(i, -64.0)
    c4.unsafe_store(i + 1, -624.0)
    c4.unsafe_store(i + 2, 4576.0)
    c4.unsafe_store(i + 3, -6864.0)
    c4.unsafe_store(i + 4, 3003.0)
    c4.unsafe_store(i + 5, 135135.0)
    i += 6
    c4.unsafe_store(i, 8.0)
    c4.unsafe_store(i + 1, 10725.0)
    i += 2
    c4.unsafe_store(i, 1856.0)
    c4.unsafe_store(i + 1, -936.0)
    c4.unsafe_store(i + 2, 225225.0)
    i += 3
    c4.unsafe_store(i, -8448.0)
    c4.unsafe_store(i + 1, 4992.0)
    c4.unsafe_store(i + 2, -1144.0)
    c4.unsafe_store(i + 3, 225225.0)
    i += 4
    c4.unsafe_store(i, -1440.0)
    c4.unsafe_store(i + 1, 4160.0)
    c4.unsafe_store(i + 2, -4576.0)
    c4.unsafe_store(i + 3, 1716.0)
    c4.unsafe_store(i + 4, 225225.0)
    i += 5
    c4.unsafe_store(i, -136.0)
    c4.unsafe_store(i + 1, 63063.0)
    i += 2
    c4.unsafe_store(i, 1024.0)
    c4.unsafe_store(i + 1, -208.0)
    c4.unsafe_store(i + 2, 105105.0)
    i += 3
    c4.unsafe_store(i, 3584.0)
    c4.unsafe_store(i + 1, -3328.0)
    c4.unsafe_store(i + 2, 1144.0)
    c4.unsafe_store(i + 3, 315315.0)
    i += 4
    c4.unsafe_store(i, -128.0)
    c4.unsafe_store(i + 1, 135135.0)
    i += 2
    c4.unsafe_store(i, -2560.0)
    c4.unsafe_store(i + 1, 832.0)
    c4.unsafe_store(i + 2, 405405.0)
    i += 3
    c4.unsafe_store(i, 128.0)
    c4.unsafe_store(i + 1, 99099.0)


@export("mg_geodesic_init")
def mg_geodesic_init(a: Float64, f: Float64, block: Int) abi("C") -> Int:
    """Fill a Geodesic block.  0 ok, 1 bad a, 2 bad b."""
    setcoeff(p(block))
    var g = Geodesic(a, f, p(block), False)
    if not (isfinite(a) and a > 0.0):
        return 1
    if not (isfinite(g._b) and g._b > 0.0):
        return 2
    return 0


@export("mg_geodesic_c2")
def mg_geodesic_c2(a: Float64, f: Float64, block: Int) abi("C") -> Float64:
    """The authalic radius squared, for PolygonArea.area0."""
    return Geodesic(a, f, p(block), True)._c2


# ----------------------------------------------------------------- inverse
def inv_range(
    a: Float64,
    f: Float64,
    blk: Int,
    lat1: Int,
    lon1: Int,
    lat2: Int,
    lon2: Int,
    outmask: Int,
    dst: Int,
    n: Int,
    lo: Int,
    hi: Int,
):
    """The inverse kernel over [lo, hi), writing into the full-length block."""
    var g = Geodesic(a, f, p(blk), True)
    var x1 = p(lat1)
    var y1 = p(lon1)
    var x2 = p(lat2)
    var y2 = p(lon2)
    var o = p(dst)
    var i = lo
    while i < hi:
        var la1 = x1.unsafe_load(i)
        var la2 = x2.unsafe_load(i)
        var r = g._GenInverse(
            la1, y1.unsafe_load(i), la2, y2.unsafe_load(i), outmask
        )
        o.unsafe_store(0 * n + i, r[0])
        o.unsafe_store(1 * n + i, r[1])
        o.unsafe_store(2 * n + i, r[2])
        o.unsafe_store(3 * n + i, r[3])
        o.unsafe_store(4 * n + i, r[4])
        o.unsafe_store(5 * n + i, r[5])
        o.unsafe_store(6 * n + i, r[6])
        o.unsafe_store(7 * n + i, r[7])
        o.unsafe_store(8 * n + i, r[8])
        o.unsafe_store(9 * n + i, r[9])
        i += 1


def split(n: Int, w: Int) -> Int:
    """Points per worker; w chunks, so chunk j owns [j*c, min(n, (j+1)*c))."""
    return (n + w - 1) // w


@export("mg_inverse")
def mg_inverse(
    a: Float64,
    f: Float64,
    block: Int,
    lat1: Int,
    lon1: Int,
    lat2: Int,
    lon2: Int,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var w = nworkers(n, workers)
    if w <= 1:
        inv_range(a, f, block, lat1, lon1, lat2, lon2, outmask, dst, n, 0, n)
        return
    var c = split(n, w)

    def work(j: Int) {imm}:
        var lo = j * c
        var hi = lo + c
        if hi > n:
            hi = n
        inv_range(
            a, f, block + 8 * GEOD_BLOCK * j, lat1, lon1, lat2, lon2, outmask,
            dst, n, lo, hi,
        )

    parallelize(work, w, w)


# ------------------------------------------------------------------ direct
def gendir_range(
    a: Float64,
    f: Float64,
    blk: Int,
    x1: Int,
    y1: Int,
    z: Int,
    d: Int,
    arcmode: Bool,
    outmask: Int,
    dst: Int,
    n: Int,
    lo: Int,
    hi: Int,
):
    var g = Geodesic(a, f, p(blk), True)
    var q1 = p(x1)
    var q2 = p(y1)
    var q3 = p(z)
    var q4 = p(d)
    var o = p(dst)
    var i = lo
    while i < hi:
        var r = g._GenDirect(
            q1.unsafe_load(i), q2.unsafe_load(i), q3.unsafe_load(i), arcmode,
            q4.unsafe_load(i), outmask,
        )
        o.unsafe_store(0 * n + i, r[0])
        o.unsafe_store(1 * n + i, r[1])
        o.unsafe_store(2 * n + i, r[2])
        o.unsafe_store(3 * n + i, r[3])
        o.unsafe_store(4 * n + i, r[4])
        o.unsafe_store(5 * n + i, r[5])
        o.unsafe_store(6 * n + i, r[6])
        o.unsafe_store(7 * n + i, r[7])
        o.unsafe_store(8 * n + i, r[8])
        i += 1


def gendir(
    a: Float64,
    f: Float64,
    blk: Int,
    x1: Int,
    y1: Int,
    z: Int,
    d: Int,
    arcmode: Bool,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
):
    var w = nworkers(n, workers)
    if w <= 1:
        gendir_range(
            a, f, blk, x1, y1, z, d, arcmode, outmask, dst, n, 0, n
        )
        return
    var c = split(n, w)

    def work(j: Int) {imm}:
        var lo = j * c
        var hi = lo + c
        if hi > n:
            hi = n
        gendir_range(
            a, f, blk + 8 * GEOD_BLOCK * j, x1, y1, z, d, arcmode, outmask, dst,
            n, lo, hi,
        )

    parallelize(work, w, w)


@export("mg_direct")
def mg_direct(
    a: Float64,
    f: Float64,
    block: Int,
    lat1: Int,
    lon1: Int,
    azi1: Int,
    s12: Int,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    gendir(a, f, block, lat1, lon1, azi1, s12, False, outmask, dst, n, workers)


@export("mg_arcdirect")
def mg_arcdirect(
    a: Float64,
    f: Float64,
    block: Int,
    lat1: Int,
    lon1: Int,
    azi1: Int,
    a12: Int,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    gendir(a, f, block, lat1, lon1, azi1, a12, True, outmask, dst, n, workers)


# -------------------------------------------------------------------- line
@export("mg_line_new")
def mg_line_new(
    a: Float64,
    f: Float64,
    block: Int,
    lat1: Float64,
    lon1: Float64,
    azi1: Float64,
    salp1: Float64,
    calp1: Float64,
    caps: Int,
    state: Int,
) abi("C"):
    var g = Geodesic(a, f, p(block), True)
    var line = GeodesicLine(g, lat1, lon1, azi1, caps, salp1, calp1, p(state), False)
    line._save(p(state))


def linepos_range(
    a: Float64,
    f: Float64,
    blk: Int,
    state: Int,
    s: Int,
    arcmode: Bool,
    outmask: Int,
    dst: Int,
    n: Int,
    lo: Int,
    hi: Int,
):
    """_GenPosition over [lo, hi).

    `_GenPosition` reads only the line's own state and writes only to the
    output block, so the chunks share nothing and the Geodesic block's
    Clenshaw scratch is not touched at all.
    """
    var g = Geodesic(a, f, p(blk), True)
    var line = GeodesicLine(g, NAN, NAN, NAN, 0, NAN, NAN, p(state), True)
    var q = p(s)
    var o = p(dst)
    var i = lo
    while i < hi:
        var r = line._GenPosition(arcmode, q.unsafe_load(i), outmask)
        o.unsafe_store(0 * n + i, r[0])
        o.unsafe_store(1 * n + i, r[1])
        o.unsafe_store(2 * n + i, r[2])
        o.unsafe_store(3 * n + i, r[3])
        o.unsafe_store(4 * n + i, r[4])
        o.unsafe_store(5 * n + i, r[5])
        o.unsafe_store(6 * n + i, r[6])
        o.unsafe_store(7 * n + i, r[7])
        o.unsafe_store(8 * n + i, r[8])
        i += 1


def linepos(
    a: Float64,
    f: Float64,
    blk: Int,
    state: Int,
    s: Int,
    arcmode: Bool,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
):
    var w = nworkers(n, workers)
    if w <= 1:
        linepos_range(a, f, blk, state, s, arcmode, outmask, dst, n, 0, n)
    else:
        var c = split(n, w)

        def work(j: Int) {imm}:
            var lo = j * c
            var hi = lo + c
            if hi > n:
                hi = n
            linepos_range(
                a, f, blk, state, s, arcmode, outmask, dst, n, lo, hi
            )

        parallelize(work, w, w)
    # _GenPosition does not mutate the line, so this writes back the state
    # the call was given; it stays after the chunks so it is not racing them.
    var g = Geodesic(a, f, p(blk), True)
    var line = GeodesicLine(g, NAN, NAN, NAN, 0, NAN, NAN, p(state), True)
    line._save(p(state))


@export("mg_line_position")
def mg_line_position(
    a: Float64,
    f: Float64,
    block: Int,
    state: Int,
    s12: Int,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    linepos(a, f, block, state, s12, False, outmask, dst, n, workers)


@export("mg_line_arcposition")
def mg_line_arcposition(
    a: Float64,
    f: Float64,
    block: Int,
    state: Int,
    a12: Int,
    outmask: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    linepos(a, f, block, state, a12, True, outmask, dst, n, workers)


@export("mg_line_set_distance")
def mg_line_set_distance(
    a: Float64, f: Float64, block: Int, state: Int, s13: Float64
) abi("C") -> Float64:
    """Set point 3 from a distance; returns the arc length a13."""
    var g = Geodesic(a, f, p(block), True)
    var line = GeodesicLine(g, NAN, NAN, NAN, 0, NAN, NAN, p(state), True)
    line.SetDistance(s13)
    line._save(p(state))
    return line.a13


@export("mg_line_set_arc")
def mg_line_set_arc(
    a: Float64, f: Float64, block: Int, state: Int, a13: Float64
) abi("C") -> Float64:
    """Set point 3 from an arc length; returns the distance s13."""
    var g = Geodesic(a, f, p(block), True)
    var line = GeodesicLine(g, NAN, NAN, NAN, 0, NAN, NAN, p(state), True)
    line.SetArc(a13)
    line._save(p(state))
    return line.s13


# ----------------------------------------------------------------- polygon
# The float64 half of the block starts at slot PA_F; PA_LAT0 .. PA_PT index it.
def polf(state: Int) -> CP:
    return p(state + 8 * PA_F)


def polload(g: Geodesic, state: Int, polyline: Bool) -> PolygonArea:
    var pa = PolygonArea(g, polyline)
    var q = ip(state)
    var f = polf(state)
    pa.num = q.unsafe_load(PA_NUM)
    pa._crossings = q.unsafe_load(PA_CROSS)
    pa._lat0 = f.unsafe_load(PA_LAT0)
    pa._lon0 = f.unsafe_load(PA_LON0)
    pa.lat1 = f.unsafe_load(PA_LAT1)
    pa.lon1 = f.unsafe_load(PA_LON1)
    pa._areasum = Accumulator(0.0)
    pa._areasum.Set(f.unsafe_load(PA_AS), f.unsafe_load(PA_AT))
    pa._perimetersum = Accumulator(0.0)
    pa._perimetersum.Set(f.unsafe_load(PA_PS), f.unsafe_load(PA_PT))
    return pa^


def polsave(ref pa: PolygonArea, state: Int):
    var q = ip(state)
    var f = polf(state)
    q.unsafe_store(PA_NUM, pa.num)
    q.unsafe_store(PA_CROSS, pa._crossings)
    f.unsafe_store(PA_LAT0, pa._lat0)
    f.unsafe_store(PA_LON0, pa._lon0)
    f.unsafe_store(PA_LAT1, pa.lat1)
    f.unsafe_store(PA_LON1, pa.lon1)
    f.unsafe_store(PA_AS, pa._areasum._s)
    f.unsafe_store(PA_AT, pa._areasum._t)
    f.unsafe_store(PA_PS, pa._perimetersum._s)
    f.unsafe_store(PA_PT, pa._perimetersum._t)


@export("mg_polygon_clear")
def mg_polygon_clear(a: Float64, f: Float64, block: Int, state: Int) abi("C"):
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, False)
    pa.Clear()
    polsave(pa, state)


@export("mg_polygon_addpoint")
def mg_polygon_addpoint(
    a: Float64, f: Float64, block: Int, state: Int, lat: Float64, lon: Float64
) abi("C") -> Int:
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, False)
    pa.AddPoint(lat, lon)
    polsave(pa, state)
    return pa.num


@export("mg_polygon_addpoints")
def mg_polygon_addpoints(
    a: Float64, f: Float64, block: Int, state: Int, lat: Int, lon: Int, n: Int
) abi("C") -> Int:
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, False)
    var x = p(lat)
    var y = p(lon)
    var i = 0
    while i < n:
        pa.AddPoint(x.unsafe_load(i), y.unsafe_load(i))
        i += 1
    polsave(pa, state)
    return pa.num


@export("mg_polygon_addedge")
def mg_polygon_addedge(
    a: Float64, f: Float64, block: Int, state: Int, azi: Float64, s: Float64
) abi("C") -> Int:
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, False)
    pa.AddEdge(azi, s)
    polsave(pa, state)
    return pa.num


@export("mg_polygon_compute")
def mg_polygon_compute(
    a: Float64,
    f: Float64,
    block: Int,
    state: Int,
    reverse: Int,
    sign: Int,
    polyline: Int,
    dst: Int,
) abi("C") -> Int:
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, polyline != 0)
    var r = pa.Compute(reverse != 0, sign != 0)
    polsave(pa, state)
    var o = p(dst)
    o.unsafe_store(0, Float64(r[1]))
    o.unsafe_store(1, r[2])
    return r[0]


@export("mg_polygon_testpoint")
def mg_polygon_testpoint(
    a: Float64,
    f: Float64,
    block: Int,
    state: Int,
    lat: Float64,
    lon: Float64,
    reverse: Int,
    sign: Int,
    polyline: Int,
    dst: Int,
) abi("C") -> Int:
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, polyline != 0)
    var r = pa.TestPoint(lat, lon, reverse != 0, sign != 0)
    var o = p(dst)
    o.unsafe_store(0, Float64(r[1]))
    o.unsafe_store(1, r[2])
    return r[0]


@export("mg_polygon_testedge")
def mg_polygon_testedge(
    a: Float64,
    f: Float64,
    block: Int,
    state: Int,
    azi: Float64,
    s: Float64,
    reverse: Int,
    sign: Int,
    polyline: Int,
    dst: Int,
) abi("C") -> Int:
    var g = Geodesic(a, f, p(block), True)
    var pa = polload(g, state, polyline != 0)
    var r = pa.TestEdge(azi, s, reverse != 0, sign != 0)
    var o = p(dst)
    o.unsafe_store(0, Float64(r[1]))
    o.unsafe_store(1, r[2])
    return r[0]
