# mojo-geographiclib

GeographicLib's geodesic routines with the numerics written in Mojo.

`geographiclib` is Charles Karney's reference implementation of geodesic
problems on an ellipsoid. This repository re-implements it in Mojo, keeps the
Python API name for name and signature for signature, and checks the results
against the real upstream package field by field. The port is a single
compilation unit, reached from Python through `ctypes`. A batch call is spread
across worker threads; a single-point call is not, because there is nothing to
spread.

On WGS84 the two implementations return *the same doubles* for the large
majority of inputs; where they differ at all it is by one or two units in the
last place.

## Install

```bash
pixi install
pixi run build      # src/capi.mojo -> dist/libmojo-geographiclib.so
pixi run test       # parity against upstream geographiclib
pixi run bench      # benchmarks against upstream, markdown table on stdout
pixi run python bench/profile.py   # per-path cost breakdown
pixi run python bench/sweep.py     # the batch thresholds, re-measured
```

Everything runs inside the pixi environment; the pinned toolchain is
`mojo 1.2.0.dev2026092605` with `max 26.7.0.dev2026092605` beside it.

## Usage

```python
from mojogeographiclib.geodesic import Geodesic

Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)
# {'lat1': -41.32, 'lon1': 174.81, 'lat2': 40.96, 'lon2': -5.5,
#  'a12': 179.6197069334283, 's12': 19959679.26735382,
#  'azi1': 161.06766998616015, 'azi2': 18.825195123247063}

line = Geodesic.WGS84.Line(37.0, -122.0, 51.0)
line.Position(2_000_000.0)['lat2']
# 46.7007763420085

# Many points through one kernel call; upstream has no batch API, so this is
# a drop-in for a loop over Inverse.
import numpy as np
lat, lon = np.array([0.0, 10.0]), np.array([0.0, 10.0])
r = Geodesic.WGS84.InversePairs(lat, lon, lat + 1.0, lon + 1.0,
                                outmask=Geodesic.WGS84.ALL)
r['s12'].shape  # (2,)

p = Geodesic.WGS84.Polygon()
p.AddPoints(*zip(*[(0, 0), (0, 90), (90, 0)]))
p.Compute()  # (3, 30022685.630020067, 63758202715511.055)
```

`examples/quickstart.py` runs the same thing; `pixi run python examples/quickstart.py`.

## What is covered

The upstream Python package has eight modules. Seven of them are here, with the
same names, arguments, defaults, return shapes and exception behaviour:

| upstream module | here | notes |
| --- | --- | --- |
| `geographiclib.constants` | `mojogeographiclib.constants` | `Constants.WGS84_a/f` |
| `geographiclib.geodesiccapability` | `mojogeographiclib.geodesiccapability` | every bit constant |
| `geographiclib.geomath` | `mojogeographiclib.geomath` | `Math` (see below) |
| `geographiclib.geodesic` | `mojogeographiclib.geodesic` | `Geodesic`, all methods |
| `geographiclib.geodesicline` | `mojogeographiclib.geodesicline` | `GeodesicLine`, all methods |
| `geographiclib.polygonarea` | `mojogeographiclib.polygonarea` | `PolygonArea`, all methods |
| `geographiclib.accumulator` | *not exposed* | `Accumulator` is only ever used inside `PolygonArea`, and all of `PolygonArea` -- accumulator included -- is computed in Mojo. |

Numerically ported in `src/ported.mojo`, in upstream's module order:
`Math` (geomath), `Accumulator` (accumulator), `Geodesic` (geodesic),
`GeodesicLine` (geodesicline), `PolygonArea` (polygonarea). Every private
helper is there too: `_SinCosSeries`, `_Astroid`, `_A1m1f`, `_C1f`, `_C1pf`,
`_A2m1f`, `_C2f`, `_A3coeff`, `_C3coeff`, `_C4coeff`, `_A3f`, `_C3f`, `_C4f`,
`_Lengths`, `_InverseStart`, `_Lambda12`, `_GenInverse`, `_GenDirect`,
`_GenDirectLine`, `_GenPosition`, `_transit`, `_transitdirect`,
`_areareduceA`, `_areareduceB`. Series order 6, as upstream.

`Geodesic` handles any ellipsoid, not just WGS84: the tests run the whole
parity suite on f = 0, 1/150, -1/100, 1/10, -1/5 and a deliberately silly
one-metre ellipsoid.

### Extensions (not in upstream)

These exist because the port is batch-first; each is a drop-in for the obvious
loop, and each returns what the loop would have returned, bit for bit. The
five `*Many` / `*s` entry points spread a batch across worker threads once it
is large enough to pay for the launch; `PolygonArea.AddPoints` cannot,
because its Shewchuk accumulation is inherently sequential.

| name | loop it replaces |
| --- | --- |
| `Geodesic.InversePairs(lat1, lon1, lat2, lon2, outmask)` | `Inverse` over arrays |
| `Geodesic.DirectMany(lat1, lon1, azi1, s12, outmask)` | `Direct` over arrays |
| `Geodesic.ArcDirectMany(lat1, lon1, azi1, a12, outmask)` | `ArcDirect` over arrays |
| `GeodesicLine.Positions(s12, outmask)` | `Position` over arrays |
| `GeodesicLine.ArcPositions(a12, outmask)` | `ArcPosition` over arrays |
| `PolygonArea.AddPoints(lat, lon)` | `AddPoint` over arrays |

### Documented divergences from the upstream source

Where Mojo forced a change, the change is listed here. Everything else -- the
branch structure, the order of the arithmetic, the tolerance constants, the
Newton iteration and its bracketing -- follows upstream line for line.

**Python-visible signatures are unchanged.** The changes below are all either
inside the Mojo kernel file or inside the ctypes glue.

1. **`Math.fmod` is new.** `std.math` has no `fmod`, and Mojo's `%` on floats is
   Python-style modulo (sign of the divisor), not C `fmod`. Upstream
   `Math.sincosd` calls `math.fmod`, so the port carries an exact dyadic
   reduction instead.
2. **`Math.radians` / `Math.degrees` are new helpers.** Upstream calls the
   `math` module inline; there is no such module here.
3. **`Math.cbrt` uses `std.math.cbrt`.** Upstream spells it
   `copysign(pow(abs(x), 1/3.0), x)`; `std.math`'s `pow` is a different
   implementation and disagrees with libm's `pow` in the last place, so the
   port uses the intrinsic, which agrees with libm except (1 ulp) at
   extreme magnitudes. This is the only primitive used by the port that is not
   bit-identical to its Python counterpart.
4. **Coefficient tables are a caller-owned buffer, not struct fields.** A Mojo
   struct field cannot hold a pointer of parametric origin, and a struct with
   a user `__init__` is not implicitly copyable, so `Geodesic.__init__` and
   `GeodesicLine.__init__` take the block plus a `ready` flag instead of owning
   their tables. The Python `Geodesic(a, f)` and
   `GeodesicLine(geod, lat1, lon1, azi1, caps, salp1, calp1)` signatures are
   upstream's. A `Geodesic` owns a *pool* of identical blocks, one per batch
   worker, because the Clenshaw scratch inside a block is written per point;
   see *How it works*.
5. **Array arguments where upstream passes a list.** `Math.polyval` takes
   `(N, pointer, start, x)`, `Geodesic._SinCosSeries` takes
   `(sinp, sinx, cosx, c, len(c))`, and the static coefficient tables are
   reached through the block. `_Lengths` still takes `C1a` and `C2a` as
   explicit scratch arguments, exactly as upstream.
6. **Multiple returns are tuples, unpacked by index.** Mojo has no Python-style
   "rebind these names from a tuple" that reaches an enclosing scope, so
   `s12x, m12x, M12, M21 = self._Lengths(...)` becomes
   `r = self._Lengths(...); s12x = r[0]` and so on.
7. **`outmask` is masked into a local.** Upstream writes
   `outmask &= Geodesic.OUT_MASK`; Mojo parameters are immutable, so the port
   writes `var om = outmask & OUT_MASK` and uses `om`. Same for `caps` in
   `_GenDirectLine` and the two latitude arguments of `_GenInverse`,
   `_Lambda12` and `AddPoint`, which upstream rebinds and which are therefore
   bound to mutable locals at the call site.
8. **Upstream's `real x` forward declarations become ordinary locals**, and
   the caps-gated members of `GeodesicLine` are initialised to NaN because a
   Mojo struct has no partially-constructed states.
9. **`Accumulator.Set` has a two-float overload**, used only by the ctypes
   layer to reload the accumulator from its state block.
10. **Floating-point contraction is disabled** (`--fp-mode contract=off`, in
    both `build/build.sh` and the fallback build in `_lib.py`). By default
    this toolchain fuses `a + b*c` into an FMA, which is what Clang does with
    `-ffp-contract=fast` and is normally harmless -- but here it both breaks
    agreement with libm and, on this toolchain, produces outright wrong
    results in the series expansions. Turn it off and the kernels match
    upstream to the last place.
11. **`PolygonArea.num`, `.lat1` and `.lon1` are read-only properties** read
    back from the native state block rather than plain Python attributes.
    Upstream documents all three as readonly.
12. **The batch kernels take a trailing `workers` count.** It is how deep the
    caller's block pool is; the kernel clamps it to what the batch size can
    keep busy and to `MG_MAX_WORKERS`, and takes the serial path when it comes
    out at 1. It is an internal ABI detail -- the Python signatures are
    upstream's -- and the answer does not depend on it, which is what
    `tests/test_geodesic.py` asserts.

`mojogeographiclib.geomath.Math` is a faithful copy of upstream's `Math` and is
used only where upstream uses it too: assembling the result dicts and
reporting a `GeodesicLine`'s own attributes. The array forms
(`ang_normalize`, `lat_fix`, `ang_diff`, `atan2d`) exist only for the batch
entry points and follow the scalar code step for step; the one place they
differ is `_rem`, which falls back to the scalar routine when `|x/y| >= 2**52`
and the closed form stops being exact.

## Benchmarks

Produced by `pixi run bench` (which holds a machine-wide lock, so the numbers
are not distorted by other jobs). Upstream has no batch API, so every
"upstream" figure is a plain Python loop over the same inputs; the scalar rows
loop on both sides and are the fair comparison of the single-point path.

Measured on:

    Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 threads, Linux x86_64
    python 3.13.14, numpy 2.5.1, best of 3

"before" is the table this file shipped with; "after" is the current code.
This is a shared box, so treat the short rows as a range rather than a point:
two runs of the same binary gave 44.13 and 47.27 ms for `Inverse, 100k, all
outputs`, 17.42 and 26.44 ms for `Direct, 100k`, and 79.55 and 88.72 ms for
`AddPoint, 20k`. The table below is the first of the two.

| case | note | before | after | upstream | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Inverse, 20k scalar calls (Python loop) | scalar API, STANDARD mask | 484.22 ms | 264.87 ms | 2617.71 ms | 9.9x |
| Inverse, 100k points, all outputs | batch API vs upstream loop | 202.67 ms | 44.13 ms | 13643.09 ms | 309.2x |
| Inverse, 100k points, STANDARD mask | batch API vs upstream loop | 182.49 ms | 42.77 ms | 11375.38 ms | 265.9x |
| Direct, 100k points, all outputs | batch API vs upstream loop | 76.45 ms | 17.42 ms | 6213.39 ms | 356.8x |
| ArcDirect, 100k points, all outputs | batch API vs upstream loop | 65.04 ms | 18.29 ms | 5535.17 ms | 302.6x |
| GeodesicLine.Position, 20k calls | scalar API, one line | 160.47 ms | 105.94 ms | 169.10 ms | 1.6x |
| GeodesicLine.Positions, 100k | batch API vs upstream loop | 36.23 ms | 10.55 ms | 838.14 ms | 79.5x |
| PolygonArea, 20k vertices (AddPoints) | batch API vs upstream loop | 34.57 ms | 33.91 ms | 2609.33 ms | 77.0x |
| PolygonArea, 20k vertices (AddPoint) | scalar API on both sides | 81.33 ms | 79.55 ms | 2583.17 ms | 32.5x |

Reading the table honestly:

* The batch rows are not a like-for-like fight. Upstream has no batch API, so
  its column is the cost of 100,000 Python function calls, and most of the
  speedup is the interpreter loop that disappears.
* The scalar rows are. `Inverse` in a Python loop went from 5.5x to 9.9x, but
  almost none of that is arithmetic -- the arithmetic was already 5x faster
  and the remaining cost was the interpreter on both sides. It is the
  Python-side glue around the FFI call that this pass removed (see *How it
  works*): numpy element access, address lookups and per-call library
  lookups were costing about 2 us a call, twice the kernel.
* `GeodesicLine.Position` is the one case that is still close to a wash
  (1.6x), and it is close to its floor. A single position is about 0.23 us of
  arithmetic; the call is 5.3 us, of which 2.5 us is the ctypes boundary
  itself -- a nine-argument `ctypes` call costs about 2.5 us whatever is on
  the other end of it. Everything above that has been taken out. Use
  `Positions` for anything repeated: it is 79x.
* `PolygonArea` barely moved, and that is expected. `AddPoints` is a running
  Shewchuk accumulation, so it cannot be split across workers at all, and
  `AddPoint` is one ctypes call and one geodesic per vertex with no
  redundant work left around it.
* There is no multithreaded BLAS to race against anywhere in this benchmark,
  and the batch rows are now multithreaded here and single-threaded there.

### What the batch speedup actually is

The batch rows are faster for two separate reasons, and they are worth
separating because only one of them is arithmetic.

1. **The batch kernels are now parallel.** `mg_inverse`, `mg_direct`,
   `mg_arcdirect`, `mg_line_position` and `mg_line_arcposition` split a batch
   into one chunk per worker with `max.algorithm.parallelize`. Every point is
   independent, so the only thing workers cannot share is the Clenshaw
   scratch inside the `Geodesic` block; a `Geodesic` therefore owns a pool of
   identical blocks and worker *j* uses slot *j*. Since each point is
   computed by exactly the same code on exactly the same inputs, the parallel
   answer is the serial answer bit for bit -- `tests/test_geodesic.py` asserts
   that for sizes that straddle the threshold and are not multiples of the
   worker count.
2. **The single-point glue got cheaper.** Measured with
   `pixi run python bench/profile.py`: `Geodesic.Inverse` went from 21.2 us
   to about 10 us, `GeodesicLine.Position` from 8.0 us to 5.2 us.

Both thresholds in `src/capi.mojo` are measured, and `bench/sweep.py`
re-measures them:

* A parallel launch costs several milliseconds, so below **8192 points** the
   kernel stays serial: 4096 points measured 1.00x, 8191 measured 0.99x, 8192
   measured 1.65x, 200k measured 3.99x.
* The worker curve is flat past about 24, so the cap is **32** rather than
   the core count: 100k inverse points took 210 ms on one worker, 56 at 8, 46
   at 16, 38.5 at 24 and 37.9 at 64.

### What did not pay, and why

**SIMD.** There is no contiguous elementwise loop in the kernels to widen.
`_GenInverse` is a Newton iteration whose trip count depends on the point,
with a per-point Clenshaw scratch (`C1a`, `C2a`, `C3a`) recomputed inside
`_Lengths` for every point, and `_GenDirect` rebuilds a whole `GeodesicLine`
-- four polynomial coefficient sets, four Clenshaw sums, `sincosd`, `hypot` --
per point. The arithmetic between the transcendentals is maybe 20 operations
against a `sin`, a `cos` or a `sqrt`, and the vector forms of those in this
toolchain are not the `libm` ones the port calls: the agreement with upstream
down to the last place comes from calling the same `libm`, and a vectorised
`sin` would give it up. So widening the vector buys at most a few percent of
a kernel that is already 3-7x faster than upstream per point.

The one genuinely contiguous elementwise stage this port has is the NumPy
post-processing of a batch result -- `ang_diff`, `lat_fix`, `atan2d` -- and
`bench/profile.py` measures it at about 19 ms against a 22 ms kernel for 100k
points. A fused SIMD pass would remove most of that, but half of `atan2d` is
`np.arctan2` (3.9 ms of its 8.4 ms) with no SIMD form that keeps `libm`'s
result, and the rows it would help are already 265-357x. It was measured and
left out.

**ctypes calling conventions.** Calling the kernels through a `CFUNCTYPE`
prototype instead of `getattr(lib, name)` was tried, on the theory that it
skips a per-call signature lookup. A/B'd in-process on `line.Position` it came
out 5.09 us against 4.92 us -- no better, in one run slightly worse. Reverted.

**The GPU.** The host API is available (`max.gpu.host.DeviceContext`) and the
box has a card with 15.8 GB free, so nothing was skipped for want of one.
This kernel is not a GPU kernel. A `Geodesic.InversePairs` call moves 32 bytes
of input and 80 bytes of output per point, against on the order of 200 flops
(a count of the series expansions, not a measurement) -- under 2 flops per
byte -- and those flops are dominated by `sin`, `cos`, `atan2`, `atanh` and
`pow` inside a data-dependent Newton loop, which is the shape a GPU handles
worst. The 100k points would also have to be copied up (3.2 MB) and back
(8 MB) against a 22 ms CPU kernel. A GPU path here would be slower and would
give up bit-parity; the CPU port is the answer, and it is a deliberate one.

## How it works

**FFI.** `src/ported.mojo` holds the kernels; `src/capi.mojo` is the only file
with `@export`, and it contains nothing but the wrappers. Every buffer crosses
as an `Int` address, because `@export` refuses parametric functions and a
pointer with an inferred origin is parametric; the address is turned back into
a `Pointer[Float64, UntrackedOrigin]` inside the wrapper. Nothing on the Mojo
side allocates, so nothing can leak. The library is opened with
`ctypes.PyDLL` rather than `ctypes.CDLL`: the kernels keep a small amount of
per-ellipsoid scratch in a buffer owned by the calling Python object, so two
Python threads must not be inside the same `Geodesic` at once, and PyDLL's
refusal to release the GIL is exactly that guarantee. The workers a batch
kernel starts are inside one ctypes call, so the GIL guarantee is not what
protects them -- the per-worker blocks are.

**Single-point calls.** `Inverse` with four floats used to cost 21 us, of
which the kernel was 3. Three things were eating the rest, all of them on
the Python side of the boundary and none of them visible in a kernel-level
profile:

* Each scalar went into and out of a NumPy array one element at a time, and
  `ndarray.__setitem__` / `ndarray.__getitem__` each build a fresh scalar
  object: 4 writes plus 10 reads came to about 2.5 us a call. A single
  thread-local `bytearray` per call site, written with one
  `struct.pack_into` and read with one `struct.unpack_from`, costs 0.3 us for
  the same work, and hands the kernel one base address from which the four
  input slices are `+0`, `+8`, `+16`, `+24`.
* `ndarray.ctypes.data` builds a new ctypes object on every access, and the
  code called it four times per `Inverse` looking up addresses of buffers
  that never moved. Every address is now taken once.
* `getattr(lib, "mg_inverse")` ran on every call. The entry points are
  resolved once, in `_lib.fn`.

**Memory layout.** Three caller-owned blocks, all allocated on the Python
side and all sized by constants in `ported.mojo`:

* *Geodesic blocks*, 337 float64 each, a pool of one per batch worker plus
  one spare. Every slot is a copy of slot 0. The static series coefficient
  tables (identical for every ellipsoid) are at `[0..203]`, the ellipsoid's
  own `A3x`, `C3x`, `C4x` at `[204..245]`, the Clenshaw scratch `C1a`,
  `C2a`, `C3a`, `C4a` at `[246..271]`, and a scratch `GeodesicLine` state at
  `[272..336]` used by `_GenDirect`. Only `[246..336]` is written during a
  call, and that is exactly what the pool exists for. At 32 workers this is
  89 KB per `Geodesic`. A `GeodesicLine` object gets its own 65-double block
  so a `Position` call never rebuilds the line.
* *PolygonArea state*, 11 int64: `num` and `crossings`, then eight float64
  from slot 2 (`lat0`, `lon0`, `lat1`, `lon1`, and the two Shewchuk
  accumulators).

`GeodesicLine` state is 32 scalars followed by its five coefficient arrays, so
that a `Position` call is a 32-word load rather than a rebuild of the line.

**Batch entry points.** `mg_inverse`, `mg_direct`, `mg_arcdirect`,
`mg_line_position` and `mg_line_arcposition` take parallel float64 input
arrays and write one structure-of-arrays output block, field-major:
`out[field * n + i]`. The loop inside is the scalar solve, once per element;
what this pass added is that the loop is split into one chunk per worker
above 8192 points, and that the single-point path no longer pays NumPy to
move four doubles. `mg_polygon_addpoints` is deliberately still serial: it
is a running Shewchuk accumulation, and splitting it would change the sum.

## Layout

```
src/ported.mojo        the port, in upstream's module order
src/capi.mojo          @export wrappers and the static coefficient tables
build/build.sh         mojo build --emit shared-lib -> dist/
python/mojogeographiclib/   the Python API, module for module with upstream
tests/                 pytest; parity against the real upstream package
bench/bench.py         benchmarks, prints a markdown table
bench/profile.py       per-path cost breakdown (us or ms per call)
bench/sweep.py         the worker-count and batch-size curves the thresholds come from
examples/quickstart.py a runnable tour
```

## Licence

MIT/X11, as upstream GeographicLib. See `LICENSE`.
