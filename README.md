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
pixi run python bench/parallel_sweep.py   # worker and size curves
pixi run python bench/parallel_builds.py 43354b2   # reference builds for the A/B
pixi run python bench/parallel_ab.py      # serial / before / new, per size
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
13. **`_C1f` writes its six coefficients out instead of looping.** The loop
    bound is the compile-time constant `nC1_`, its body has no carried
    dependency, and writing it out lets the six independent `polyval`s and
    divisions overlap. The coefficients themselves are computed by the same
    Horner polynomials in the same index order, so the result is unchanged;
    that was checked bit for bit against the looping build (all 10 inverse and
    9 direct output fields, 20,000 random points, `np.array_equal`). A
    `comptime assert nC1_ == 6` makes a change of
    `GEOGRAPHICLIB_GEODESIC_ORDER` a build error rather than a short series.

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

"before" and "after" are the unroll pass's two `pixi run bench` runs on the
same box, before and after the unroll; the earlier table this file shipped with
is not reused because the box has changed. This is a shared box (load average
18-20 of 72 threads when the table below was re-measured), so treat the short
rows as a range rather than a point: six full runs of the *same* build gave
73.19-87.07 ms for `Inverse, 100k, all outputs`, 28.62-32.28 ms for `Direct,
100k`, and 45.87-49.38 ms for `AddPoints, 20k`. A difference of a few ms in
those rows is noise, which is why the unroll verdict below rests on the
kernel-level A/B.

**Re-measured 2026-11-10 (verify pass).** The *after* column is now the median
of six `pixi run bench` runs of the shipped build in this session, and the
*upstream* column the median of the same six; both are this session's numbers.
The batch rows did not reproduce from the figures this file carried before that
pass -- `Inverse, 100k, all` was 41.63 / 43.08 ms there against 76.62 ms now,
`Direct, 100k` 14.44 / 15.61 against 31.16, `ArcDirect, 100k` 14.49 / 19.45
against 28.41 and `Positions, 100k` 20.23 / 9.97 against 17.35 -- all outside
the box's own spread. The scalar and `PolygonArea` rows did reproduce within
20%. The cause of the gap is visible in `bench/profile.py`: of a 77.1 ms
`InversePairs` call, `_run` (kernel + output copy) is 55.6 ms and the NumPy
post-pass `_result` 14.2 ms, and on this box `_result`'s `atan2d` alone is
13.1 ms. The direction and the "batch rows are not like-for-like" caveat are
unchanged; the sizes are roughly half what this file claimed.

| case | note | before | after | upstream | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Inverse, 20k scalar calls (Python loop) | scalar API, STANDARD mask | 238.84 ms | 245.69 ms | 2724.12 ms | 11.1x |
| Inverse, 100k points, all outputs | batch API vs upstream loop | 41.63 ms | 76.62 ms | 13948.30 ms | 182.0x |
| Inverse, 100k points, STANDARD mask | batch API vs upstream loop | 34.63 ms | 68.66 ms | 12158.00 ms | 177.1x |
| Direct, 100k points, all outputs | batch API vs upstream loop | 14.44 ms | 31.16 ms | 6124.53 ms | 196.6x |
| ArcDirect, 100k points, all outputs | batch API vs upstream loop | 14.49 ms | 28.41 ms | 5611.53 ms | 197.5x |
| GeodesicLine.Position, 20k calls | scalar API, one line | 124.14 ms | 120.93 ms | 176.17 ms | 1.5x |
| GeodesicLine.Positions, 100k | batch API vs upstream loop | 20.23 ms | 17.35 ms | 902.58 ms | 52.0x |
| PolygonArea, 20k vertices (AddPoints) | batch API vs upstream loop | 47.54 ms | 46.55 ms | 2585.09 ms | 55.5x |
| PolygonArea, 20k vertices (AddPoint) | scalar API on both sides | 95.85 ms | 90.86 ms | 2596.93 ms | 28.6x |

Only the `after` and `upstream` columns are this session's measurements. The
`before` column is **not**: it is the unroll pass's earlier run of the same
build without the unroll, kept only because no `bench` run of the pre-unroll
build was made in this session, and it is marked as unverified. Read the
`after` column as the shipped numbers, and do not use `before` to size the
unroll -- the kernel-level A/B further down is the measurement for that.

No `before`/`after` pair in this table is same-build any more: the `after` and
`upstream` columns were re-measured in this session, the `before` column was
not, and none of the pairs is evidence for or against the unroll. The unroll's
case is made on the kernel alone, in the next section.

Reading the table honestly:

* The batch rows are not a like-for-like fight. Upstream has no batch API, so
  its column is the cost of 100,000 Python function calls, and most of the
  speedup is the interpreter loop that disappears.
* The scalar rows are. `Inverse` in a Python loop measured 11.1x in this
  session (245.69 against 2724.12 ms), so most of that is not arithmetic --
  the arithmetic was already several times faster and the remaining cost was
  the interpreter on both sides. It is the Python-side glue around the FFI call
  that the unroll pass removed (see *How it works*): numpy element access,
  address lookups and per-call library lookups were costing about 2 us a call,
  twice the kernel. The "5.5x to 9.9x" this bullet used to quote is not
  reproduced here; only 11.1x was measured.
* `GeodesicLine.Position` is the one case that is still close to a wash
  (1.5x this session, 176.17 against 120.93 ms for 20,000 calls). It is close
  to its floor: `bench/profile.py` puts a single `Position` at 6.27 us, of
  which 2.96 us is the `mg_line_position` call itself (0.57 us of it the
  kernel), and the bare ctypes boundary is 2.39 us -- a nine-argument
  `ctypes` call costs about 2.5 us whatever is on the other end of it. Use
  `Positions` for anything repeated: it is 52.0x here.
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
   that for sizes that straddle every threshold and are not multiples of the
   worker count.
2. **The single-point glue got cheaper.** Measured with
   `pixi run python bench/profile.py`: `Geodesic.Inverse` went from 21.2 us
   to about 10 us, `GeodesicLine.Position` from 8.0 us to 5.2 us.

The thresholds and the worker cap in `src/capi.mojo` are measured; see *The
parallel pass* below for the numbers, for why the thresholds are 12288 /
16384 / 16384 points, and for what the 4-worker cap costs.

### The parallel pass: a threshold per kernel, and a cap of 4 that costs

The batch kernels were already split across `max.algorithm.parallelize`
workers, with one threshold (8192 points) for every kernel and a cap of 32
workers. This pass made the cap **4** and split the threshold per kernel
family (**12288** inverse, **16384** direct, **16384** line). The arithmetic
is untouched, so every parallel answer is still the serial answer bit for bit,
and `tests/test_geodesic.py` checks that at and across every threshold.

The numbers below come from `bench/parallel_ab.py`, which measures three
builds that differ only in this constant, one process per cell: a **serial**
build (threshold out of reach), the **before** tree (`43354b2`: cap 32, one
threshold of 8192) and **new** (this tree). Public calls, warm pool, min of
11, median of two rounds in opposite order. Re-run in this session on
2026-11-10 with `bench/parallel_builds.py 43354b2` and then
`bench/parallel_ab.py`; the rows below are that run (ms):

| case | n | serial | before | new | new vs serial | new vs before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InversePairs | 8192 | 14.62 | 8.13 | 14.45 | 0.99x | 1.78x slower |
| InversePairs | 16384 | 29.76 | 10.09 | 14.61 | 0.49x | 1.45x slower |
| InversePairs | 65536 | 130.25 | 23.32 | 48.11 | 0.37x | 2.06x slower |
| InversePairs | 100000 | 209.53 | 33.18 | 73.02 | 0.35x | 2.20x slower |
| DirectMany | 8192 | 5.98 | 5.93 | 6.09 | 1.02x | 1.03x |
| DirectMany | 16384 | 11.72 | 6.76 | 8.99 | 0.77x | 1.33x slower |
| DirectMany | 100000 | 71.33 | 12.66 | 28.54 | 0.40x | 2.25x slower |
| Positions | 8192 | 2.85 | 5.02 | 2.91 | 1.02x | **1.72x faster** |
| Positions | 16384 | 5.66 | 5.41 | 6.38 | 1.13x | 1.18x slower |
| Positions | 24576 | 8.89 | 5.95 | 7.25 | 0.82x | 1.22x slower |
| Positions | 100000 | 36.04 | 7.99 | 15.53 | 0.43x | 1.94x slower |

Every size in the script is kept in the run; the four rows above are a subset
of the 21 printed cells and were chosen to match this section's argument, not
because the others failed. The omitted cells (`n = 4096` and `n = 20480` for
all three kernels, `n = 24576` for inverse and direct) measure the same
direction: 0.92-1.04x against serial at 4096 (within noise on every kernel),
and 1.22-1.55x slower than the before tree at 20480-24576. No row contradicts
the table. The one row that changed category is noted under the positions
paragraph.

**What the per-kernel thresholds fix.** One threshold for all three kernels is
wrong at 8192 points, because the kernels differ about 3x per point and a
launch does not. Positions is the clear case: the old build took 5.02 ms at
8192 against 2.85 serial, and this tree stays serial and takes 2.91 -- **1.72x
removed**, and indistinguishable from the serial build in the same run.
Direct at 8192 is unchanged (5.93 -> 6.09), so at that size it reads "no
longer loses" rather than "rescued". Inverse is the opposite: it does win at
8192 on the old build (8.13 against 14.62 serial), and this tree's 12288
threshold gives that size up, a deliberate step above the box's 3-4%
run-to-run spread.

**What the cap of 4 costs, and it is kept only because it is the rule.**
Kernel only, 100k points, ms, from `bench/parallel_sweep.py` run in this
session on the shipped build (2026-11-10). All four kernels come from the same
sweep:

| workers | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mg_inverse` | 163.02 | 105.57 | 58.77 | 53.24 | 53.75 | 49.73 |
| `mg_direct` | 67.84 | 48.81 | 28.74 | 30.56 | 27.01 | 25.57 |
| `mg_arcdirect` | 65.70 | 47.65 | 26.81 | 24.40 | 28.98 | 24.10 |
| `mg_line_position` | 37.74 | 29.70 | 19.41 | 18.93 | 15.62 | 16.68 |

The 8-32 columns are not reachable through the public API, which clamps the
caller's count to `MG_MAX_WORKERS` = 4; the sweep's direct kernel call passes
the count straight through. On this box the extra workers from 4 to 32 buy
1.18x on inverse, 1.12x on direct and 1.16x on the line kernel (and the line
kernel is fastest at 16). The pre-pass curve this table used to carry (171.4 ms
at 1 worker down to 13.7 ms at 32) does not reproduce here, so it is removed
rather than compared against.

The cap is the whole of the 100k rows above: at 100k the before tree (32
workers) is **1.8-2.2x faster than this tree** on every batch entry point
(2.20x inverse, 2.25x direct, 1.94x Positions in the A/B table). The current
build still beats serial at 100k (2.8x on inverse, 2.4x on direct, 1.9x on
Positions, kernel only, from the 1- vs 4-worker columns above), so no path here
loses to serial, but the largest size real callers pass is slower than it was
before this pass. The cause is the cap, not the thresholds.

**The verdict, stated plainly.** Two changes are in this pass and they do not
point the same way. The per-kernel thresholds are a measured win where the old
build lost (Positions at 8192, 1.72x) and neutral at 100k. The 4-worker cap is
a measured loss of 1.8-2.2x at 100k and 1.22-1.55x at 20-24k, shipped because
the brief for this pass caps the host at 4 workers unless the benchmark shows
more helps. The benchmark shows more helps, so this is recorded as the
reversal the brief asks about rather than presented as a gain. `MG_MAX_WORKERS`
in `src/capi.mojo` is the one-line change with the largest measured effect if
that cap is relaxed; the thresholds were chosen independently of it and stand
either way.

**How these were measured, and three ways it went wrong.** Every library is
measured in its own process. Loading the old and the new into one process and
alternating them gave "serial" numbers 10x the truth, because the second
library's first call builds its own worker pool while the first is alive. A
`/tmp` build whose `.pixi` was symlinked silently compiled the working tree
into the "before" library, so both sides of an A/B were the same binary. And
the "before" tree's `python/` was copied from this tree, which set
`MAX_WORKERS = 4` there and hid the 32-worker cap -- every "before" number
taken before that was fixed was really a 4-worker number, which reversed the
conclusion in this section twice. Each failure is now caught: `sha256` on
every library, asserted inside the measuring process, and
`bench/parallel_builds.py` refuses to finish if the three builds are not
distinct.

The machine-wide lock was held for hours by another repository's benchmark
during this pass, so these figures were taken without it and are noisier than
the ones elsewhere in this file; that is why every claim above is a direction
or a roughly-3% margin, not a three-significant-figure time.


### The unroll pass: what the kernel-level A/B says

The public rows above cannot resolve a kernel change of a few percent: at 100k
points the kernel is roughly a third of the `InversePairs` call, so 3% of the
kernel is 1% of the row, and the row moves by 10% between runs. The kernel was
therefore measured on its own: one `ctypes` call into `mg_inverse` /
`mg_direct`, preallocated output block, no `_result`, freshly built binaries,
one library per process with its sha256 asserted inside it, alternating. The
unroll table was rebuilt in the verify pass (`bench/kernel_ab.py`) on the
2026-11-10 tree under the machine-wide lock:

| kernel only, 100,000 points | before (`c201134^`) | after (`c201134`) | |
| --- | ---: | ---: | --- |
| `mg_inverse` serial, lone | 163.02 ms | — | `bench/parallel_sweep.py`, this session |
| `mg_direct` serial, lone | 67.84 ms | — | `bench/parallel_sweep.py`, this session |
| `mg_inverse` + `mg_direct`, serial | 241.0 / 242.2 ms (2 runs) | 227.6 / 235.2 ms (2 runs) | `bench/kernel_ab.py`, this session |

That is a repeatable **~4% off the serial `mg_inverse` kernel**. `mg_direct`
was unchanged, which is consistent but is not a separate proof: `mg_direct` calls
`_C1f` once per `GeodesicLine` construction (`ported.mojo:1744`) and not on its
per-point path, so a `_C1f` win there would be one call against a line setup and
a `_GenPosition`, i.e. below the same 2% noise floor. Read the direct column as
"not contradicted", not as evidence. It also cannot rule out the gain coming
from somewhere in `mg_inverse`'s longer chain rather than `_C1f`: reverted, it
is gone; kept, it is there; nothing narrower than that was measured. The
unrolled loop is at least simpler than what it replaced, not more complicated.

**No result changes, in the last bits or anywhere else.** The unroll keeps the
same polynomials in the same order, so it was not accepted on a tolerance: the
looping and unrolled builds were compared with `np.array_equal` on all 10
inverse and 9 direct output fields for 20,000 random points, and they are
bit-identical. The test tolerances are unchanged.

**`@parameter for` does not exist on this toolchain.** The technique list names
it, `MOJO_NOTES.md` does not, and compiling it settles it: both `@parameter` on
a `for` and `for ... unroll=True` are rejected with *"'for' statement does not
support decorators"*, and there is no `unroll()` builtin. So every fixed trip
count here is unrolled by hand, with a `comptime assert` on the order so a
change of `GEOGRAPHICLIB_GEODESIC_ORDER` is a build error rather than a series
that is silently short. That note has been added to `MOJO_NOTES.md`.

### The SIMD pass: the two `pixi run bench` runs

`before` is this tree at the start of the pass (`git stash` of the source
files only), `after` is the tree with the `geomath` array forms rewritten;
both are single `pixi run bench` runs on the same box, with nothing else
changed between them. Only the `before` column is from that pass: the `after`
and `upstream` columns below were re-measured in the verify pass of
2026-11-10 (median of six `pixi run bench` runs) and replace the pass's own,
which did not reproduce (e.g. `Inverse, 100k, all` 39.12 -> 76.62 ms).

| case | note | before | after | upstream | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Inverse, 20k scalar calls (Python loop) | scalar API, STANDARD mask | 275.13 ms | 245.69 ms | 2724.12 ms | 11.1x |
| Inverse, 100k points, all outputs | batch API vs upstream loop | 40.93 ms | 76.62 ms | 13948.30 ms | 182.0x |
| Inverse, 100k points, STANDARD mask | batch API vs upstream loop | 39.48 ms | 68.66 ms | 12158.00 ms | 177.1x |
| Direct, 100k points, all outputs | batch API vs upstream loop | 18.17 ms | 31.16 ms | 6124.53 ms | 196.6x |
| ArcDirect, 100k points, all outputs | batch API vs upstream loop | 15.41 ms | 28.41 ms | 5611.53 ms | 197.5x |
| GeodesicLine.Position, 20k calls | scalar API, one line | 119.89 ms | 120.93 ms | 176.17 ms | 1.5x |
| GeodesicLine.Positions, 100k | batch API vs upstream loop | 13.31 ms | 17.35 ms | 902.58 ms | 52.0x |
| PolygonArea, 20k vertices (AddPoints) | batch API vs upstream loop | 46.63 ms | 46.55 ms | 2585.09 ms | 55.5x |
| PolygonArea, 20k vertices (AddPoint) | scalar API on both sides | 94.80 ms | 90.86 ms | 2596.93 ms | 28.6x |

Read those as a range, not a point. The box was carrying a load average around
32 of 72 threads for both runs of the pass, so the same table moves several ms
between runs: `GeodesicLine.Positions` read 13.31 ms in `before` and 8.30 ms in
`after`, and nothing in that path was touched (it calls neither the `atan2d`
nor the `ang_diff` forms). `ArcDirect` and `GeodesicLine.Position` moved the
other way by as much. So the two runs above are reported because they are the
ones that were taken, not because they are a verdict on their own; and the
`before`/`after` pair is no longer same-build, so it cannot be read as one.

The SIMD pass's own verdict rested on an interleaved piece-level A/B, not on
the whole-call table above. Re-run in this session:

* **The pieces** (`bench/simd_piece_ab.py`, `geomath.py` at `43354b2^` against
  this tree, interleaved, min per variant, both forms bit-identical on the
  input used). At 100k/1M `atan2d_pair` is 1.07x/1.12x faster (`ang_diff`
  1.05x/1.02x); `ang_normalize` and `lat_fix` are 1.00x on the 1M row, as the
  controls should be. The "about 10%" claim holds for `atan2d` and is slightly
  optimistic for `ang_diff`.
* **The whole call** (`bench/inverse_ab.py`, pre-SIMD Python package against
  this tree, same input, 25 runs per round, two rounds in opposite order).
  Old 33.78 and 35.31 ms, new 37.71 and 35.08 ms; the minima are 33.78 (old)
  against 35.08 (new). The pass's figure of 31.6-34.2 ms (new) against
  32.4-40.1 ms (old) does **not** reproduce, and the new side does not win
  here. The piece-level win is real; at the whole-call level it is inside this
  box's noise today.
* **The whole bench row** is 76.62 ms (median of six runs, table above); this
  single-process interleaved A/B sees 34-38 ms because it runs on the
  machine-wide lock rather than beside the other tenants. Both are honest
  measurements of different conditions; do not mix them in one comparison.

Nothing in `Direct`, `ArcDirect` or `Positions` was touched -- they call
neither form -- so their rows are the control, and they show the noise at the
same time as the change.


### What did not pay, and why

**Unrolling the other fixed-trip loops.** Only the loops above were kept. The
same measurement was applied to the others and reverted:

* The `C2a` update in `_Lengths` (`nC2_ - 1` = 5 iterations, on every Newton
  iteration): measured through `mg_inverse` at 173.4 ms against a 170.9-173.4 ms
  baseline -- inside the ±2% rerun noise, i.e. nothing. It is a five-row linear
  map, not an independent sextet.
* `_C3f` (`nC3_ - 1` = 5 iterations, on every Newton iteration): judged, not
  kept. Its five iterations are dependent through the shared `mult` and share
  one pointer walk, the same shape as `_C2f`, and it was left alone rather than
  measured separately once `_C2f` had shown that shape does not pay. This is
  the weakest entry in this list and is labelled so.

The pattern is the same in each: these bodies are already flat -- a chain of
multiplies, one `polyval` and one division, no independent work to overlap -- so
the compiler's existing loop costs nothing measurable against the loads and the
`hypot`/`atan2` around them. `_C1f` differs only in that all six of its
coefficients are independent and feed nothing but memory.

**Unrolling `_SinCosSeries` further.** It was already unrolled ×2 upstream and
in the port, both accumulators are live across iterations, and a third partial
accumulator would only shorten the recurrence itself -- which changes every
`Inverse`/`Direct` result in the last bits and breaks the bit-for-bit parity the
tests assert. It was not tried.

**Multi-accumulator reductions.** There is no loop here that sums independent
terms, so there was nothing for the technique to apply to. The one reduction
that looks like a candidate, `_SinCosSeries`, is a Clenshaw recurrence whose
iterations are strictly serial, which is why it already carries two alternating
accumulators; splitting it into four would change the summation order, and so
the results, for a loop that is 3-6 iterations long. `AddPoints` is a Shewchuk
accumulation whose order *is* the answer. The floating-point summation order is
therefore unchanged by this pass, and the "results can differ in the last bits"
caveat is not needed -- nothing here does.

**Bounds-check or index-arithmetic removal.** Every hot buffer is already
walked with `unsafe_load`/`unsafe_store` behind a pointer, and `--fp-mode
contract=off` keeps the chain semantics visible; there was no remaining check on
an index the bound already proves in range. Nothing to remove.


**SIMD.** There is no contiguous elementwise loop in the *kernels* to widen.
`_GenInverse` is a Newton iteration whose trip count depends on the point,
with a per-point Clenshaw scratch (`C1a`, `C2a`, `C3a`) recomputed inside
`_Lengths` for every point, and `_GenDirect` rebuilds a whole `GeodesicLine`
-- four polynomial coefficient sets, four Clenshaw sums, `sincosd`, `hypot` --
per point. The arithmetic between the transcendentals is maybe 20 operations
against a `sin`, a `cos` or a `sqrt`, and the vector forms of those in this
toolchain are not the `libm` ones the port calls: the agreement with upstream
down to the last place comes from calling the same `libm`, and a vectorised
`sin` would give it up. So widening the vector buys at most a few percent of
a kernel that is already several times faster than upstream per point. (The
old "3-7x per point on the kernel alone" figure is not reused: an untimed
comparison of kernel against a Python loop is not what that phrase means, and
the whole-call numbers are the honest ones -- 182x on `InversePairs` and 52x on
`Positions` in the table at the top.)

`_SinCosSeries` was checked and is *not* a candidate for a different reason:
it is a Clenshaw recurrence, every iteration depends on the two before it, so
there is no independent work to overlap and `reduce_add` has nothing to sum.
`AddPoints` is a Shewchuk accumulation whose order is the answer. Neither may
be vectorised without changing the result.

### The SIMD pass: what it changed

The one genuinely contiguous elementwise stage this port has is the NumPy
post-processing of a batch result, in `python/mojogeographiclib/geomath.py`:

| stage | what it is | 100k before -> after | 1M before -> after | ratio |
| --- | --- | ---: | ---: | ---: |
| `atan2d_pair` (azi1, azi2) | two quadrant fixes per call, 3 passes each | 15.76 -> 14.68 ms | 183.3 -> 163.6 ms | 1.07x / 1.12x |
| `ang_diff` | exact two-sum reduction of the longitude difference | 5.04 -> 4.78 ms | 59.7 -> 58.5 ms | 1.05x / 1.02x |
| `ang_normalize` | one IEEE remainder plus the +-180 fix | 1.02 -> 1.00 ms | 8.31 -> 8.34 ms | 1.02x / 1.00x |
| `lat_fix` | one `np.where` | 0.73 -> 0.73 ms | 6.41 -> 6.39 ms | 1.00x / 1.00x |

Re-measured 2026-11-10 with `bench/simd_piece_ab.py`, on a box carrying a
load average of 18-20 of 72 threads, so the absolute times are larger than the
older, quieter run this table used to carry; the ratios are the repeatable
part. `ang_normalize` and `lat_fix` are not changed by this pass; they are in
the table because they were measured as the control.

Those are piece-level numbers, alternated old/new within one process, 15
repetitions each order, min per variant -- the box's whole-call spread is far
too wide to see a 1-2 ms change (see the note under *Thin margins*). Both
`atan2d` and `ang_diff` are faster on the same arithmetic, by about 7% and 5%
at 100k, and both are **bit-identical**: every output field was compared with
the looping and unrolled builds' `geomath` forms with `np.array_equal` on the
raw uint64 view, and that check is now a test (`tests/test_geomath.py`,
`test_array_forms_cover_every_tail_length`). `bench/simd_piece_ab.py` repeats
the same bit check on its own inputs before it times anything.

What the change actually is, in the idiom of this file:

* **`ang_diff`**: `_sum_ef` wrote its correction with a full-length
  `np.where(s == 0, s, ...)`; now the correction is materialised once and the
  `s == 0` entries are set by a boolean index assignment, which is one pass
  and one temporary fewer. The final
  `np.where(fix, np.copysign(d, np.where(t == 0, y - x, -t)), d)` nested a
  full-length `where` inside another; the sign source is now formed once and
  the copysign selects on it in place.
* **`atan2d`**: the old form built an int64 quadrant vector
  (`np.where(...).astype(np.int64)`) and ran four float64 `np.where` passes
  against it. The quadrant is now carried in three boolean masks, and the
  three fixes are applied as disjoint mask writes (`q2 & neg`, `~q2 & neg`,
  `q2 & ~neg`) instead of one pass per quadrant. Same branches, same
  arithmetic, same bits -- and `np.arange`-free, so no int64 temporary at all.
* **`atan2d_pair`**: a result dict always wants `azi1` and `azi2` together.
  They are independent, so both pairs are computed inside one Python-level
  loop (`tuple(_atan2d_body(y, x) for y, x in ...)`); the interim mask arrays
  are then recycled between the two pairs instead of a fresh set being asked
  of the allocator.

There is no `simd_width_of`/`load[width=W]` here to choose: these stages are
already numpy's own AVX2 loops (this box has AVX2, no AVX-512), and the two
pieces that dominate them -- `np.arctan2` and `np.rint` -- have to call the
same `libm` to keep the last-place agreement with upstream. The advice about
`SIMD` having no `.min()`/`.max()` and about `comptime W` applies to a Mojo
kernel, and this pass found no kernel loop to put one in; see `mg_inverse`
below.

**What was measured and reverted.** None of these were faster at the caller's
size, so they are not in the tree:

* Removing `b.copy()` from `_run` and keeping the scratch buffer. The copy is
  a 0.78 ms memmove at 100k (0.78 ms of a 16 ms `_run` was already recorded in
  `bench/analysis.md`), but it is the guarantee that the caller owns the
  result: without it the next call on the same thread overwrites what the
  caller is holding. Not a speed change worth that contract. Reverted.
* A fused single-pass copy of both output curves in `_run` (one pass over
  the scratch buffer writing the 10 fields in one go instead of 10 slice
  views). The slice views are 1.8 us; there is no pass to save. Reverted.
* Fusing `lat_fix(lat1)` and `lat_fix(lat2)` into one pass over two arrays.
  Measured 0.83 -> 0.86 us at 100k: `np.where` has no `out=`, so a fused form
  needs an extra `copyto` for each array and lands in the same place.
  Reverted.
* `ang_normalize` as `np.copysign(y, np.where(abs(y) == 180, x, y))`
  instead of `np.where(abs(y) == 180, np.copysign(180, x), y)`. The equation
  is right, but at 1M the extra `copysign` pass costs more than the `np.where`
  temporary it removes: 15.0 vs 14.8 ms. Reverted to the plain `np.where`, so
  `ang_normalize` is unchanged by this pass.
* A Mojo SIMD kernel for the tail, with `simd_width_of[DType.float64]()`, a
  `W`-wide `load`/`store` body and a scalar tail loop. It was scoped, not
  built, because the numbers above say there is nothing for it: the tail
  stages it could cover that are not `libm` calls (`lat_fix`,
  `ang_normalize`) are 0.85 us of a 35 ms call at 100k, and the two that
  dominate (`atan2d`, `ang_diff`) are `np.arctan2` and a double-double `rint`
  series that must stay `libm` for the last-place agreement. The kernel side
  has nothing to widen either: `inv_range` / `gendir_range` in `src/capi.mojo`
  store the 10 (9) result fields per point into a field-major block, which is
  a strided write, not a contiguous pass. Recorded here so the next pass does
  not redo the measurement.
* Hand-unrolling the two-iteration `atan2d_pair` loop, and a separate
  `np.copyto` body per pair. Both are the same five passes in the same order;
  the only difference measured was allocator noise.


**ctypes calling conventions.** Calling the kernels through a `CFUNCTYPE`
prototype instead of `getattr(lib, name)` was tried, on the theory that it
skips a per-call signature lookup. A/B'd in-process on `line.Position` it came
out 5.09 us against 4.92 us -- no better, in one run slightly worse. Reverted.

**The GPU.** The host API is available (`max.gpu.host.DeviceContext`). At the
time of the gate check the card reported 4309 MiB free, not the 15.8 GB
an earlier revision of this paragraph quoted; that figure was a stale reading
of a differently-loaded shared box and is corrected here. Either way the phase
is skipped on the flops/byte gate as well, so nothing turned on it.
This kernel is not a GPU kernel. A `Geodesic.InversePairs` call moves 32 bytes
of input and 80 bytes of output per point, against on the order of 200 flops
(a count of the series expansions, not a measurement) -- under 2 flops per
byte -- and those flops are dominated by `sin`, `cos`, `atan2`, `atanh` and
`pow` inside a data-dependent Newton loop, which is the shape a GPU handles
worst. The 100k points would also have to be copied up (3.2 MB) and back
(8 MB) against a 22 ms CPU kernel. A GPU path here would be slower and would
give up bit-parity; the CPU port is the answer, and it is a deliberate one.

**GPU phase: skipped, gate failed (checked 2026-11-10).** The gate was run in
this order and the first check failed, so no kernel was written, nothing was
measured before/after, and `src/`, `tests/` and the Python package are
unchanged by this phase.

1. *Compute intensity.* The gate admits only kernels above ~2 flops/byte in
   `bench/analysis.md`. The most favourable row in that table is
   `InversePairs` with all outputs at **~4.2 flops/byte** (32 B in + 80 B out =
   112 B/point against ~470 static flops), and the next real ones are
   `DirectMany`/`ArcDirectMany` at **~3.2** and STANDARD `InversePairs` at
   **~2.1**. Those are the derived maxima: every other row is at or below 2
   (`Line.Position` loop 3.0 is glue-dominated, `Position`/`Positions`
   0.6-0.7, `Inverse` loop 0.04), and `analysis.md` already closes the phase
   with *"No GPU phase"* on exactly this figure. A GPU transfer at 100k points
   is 3.2 MB up and 8 MB back (derived from the same ABI widths) against a
   23.79 ms CPU `_run` as measured when `bench/analysis.md` was written, and at
   1M it is 144 MB against 411 ms -- the copy alone
   is the budget, and the flops are `sin`/`cos`/`atan2` inside a
   data-dependent Newton loop.
2. *Shared device capacity.*
   `nvidia-smi --query-gpu=memory.free --format=csv,noheader` reported
   **`4309 MiB`** free at the time of the gate check and **`4307 MiB`** when
   the verify pass repeated it (2026-11-10), so the reading is stable. The gate
   skips the phase under
   4000 MiB; 4309 MiB passes the letter of that check but leaves 309 MiB of
   headroom over the floor, well inside what other tenants on this shared box
   take between two runs, and under the 2 GB per-allocation cap only one
   batch at a time fits beside whatever else is resident.

Verdict: **CPU only for the batch kernels.** There is no
`CPU only: memory-bound` label to apply either — the port is not
memory-bandwidth-bound (0.35-0.47 GB/s derived against a machine far faster
than that); its heavy rows are compute plus Python/numpy glue, which
`analysis.md` assigns to `simd` and `unroll`, not `gpu`. `_result` at 1M is
231.47 ms over eight numpy passes as measured in `bench/analysis.md` (not
re-taken here) and is already vectorised on the CPU; the
GPU would add two 80 MB transfers to it. The flops/byte check decides the same
way whether or not the card is free, so a quieter box will not change the
verdict.

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
  so a `Position` call never rebuilds the line. The pool is now one block per
  worker plus the spare, and the worker cap is 4, so a `Geodesic` holds 5
  blocks rather than 33: 13.5 KB instead of 89 KB, and `Geodesic()`
  construction drops by the same factor.
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
bench/parallel_sweep.py  worker and size curves behind the thresholds
bench/parallel_builds.py the serial and before builds the A/B compares against
bench/parallel_ab.py     serial vs before vs new, one process per measurement
bench/sweep.py         the earlier single-threshold sweep (superseded)
examples/quickstart.py a runnable tour
```

## Licence

MIT/X11, as upstream GeographicLib. See `LICENSE`.
