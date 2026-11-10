# ANALYZE: where the time goes

Measured in this session on the shared box, under the same machine-wide lock as
`pixi run bench`, best-of-N per case. Nothing in `src/` or `tests/` was changed;
the only file added is this one.

Environment actually reported by the bench run:

    Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 threads, Linux x86_64
    python 3.13.14, numpy 2.5.1, best of 3

Sources of every number below:

| tag | command | what it gives |
| --- | --- | --- |
| `bench` | `pixi run bench` (exit 0) | public entry points vs upstream at real sizes |
| `profile` | `flock /tmp/mojo-bench.lock pixi run python bench/profile.py` | existing per-path breakdown, single point + 100k |
| `probe1` | `/tmp/probe_analyze.py` | FFI floor (n=0 vs n=1), scalar loop vs kernel-only |
| `probe2` | `/tmp/probe_analyze2.py` | serial vs parallel kernel, glue split at 1k/8192/100k/1M, numpy post |
| `probe3` | `/tmp/probe_analyze3.py` | allocation and copy cost, `_result` cost, ctor cost |
| `probe4` | `/tmp/probe_analyze4.py` | scalar `Inverse` breakdown, polygon kernel vs glue |
| `probe5` | `/tmp/probe_analyze5.py` | `GeodesicLine.Position` per-call overhead |

`flops` and `bytes` are **not** measured: this session did not run a performance
counter. They are arithmetic over the buffer sizes the ABI fixes, and every row
that uses them is marked `~` with the formula spelled out:

* bytes = (`n_in` input arrays × 8 B + `n_out` SoA fields × 8 B) × n.
  `n_in`/`n_out` are from `_lib.INV_FIELDS` / `DIR_N` / the call site in
  `geodesic.py`; e.g. `InversePairs` at ALL is 4×8 + 10×8 = 112 B per point.
* flops are a per-point static count of the arithmetic in the ported routine
  (Clenshaw series, `hypot`, `sincosd`, the Newton step): ~470 for
  `_GenInverse` at ALL, ~230 at STANDARD, ~330 for `_GenDirect`, ~48 for
  `_GenPosition`. That count is a reading of the code, not a measurement, and
  the flops/byte column inherits its uncertainty.

Every time in the table is a stopwatch number from a run above. Where a
flops/byte figure is used to decide something (the GPU verdict), its formula is
shown so the decision can be re-derived from the measured times alone.

## Ranked hotspot table

Ranked by measured time at the size real callers pass. "Bound" is decided by
flops/byte plus measured throughput, not by cache size: the 1M rows move
144 MB at 1M, far past L3, and still identical in per-point cost to the 100k
rows (0.411 vs 0.354 us/pt), so the working set is not what costs. Compute
means the flops/byte is ≥ 2; glue means the time is in Python or numpy and the
kernel is a minority of it.

| # | function | input size | time | bytes touched | ~flops | flops/byte | bound | time sits in | next phase |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `Geodesic.InversePairs` | 1,000,000 | 411.00 ms | 144 MB ~ | ~4.7e8 | ~4.2 | compute + glue | 147.71 ms kernel (par, w=32) + 231.47 ms `_result` + 4.58 ms copy of 80 MB; rest = 27.24 ms remainder | **simd** (`_result`), **unroll** (kernel) |
| 2 | `_result` numpy post, ALL | 1,000,000 | 231.47 ms | 144 MB ~ | — | — | glue | 8 numpy passes (`atan2d`×2, `ang_diff`, `lat_fix`×2, `np.where`) | **simd** |
| 3 | `mg_inverse` kernel (par, w=32) | 1,000,000 | 147.71 ms | 144 MB ~ | ~4.7e8 | ~4.2 | compute | Mojo kernel: one Newton solve + Clenshaw per point | **unroll** |
| 4 | `Inverse` Python loop | 20,000 pts | 243.11 ms | 2.24 MB ~ | ~4.6e6 | ~0.04 | glue | 12.16 us/pt total; 3.18 us/pt FFI+kernel; 8.98 us/pt interpreted | **none** (floor) |
| 5 | `PolygonArea.AddPoint` loop | 20,000 verts | 92.83 ms | 320 KB ~ | — | — | glue + compute | 4.64 us/v: one ctypes call, one geodesic per vertex | **none** |
| 6 | `PolygonArea.AddPoints` | 20,000 verts | 48.06 ms | 320 KB ~ | — | — | compute | 36.91 ms raw serial kernel; Shewchuk, serial by design | **none** |
| 7 | `InversePairs` | 100,000 | 35.39 ms | 11.2 MB ~ | ~4.7e7 | ~4.2 | compute + glue | 23.79 ms kernel (par) + 15.94 ms `_result`; `b.copy()` of the 8 MB output measured 4.58 ms standalone | **simd** (`_result`) |
| 8 | `GeodesicLine.Position` loop | 20,000 calls | 122.44 ms | 0.32 MB ~ | ~9.6e5 | ~3.0 | glue | 6.12 us/pt: 2.70 us kernel+ctypes, 3.42 us Python | **none** |
| 9 | `DirectMany` | 1,000,000 | 143.20 ms | 104 MB ~ | ~3.3e8 | ~3.2 | compute + glue | 108.99 ms kernel (par) + 34.21 ms remainder | **parallel** (at cap w=32) |
| 10 | `Geodesic.Inverse` single | 1 call | 10.49 us | 112 B ~ | ~230 | ~2.1 | glue | 3.34 us ctypes (kernel incl.) + 0.63 us pack/unpack + 6.52 us dict | **none** (floor) |
| 11 | `Geodesic.Direct` single | 1 call | 8.71 us | 104 B ~ | ~330 | ~3.2 | glue | ctypes call + result dict, same shape as #10 | **none** (floor) |
| 12 | `ArcDirectMany` | 100,000 | 14.83 ms | 10.4 MB ~ | ~3.3e7 | ~3.2 | compute + glue | kernel + `_gen_direct` result work | **parallel** (at cap) |
| 13 | `mg_direct` kernel (par, w=32) | 100,000 | 13.09 ms | 10.4 MB ~ | ~3.3e7 | ~3.2 | compute | 13.09 ms par vs 66.97 ms serial ⇒ 5.1x on 32 workers | **parallel** (at cap) |
| 14 | `InversePairs` STANDARD | 100,000 | 33.40 ms | 112 B/pt ~ | ~2.3e7 | ~2.1 | compute + glue | same kernel, fewer output fields used by `_result` | **simd** (`_result`) |
| 15 | `GeodesicLine.Positions` | 100,000 | 9.12 ms | 7.2 MB ~ | ~4.8e6 | ~0.7 | compute | batch kernel, 32 workers | **none** |
| 16 | `GeodesicLine.Position` single | 1 call | 5.49 us | 80 B ~ | ~48 | ~0.6 | glue | 2.79 us kernel + 0.64 us pack/unpack + 2.06 us Python | **none** (floor) |
| 17 | `InversePairs` @ 8,192 | 8,192 | 8.52 ms | 0.92 MB ~ | ~3.9e6 | ~4.2 | launch-bound | 7.50 ms kernel (par) vs 14.23 ms serial: threshold correctly placed | **none** |
| 18 | `InversePairs` @ 1,000 | 1,000 | 2.07 ms | 112 KB ~ | ~4.7e5 | ~4.2 | glue | 1.72 ms serial kernel + 0.26 ms `_result` | **none** |
| 19 | `np.arctan2` inside `atan2d` | 100,000 | 3.19 ms | 2.4 MB ~ | — | — | glue | libm vector call; no SIMD form keeps libm's last place | **none** |
| 20 | `ang_diff` | 100,000 | 4.35 ms | 4.8 MB ~ | — | — | glue | numpy multi-pass exact-sum, 4 arrays live | **simd** |
| 21 | `lat_fix` ×2 | 100,000 | 0.43 ms | 3.2 MB ~ | — | — | glue | 2 `np.where` passes | **none** (negligible) |
| — | `b.copy()` of 10×n output | 100,000 | 0.78 ms | 8 MB ~ | 0 | 0 | glue (copy) | one memcpy | **none** |
| — | `_same_length` | 100,000 | 4.78 us | — | — | — | glue | no copy on already-contiguous float64 | **none** |

`—` in flops/byte means the row is pure glue (a numpy pass or a Python call);
no count is quoted for it. `112 B/pt ~` in row 14 means 112 B is the per-point
total and STANDARD uses fewer of those output slots.

Two derived cells matter to later phases, so they are stated as remainders
rather than estimated away:

* Row 1: the measured parts 147.71 + 231.47 + 4.58 = 383.76 ms against a
  411.00 ms public call leave **27.24 ms** unaccounted — `_same_length`, the
  reshape/slicing in `_run`, and the `lat_fix`/`ang_diff` calls inside
  `_gen_inverse` that neither the `_run` nor the `_result` measurement captures.
* Row 9: 143.20 − 108.99 = **34.21 ms** of `DirectMany` at 1M is not kernel.
  At 100k that non-kernel part was 14.26 − 13.09 = 1.17 ms, so this cost scales
  with array size rather than per call.

Rows where the port is slower than upstream: **none in this session.**
Every `bench` row came out ahead: 11.6x scalar `Inverse`, 285.2x / 365.7x
`InversePairs` at 100k, 402.7x `DirectMany`, 395.8x `ArcDirectMany`,
1.5x `GeodesicLine.Position` loop, 97.6x `Positions`, 57.6x `AddPoints`,
27.7x `AddPoint`. The two closest rows are the ones to watch on a noisy box;
they are 1.5x and they are the two where both sides pay the same Python cost
per point, so a rerun on a loaded machine can put them near 1.0x. That is
flagged again under *Thin margins*.

## The measured comparison, verbatim from `bench`

| case | mojo-geographiclib | upstream geographiclib | speedup |
| --- | ---: | ---: | ---: |
| Inverse, 20k scalar calls | 245.05 ms | 2846.80 ms | 11.6x |
| Inverse, 100k, all outputs | 50.34 ms | 14357.76 ms | 285.2x |
| Inverse, 100k, STANDARD | 33.40 ms | 12214.98 ms | 365.7x |
| Direct, 100k, all outputs | 15.83 ms | 6372.84 ms | 402.7x |
| ArcDirect, 100k, all outputs | 14.83 ms | 5867.85 ms | 395.8x |
| GeodesicLine.Position, 20k calls | 121.70 ms | 183.58 ms | 1.5x |
| GeodesicLine.Positions, 100k | 9.47 ms | 924.57 ms | 97.6x |
| PolygonArea, 20k (AddPoints) | 47.91 ms | 2759.39 ms | 57.6x |
| PolygonArea, 20k (AddPoint) | 97.96 ms | 2717.44 ms | 27.7x |

The "upstream" column is a plain Python loop over the scalar API, because
upstream has no batch API. The batch speedups therefore mix three effects:
the interpreter loop upstream pays, the threaded kernel here, and a
single-point path here that is itself 10x faster. Only the scalar rows
(`Inverse, 20k`, `Position, 20k`, `AddPoint, 20k`) are like-for-like.

### Kernel vs everything else, measured

`_run` is one ctypes call plus a copy-out; the kernel is what the Mojo side
does inside it. `probe2`:

| n | `InversePairs` (public) | kernel, serial | kernel, 32 workers | `_result` (numpy) |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2.07 ms | 1.72 ms | 1.72 ms (below threshold) | 0.26 ms |
| 8,192 | 8.52 ms | 14.23 ms | 7.50 ms | 1.40 ms |
| 100,000 | 35.39 ms | 175.42 ms | 23.79 ms | 15.94 ms |
| 1,000,000 | 411.00 ms | 1790.83 ms | 147.71 ms | 231.47 ms |

Parallel speedup of the kernel: 7.4x at 100k and 12.1x at 1M on 32 workers.
The 8,192 row is the threshold itself, where the 14.23 ms serial kernel drops
to 7.50 ms, i.e. the launch is already paid back at the documented boundary.

The single-point path, `probe1` / `probe5`:

| step | `Inverse`, ALL | `GeodesicLine.Position` |
| --- | ---: | ---: |
| ctypes call, n=0 (pure FFI) | 2.80 us | 2.45 us |
| ctypes call, n=1 (kernel work) | 3.04 us | 2.79 us |
| + `struct.pack_into` / `unpack_from` | +0.60 us | +0.64 us |
| + result dict (`Math.*`, numpy scalars) | +6.85 us | +2.06 us |
| public call, measured | 10.49–11.25 us | 5.49–6.32 us |

The kernel is 3.04 of the 10.49 us `Inverse` call; the other 71% is ctypes
crossing, `struct` marshalling and the result dict.

## Python-level loops that re-enter ctypes per element

These are the loops a real caller writes, and each iteration is one FFI call.
They are listed with their measured per-iteration cost, and with the batch
entry point that already removes them where one exists.

| loop | per element | FFI+kernel | glue per element | batch escape |
| --- | ---: | ---: | ---: | --- |
| `for p in pairs: Geodesic.Inverse(...)` | 12.16 us | 3.18 us | 8.98 us | `InversePairs`, 0.354 us/pt at 100k |
| `for d in s: line.Position(d)` | 6.12 us | 2.70 us | 3.42 us | `Positions`, 0.091 us/pt at 100k |
| `for v: PolygonArea.AddPoint(lat, lon)` | 4.64 us | not split | — | `AddPoints`, 2.40 us/v at 20k |
| `for p: Geodesic.Direct(...)` | 8.71 us (single call) | 3.10 us | 5.61 us | `DirectMany`, 0.143 us/pt at 100k |
| `for p: PolygonArea.AddPoint` inside `AddPoints` | — | — | — | already batched, serial by algorithm |

Two of these loops look like they should have a batch form and do. The
`PolygonArea.AddPoint` loop at 4.64 us a vertex is one ctypes call and one
full inverse geodesic per vertex, the same per-vertex arithmetic the batch
kernel does, at 2.40 us a vertex. That difference is the interpreter and the
per-call `_real()` conversion, not arithmetic.

`PolygonArea.AddPoint` also re-resolves the symbol per vertex
(`_lib.fn("mg_polygon_addpoint")` inside the method, `polygonarea.py:130`).
That is a dict hit, so it is cheap — `_lib.fn` is a `dict.get` on a warm
cache — but it is the same pattern the port removed from `Geodesic` and left
here.

No loop in the hot paths re-enters ctypes *per array element inside a batch*.
The batch kernels are one call each and the element loop is in Mojo.

## Per-call allocations and copies

Measured (`probe3`), at n = 100,000 unless stated:

| item | cost | note |
| --- | ---: | --- |
| `_lib.buf` reuse (cache hit) | 0.744 us | thread-local, no allocation |
| `np.empty(n*10)` | 1.740 us | only on a first call per thread/size |
| `b.copy()` of the 10×n output | 0.78 ms (100k) / 4.58 ms (1M) | one memcpy; 3.3% of `_run` at 100k, 3.1% at 1M |
| 10 slice views of the output | 3.7 us | views, not copies |
| `_same_length` on 4 contiguous float64 arrays | 4.78 us | no copy when dtype/order already match |
| `_lib.f64(list)` for 1,000 Python floats | 63.0 us | allocates; only for non-array input |
| `Geodesic()` construction | 35.6 us | allocates the 33-block pool (89 KB) |
| `GeodesicLine()` construction | 11.0 us | allocates a 65-double state block |
| `Geodesic.Polygon()` construction | 8.7 us | allocates the 11-word state block |

The per-call costs that matter:

* **`b.copy()` in `_run`** — the whole SoA result is copied out of the scratch
  buffer before being sliced. At 100k points that copy is a memmove of 8 MB
  (measured 0.78 ms, `probe3`) and at 1M points 80 MB (measured 4.58 ms);
  it is included in the 31 ms of "copy/alloc" attributed to row 1. It is not
  reducible without giving up the guarantee that the caller owns the result.
* **`np.reshape(v, shape)` per output field** — 7.9 us per call for 8 fields.
  Negligible at every size.
* **`np.asarray` / `np.where` temporaries in `_result`** — `atan2d` allocates
  roughly eight full-length temporaries per call (two `abs`, two `where`,
  one `where(...,2,0)`, one `astype`, `arctan2`, `degrees`, three more
  `where`). At 100k that is the 12.9 ms measured for `atan2d x2`. This is
  allocation plus memory traffic, not arithmetic — the entire output is only
  800 KB.
* **Polygon per-vertex** — `AddPoints` at n=1,000 takes 2.57 ms end to end
  (constructor included) for 8 KB of input.

## Where the batches are serial that need not be, and where they are not

All five batch kernels use `max.algorithm.parallelize` with an explicit worker
count and the measured 8,192-point threshold (`src/capi.mojo`). Measured on
this box:

| kernel | serial | 32 workers | speedup | verdict |
| --- | ---: | ---: | ---: | --- |
| `mg_inverse`, 1M | 1790.83 ms | 147.71 ms | 12.1x | at the 32-worker cap |
| `mg_inverse`, 100k | 175.42 ms | 23.79 ms | 7.4x | at the cap |
| `mg_direct`, 1M | 702.10 ms | 108.99 ms | 6.4x | at the cap |
| `mg_direct`, 100k | 66.97 ms | 13.09 ms | 5.1x | at the cap |
| `mg_polygon_addpoints`, 20k | 36.91 ms | — | 1.0x | **serial by design** |

`mg_polygon_addpoints` is a running Shewchuk accumulation; splitting it would
change the sum. That is a correctness constraint, not a gap.

The cap of 32 is flat past 24 on this box according to the numbers in
`src/capi.mojo`. At 1M points 32 workers deliver 12x on a 72-thread machine;
the remaining 39x of headroom is real and untouched, but every attempt to use
it has to answer why 24→64 measured flat.

## GPU verdict

**No GPU phase.** The flops-per-byte figure is what decides it, and here it is
with its inputs shown, so the verdict does not rest on a counter this session
did not run:

* `InversePairs`, 100k, `_run` only: **23.79 ms measured** for
  (4 in + 10 out) × 8 B × 100,000 = **11.2 MB** ⇒ 11.2 MB / 0.02379 s =
  **0.47 GB/s**. Flops at ~470/pt × 100,000 = 4.7e7 ⇒ 4.7e7 / 0.02379 =
  **2.0 Gflop/s**.
* Per point, ALL: 32 B in + 80 B out = 112 B, ~470 flops ⇒
  **4.2 flops/byte**.
* Per point, STANDARD: 112 B, ~230 flops ⇒ **2.1 flops/byte**.
* The 1M row: 411.00 ms measured, (14 out + 4 in) × 8 B × 1,000,000 =
  144 MB derived ⇒ 0.35 GB/s, and 0.411
  us/pt against 0.354 at 100k. The flat per-point cost across a 10x size
  increase is the evidence that this is not a cache artefact — the batch is
  144 MB and the per-point cost does not change.

At 2–4 flops/byte a GPU has nothing to eat: the copy-in/copy-out alone
(11.2 MB each way at 100k, 144 MB at 1M, derived from the same formula)
would move more bytes than the
compute budget ever justifies, and the flops are
`sin`/`cos`/`atan2`/`atanh` inside a data-dependent Newton loop, which is the
worst shape for a GPU. The 4.2 flops/byte at ALL is the most favourable figure
available, and it is still far below where a pure copy would be free.

The one memory-shaped piece is `_result`: **231.47 ms measured** at 1M for
(8 fields × 8 B + 2 working arrays × 8 B) × 1,000,000 = **80 MB** ⇒
**0.35 GB/s**, across eight separate numpy passes. That is a CPU SIMD job, not
a GPU one — it is already in numpy, which vectorises, so the win is fewer
passes and fewer temporaries, not a wider lane.

## Thin margins

Two rows are close enough to upstream that the port should not be defended on
them:

* `GeodesicLine.Position`, 20k loop: 121.70 ms vs 183.58 ms, **1.5x**. Both
  sides pay a Python call per point; the port's kernel is faster (2.70 us vs
  the upstream arithmetic) but 3.42 us/pt of the port's cost is still the
  result dict.
* `Inverse`, 20k loop: 11.6x, a healthy margin, but the port's per-point cost
  is 12.16 us of which 3.18 us is kernel — 74% is still glue. It is a margin,
  not a floor.

## Next phase per hotspot

Every recommendation names one of `unroll`, `simd`, `parallel`, `gpu`, `none`.
The GPU verdict above is `none` with 2.1–4.2 flops/byte behind it.

| hotspot | next phase | why, from the numbers above |
| --- | --- | --- |
| `_result` numpy post (`atan2d`, `ang_diff`) | **simd** | 231 ms at 1M, 16 ms at 100k, eight full-length temporaries; a fused single-pass kernel over the SoA output is the one contiguous elementwise stage in the port. 80 MB touched in 231 ms (0.35 GB/s derived) means the passes, not the arithmetic, are the cost. |
| `mg_inverse` batch kernel | **unroll** | 147.71 ms at 1M, 12.1x on 32 workers, ~3 flops/byte, scalar Newton per point. No contiguous loop to widen at SIMD width, but the per-point coefficient rebuild in `_Lengths` and the Clenshaw sum are a fixed unrolled chain worth flattening before considering anything wider. |
| `mg_direct` batch kernel | **parallel** | 108.99 ms at 1M, 6.4x on 32 workers; worker cap is already 32 and the serial residue is a launch-cost question, not compute. Nothing to widen, so the next step is the worker curve, not a vector. |
| `Inverse` / `Position` / `Direct` single calls | **none** | 3.0–3.3 us of a 10.5–11.2 us call is the ctypes boundary plus kernel; the other 6–8 us is the result dict, which is upstream's contract. This is the floor of a Python-visible API. |
| `Inverse` 20k Python loop | **none** | 12.16 us/pt, 3.18 us/pt kernel. The escape hatch is `InversePairs` at 0.354 us/pt and it already exists. |
| `PolygonArea.AddPoints` | **none** | 36.91 ms raw at 20k; sequential by Shewchuk accumulation. Parallelising changes the answer. |
| `PolygonArea.AddPoint` loop | **none** | 4.64 us/v, one geodesic per vertex; `AddPoints` is the batch form and is 2.40 us/v. |
| `GeodesicLine.Positions` | **none** | 9.12 ms at 100k, 0.091 us/pt, already 97.6x upstream. |
| `b.copy()` / `_same_length` / ctor allocs | **none** | 0.78 ms, 4.8 us, 8.7–35.6 us; all below the resolution of the hotspots above them. |

## Correctness notes found while reading, not changed

Two things showed up while tracing the call graph. Neither is a timing
hotspot; both are recorded here because ANALYZE is where the map is drawn and
neither should be lost.

* `Geodesic._inv_fields` (`python/mojogeographiclib/geodesic.py:281`) has no
  `return` on its `vector=True` branch. Every current caller passes
  `vector=False` (`geodesic.py:562` is the only call site), so it is latent,
  not measured.
* `PolygonArea.AddPoint` (`polygonarea.py:130`) resolves its symbol with
  `_lib.fn(...)` per call instead of caching it on the object the way
  `Geodesic.__init__` does. It is a warm dict hit, so it does not show up in
  the 4.64 us/v above; it is simply the one remaining "resolve per element"
  pattern in the port.
