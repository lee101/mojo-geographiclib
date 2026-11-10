# Verification report, 2026-11-10

Scope: the shipped tree at `a85cb22` (phase gpu), on the box in
`pixi.toml` (`Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz`, 72 threads,
Python 3.13.14, numpy 2.5.1), with a **shared** box: load average 18-20 of
72 threads for the whole session. Nothing here was run on an idle machine, so
absolute times are larger than the earlier passes' and repeatable *ratios* are
what the README relies on.

## What was run, in order

| command | result |
| --- | --- |
| `pixi run build && pixi run test` | built; **1458 passed**, exit 0 (~21 s) |
| `pixi run bench` | ran; first run recorded, then 5 more (6 total) |
| `pixi run bench/verify_inputs.py` (new) | OK: both sides same inputs, same masks, same outputs |
| `pixi run bench/parallel_builds.py 43354b2` then `parallel_ab.py` | rebuilt reference trees, 21 cells measured |
| `pixi run bench/parallel_sweep.py` | worker curve + size curves of the shipped build |
| `pixi run bench/profile.py` | glue breakdown of the 100k batch call |
| `pixi run bench/simd_piece_ab.py` (new) | piece-level A/B, pre-SIMD vs shipped |
| `pixi run bench/inverse_ab.py` (new) | interleaved `InversePairs` 100k A/B |
| `pixi run bench/kernel_ab.py` (new, x2) | serial-kernel unroll A/B, two runs |
| `pixi run build && pixi run test` | **1458 passed**, exit 0 (final gate) |

All timing commands were run under `flock /tmp/mojo-bench.lock`.

## Headline table: reproduced or not

Six `pixi run bench` runs of the shipped build. The README previously quoted
the unroll pass's "after" column as `pixi run bench` on this same box; the
numbers below are this session's medians of six, against that column.

| case | shipped, median (range) of 6 | README before | reproduced? |
| --- | ---: | ---: | --- |
| Inverse, 20k scalar | 245.69 ms (242.03-266.61) | 257.95 | yes (-5%) |
| Inverse, 100k, all | 76.62 ms (73.19-87.07) | 43.08 | **no (+78%)** |
| Inverse, 100k, STANDARD | 68.66 ms (64.93-69.31) | 40.60 | **no (+69%)** |
| Direct, 100k | 31.16 ms (28.62-32.28) | 15.61 | **no (+100%)** |
| ArcDirect, 100k | 28.41 ms (27.56-29.84) | 19.45 | **no (+46%)** |
| Position, 20k calls | 120.93 ms (111.53-125.23) | 115.84 | yes (+4%) |
| Positions, 100k | 17.35 ms (17.03-19.63) | 9.97 | **no (+74%)** |
| AddPoints, 20k | 46.55 ms (45.87-49.38) | 47.12 | yes (-1%) |
| AddPoint, 20k | 90.86 ms (89.46-95.44) | 97.77 | yes (-7%) |

Upstream was re-timed in the same six runs (e.g. 13948 ms for `Inverse, 100k,
all` against the README's 14187.75) and every upstream figure reproduced; the
README's table now carries this session's `after` and `upstream` medians.

The README's "before" column (the unroll pass's pre-unroll run) was **not**
reproduced: no pre-unroll `bench` run was made here, and re-basing the after
side made a same-build comparison impossible anyway. That column is kept and
labelled unverified in the README, not overwritten.

**Why the batch rows shrank.** `bench/profile.py` on the shipped build splits
a 100k `InversePairs` call into `_run` (kernel + output copy) 55.6 ms,
`_result` (NumPy post) 14.2 ms, total 77.1 ms; `_result`'s `atan2d` alone is
13.1 ms. The kernel itself is 176-179 ms serial and 56.16 ms at 4 workers in
this session (`profile.py` and the `InversePairs 56.16 ms vs upstream` probe). Nothing in the Python package changed since the "43.08 ms" run
(`git log` shows only README changes after `70ebb4c`), so the gap is machine
state -- the batch rows on this box are ~1.7x slower than the quiet-box run
the README recorded. The README now says so and quotes the measured numbers.

## Other tables

| README table | what was done | verdict |
| --- | --- | --- |
| Parallel A/B (11 rows) | rebuilt `43354b2` + serial tree, re-ran `parallel_ab.py` | reproduced: inverse 2.20x vs before at 100k (was 2.12x), direct 2.25x (2.28x), positions 1.94x (1.88x); Positions@8192 1.72x faster than before (was 1.89x). All 21 cells measured; table now carries this run. |
| Worker cap curve | `parallel_sweep.py` | the shipped build's curve measured for all four kernels and replaces the three old rows (inverse/direct/line for the pre-pass and new builds); the pre-pass rows were not re-run and were **removed**, not kept |
| Unroll kernel table | new `bench/kernel_ab.py`, `c201134^` vs `c201134`, one process per library, sha256 asserted | serial `mg_inverse`+`mg_direct` 241.0/242.2 ms before vs 227.6/235.2 ms after -> ~3.5-4% faster; direct alone unchanged. The old "32 workers 16.3 -> 15.8 ms" column was untestable (the cap is 4) and was removed |
| SIMD two-bench-run table | `after`/`upstream` re-measured (6 runs) | reproduced the *direction*, not the old absolute values; the `after` column was replaced |
| SIMD piece table | new `bench/simd_piece_ab.py`, `geomath.py` at `43354b2^` vs shipped | `atan2d_pair` 1.07x/1.12x (was "about 10%"), `ang_diff` 1.05x/1.02x, controls 1.00x. Both forms bit-identical on the inputs used. The old µs values (184.32 etc.) were "quiet box" numbers and did not reproduce in absolute terms; replaced with ms |
| SIMD `InversePairs` verdict | new `bench/inverse_ab.py`, interleaved, 25 runs/round, 2 rounds | old 33.78/35.31 ms, new 37.71/35.08 ms -- the new side does **not** win here. The pass's "31.6-34.2 vs 32.4-40.1" claim did not reproduce; README now says the whole-call win is inside the noise |
| GPU section | `nvidia-smi --query-gpu=memory.free` -> 4307 MiB; read `analysis.md` flops/byte rows | capacity reading stable (was 4309 MiB); flops/byte figures are derivations from `analysis.md`, not re-derived. `_run` at 100k now measures 55.6-58.5 ms against the 23.79 ms quoted from `analysis.md`; the README now marks that number as from `analysis.md` and not re-taken |
| `91-93x` in the reading bullets | not reproducible from any current number | replaced with the measured `Positions` 52.0x (in the reading bullets) |
| `3-7x faster per point` (SIMD section) | no measurement backs "<7x" | replaced with a pointer to the whole-call table; the number itself could be *exceeded* by an untimed kernel-vs-Python-loop comparison (239x), which is not what the claim meant |

## Same input, same work? (step 4)

`bench/verify_inputs.py` (new, committed) re-runs every bench case once on the
shipped build and checks, without timing:

* `ALL` (32671) and `STANDARD` (1929) masks are the same integer on both sides,
  and the `GeodesicLine` caps are the same (65439) -- so neither side is asked
  for fewer fields.
* Both sides in `bench/bench.py` are handed *the same numpy arrays* (`BIG`,
  `LOOP`, `SMALL` from one generator); there is no per-side input construction
  to drift.
* Batch vs upstream loop, on the bench's own 100k input: 12 fields each,
  identical keys, NaN pattern and shape; worst relative error 2.93e-13
  (`InversePairs`), 9.05e-14 (`DirectMany`), 1.68e-11 (`ArcDirectMany`, lon2),
  2.48e-16 (`Positions`). All port outputs finite and full length.
* `PolygonArea`: same 20,000 vertices, same `num` (20000), **bit-identical**
  perimeter, area relative error 2.4e-15.

No side skips work: the upstream column is a plain Python loop over the same
arrays calling the same public API with the same mask, and the port column is
the equivalent batch call; the scalar rows loop on both sides.

One thing worth stating: the line rows are gated on `line.caps` on both sides
(the kernel receives `outmask & caps` in both implementations), so the
`Positions` rows compute the same 12 fields, not the 9 that a bare mask
comparison would suggest.

## Tests (step 5)

`git log -p -- tests/` shows four commits touching `tests/`. The only removed
lines are:

1. `tests/test_geodesic.py`, parallel pass: a three-line comment and
   `PARALLEL_THRESHOLD = 8192` -> `12288`. That is a constant matching
   `src/capi.mojo`'s `MG_MIN_INVERSE`, not an assertion; the tests on either
   side of it are bit-for-bit equality against the serial path. No loosening.
2. `tests/test_parity.py`, test pass: a docstring sentence and the key-set
   assertion `sorted(set(ours) ^ set(theirs))`, which was replaced by a
   **stricter** assertion naming ours-only and upstream-only keys separately.

No tolerance was loosened (`ULP = 1e-9` unchanged; the per-ellipsoid `1e-7`
for the one-metre ellipsoid predates all passes). No parity test was deleted.
The later commits only add tests (tail lengths, threshold pinning, per-threshold
bit-identity) and extend an existing comment. Nothing to restore.

`tests/test_parity.py`'s docstring refers to `test_degenerate.py`, which does
not exist in the tree. That is a documentation inaccuracy in a test file, not a
weakened assertion; it is left alone because fixing it is not needed for
correctness and changing tests was outside what this pass was asked to do.

## Code changes

**None.** No failing test was found: the suite is green at every stage, the
parity checks above pass, and no bug was demonstrated. Per the rules, nothing
in `src/` or `python/` was touched.

## Added files

All are verification tools, not part of the build or test gate:

* `bench/verify_inputs.py` -- same-input / same-work / same-answer check.
* `bench/simd_piece_ab.py` -- rebuilds the SIMD piece table.
* `bench/inverse_ab.py` -- interleaved whole-call `InversePairs` A/B.
* `bench/kernel_ab.py` -- rebuilds the unroll kernel table.
* `bench/verify.md` -- this report.

## What could not be reproduced or measured

* The pre-unroll and pre-SIMD **`pixi run bench` rows** (the `before` columns):
  could not be re-run without checking out and rebuilding those trees for a
  full bench, which was not done; those columns are labelled as old and
  unverified in the README.
* The **GPU flops/byte** figures in `bench/analysis.md` (4.2, 3.2, 2.1): those
  are derived, not measured, and were read, not re-derived.
* The **worker-count curve of the pre-pass 32-worker build**: removed from the
  README rather than re-measured.
* Anything requiring an **idle box**: the load average never dropped below 17.
  Every absolute figure here is from a loaded machine and will read faster on
  an idle one; the ratios are the transferable part.

## Final gate

`pixi run build && pixi run test` -- green, 1458 passed (run after all of the
above; no source file changed during the pass).
