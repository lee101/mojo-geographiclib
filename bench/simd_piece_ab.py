"""Rebuild the SIMD-pass piece-level A/B, the README's second SIMD table.

    pixi run bash -c 'flock /tmp/mojo-bench.lock python bench/simd_piece_ab.py'

That table names four pieces of python/mojogeographiclib/geomath.py and gives a
100k and a 1M timing for the before and after form of each.  This script
measures them again, interleaved inside one process (old, new, new, old, ...
rounds), min per variant, and checks the two forms agree bit for bit.

"Before" is geomath.py at 43354b2^ -- the tree the SIMD pass started from --
loaded from a scratch copy under its own module name.  "After" is this tree's.
The atan2d row is two scalar calls on each side when that is all the old
module has.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojogeographiclib.geomath as new  # noqa: E402

BASE = "43354b2^"
OLD_PATH = os.path.join(tempfile.gettempdir(), "mojogeo_geomath_before_simd.py")
with open(OLD_PATH, "w") as fh:
    fh.write(subprocess.run(["git", "-C", ROOT, "show",
                             f"{BASE}:python/mojogeographiclib/geomath.py"],
                            capture_output=True, text=True, check=True).stdout)
_spec = importlib.util.spec_from_file_location("geomath_before_simd", OLD_PATH)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)

ROUNDS = 15


def ab(fo, fn):
    fo(); fn()
    to, tn = [], []
    for _ in range(ROUNDS):
        t0 = time.perf_counter(); fo(); to.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fn(); tn.append(time.perf_counter() - t0)
    for _ in range(ROUNDS):  # opposite order for the second half
        t0 = time.perf_counter(); fn(); tn.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fo(); to.append(time.perf_counter() - t0)
    return min(to), min(tn)


def main():
    print(f"before: geomath.py at {BASE}; after: this tree")
    print(f"interleaved, {ROUNDS} rounds each order, min per variant")
    print("| piece | n | before | after | old/new |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for n in (100_000, 1_000_000):
        rng = np.random.default_rng(0)
        y = np.ascontiguousarray(rng.uniform(-1, 1, n))
        x = np.ascontiguousarray(rng.uniform(-1, 1, n))
        lon = np.ascontiguousarray(rng.uniform(-180, 180, n))
        lon2 = np.ascontiguousarray(rng.uniform(-180, 180, n))
        us = 1e6

        # The old module has no pair form, so this is two scalar calls per
        # side; two calls is what a pair form replaces in the result dict.
        assert np.array_equal(new.atan2d(y, x), old.atan2d(y, x))
        assert np.array_equal(new.atan2d(lon, lon2), old.atan2d(lon, lon2))
        a, b = ab(lambda: (old.atan2d(y, x), old.atan2d(lon, lon2)),
                  lambda: new.atan2d_pair(y, x, lon, lon2))
        print(f"| atan2d_pair (azi1, azi2) | {n} | {a*us:.2f} us | {b*us:.2f} us | {a/b:.2f} |")

        o, w = old.ang_diff(lon, lon2), new.ang_diff(lon, lon2)
        assert np.array_equal(o[0].view(np.uint64), w[0].view(np.uint64))
        assert np.array_equal(o[1].view(np.uint64), w[1].view(np.uint64))
        a, b = ab(lambda: old.ang_diff(lon, lon2), lambda: new.ang_diff(lon, lon2))
        print(f"| ang_diff | {n} | {a*us:.2f} us | {b*us:.2f} us | {a/b:.2f} |")

        assert np.array_equal(new.ang_normalize(lon), old.ang_normalize(lon))
        a, b = ab(lambda: old.ang_normalize(lon), lambda: new.ang_normalize(lon))
        print(f"| ang_normalize | {n} | {a*us:.2f} us | {b*us:.2f} us | {a/b:.2f} |")

        # NaN inputs (|lon| > 90) come out NaN on both sides, and NaN != NaN,
        # so compare on bits with a NaN pattern that is expected.
        assert np.array_equal(np.isnan(new.lat_fix(lon)), np.isnan(old.lat_fix(lon)))
        assert np.array_equal(np.isnan(new.lat_fix(lon)), np.abs(lon) > 90.0)
        a, b = ab(lambda: old.lat_fix(lon), lambda: new.lat_fix(lon))
        print(f"| lat_fix | {n} | {a*us:.2f} us | {b*us:.2f} us | {a/b:.2f} |")
        sys.stdout.flush()
    print("\n(us = microseconds; 1M rows are printed in us too, as in the README)")


if __name__ == "__main__":
    main()
