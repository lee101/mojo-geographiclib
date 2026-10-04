"""Loads (and if necessary builds) the compiled Mojo library.

The library is opened with ``ctypes.PyDLL`` rather than ``ctypes.CDLL``: the
Mojo kernels keep a small amount of per-ellipsoid scratch (the series
coefficients and the Clenshaw work areas) in a buffer owned by the calling
Python object, so two threads must not be inside the same ``Geodesic`` at once.
PyDLL does not release the GIL, which is exactly that guarantee, and it costs
nothing measurable against a multi-microsecond kernel.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading

import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOGEOGRAPHICLIB_LIB") or os.path.join(
    ROOT, "dist", "libmojo-geographiclib.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# Offsets into the caller-owned blocks, mirrored from src/ported.mojo.
GEOD_BLOCK = 337
LINE_BLOCK = 65
PA_BLOCK = 11  # int64 slots

# Batch workers.  Mirrors MG_MAX_WORKERS in src/capi.mojo: a Geodesic owns a
# pool of that many identical blocks so that concurrent workers never share
# the Clenshaw scratch that lives in one, and the kernel clamps itself to
# what the batch size can keep busy.  One is the serial path.
MAX_WORKERS = 32

# name -> (argtypes, restype)
_SIGNATURES = {
    "mg_geodesic_init": ([F, F, I], I),
    "mg_geodesic_c2": ([F, F, I], F),
    "mg_inverse": ([F, F, I, I, I, I, I, I, I, I, I], None),
    "mg_direct": ([F, F, I, I, I, I, I, I, I, I, I], None),
    "mg_arcdirect": ([F, F, I, I, I, I, I, I, I, I, I], None),
    "mg_line_new": ([F, F, I, F, F, F, F, F, I, I], None),
    "mg_line_position": ([F, F, I, I, I, I, I, I, I], None),
    "mg_line_arcposition": ([F, F, I, I, I, I, I, I, I], None),
    "mg_line_set_distance": ([F, F, I, I, F], F),
    "mg_line_set_arc": ([F, F, I, I, F], F),
    "mg_polygon_clear": ([F, F, I, I], None),
    "mg_polygon_addpoint": ([F, F, I, I, F, F], I),
    "mg_polygon_addpoints": ([F, F, I, I, I, I, I], I),
    "mg_polygon_addedge": ([F, F, I, I, F, F], I),
    "mg_polygon_compute": ([F, F, I, I, I, I, I, I], I),
    "mg_polygon_testpoint": ([F, F, I, I, F, F, I, I, I, I], I),
    "mg_polygon_testedge": ([F, F, I, I, F, F, I, I, I, I], I),
}

# Field order of the structure-of-arrays output blocks (see src/capi.mojo).
INV_FIELDS = ("a12", "s12", "salp1", "calp1", "salp2", "calp2", "m12", "M12", "M21", "S12")
DIR_FIELDS = ("a12", "lat2", "lon2", "azi2", "s12", "m12", "M12", "M21", "S12")
DIR_N = len(DIR_FIELDS)
INV_N = len(INV_FIELDS)


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOGEOGRAPHICLIB_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOGEOGRAPHICLIB_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-geographiclib.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    # Must match build/build.sh: without contract=off the compiler fuses
    # a + b*c into an FMA, which is correct but no longer agrees with libm.
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "--fp-mode", "contract=off", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None
_fns = {}


def lib() -> ctypes.PyDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.PyDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def fn(name: str):
    """The exported kernel `name`, looked up once.

    `getattr` on a shared library plus ctypes' per-call signature handling
    is not free -- measured at 0.12 us a call here, which is real money on a
    single-point path where the whole call is 5 us.  (Going through a
    `CFUNCTYPE` instead was tried and measured no faster, so it is not used.)
    """
    f = _fns.get(name)
    if f is None:
        f = getattr(lib(), name)
        _fns[name] = f
    return f


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 array of `a`, copying only when it has to."""
    return (
        np.array(a, dtype=np.float64, order="C", copy=True)
        if copy
        else np.ascontiguousarray(a, dtype=np.float64)
    )


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


class _Local(threading.local):
    def __init__(self):
        self.cache = {}


_tls = _Local()


def buf(key: str, n: int) -> tuple[np.ndarray, int]:
    """A reusable thread-local float64 scratch array of at least n elements,
    and its address.

    Reusing the buffers keeps a single-point call from being dominated by
    numpy allocation -- and caching the address keeps it from paying for
    `ndarray.ctypes.data`, which builds a fresh ctypes object on every access.
    """
    cache = _tls.cache
    entry = cache.get(key)
    if entry is None or entry[0].size < n:
        b = np.empty(max(n, 256), dtype=np.float64)
        entry = (b, b.ctypes.data)
        cache[key] = entry
    return (entry[0][:n], entry[1])

pack = struct.pack_into
unpack = struct.unpack_from


class Io:
    """One thread-local byte buffer for a whole single-point kernel call, with
    its address.

    A single-point call writes four doubles of input and reads back up to ten
    of output.  Doing that through numpy costs a `ndarray.__setitem__` or a
    `ndarray.__getitem__` per value -- each of which builds a fresh scalar
    object -- which measured several times the cost of the arithmetic itself.
    A `bytearray` written with one `struct.pack_into` and read with one
    `struct.unpack_from` costs one call each, and it hands the kernel a single
    base address from which the four input slices are offsets.
    """

    __slots__ = ("mem", "addr")

    def __init__(self, n: int):
        self.mem = bytearray(8 * n)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mem))


def io(key: str, n: int) -> Io:
    """A reusable thread-local `Io` holding at least n float64 slots."""
    cache = _tls.cache
    entry = cache.get(key)
    if entry is None or len(entry.mem) < 8 * n:
        entry = Io(n if n > 256 else 256)
        cache[key] = entry
    return entry


def main() -> int:
    """`python -m mojogeographiclib._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
