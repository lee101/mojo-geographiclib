"""Build the two reference trees `bench/parallel_ab.py` measures against.

    pixi run python bench/parallel_builds.py BASE_COMMIT

BASE_COMMIT is the tree as it was before the parallel pass (here `43354b2`).

* `/tmp/b_before` -- that tree as-is: worker cap 32, one threshold of 8192.
* `/tmp/b_serial` -- the same, with the one threshold pushed out of reach, so
  every batch kernel runs on one worker.  This is the reference the parallel
  numbers are judged against.

Three details, each of which cost a wrong measurement before it was handled:

* **Own sources.**  They are copied from the commit into `src/`, not
  symlinked to the working tree.  `_lib.build()` recompiles when a
  `src/*.mojo` is newer than the `.so`, and a symlink made the "before"
  build silently compile the working tree instead.
* **Own Python package.**  `python/mojogeographiclib/_lib.py` is restored
  from the commit, because `MAX_WORKERS` lives there: taking this tree's copy
  set the old build to 4 workers and hid its 32-worker cap entirely.
* **Backdated, so nothing rebuilds.**  The sources are stamped 1980 and the
  `.so` 1990.

At the end it prints the sha256 of all three libraries and fails if they are
not distinct.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOJO = os.path.join(ROOT, ".pixi/envs/default/bin/mojo")
BEFORE = "/tmp/b_before"
SERIAL = "/tmp/b_serial"


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_show(commit, path):
    return subprocess.run(["git", "-C", ROOT, "show", f"{commit}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def build(src_dir, out_so):
    cmd = [MOJO, "build", "--emit", "shared-lib", "--fp-mode", "contract=off",
           "-I", src_dir, os.path.join(src_dir, "capi.mojo"), "-o", out_so]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_so):
        sys.exit((proc.stderr or proc.stdout)[-3000:])


def make(base, files):
    """A tree holding `files` under src/, built into dist/, plus the package.

    The package comes from this tree and is overwritten with the base
    commit's `_lib.py` by the caller.
    """
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(os.path.join(base, "src"))
    os.makedirs(os.path.join(base, "dist"))
    os.makedirs(os.path.join(base, "python"))
    for name, text in files.items():
        with open(os.path.join(base, "src", name), "w") as fh:
            fh.write(text)
    lib = os.path.join(base, "dist", "libmojo-geographiclib.so")
    build(os.path.join(base, "src"), lib)
    shutil.copytree(os.path.join(ROOT, "python", "mojogeographiclib"),
                    os.path.join(base, "python", "mojogeographiclib"))
    return lib


def main(commit):
    capi = git_show(commit, "src/capi.mojo")
    ported = git_show(commit, "src/ported.mojo")
    lib_py = git_show(commit, "python/mojogeographiclib/_lib.py")
    serial_capi = capi.replace(
        "comptime MG_MIN_PARALLEL: Int = 8192",
        "comptime MG_MIN_PARALLEL: Int = 4611686018427387904")
    if serial_capi == capi:
        sys.exit("MG_MIN_PARALLEL not found in the base commit")

    before_lib = make(BEFORE, {"capi.mojo": capi, "ported.mojo": ported})
    serial_lib = make(SERIAL, {"capi.mojo": serial_capi,
                               "ported.mojo": ported})

    for base in (BEFORE, SERIAL):
        with open(os.path.join(base, "python", "mojogeographiclib",
                               "_lib.py"), "w") as fh:
            fh.write(lib_py)
        for name in ("capi.mojo", "ported.mojo"):
            os.utime(os.path.join(base, "src", name), (315532800, 315532800))
        os.utime(os.path.join(base, "dist",
                              "libmojo-geographiclib.so"), (631152000, 631152000))

    shipped = os.path.join(ROOT, "dist", "libmojo-geographiclib.so")
    hashes = {"shipped": sha(shipped), "before": sha(before_lib),
              "serial": sha(serial_lib)}
    for k, v in hashes.items():
        print(f"{k:8s} {v}")
    if len(set(hashes.values())) != 3:
        sys.exit("reference builds are not distinct; the A/B would be void")


if __name__ == "__main__":
    main(sys.argv[1])
