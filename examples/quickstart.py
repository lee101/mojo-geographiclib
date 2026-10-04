"""A runnable tour of the port.  `pixi run python examples/quickstart.py`."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojogeographiclib.geodesic import Geodesic


def main() -> None:
    geod = Geodesic.WGS84

    print("inverse problem")
    for k, v in geod.Inverse(-41.32, 174.81, 40.96, -5.50).items():
        print(f"  {k:5s} {v}")

    print("\ndirect problem, and back again")
    r = geod.Inverse(-41.32, 174.81, 40.96, -5.50, outmask=geod.ALL)
    d = geod.Direct(r["lat1"], r["lon1"], r["azi1"], r["s12"], outmask=geod.ALL)
    print(f"  lat2 {d['lat2']:.12f}  (inverse said {r['lat2']:.12f})")
    print(f"  s12  {d['s12']:.6f}  (inverse said {r['s12']:.6f})")
    print(f"  residual {abs(d['lat2'] - r['lat2']) * 111320.0:.3e} m")

    print("\nfollowing a geodesic")
    line = geod.Line(37.0, -122.0, 51.0)
    for s in (0.0, 5e5, 1e6, 2e6):
        p = line.Position(s)
        print(f"  s={s/1000:7.1f} km  lat2={p['lat2']:9.5f}  lon2={p['lon2']:10.5f}"
              f"  azi2={p['azi2']:7.3f}")

    print("\nten thousand points through one call")
    n = 10_000
    rng = np.random.default_rng(0)
    lat1 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon1 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    lat2 = np.ascontiguousarray(np.degrees(np.arcsin(rng.uniform(-1, 1, n))))
    lon2 = np.ascontiguousarray(rng.uniform(-180, 180, n))
    out = geod.InversePairs(lat1, lon1, lat2, lon2, outmask=geod.ALL)
    print(f"  s12  min {out['s12'].min():.3f} m  max {out['s12'].max():.3f} m")
    print(f"  a12  min {out['a12'].min():.6f} deg  max {out['a12'].max():.6f} deg")

    print("\narea of a polygon")
    poly = geod.Polygon()
    poly.AddPoints(*zip(*[(0, 0), (0, 90), (90, 0)]))
    n, perimeter, area = poly.Compute()
    print(f"  {n} vertices, perimeter {perimeter/1e3:.3f} km, area {area:.6e} m^2")
    print(f"  the ellipsoid is {poly.area0:.6e} m^2, so this is a "
          f"{100 * area / poly.area0:.4f}% of it")

    print("\nanother ellipsoid")
    mars = Geodesic(3396190.0, 1 / 169.8)
    print(f"  Mars, 0 0 -> 0 90: s12 = {mars.Inverse(0, 0, 0, 90)['s12']:.3f} m")
    print(f"  WGS84, 0 0 -> 0 90: s12 = {geod.Inverse(0, 0, 0, 90)['s12']:.3f} m")
    assert math.isfinite(mars._c2())


if __name__ == "__main__":
    main()
