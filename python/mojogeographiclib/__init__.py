"""mojo-geographiclib: GeographicLib's geodesic routines with the numerics in Mojo.

The public names and signatures mirror the pure-Python `geographiclib`
package, so a drop-in for the covered subset:

    from mojogeographiclib.geodesic import Geodesic
    Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)

Every number is produced by `src/ported.mojo` through `src/capi.mojo`; see
README.md for what is covered and how.
"""

from mojogeographiclib.constants import Constants
from mojogeographiclib.geodesiccapability import GeodesicCapability
from mojogeographiclib.geodesic import Geodesic
from mojogeographiclib.geodesicline import GeodesicLine
from mojogeographiclib.polygonarea import PolygonArea

__version__ = "0.1.0"

__all__ = [
    "Constants",
    "Geodesic",
    "GeodesicCapability",
    "GeodesicLine",
    "PolygonArea",
    "__version__",
]
