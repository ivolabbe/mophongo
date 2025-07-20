import numpy as np
import shapely.geometry as sgeom

from mophongo.psf_map import PSFRegionMap
from mophongo.kernels import KernelLookup


def test_kernel_lookup_caching():
    footprints = {
        "A": sgeom.box(0, 0, 1, 1),
        "B": sgeom.box(1.1, 0, 2.1, 1),
    }
    prm = PSFRegionMap.from_footprints(footprints, crs=None)
    kernels = np.zeros((len(prm.regions), 3, 3))
    kernels[0, 1, 1] = 1.0
    kernels[1, 1, 1] = 2.0

    lookup = KernelLookup(prm, kernels)

    k1 = lookup.get_kernel(0.5, 0.5)
    k2 = lookup.get_kernel(1.6, 0.5)

    assert np.allclose(k1, kernels[prm.lookup_key(0.5, 0.5)])
    assert np.allclose(k2, kernels[prm.lookup_key(1.6, 0.5)])

    # Cached object identity
    k1_again = lookup.get_kernel(0.5, 0.5)
    assert k1_again is k1
