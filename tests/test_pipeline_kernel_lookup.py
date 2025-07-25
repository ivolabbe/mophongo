import numpy as np
import shapely.geometry as sgeom

from mophongo.pipeline import run_photometry
from mophongo.psf_map import PSFRegionMap
from mophongo.kernels import KernelLookup
from mophongo.psf import PSF
from utils import make_simple_data


def test_run_photometry_with_lookup(tmp_path):
    images, segmap, catalog, psfs, truth, wht = make_simple_data(seed=42, nsrc=20, size=101)

    footprints = {"A": sgeom.box(0, 0, 1, 1)}
    prm = PSFRegionMap.from_footprints(footprints, crs=None)

    kernel = PSF.from_array(psfs[0]).matching_kernel(psfs[1])
    lookup = KernelLookup(prm, np.stack([kernel]))

    table, resid, _ = run_photometry(
        images, segmap, catalog, psfs, wht_images=wht, kernels=[None, lookup]
    )

    assert "flux_1" in table.colnames
    assert resid[0].shape == images[1].shape
