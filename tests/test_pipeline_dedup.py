import numpy as np
import mophongo.pipeline as pipeline
import mophongo.utils as mutils
from utils import make_simple_data


def test_pipeline_deduplicates_templates():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=3, size=51)
    dup_catalog = catalog[:2].copy()
    dup_catalog.add_row(catalog[0])  # duplicate first source
    kernel = [mutils.matching_kernel(psfs[0], p) for p in psfs]
    kernel[0] = np.array([[1.0]])
    from mophongo.fit import FitConfig
    table, resid, fitter = pipeline.run(
        images,
        segmap,
        catalog=dup_catalog,
        psfs=psfs,
        weights=wht,
        kernels=kernel,
        config=FitConfig(fit_astrometry_niter=0),
    )
    flux_col = "flux_1"
    mask = dup_catalog['id'] == dup_catalog['id'][0]
    assert np.count_nonzero(np.isfinite(table[flux_col][mask])) == 1
