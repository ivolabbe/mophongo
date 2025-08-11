import numpy as np
import mophongo.pipeline as pipeline
from utils import make_simple_data
from mophongo.fit import FitConfig
import mophongo.utils as mutils


def test_pipeline_class_attributes():
    images, segmap, catalog, psfs, _, wht = make_simple_data(nsrc=3, size=51)
    kernel = [mutils.matching_kernel(psfs[0], p) for p in psfs]
    kernel[0] = np.array([[1.0]])
    pl = pipeline.Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=wht,
        kernels=kernel,
        config=FitConfig(fit_astrometry_niter=0),
    )
    cat, residuals, fitter = pl.run()

    assert pl.catalog is cat
    assert pl.residuals == residuals
    assert pl.fitter is fitter
    assert pl.astro is not None
