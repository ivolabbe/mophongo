import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates
from mophongo.fit import SparseFitter
from utils import make_simple_data


def test_benchmark_pipeline_steps():
    images, segmap, catalog, psfs, _, _ = make_simple_data(nsrc=100, size=201, ndilate=2)
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    start = time.perf_counter()
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel)
    extract_time = time.perf_counter() - start

    start = time.perf_counter()
    tmpls.extend_with_moffat(kernel)
    extend_time = time.perf_counter() - start

    start = time.perf_counter()
    fitter = SparseFitter(tmpls.templates, images[1])
    fitter.solve()
    fit_time = time.perf_counter() - start

    print(f"Extraction time: {extract_time:.4f} s")
    print(f"Extension time: {extend_time:.4f} s")
    print(f"Fitting time: {fit_time:.4f} s")

    assert extract_time > 0
    assert extend_time > 0
    assert fit_time > 0

