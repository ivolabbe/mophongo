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
    images, segmap, catalog, psfs, _, _ = make_simple_data(nsrc=20, size=101, ndilate=2)
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    start = time.perf_counter()
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)
    extract_time = time.perf_counter() - start

  
    start = time.perf_counter()
    fitter = SparseFitter(tmpls.templates, images[1])
    fitter.solve()
    fit_time = time.perf_counter() - start

    print(f"Extraction time: {extract_time:.4f} s")
#    print(f"Extension time: {extend_time:.4f} s")
    print(f"Fitting time: {fit_time:.4f} s")

    assert extract_time > 0
#    assert extend_time > 0
    assert fit_time > 0

#!/usr/bin/env python3
import time
import numpy as np

# pure‐NumPy direct sliding‐window
def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    padded = np.pad(image, ((pad_y, ky-1-pad_y), (pad_x, kx-1-pad_x)), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, kernel.shape)
    # windows shape = (H, W, ky, kx)
    return np.einsum("ijkl,kl->ij", windows, kernel)

# scipy / astropy imports
from scipy.ndimage       import convolve as nd_convolve
from scipy.signal       import convolve2d, fftconvolve
from astropy.convolution import convolve as astro_convolve, convolve_fft

def test_benchmark_convolution():
    def run_benchmark(image, kernel, niter=5):
        methods = {
            "direct (_convolve2d)"       : lambda: _convolve2d(image, kernel),
            "scipy.ndimage.convolve"     : lambda: nd_convolve(image, kernel, mode="constant", cval=0.0),
            "scipy.signal.convolve2d"    : lambda: convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0.0),
            "scipy.signal.fftconvolve"   : lambda: fftconvolve(image, kernel, mode="same"),
            "astropy.convolution.convolve": lambda: astro_convolve(image, kernel, boundary="fill", fill_value=0.0, normalize_kernel=False),
            "astropy.convolution.convolve_fft": lambda: convolve_fft(image, kernel, normalize_kernel=False, boundary="fill", fill_value=0.0),
        }

        print(f"{'method':35s}  {'time [s]':>10s}")
        print("-"*50)
        for name, func in methods.items():
            # warm up
            func()
            t0 = time.perf_counter()
            for _ in range(niter):
                out = func()
            dt = (time.perf_counter() - t0) / niter
            print(f"{name:35s}  {dt:10.5f}")

#    if __name__ == "__main__":
        # configure sizes here
    # hands down scipy.signal.fftconvolve is the fastest 
 #       IMG_SIZE = 50
 #       KERNEL_SIZE = 201

        IMG_SIZE = 50
        KERNEL_SIZE = 201


        # random test data
        np.random.seed(0)
        image  = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
        # e.g. Gaussian‐like kernel
        x = np.linspace(-1, 1, KERNEL_SIZE)
        y = x[:,None]
        gauss = np.exp(-(x[None,:]**2 + y**2)/(2*(0.2**2)))
        kernel = (gauss/gauss.sum()).astype(np.float32)

        print(f"Image size: {image.shape}, kernel size: {kernel.shape}")
        run_benchmark(image, kernel, niter=20)
