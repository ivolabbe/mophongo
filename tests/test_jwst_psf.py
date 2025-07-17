import numpy as np
from astropy.io import fits
import mophongo.jwst_psf as jwst_psf

class DummySTDPSFGrid:
    def __init__(self, data, oversampling=4):
        self.data = np.asarray(data)
        self.oversampling = oversampling
        self.grid_xypos = [(0, 0)] * len(data)
        self.meta = {
            "detector": "NRCAL",
            "filter": "F444W",
            "grid_shape": (1, 1),
        }

class DummyNIRCam:
    def __init__(self):
        self.filter = None
        self.detector = None

    def psf_grid(self, num_psfs, all_detectors=False, oversample=4, fov_arcsec=1.0):
        arr = np.zeros((num_psfs, 11, 11))
        cy = cx = 5
        arr[:, cy, cx] = 1.0
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(arr, name="DET_SAMP")])
        return hdul


def test_make_extended_grid(monkeypatch):
    emp = DummySTDPSFGrid(np.ones((1, 5, 5)))
    monkeypatch.setattr(jwst_psf.stpsf, "NIRCam", DummyNIRCam)
    grid = jwst_psf.make_extended_grid(emp, Rmax=0.5, Rtaper=0.1, pixscale=0.5)
    assert grid.data.shape == (1, 9, 9)
    np.testing.assert_allclose(grid.data.sum(), 1.0)
