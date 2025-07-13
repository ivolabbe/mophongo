import numpy as np
from pathlib import Path
from astropy.io import fits

from mophongo.catalog import Catalog, deblend_sources_lutz, safe_dilate_segmentation

from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.aperture import CircularAperture, aperture_photometry
import matplotlib.pyplot as plt
from utils import lupton_norm, label_segmap


def test_catalog(tmp_path):
    sci = Path('data/uds-test-f444w_sci.fits')
    wht = Path('data/uds-test-f444w_wht.fits')
    out = tmp_path / 'uds-test-f444w_ivar.fits'
    params = {
        "kernel_size": 4.0,
        "detect_threshold": 1.0,
        "dilate_segmap": 3,
        "deblend_mode": 'sinh',
        "detect_npixels": 5,
        "deblend_nlevels": 32,
        "deblend_contrast": 1e-3
    }
    cat = Catalog.from_fits(sci, wht, ivar_outfile=out, params=params)
    cat.catalog["x"] = cat.catalog["xcentroid"]
    cat.catalog["y"] = cat.catalog["ycentroid"]

    assert cat.segmap.shape == cat.sci.shape
    assert cat.ivar.shape == cat.sci.shape
    assert len(cat.catalog) > 0
    assert np.all(np.isfinite(cat.ivar))
    assert out.exists()
    hdr = fits.getheader(out)
    assert 'CRPIX1' in hdr

    #    segimage = fits.getdata('data/uds-test-f444w_seg.fits')
    #    seg = SegmentationImage(segimage)
    segmap = cat.segmap
    segmap = deblend_sources_lutz(cat.det_img,
                                  segmap,
                                  npixels=cat.params['detect_npixels'],
                                  contrast=cat.params['deblend_contrast'])
    cmap_seg = segmap.cmap

    print('UNIQUE ids in lutzmap', np.unique(segmap.data))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    ax.imshow(cat.det_img,
              origin="lower",
              cmap="gray",
              norm=lupton_norm(cat.det_img))
    ax.imshow(segmap.data, origin="lower", cmap=cmap_seg, alpha=0.3)
    if len(cat.catalog) < 50:
        label_segmap(ax, segmap.data, cat.catalog, fontsize=5)

        for aper in cat.catalog.kron_aperture:
            aper.plot(ax=ax, color="white", lw=0.4, alpha=0.6)

    diag = tmp_path / "catalog_diagnostic.png"
    fig.savefig(diag, dpi=150)
    plt.close(fig)
    assert diag.exists()

    # Verify that small unweighted aperture errors reflect the weight map
    positions = np.column_stack([cat.catalog["x"], cat.catalog["y"]])
    apertures = CircularAperture(positions, r=4.0)
    phot = aperture_photometry(cat.sci,
                               apertures,
                               error=np.sqrt(1.0 / cat.ivar))
    measured_err = phot["aperture_sum_err"].data
    expected_err = []
    for mask in apertures.to_mask(method="exact"):
        cutout = mask.multiply(1.0 / cat.ivar)
        expected_err.append(np.sqrt(np.sum(cutout[mask.data > 0])))
    expected_err = np.asarray(expected_err)
    assert np.allclose(measured_err, expected_err)
