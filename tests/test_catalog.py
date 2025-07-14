import numpy as np
from pathlib import Path
from astropy.io import fits

from mophongo.catalog import Catalog, safe_dilate_segmentation
from mophongo.psf import PSF
from mophongo.templates import _convolve2d
from photutils.segmentation import SegmentationImage, SourceCatalog, detect_sources, deblend_sources
from photutils.aperture import CircularAperture, aperture_photometry
import matplotlib.pyplot as plt
from utils import lupton_norm, label_segmap
from utils import make_simple_data
from astropy.stats import mad_std
from skimage.morphology import dilation, disk, max_tree, square
from skimage.feature import peak_local_max
from astropy.visualization import make_lupton_rgb


def test_deblend_sources(tmp_path):
    images, segmap, catalog, psfs, truth, wht = make_simple_data(seed=3,
                                                                 nsrc=50,
                                                                 size=101,
                                                                 ndilate=2,
                                                                 peak_snr=2)

    psf_det = PSF.gaussian(9, 2, 2)  # delta function = no smoothing
    detimg = images[0]
    print('reading detection image')
    sci1 = fits.getdata('data/uds-test-f115w_sci.fits')
    wht1 = fits.getdata('data/uds-test-f115w_wht.fits')
    sci4 = fits.getdata('data/uds-test-f444w_sci.fits')
    wht4 = fits.getdata('data/uds-test-f444w_wht.fits')

    print('convolving detection image')
    sci1c = _convolve2d(sci1, psf_det.array)
    sci4c = _convolve2d(sci4, psf_det.array)
    detimg = (sci1c * np.sqrt(wht1))**2 + (sci4c * np.sqrt(wht4))**2
 
    print('detecing sources')
    seg = detect_sources(detimg, threshold=3.0, npixels=5)
    print('dilating segmentation')
    seg.data = safe_dilate_segmentation(seg.data, selem=square(3))

    # Find local maxima per segment
  #  print('finding local maxima per segment')
  #  local_maxima, segment_counts = find_local_max(
  #      detimg, seg.data, min_distance=3, threshold_abs=5, threshold_rel=1e-5,
  #  )
    
  #  total_maxima = len(local_maxima)
  #  print(f'Found {total_maxima} total local maxima across all segments')
  #  print(f'Segment breakdown: {segment_counts}')

    # Try with very relaxed parameters
    print('deblending sources')
#    seg = deblend_sources(detimg, seg, npixels=5, contrast=1e-10, nlevels=48, mode='exponential')
    seg = deblend_sources(detimg, seg, npixels=5, contrast=1e-10, nlevels=64, mode='exponential')

    rgb = make_lupton_rgb(
        sci4c,  # red channel (F444W)
        (sci1c + sci4c) / 2.0,  # green channel (average)
        sci1c,  # blue channel (F115W)
        Q=10,  # softening parameter
        stretch=0.1  # stretch parameter
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left panel: Detection image with segmentation and local maxima
    ax1.axis("off")
    ax1.imshow(detimg, origin="lower", cmap="gray", norm=lupton_norm(detimg))
    ax1.imshow(seg.data, origin="lower", cmap=seg.cmap, alpha=0.3)
    
    # Plot local maxima as red crosses
 #   if len(local_maxima) > 0:
 #       ax1.scatter(local_maxima[:, 1], local_maxima[:, 0], 
 #                  marker='x', s=20, c='red', linewidths=1, alpha=0.8) 
  #  ax1.set_title(f"Detection + Segmentation + {total_maxima} Local Maxima")

    # Right panel: RGB image with segmentation
    ax2.axis("off")
    ax2.imshow(rgb, origin="lower")
    ax2.set_title("F115W/F444W RGB")

    plt.tight_layout()
    out = tmp_path / "deblend_diagnostic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    assert out.exists()


def test_catalog(tmp_path):
    sci = Path("data/uds-test-f444w_sci.fits")
    wht = Path("data/uds-test-f444w_wht.fits")
    out = tmp_path / "uds-test-f444w_ivar.fits"
    params = {
        "kernel_size": 4.0,
        "detect_threshold": 1.0,
        "dilate_segmap": 3,
        "deblend_mode": "exponential",
        "detect_npixels": 5,
        "deblend_nlevels": 32,
        "deblend_contrast": 1e-3,
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
    assert "CRPIX1" in hdr

    #    segimage = fits.getdata('data/uds-test-f444w_seg.fits')
    #    seg = SegmentationImage(segimage)
    segmap = cat.segmap
    #    segmap = deblend_sources_lutz(cat.det_img,
    #                                  segmap,
    #                                  npixels=cat.params['detect_npixels'],
    #                                  contrast=cat.params['deblend_contrast'])
    cmap_seg = segmap.cmap

    #    print("UNIQUE ids in segmap", np.unique(segmap.data))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    ax.imshow(cat.det_img, origin="lower", cmap="gray", norm=lupton_norm(cat.det_img))
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
    phot = aperture_photometry(cat.sci, apertures, error=np.sqrt(1.0 / cat.ivar))
    measured_err = phot["aperture_sum_err"].data
    expected_err = []
    for mask in apertures.to_mask(method="exact"):
        cutout = mask.multiply(1.0 / cat.ivar)
        expected_err.append(np.sqrt(np.sum(cutout[mask.data > 0])))
    expected_err = np.asarray(expected_err)
    assert np.allclose(measured_err, expected_err)
