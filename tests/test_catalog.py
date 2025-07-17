import numpy as np
from pathlib import Path
from astropy.io import fits

from mophongo.catalog import Catalog, safe_dilate_segmentation
from mophongo.psf import PSF
from mophongo.templates import _convolve2d
from mophongo.deblender import deblend_sources_hybrid
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
    return
    images, segmap, catalog, psfs, truth, wht = make_simple_data(seed=3,
                                                                 nsrc=20,
                                                                 size=51,
                                                                 ndilate=2,
                                                                 peak_snr=2)

    psf_det = PSF.gaussian(5, 2, 2)  # delta function = no smoothing
    detimg = images[0]
    print('reading detection image')
    sci1 = fits.getdata('data/uds-test-f115w_sci.fits')
    wht1 = fits.getdata('data/uds-test-f115w_wht.fits')
    sci4 = fits.getdata('data/uds-test-f444w_sci.fits')
    wht4 = fits.getdata('data/uds-test-f444w_wht.fits')

    print('convolving detection image')
    sci1c = _convolve2d(sci1, psf_det.array)
    sci1c = sci1
    sci4c = _convolve2d(sci4, psf_det.array)
    detimg = (sci1c * np.sqrt(wht1))**2 + (sci4c * np.sqrt(wht4))**2

    print('detecing sources')
    seg = detect_sources(detimg, threshold=3.0, npixels=5)
    print('dilating segmentation')
    seg.data = safe_dilate_segmentation(seg.data, selem=square(3))

    # Try with very relaxed parameters and debug output
    print('deblending sources')
    hybrid = False
    if hybrid:
        seg_deb, debug_info = deblend_sources_hybrid(
            detimg,
            seg,
            nmax=10,  # Allow more peaks for debugging
            peak_threshold_rel=1e-6,
            return_debug=True)

        # Find segments with more than 5 peaks
        segments_with_many_peaks = {
            seg_id: peak_list
            for seg_id, peak_list in debug_info['peaks'].items()
            if len(peak_list) > 5
        }

        print(f"Found {len(segments_with_many_peaks)} segments with >5 peaks")

        for seg_id, peak_list in segments_with_many_peaks.items():
            print(
                f"Creating diagnostic for segment {seg_id} with {len(peak_list)} peaks"
            )

            # Get templates for this segment
            segment_templates = {
                tid: (tmpl, pos)
                for tid, (tmpl, pos) in debug_info['templates'].items()
                if debug_info['mapping'].get(tid) == seg_id
            }
            segment_models = {
                tid: model
                for tid, model in debug_info['models'].items()
                if debug_info['mapping'].get(tid) == seg_id
            }

            if not segment_templates:
                continue

            # Get segment region
            seg_mask = seg.data == seg_id
            if not seg_mask.any():
                continue

            y_coords, x_coords = np.where(seg_mask)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Add padding
            pad = 20
            y_slice = slice(max(0, y_min - pad),
                            min(detimg.shape[0], y_max + pad + 1))
            x_slice = slice(max(0, x_min - pad),
                            min(detimg.shape[1], x_max + pad + 1))

            # Extract region
            det_crop = detimg[y_slice, x_slice]
            seg_crop = seg.data[y_slice, x_slice]
            parent_mask_crop = (seg_crop == seg_id)

            n_templates = len(segment_templates)
            ncols = max(n_templates, 1)
            fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))
            if ncols == 1:
                axes = axes.reshape(3, 1)

            # Row 0: parent segment mask over det_img
            for i in range(ncols):
                ax = axes[0, i]
                ax.axis("off")
                ax.imshow(det_crop,
                          origin="lower",
                          cmap="gray",
                          norm=lupton_norm(det_crop))
                ax.imshow(np.ma.masked_where(~parent_mask_crop,
                                             parent_mask_crop),
                          origin="lower",
                          cmap="autumn",
                          alpha=0.5)
                ax.set_title(f"Parent Segment {seg_id}", fontsize=10)

            # Row 1: templates (grayscale, lupton norm)
            for i, (tid, (template,
                          (t_y0,
                           t_x0))) in enumerate(segment_templates.items()):
                ax = axes[1, i]
                ax.axis("off")
                ax.imshow(template,
                          origin="lower",
                          cmap="gray",
                          norm=lupton_norm(det_crop))
                ax.set_title(f"Symmetry Template {tid}", fontsize=10)
                # Mark the template center (peak location)
                template_peaks = [
                    p for p in peak_list
                    if debug_info['mapping'].get(tid) == seg_id
                ]
                if template_peaks:
                    for py, px in template_peaks:
                        local_py = py - t_y0
                        local_px = px - t_x0
                        if (0 <= local_py < template.shape[0]
                                and 0 <= local_px < template.shape[1]):
                            ax.scatter(local_px,
                                       local_py,
                                       marker='x',
                                       s=30,
                                       c='red',
                                       linewidths=2)

            # Row 2: models (grayscale, lupton norm)
            for i, (tid, (template,
                          (t_y0,
                           t_x0))) in enumerate(segment_templates.items()):
                ax = axes[2, i]
                ax.axis("off")
                if tid in segment_models:
                    model = segment_models[tid]
                    ax.imshow(model,
                              origin="lower",
                              cmap="gray",
                              norm=lupton_norm(det_crop))
                    ax.set_title(f"Analytic Model {tid}", fontsize=10)
                    for py, px in template_peaks:
                        local_py = py - t_y0
                        local_px = px - t_x0
                        if (0 <= local_py < model.shape[0]
                                and 0 <= local_px < model.shape[1]):
                            ax.scatter(local_px,
                                       local_py,
                                       marker='x',
                                       s=30,
                                       c='yellow',
                                       linewidths=2)
                else:
                    ax.text(0.5,
                            0.5,
                            'No Model',
                            ha='center',
                            va='center',
                            transform=ax.transAxes)
                    ax.set_title(f"No Model {tid}", fontsize=10)

            # Hide unused subplots
            for i in range(n_templates, axes.shape[1]):
                axes[0, i].axis("off")
                axes[1, i].axis("off")
                axes[2, i].axis("off")

            plt.suptitle(
                f"Segment {seg_id}: {len(peak_list)} peaks, {len(segment_templates)} templates",
                fontsize=14)
            plt.tight_layout()

            debug_out = tmp_path / f"segment_{seg_id}_debug.png"
            fig.savefig(debug_out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved debug plot: {debug_out}")
    else:
        seg_deb = deblend_sources(
            detimg,
            seg,
            npixels=5,  # Minimum number of pixels in a source
            nlevels=32,  # Number of deblending levels
            mode='exponential',  # Deblending mode
            contrast=1e-5)

    # **ORIGINAL VISUALIZATION**
    rgb = make_lupton_rgb(
        sci4c,  # red channel (F444W)
        (sci1c + sci4c) / 2.0,  # green channel (average)
        sci1c,  # blue channel (F115W)
        Q=10,  # softening parameter
        stretch=0.1  # stretch parameter
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left panel: Detection image with segmentation
    ax1.axis("off")
    ax1.imshow(detimg, origin="lower", cmap="gray", norm=lupton_norm(detimg))
    ax1.imshow(seg.data, origin="lower", cmap=seg.cmap, alpha=0.3)
    ax1.set_title(
        f"Detection + Segmentation\n{seg.nlabels} → {seg_deb.nlabels} sources")

    # Right panel: RGB image with final segmentation
    ax2.axis("off")
    ax2.imshow(rgb, origin="lower")
    ax2.imshow(seg_deb.data, origin="lower", cmap=seg_deb.cmap, alpha=0.3)
    ax2.set_title("F115W/F444W RGB + Deblended")

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
    cmap_seg = segmap.cmap
    
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
