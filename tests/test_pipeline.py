import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.nddata import Cutout2D

from mophongo.pipeline import run_photometry
from utils import (
    make_simple_data,
    save_diagnostic_image,
    save_flux_vs_truth_plot,
)


def test_pipeline_flux_recovery(tmp_path):
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(nsrc=30, size=151, ndilate=2, peak_snr=3)
    table, resid, templates = run_photometry(images, segmap, catalog, psfs, rms)
    
    table["flux_true"] = catalog["flux_true"]  # add flux_true to the table

    # Plot for high-res (flux_0) vs truth
    flux_hi_plot = tmp_path / "flux_hi_vs_true.png"
    save_flux_vs_truth_plot(
        flux_hi_plot,
        table["flux_true"],
        table["flux_0"],
        error=table["err_0"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)"
    )
    assert flux_hi_plot.exists()

    # Plot for low-res (flux_1) vs truth
    flux_lo_plot = tmp_path / "flux_lo_vs_true.png"
    save_flux_vs_truth_plot(
        flux_lo_plot,
        table["flux_true"],
        table["flux_1"],
        error=table["err_1"],  # Add error column
        label="Flux (lowres)",
        xlabel="True Flux",
        ylabel="Recovered Flux (lowres)"
    )
    assert flux_lo_plot.exists()

    # Plot for flux_lo vs flux_hi with error propagation
    flux_lo_hi_plot = tmp_path / "flux_lo_vs_hi.png"
    # Calculate combined error for hires vs lowres comparison
    combined_error = np.sqrt(table["err_0"]**2 + table["err_1"]**2)
    save_flux_vs_truth_plot(
        flux_lo_hi_plot,
        table["flux_0"],
        table["flux_1"],
        error=combined_error,
        label="Flux (lowres) vs (hires)",
        xlabel="Recovered Flux (hires)",
        ylabel="Recovered Flux (lowres)"
    )
    assert flux_lo_hi_plot.exists()

    model = images[1] - resid[1]
    fname = tmp_path / "diagnostic.png"
    save_diagnostic_image(fname, truth_img, images[0], images[1], model, resid[1], segmap=segmap, catalog=catalog)
    assert fname.exists()

    # Report statistics for flux recovery
    for idx in range(len(psfs)):
        col = f"flux_{idx}"
        ratio = np.array(table[col]) / np.array(table["flux_true"])
        print(f"flux_{idx}/flux_true percentiles: 5th={np.percentile(ratio,5):.2f}, "
              f"16th={np.percentile(ratio,16):.2f}, 50th={np.percentile(ratio,50):.2f}, "
              f"84th={np.percentile(ratio,84):.2f}, 95th={np.percentile(ratio,95):.2f}")

    # sanity check on propagated errors for low-res image
    from mophongo.psf import PSF
    from mophongo.templates import Templates
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel)
    noise_std = rms[1][0, 0]
    err_pred = np.array([noise_std / np.sqrt((t.array**2).sum()) for t in tmpls.templates])
    ratio_err = table["err_1"] / err_pred
    assert np.allclose(np.mean(ratio_err), 1.0, atol=3)

    # Write catalog with all columns formatted to 3 digits after the decimal
    for col in table.colnames:
        if table[col].dtype.kind in "fc":  # float or complex
            table[col].info.format = ".3f"

    cat_file = tmp_path / "photometry.cat"
    table.write(cat_file, format="ascii.commented_header")
    assert cat_file.exists()

    loaded = Table.read(cat_file, format="ascii.commented_header")
    assert len(loaded) == len(table)

    assert resid.shape[0] == len(images)


def _calculate_growth_curve(image, center_xy, max_radius=15):
    """Helper to calculate a cumulative flux growth curve."""
    y, x = np.indices(image.shape)
    cx, cy = center_xy
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    radii = np.arange(0, max_radius, 0.5)
    fluxes = []
    
    for rad in radii:
        mask = r <= rad
        fluxes.append(image[mask].sum())
        
    return radii, np.array(fluxes)


def test_template_growth_curve_diagnostic(tmp_path):
    """
    Compares the flux growth curves of templates created by two methods:
    1. Using a dilated segmentation map.
    2. Using the `psf_dilation_simple` template extension.
    """
    n_sources_to_plot = 5
    seed = 42

    # --- Generate Base Data (ndilate=0) ---
    images_base, segmap_base, catalog_base, psfs, truth_img, _ = make_simple_data(
        nsrc=15, size=101, ndilate=0, peak_snr=10, seed=seed
    )

    # --- Case 1: Dilated Segmap ---
    images_dil, segmap_dil, catalog_dil, _, _, _ = make_simple_data(
        nsrc=15, size=101, ndilate=3, peak_snr=10, seed=seed
    )
    # Run photometry just to get the templates generated from the dilated segmap
    _, _, templates_dilated = run_photometry(
        images_dil, segmap_dil, catalog_dil, psfs, extend_templates=None
    )

    # --- Case 2: Extended Template ---
    # Run photometry on base data to get the extended templates
    _, _, templates_extended = run_photometry(
        images_base, segmap_base, catalog_base, psfs, extend_templates='psf_dilation_simple'
    )

    # --- Plotting: Growth Curves + Stamps ---
    fig = plt.figure(figsize=(25, 16))
    
    # Create a custom grid: 4 rows x 5 columns for 10 sources
    # Row 1: Growth curves
    # Row 2: Dilated segmap stamps  
    # Row 3: Extended template stamps
    # Row 4: Difference stamps (extended - dilated)
    
    gs = fig.add_gridspec(4, 5, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.2)
    
    fig.suptitle("Template Growth Curve Comparison with Stamps", fontsize=16)

    for i in range(n_sources_to_plot):
        col = i % 5
        source_id = i + 1
        
        # Get the original segmap mask for this source
        source_mask_base = (segmap_base == source_id)
        
        # Get the template objects
        tmpl_dil = templates_dilated._templates_hires[i]
        tmpl_ext = templates_extended._templates_hires[i]
        
        # Create a truth cutout from the noise-free image
        y_c, x_c = catalog_base['y'][i], catalog_base['x'][i]
        truth_cutout_obj = Cutout2D(truth_img, (x_c, y_c), tmpl_dil.array.shape)
        truth_cutout = truth_cutout_obj.data

        # --- Normalization Factor ---
        # Flux in the original (ndilate=0) segmentation map
        norm_flux = truth_img[source_mask_base].sum()
        if norm_flux == 0:
            continue

        # === ROW 1: GROWTH CURVES ===
        ax_growth = fig.add_subplot(gs[0, col])
        
        # Get the source position from the Cutout2D object
        radii, gc_truth = _calculate_growth_curve(truth_cutout, truth_cutout_obj.input_position_cutout)

        # For the templates, use the stored source position
        _, gc_dilated = _calculate_growth_curve(tmpl_dil.array, tmpl_dil.position_cutout)
        _, gc_extended = _calculate_growth_curve(tmpl_ext.array, tmpl_ext.position_cutout)

        # Plot growth curves
        ax_growth.plot(radii, gc_truth / norm_flux, 'k-', label='Ground Truth', lw=2)
        ax_growth.plot(radii, gc_dilated / norm_flux, 'b--', label='Dilated Segmap', lw=2)
        ax_growth.plot(radii, gc_extended / norm_flux, 'r:', label='Extended Template', lw=2)
        
        ax_growth.axhline(1.0, color='grey', linestyle='--', alpha=0.7)
        ax_growth.set_ylim(0, 1.5)
        ax_growth.set_title(f"Source {source_id}")
        ax_growth.grid(True, alpha=0.3)
        
        if col == 0:
            ax_growth.set_ylabel('Cumulative Flux / Base Flux')
        if i >= 5:  # Bottom row of growth curves
            ax_growth.set_xlabel('Radius (pixels)')

        # === ROW 2: DILATED SEGMAP STAMPS ===
        ax_dil = fig.add_subplot(gs[1, col])
        
        # Linear stretch min-max
        vmin, vmax =  tmpl_dil.array.min(), tmpl_dil.array.max()/5
        im_dil = ax_dil.imshow(tmpl_dil.array, origin='lower', cmap='gray', 
                               vmin=vmin, vmax=vmax)
        
        # Mark source position
        if tmpl_dil.position_cutout is not None:
            ax_dil.plot(tmpl_dil.position_cutout[0], tmpl_dil.position_cutout[1], 'r+', markersize=8)
        
        ax_dil.set_title(f'Dilated Segmap\nFlux: {tmpl_dil.array.sum():.1f}', fontsize=10)
        ax_dil.set_xticks([])
        ax_dil.set_yticks([])

        # === ROW 3: EXTENDED TEMPLATE STAMPS ===
        ax_ext = fig.add_subplot(gs[2, col])
        
        # Extract 21x21 cutouts centered on source positions
        cutout_size = (21, 21)

        # Extract 21x21 cutout from extended template centered on source position
        ext_cutout = Cutout2D(tmpl_ext.array, tmpl_ext.position_cutout, cutout_size, mode='partial', fill_value=0)
        
        # Linear stretch min-max
        im_ext = ax_ext.imshow(ext_cutout.data, origin='lower', cmap='gray',
                               vmin=vmin, vmax=vmax) 
               
        # Mark source position at center of cutout
        ax_ext.plot(10.0, 10.0, 'r+', markersize=8)
        
        ax_ext.set_title(f'Extended Template\nFlux: {tmpl_ext.array.sum():.1f}', fontsize=10)
        ax_ext.set_xticks([])
        ax_ext.set_yticks([])

        # === ROW 4: DIFFERENCE STAMPS ===
        ax_diff = fig.add_subplot(gs[3, col])
        
        
        # Create cutouts from each template using their source positions
        dil_cutout = Cutout2D(tmpl_dil.array, tmpl_dil.position_cutout, cutout_size, mode='partial', fill_value=0)
        
        # Calculate difference
        diff_array = ext_cutout.data - dil_cutout.data
        flux_diff = diff_array.sum()
        
        # Source position is at center of 21x21 cutout
        source_y, source_x = 10.0, 10.0  # Center of 21x21 array
        
        # Symmetric color scale around zero
        vmax_diff = np.abs(diff_array).max() if np.abs(diff_array).max() > 0 else 1
        im_diff = ax_diff.imshow(diff_array, origin='lower', cmap='RdBu_r',
                                 vmin=-vmax_diff, vmax=vmax_diff)
        
        # Mark source position at center
        ax_diff.plot(source_x, source_y, 'k+', markersize=8)
        
        ax_diff.set_title(f'Difference\nΔFlux: {flux_diff:.1f} ({100*flux_diff/tmpl_dil.array.sum():.1f}%)', fontsize=10)
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])

    # Add legend for growth curves
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add row labels
    fig.text(0.02, 0.75, 'Growth Curves', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.55, 'Dilated Segmap', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.35, 'Extended Template', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.15, 'Difference', rotation=90, va='center', fontsize=12, weight='bold')
    
    plot_file = tmp_path / "growth_curve_diagnostic.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert plot_file.exists()
