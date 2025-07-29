import sys
import os

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))

import numpy as np
from mophongo import pipeline
from mophongo.fit import FitConfig
from utils import make_simple_data, save_diagnostic_image
from mophongo.astro_fit import GlobalAstroFitter, SparseFitter
from mophongo.templates import Templates
import mophongo.utils as mutils

 
def test_global_astro_fitter_with_correct_template_count():
    """Test GlobalAstroFitter with the actual number of templates from Templates.from_image."""
    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=10, size=101)
    
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel=None)
    
    # Use actual template count instead of assuming it equals source count
    actual_template_count = len(tmpls.templates)
    
    config = FitConfig(fit_astrometry=True, astrom_basis_order=2)
    fitter = GlobalAstroFitter(tmpls.templates, images[1], wht[1], segmap, config)
    
    # Check that astrometry parameters are set up correctly
    expected_n_alpha = (config.astrom_basis_order + 1) * (config.astrom_basis_order + 2) // 2
    assert fitter.n_alpha == expected_n_alpha
    assert fitter.n_flux == actual_template_count
    
    # GlobalAstroFitter should have more templates due to gradient templates
    expected_total_templates = actual_template_count + 2 * expected_n_alpha * actual_template_count
 
    assert len(fitter.templates) == expected_total_templates

def test_global_astro_fitter_n_flux_attribute():
    """Test that n_flux attribute is only set when astrometry is enabled."""
    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=5, size=51)
    
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel=None)
    
    # With astrometry enabled
    config_astro = FitConfig(fit_astrometry=True, astrom_basis_order=1)
    fitter_astro = GlobalAstroFitter(tmpls.templates, images[1], wht[1], segmap, config_astro)
    assert hasattr(fitter_astro, 'n_flux')
    assert fitter_astro.n_flux == len(tmpls.templates)
    
    # Without astrometry enabled
    config_no_astro = FitConfig(fit_astrometry=False)
    fitter_no_astro = GlobalAstroFitter(tmpls.templates, images[1], wht[1], segmap, config_no_astro)
    # n_flux might not be set when astrometry is disabled
    if hasattr(fitter_no_astro, 'n_flux'):
        assert fitter_no_astro.n_flux == len(tmpls.templates)

def test_solve_return_shapes_with_actual_templates(tmp_path):
    """Test solve method shapes using actual template counts."""

    from scipy.ndimage import shift as nd_shift
    from scipy.ndimage import shift as nd_shift, map_coordinates
#    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=150, size=301, peak_snr=0.5, seed=11)
    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=20, size=151, peak_snr=1, seed=11, border_size=15)
    
    # shx, shy = 0.9, -0.9
    # images[1] = nd_shift(images[0], (shy, shx))

    dirac = lambda n: ((np.arange(n)[:,None] == n//2) & (np.arange(n) == n//2)).astype(float)
    h, w = images[0].shape
    y, x = np.mgrid[0:h, 0:w]
    shift_x = -1.5 * x / w + 0.5 * (x / w)**2  # quadratic in x
    shift_y = -2.0 * y / h + 0.3 * (y / h)**2  # quadratic in y
    shift_field = np.sqrt(shift_x**2 + shift_y**2)
    images[1] = map_coordinates(images[0], [y - shift_y, x - shift_x], order=3, mode='constant')
    print(f"Shift field: {shift_field.min()} to {shift_field.max()} pixels")

# @@ fitting is ok, but goes nuts when kernel is applied. 
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(truth, segmap, positions, kernel=psfs[0])
    n_tmpl = len(tmpls.templates)
    
    sf_cfg  = FitConfig(fit_astrometry=False,reg=1e-4)     # <- no α/β any more
    fitter0 = SparseFitter(tmpls.templates, images[1], wht[1], sf_cfg)
    solution0, info0 = fitter0.solve()
    res0 = fitter0.residual()

#    config = FitConfig(fit_astrometry=True, astrom_basis_order=1, reg_astrom=1e-3)
    config = FitConfig(fit_astrometry=True, astrom_basis_order=1, reg_astrom=1e-4, snr_thresh_astrom=10.0)
    fitter = GlobalAstroFitter(tmpls.templates, images[1], wht[1], config)
    solution1, info = fitter.solve()
    res1 = fitter.residual()
    
    i=2
    print(f"Astrometry parameters:  alpha={fitter.alpha} beta={fitter.beta}")
    dx,dy = fitter.shift_at(*positions[i])
    print(f" shifts at position {positions[i]}: dx={dx}, dy={dy}")
    t = tmpls.templates[i]
    print(f"Template {i} oxy: {tmpls.templates[i].shift}")

    # 2ms / source 
    fitter2 = GlobalAstroFitter(tmpls.templates, images[1], wht[1], config)
    solution2, info2 = fitter2.solve()
    res2 = fitter2.residual()

# 864.55753 331.42455 -> 864.66969 331.87318   444 to 770 fraction of pixel shift
    print(f"Astrometry parameters:  alpha={fitter2.alpha} beta={fitter2.beta}")
    dx,dy = fitter2.shift_at(*positions[i])
    print(f" shifts at position {positions[i]}: dx={dx}, dy={dy}")
    t = tmpls.templates[i]
    print(f"Template {i} oxy: {tmpls.templates[i].shift}")
  
    catalog['flux0'] = solution0 
    catalog['flux2'] = solution2 
    catalog['err2'] = fitter.flux_errors() 
    catalog['err_pred'] = fitter.predicted_errors()
    catalog['snr'] = catalog['flux_true'] /catalog['err_pred'] 
    catalog['quick_flux'] = fitter.quick_flux()[0:fitter.n_flux]
    catalog['quick_snr'] = catalog['quick_flux'] / catalog['err_pred']

    for col in catalog.colnames:
        if catalog[col].dtype.kind in "fc":  # float or complex
            catalog[col].info.format = ".2f"

    import matplotlib.pyplot as plt
    scale = images[0].shape[0]/15
    y_grid, x_grid = np.mgrid[10:images[1].shape[0]:20, 10:images[1].shape[1]:20]
    dx_grid = np.zeros_like(x_grid, dtype=float)
    dy_grid = np.zeros_like(y_grid, dtype=float)
    dx_in_grid = np.zeros_like(x_grid, dtype=float)
    dy_in_grid = np.zeros_like(y_grid, dtype=float)

    # Ensure indices are integers for shift_x, shift_y lookup
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x_val = int(round(x_grid[i, j]))
            y_val = int(round(y_grid[i, j]))
            dx, dy = fitter.shift_at(x_val, y_val)
            dx_grid[i, j] = -dx * scale
            dy_grid[i, j] = -dy * scale
            dx_in_grid[i, j] = shift_x[y_val, x_val] * scale
            dy_in_grid[i, j] = shift_y[y_val, x_val] * scale

    offset = 3e-5
    scl = images[1].sum()
    kws = dict(vmin=-5.3, vmax=-1.5, cmap='bone_r', origin='lower')
    fig, ax = plt.subplots(2,3, figsize=(13, 8))
    ax = ax.ravel()
    ax[0].imshow(np.log10(images[1]/scl + offset), **kws)
    ax[0].imshow(np.log10(images[1]/scl + offset), **kws)
    ax[0].quiver(x_grid, y_grid, dx_in_grid, dy_in_grid, color='white', angles='xy', scale_units='xy', scale=1, label='Input', alpha=0.7)
    ax[0].quiver(x_grid+1, y_grid, dx_grid, dy_grid, color='red', angles='xy', scale_units='xy', scale=1, label='Recovered', alpha=0.7)
    ax[0].set_title("Shift Field")
    ax[0].legend(['Input', 'Recovered'])
    ax[1].imshow(np.log10((images[1]-res2)/scl + offset), **kws)
    ax[1].set_title("model")
    ax[3].imshow(np.log10(res0/scl + offset), **kws)
    ax[3].set_title("residual flux only")
    ax[4].imshow(np.log10(res1/scl + offset), **kws)
    ax[4].set_title("residual iter 1 flux + shift")
    ax[5].imshow(np.log10(res2/scl + offset), **kws)
    ax[5].set_title("residual iter 2 flux + shift")

    bins = np.linspace(0, 1.4, 30) 
#    ax[2].hist(solution1/catalog['flux_true'], bins=bins, color='red', label='pass 1', alpha=0.5)
#    ax[2].hist(solution2/catalog['flux_true'], bins=bins, color='blue', label='pass 2', alpha=0.5)
    ax[2].hist(solution1, bins=bins, color='red', label='pass 1', alpha=0.5)
    ax[2].hist(solution2, bins=bins, color='blue', label='pass 2', alpha=0.5)
    ax[2].axvline(1.0, color='black', linestyle='--', label='true flux')
    ax[2].legend()
    ax[2].set_title("flux/flux_true")
    plt.tight_layout()





    plt.savefig(tmp_path / "diagnostic_global_astro_fitter.png")


# @@@ check the prescaling vs true
# compare to initial fit and then refit 
# 