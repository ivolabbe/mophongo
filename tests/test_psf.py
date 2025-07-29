import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mophongo.psf import PSF, pad_to_shape
from mophongo.templates import _convolve2d
from utils import make_simple_data, save_psf_diagnostic, save_psf_fit_diagnostic


def test_moffat_psf_shape_and_normalization():
    psf = PSF.moffat(11, fwhm_x=3.0, fwhm_y=3.0, beta=2.5)
    assert psf.array.shape == (11, 11)
    np.testing.assert_allclose(psf.array.sum(), 1.0, atol=2e-3)


def test_psf_matching_kernel_properties(tmp_path):
    _, _, _, psfs, _, _ = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_lo.array.shape

    hi_pad = pad_to_shape(psf_hi.array, kernel.shape)
    conv = _convolve2d(hi_pad, kernel)
    np.testing.assert_allclose(conv, psf_lo.array, rtol=0, atol=3e-3)
    fname = tmp_path / "psf_kernel.png"
    save_psf_diagnostic(fname, hi_pad, psf_lo.array, kernel)
    assert fname.exists()

def test_delta_psf_default():
    psf = PSF.delta()
    assert psf.array.shape == (3, 3)
    assert psf.array[1, 1] == 1.0
    np.testing.assert_allclose(psf.array.sum(), 1.0)


def test_psf_from_data():
    from mophongo.utils import gaussian

    image = gaussian((61, 61), 2.5, 2.5, x0=30.3, y0=29.7)
    psf = PSF.from_data(image, (30.3, 29.7))

    assert psf.array.shape == (51, 51)
    np.testing.assert_allclose(psf.array.sum(), 1.0, atol=2e-3)

    cy = psf.array.shape[0] // 2
    cx = psf.array.shape[1] // 2
    maxpos = np.unravel_index(np.argmax(psf.array), psf.array.shape)
    assert maxpos == (cy, cx)


def test_matching_kernel_recenter():
    from mophongo.utils import moffat
    from photutils.centroids import centroid_quadratic

    psf_hi = PSF(moffat(41, 2.0, 2.0, beta=3.0))
    psf_lo = PSF(moffat(41, 10.0, 10.0, beta=2.5, x0=20.3, y0=20.2))

    k_off = psf_hi.matching_kernel(psf_lo, recenter=False)
    y_off, x_off = centroid_quadratic(k_off, fit_boxsize=5)
    cy = (k_off.shape[0] - 1) / 2
    cx = (k_off.shape[1] - 1) / 2
    dist_off = np.hypot(y_off - cy, x_off - cx)

    k = psf_hi.matching_kernel(psf_lo)
    y, x = centroid_quadratic(k, fit_boxsize=5)
    dist = np.hypot(y - cy, x - cx)

    assert dist < dist_off



def test_matching_kernel_basis(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import importlib
    import mophongo.utils
    from mophongo.psf import PSF
    importlib.reload(mophongo.utils)
    from mophongo.utils import multi_gaussian_basis, gauss_hermite_basis, CircularApertureProfile
    from mophongo.templates import _convolve2d
    from utils import save_psf_diagnostic 
    from matplotlib import pyplot as plt

    psf_hi = PSF.gaussian(51, 2.5, 2.0)
    psf_lo = PSF.gaussian(51, 5.0, 4.0)

    rphi = CircularApertureProfile(psf_hi.array, name='hi')
    rplo = CircularApertureProfile(psf_lo.array, name='lo')
    rplo.plot()
    rplo.plot_other(rphi)

#    basis = gauss_hermite_basis(3, [2], 51)
    basis = multi_gaussian_basis([1.0, 2.0, 3.0, 4.0, 8.0, 16.0], 51)
    kernel = psf_hi.matching_kernel_basis(psf_lo, basis)
    conv = _convolve2d(psf_hi.array, kernel)
    
    plt.imshow(psf_hi.array,vmin=-1e-3,vmax=1e-3)
    plt.imshow(basis[:,:,1],vmin=-1e-3,vmax=1e-3)
    plt.imshow(kernel,vmin=-1e-3,vmax=1e-3)

    rpconv = CircularApertureProfile(conv, name='conv')
    rplo.plot()
    rplo.plot_other(rpconv)

#    np.testing.assert_allclose(conv, psf_lo.array, rtol=0, atol=4e-2)
    fname = tmp_path / "psf_kernel_basis.png"
    save_psf_diagnostic(fname, psf_hi.array, psf_lo.array, kernel)

    assert fname.exists()


def test_effective_psf():
    return 

    import matplotlib.pyplot as plt
    from mophongo.psf import DrizzlePSF, EffectivePSF

    sw_filter = 'F090W'
    lw_filter = 'F444W'

    epsf = EffectivePSF()
    epsf.load_jwst_stdpsf(nircam_sw_filters=[sw_filter], nircam_lw_filters=[lw_filter])

    fig, axes = plt.subplots(2,5, figsize=(1.5*5, 1.5*2.2), sharex=True, sharey=True)

    # Detector coords
    x, y = 1024, 1024

    for j, module in enumerate('AB'):
        for i in range(4):
            key = f"STDPSF_NRC{module}{i+1}_{sw_filter}"
            prf = epsf.get_at_position(x=x, y=y, filter=key)
            ax = axes[j][i]
            ax.imshow(np.log10(prf), cmap='RdYlBu')
            ax.grid()

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.05, 0.05, f"{module}{i+1}", ha="left", va="bottom", transform=ax.transAxes, fontsize=7)

        key = f"STDPSF_NRC{module}LONG_{lw_filter}"
        prf = epsf.get_at_position(x=x, y=y, filter=key)
        ax = axes[j][4]
        ax.imshow(np.log10(prf), cmap='RdYlBu')
        ax.grid()

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.05, 0.05, f"{module}LONG", ha="left", va="bottom", transform=ax.transAxes, fontsize=7)

    axes[0][0].text(0.95, 0.05, sw_filter, ha="right", va="bottom", transform=axes[0][0].transAxes, fontsize=7)
    axes[0][4].text(0.95, 0.05, lw_filter, ha="right", va="bottom", transform=axes[0][4].transAxes, fontsize=7)

    fig.tight_layout(pad=1)
#%%

import pytest

@pytest.mark.skipif(1, reason="uses external data -> make smaller test")
def test_drizzle_psf():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.utils.data import download_file
    from mophongo.psf import DrizzlePSF, PSF
    from astropy.nddata import Cutout2D
    from astropy.io import fits
    import importlib
    import mophongo.psf
    importlib.reload(mophongo.psf)
    from mophongo.psf import DrizzlePSF, PSF
    from mophongo.templates import _convolve2d
    from scipy.ndimage import shift  

    filt = 'F770W'
    psf_dir = '/Users/ivo/Astro/PROJECTS/JWST/PSF/'
    if not Path(psf_dir).exists():
        pytest.skip('PSF data not available')
    filter_regex = f"STDPSF_MIRI_{filt}_EXTENDED"
#    filter_regex = f"STDPSF_NRC.._{filt}_EXTENDED"
#    filter_regex = f"STDPSF_NRC.._{filt}"
    verbose = True
    # Coordinates of a demo star
#    ra, dec = 34.30421, -5.1221624
#    ra, dec = 34.298227, -5.1262533
    ra, dec = 34.295937, -5.1294261
    size=201

    # DJA mosaic file
    # A log of the exposures that contribute to the mosaic is in drz_file.replace("_drz_sci.fits.gz", "_wcs.csv")
    data_dir = Path(__file__).resolve().parent.parent / "data"
    drz_file = str(data_dir / f"uds-test-{filt.lower()}_sci.fits")
    csv_file = str(data_dir / f"uds-test-{filt.lower()}_wcs.csv")

    # The target mosaic and its output WCS doesn't necessarily have to be the same drz_file
    dpsf = DrizzlePSF(driz_image=drz_file,csv_file=csv_file)
    dpsf.epsf_obj.load_jwst_stdpsf(local_dir=psf_dir, filter_pattern=filter_regex,verbose=verbose)

    # register on small cutout
    cutout_reg = dpsf.get_driz_cutout(ra,dec,size=15,verbose=verbose, recenter=True)
    pos_drz, cutout_reg_data, psf_data = dpsf.register(cutout_reg, filter_regex, verbose=verbose)

    # full size 
    cutout = dpsf.get_driz_cutout(ra,dec,size=size, verbose=verbose, recenter=True)
    cutout_data = cutout.data
    
    psf_hdu = dpsf.get_psf(
        ra=pos_drz[0],  dec=pos_drz[1],
        filter=filter_regex,  wcs_slice=cutout.wcs,
        kernel=dpsf.driz_header["KERNEL"],  pixfrac=dpsf.driz_header["PIXFRAC"],
        verbose=False, 
    )
    psf_data = psf_hdu[1].data

    Rnorm_as = 1.5
    ########
    # Scale the PSF model to the data
    mask = np.hypot(*np.indices(cutout_data.shape) - cutout_data.shape[0]//2) < (Rnorm_as / dpsf.driz_pscale)
    scl = (cutout.data * psf_data)[mask].sum() / (psf_data[mask]**2).sum()

    from mophongo.utils import multi_gaussian_basis, gauss_hermite_basis, CircularApertureProfile

    basis = multi_gaussian_basis([1.0, 2.0, 3.0, 4.0, 6.0], cutout_data.shape[0])
    psfd = PSF.from_array(psf_data)
    kernel = psfd.matching_kernel_basis(cutout_data, basis)
#    kernel = psfd.matching_kernel(cutout_data, recenter=False,window=mophongo.psf.TukeyWindow(0.6))
    conv = _convolve2d(psf_data, kernel)

 
    return 

#%%
    Rmax = cutout_data.shape[0] // 2    
    norm_radius = 1.5 / dpsf.driz_pscale
    Rmax_plot = 1.0

    # Center coordinates
    max_radius = np.minimum(cutout_data.shape[0] // 2, Rmax)
#    N = cutout_data.shape[0] // 2
    xycen = cutout.input_position_cutout

    # Define radii arrays (in pixels)
 #   max_radius = min(N, size//2)  # or whatever max radius you want
    radial_edges = np.linspace(0, max_radius, 101)  # 100 bins for RadialProfile
    cog_radii = np.linspace(0.5, max_radius, 100)    # must be >0 for CurveOfGrowth

    # Radial profile (azimuthal average in annuli)
    rp_star = RadialProfile(cutout_data, xycen, radial_edges)
    rp_psf  = RadialProfile(psf_data, xycen, radial_edges)

    # Normalize at R=1.0 arcsec
    star_norm = rp_star.profile[rp_star.radius >= norm_radius][0]
    psf_norm  = rp_psf.profile[rp_psf.radius >= norm_radius][0]
    star_profile = rp_star.profile / star_norm
    psf_profile  = rp_psf.profile / psf_norm

    # Curve of growth (aperture sum vs radius)
    cog_star = CurveOfGrowth(cutout_data, xycen, cog_radii)
    cog_psf  = CurveOfGrowth(psf_data, xycen, cog_radii)

    cog_star.normalize()
    cog_psf.normalize()
    cog_star.profile /= cog_star.profile[cog_star.radii >= norm_radius][0]
    cog_psf.profile  /= cog_psf.profile[cog_star.radii >= norm_radius][0]

    # EE radii
    ee20_star = cog_star.calc_radius_at_ee(0.2)
    ee80_star = cog_star.calc_radius_at_ee(0.8)
    ee20_psf  = cog_psf.calc_radius_at_ee(0.2)
    ee80_psf  = cog_psf.calc_radius_at_ee(0.8)

    # Ratio
    ratio_cog = star_cog / psf_cog

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # Panel 1: Radial profile
    axes[0].plot(rp_star.radius * dpsf.driz_pscale, star_profile, 'k-', label='Star')
    axes[0].plot(rp_psf.radius * dpsf.driz_pscale, psf_profile, 'r-', label='PSF')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Radius [arcsec]')
    axes[0].set_ylabel('Normalized Profile')
    axes[0].legend()
    axes[0].set_title(f'Radial Profile\n{filt}')
    axes[0].set_xlim(0, Rmax_plot  )

    # Panel 2: Curve of Growth
    axes[1].plot(cog_star.radius * dpsf.driz_pscale, cog_star.profile , 'k-', label='Star')
    axes[1].plot(cog_psf.radius * dpsf.driz_pscale, cog_psf.profile, 'r-', label='PSF')
    axes[1].set_xlabel('Radius [arcsec]')
    axes[1].set_ylabel('Encircled Energy')
    axes[1].set_title(f'Curve of Growth\n{filt}')
    axes[1].legend()
    axes[1].axvline(ee20_star * dpsf.driz_pscale, color='k', ls=':', label='Star EE20%')
    axes[1].axvline(ee80_star * dpsf.driz_pscale, color='k', ls='--', label='Star EE80%')
    axes[1].axvline(ee20_psf * dpsf.driz_pscale, color='r', ls=':', label='PSF EE20%')
    axes[1].axvline(ee80_psf * dpsf.driz_pscale, color='r', ls='--', label='PSF EE80%')
    axes[1].annotate(f'EE(20%)={ee20_star*dpsf.driz_pscale:.3f}"', (ee20_star*dpsf.driz_pscale, 0.2), color='k')
    axes[1].annotate(f'EE(80%)={ee80_star*dpsf.driz_pscale:.3f}"', (ee80_star*dpsf.driz_pscale, 0.8), color='k')
    axes[1].annotate(f'EE(20%)={ee20_psf*dpsf.driz_pscale:.3f}"', (ee20_psf*dpsf.driz_pscale, 0.2-0.1), color='r')
    axes[1].annotate(f'EE(80%)={ee80_psf*dpsf.driz_pscale:.3f}"', (ee80_psf*dpsf.driz_pscale, 0.8-0.1), color='r')
    axes[1].set_xlim(0, Rmax_plot  )

    # Panel 3: Ratio
    axes[2].plot(cog_star.radius * dpsf.driz_pscale, ratio_cog, 'b-')
    axes[2].axhline(1.0, color='k', ls='--', label='Unity')
    axes[2].set_xlabel('Radius [arcsec]')
    axes[2].set_ylabel('Star/PSF COG Ratio')
    axes[2].set_ylim(0.8, 1.2)
    axes[2].set_title('COG Ratio')
    axes[2].set_xlim(0, Rmax_plot  )

    fig.tight_layout()
    plt.show()

    return

#%%
    # Data centroid
    yp, xp = np.indices(cutout_hdu[0].data.shape)
    R = np.sqrt((xp-N)**2 + (yp-N)**2)
    mask = cutout_hdu[0].data > np.nanpercentile(cutout_hdu[0].data[cutout_hdu[0].data > 0], 10)
    mask &= R < 20

    sci_norm = cutout_hdu[0].data[mask].sum()
    xi = (xp * cutout_hdu[0].data)[mask].sum() / sci_norm
    yi = (yp * cutout_hdu[0].data)[mask].sum() / sci_norm

    ri, di = np.squeeze(wcs_slice.all_pix2world([xi], [yi], 0))

    fkey = f"STDPSF_MIRI_{dpsf.driz_header['FILTER']}_EXTENDED"

    psf_hdu = dpsf.get_psf(
        ra=ri,
        dec=di,
        filter=fkey,
        wcs_slice=wcs_slice,
        kernel=dpsf.driz_header['KERNEL'],
        pixfrac=dpsf.driz_header['PIXFRAC'],
        verbose=False,
        npix=npix,
    )

    psf_data = psf_hdu[1].data
    psf_norm = psf_data[mask].sum()
    pxi = (xp * psf_data)[mask].sum() / psf_norm
    pyi = (yp * psf_data)[mask].sum() / psf_norm
    xr, yr = xi, yi

    # Recenter ePSF to match data centroid
    for iter in range(3):
        ri, di = np.squeeze(wcs_slice.all_pix2world([xr], [yr], 0))

        psf_hdu = dpsf.get_psf(
            ra=ri,
            dec=di,
            filter=fkey,
            wcs_slice=wcs_slice,
            kernel=dpsf.driz_header['KERNEL'],
            pixfrac=dpsf.driz_header['PIXFRAC'],
            verbose=False,
            npix=npix,
        )
        
        psf_data = psf_hdu[1].data
        psf_norm = psf_data[mask].sum()
        pxi = (xp * psf_data)[mask].sum() / psf_norm
        pyi = (yp * psf_data)[mask].sum() / psf_norm
        xr += xi - pxi
        yr += yi - pyi
        
        print(pxi - xi, pyi - yi)

    ########
    # Scale the PSF model to the data
    scl = (cutout_hdu[0].data * psf_data)[mask].sum() / (psf_data[mask]**2).sum()

    ########
    # Make a plot showing the results
    fig, axes = plt.subplots(1,4,figsize=(12,3.4), sharex=False, sharey=False)

    # background for log scale
    offset = 2e-5
    kws = dict(vmin=-5.2, vmax=-2, cmap='bone_r')

    axes[0].imshow(np.log10(cutout_hdu[0].data/scl + offset), **kws)
    axes[1].imshow(np.log10(psf_data + offset), **kws)
    axes[2].imshow(np.log10(cutout_hdu[0].data/scl - psf_data + offset), **kws)

    for ax in axes[:3]:
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xticks([0, N*2+1]); ax.set_yticks([0, N*2+1])

    for i, label in enumerate([os.path.basename(drz_file), fkey, 'Residual']):
        axes[i].text(
            0.5, -0.02,
            label,
            ha='center', va='top', fontsize=7,
            transform=axes[i].transAxes
        )

    # Plot radial profile
    R = np.sqrt((xp-xi)**2 + (yp-yi)**2)
    ax = axes[3]
    nz = cutout_hdu[0].data != 0
    ps = dpsf.driz_pscale

    ax.scatter(R[nz]*ps, (cutout_hdu[0].data / scl + offset)[nz], alpha=0.1, color='k', label='Data')
    ax.scatter(R[nz]*ps, (psf_data + offset)[nz], alpha=0.1, color='r', label='ePSF model')
    ax.semilogy()
    ax.legend(loc='upper right')

    # Reference radii
    for rpix, rc in zip([0.4/ps, 1.2/ps], ["coral", "magenta"]):
        sr = utils.SRegion(f"circle({xi},{yi},{rpix})", wrap=False, ncirc=256)
        axes[-1].vlines([rpix*ps], 1.e-10, 10, color=rc, linestyle='-', alpha=0.5, lw=1.5)
        for ax in axes[:3]:
            # sr.add_patch_to_axis(ax, fc='None', ec='w', linestyle='-', alpha=0.5)
            sr.add_patch_to_axis(ax, fc='None', ec=rc, linestyle='-', alpha=0.5, lw=1.5)

    ax = axes[-1]
    ax.set_ylim(1.e-5, 3e-2)
    ax.set_xlim(-3*ps, 40*ps)
    ax.set_yticklabels([])
    ax.set_xlabel(r'$R$, arcsec')

    fig.tight_layout(pad=0.5)


# %%
