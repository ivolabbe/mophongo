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
    np.testing.assert_allclose(psf.array.sum(), 1.0,  rtol=0, atol=2e-3)


def test_psf_matching_kernel_properties(tmp_path):
    _, _, _, psfs, _, _ = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    assert kernel.shape == psf_lo.array.shape

    # Pad psf_hi to kernel shape for diagnostics and convolution
    hi_pad = pad_to_shape(psf_hi.array, kernel.shape)
    conv = _convolve2d(hi_pad, kernel)
    np.testing.assert_allclose(conv, psf_lo.array, rtol=0, atol=2e-3)
    fname = tmp_path / "psf_kernel.png"
    save_psf_diagnostic(fname, hi_pad, psf_lo.array, kernel)
    assert fname.exists()


def test_psf_moffat_fit(tmp_path):
    psf = PSF.moffat(31, fwhm_x=3.2, fwhm_y=4.5, beta=2.8, theta=0.3)
    params = psf.fit_moffat()
    assert np.isclose(params.beta, 2.8, rtol=0.1)
    psf_fit = PSF.moffat(psf.array.shape, params.fwhm_x, params.fwhm_y, params.beta, params.theta)
    fname = tmp_path / "psf_fit_moffat.png"
    save_psf_fit_diagnostic(fname, psf.array, psf_fit.array)
    assert fname.exists()
    np.testing.assert_allclose(psf_fit.array, psf.array, rtol=0, atol=5e-2)


def test_psf_gaussian_fit(tmp_path):
    psf = PSF.gaussian(31, fwhm_x=2.5, fwhm_y=3.0, theta=0.2)
    params = psf.fit_gaussian()

    psf_fit = PSF.gaussian(psf.array.shape, params.fwhm_x, params.fwhm_y, params.theta)
    fname = tmp_path / "psf_fit_gaussian.png"
    save_psf_fit_diagnostic(fname, psf.array, psf_fit.array)
    assert fname.exists()
    np.testing.assert_allclose(psf_fit.array, psf.array, rtol=0, atol=5e-2)


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
    np.testing.assert_allclose(psf.array.sum(), 1.0, rtol=0, atol=2e-3)

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

#%%
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import download_file
import grizli.galfit.psf
from grizli import utils
import mophongo
from mophongo.utils import read_wcs_csv
from mophongo.psf import DrizzlePSF, EffectivePSF, PSF
import matplotlib.pyplot as plt
from sregion import SRegion
from astropy.stats import mad_std
import astropy.units as u
from astropy.io import fits
from photutils.aperture import CircularAperture

# from https://github.com/gbrammer/grizli/pull/268
# but with withiout grizli dependencies
def test_effective_psf():

    sw_filter = 'F090W'
    lw_filter = 'F444W'

    epsf = EffectivePSF()
    epsf.load_jwst_stdpsf(local_dir='/Users/ivo/Astro/PROJECTS/JWST/PSF/PSF', pattern='STDPSF_NRC*(?:F200W,F444W)*')
    #    epsf.load_jwst_stdpsf(nircam_sw_filters=[sw_filter], nircam_lw_filters=[lw_filter])

    fig, axes = plt.subplots(2,
                             5,
                             figsize=(1.5 * 5, 1.5 * 2.2),
                             sharex=True,
                             sharey=True)

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
            ax.text(0.05,
                    0.05,
                    f"{module}{i+1}",
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=7)

        key = f"STDPSF_NRC{module}LONG_{lw_filter}"
        prf = epsf.get_at_position(x=x, y=y, filter=key)
        ax = axes[j][4]
        ax.imshow(np.log10(prf), cmap='RdYlBu')
        ax.grid()

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.05,
                0.05,
                f"{module}LONG",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                fontsize=7)

    axes[0][0].text(0.95,
                    0.05,
                    sw_filter,
                    ha="right",
                    va="bottom",
                    transform=axes[0][0].transAxes,
                    fontsize=7)
    axes[0][4].text(0.95,
                    0.05,
                    lw_filter,
                    ha="right",
                    va="bottom",
                    transform=axes[0][4].transAxes,
                    fontsize=7)

    fig.tight_layout(pad=1)

def test_drizzle_psf():
    # DJA mosaic file
    # A log of the exposures that contribute to the mosaic is in drz_file.replace("_drz_sci.fits.gz", "_wcs.csv")
    drz_file = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/smacs0723-grizli-v7.4-f770w_drz_sci.fits.gz"

    # Cache downloads from remote URL
    cached_drz_file = download_file(drz_file, cache=True)
    cached_csv_file = download_file(drz_file.split('_dr')[0] + "_wcs.csv", cache=True)

    info = read_wcs_csv(
        cached_drz_file,
        csv_file=cached_csv_file,
    )

    dpsf = DrizzlePSF(
        flt_files=None,
        driz_image=cached_drz_file,
        info=info
    )

    #####################
    # Coordinates of a demo star
    ra, dec = 110.7962901, -73.4660494

    # Cutout size in mosaic pixels
    N = 101
    _, _ , cutout_hdu = dpsf.get_driz_cutout(ra=ra, dec=dec, N=N, get_cutout=True)

    npix = int(np.round((N * dpsf.driz_pscale / 0.11)))
    print(f'Mosaic pixels: {N},  Detector pixels: {npix} (max=37.5)')

    # Data cutout for plotting
    _, _ , cutout_hdu = dpsf.get_driz_cutout(ra=ra, dec=dec, N=N, get_cutout=True)

    # WCS cutout for the output PSF
    slx, sly, wcs_slice = dpsf.get_driz_cutout(ra=ra, dec=dec, N=N, get_cutout=False)

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
    for iter in range(0):
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
        sr = SRegion(f"circle({xi},{yi},{rpix})", wrap=False, ncirc=256)
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
def test_NIRCam_psf():
    # A log of the exposures that contribute to the mosaic is in drz_file.replace("_drz_sci.fits.gz", "_wcs.csv")

    filt = 'F770W'
    filter_pattern = f"STDPSF_MIRI.*{filt}.*EXTENDED"
    fkey = f"STDPSF_MIRI_{filt}_EXTENDED"

    #    filt = 'F444W'
    #    filter_pattern = f"STDPSF_NRC.*{filt}*"
    #    filter_pattern=f"STDPSF_NRC.*({filter}|F444W).*"
    #    fkey = f"STDPSF_NRC*_{filt}"
    psf_dir = '/Users/ivo/Astro/PROJECTS/JWST/PSF/PSF'
    drz_file = "/Users/ivo/Desktop/mophongo/mophongo/data/uds-test-" + filt.lower(
    ) + "_sci.fits"

    #####################
    # Coordinates of a demo star
    ra, dec = 34.304205, -5.1221591
    ra, dec = 34.298222, -5.1262568  # bright star for 770 and wings
#    ra, dec = 34.295953, -5.1293929
    size = 201

    # Cache downloads from remote URL
    #    cached_drz_file = download_file(drz_file, cache=True)
    #    cached_csv_file = download_file(drz_file.split('_dr')[0] + "_wcs.csv", cache=True)
    cached_drz_file = drz_file
    cached_csv_file = drz_file.split('_sci')[0] + "_wcs.csv"

    info = read_wcs_csv(cached_drz_file, csv_file=cached_csv_file)

    dpsf = DrizzlePSF(flt_files=None, driz_image=cached_drz_file, info=info)
    dpsf.epsf_obj.load_jwst_stdpsf(local_dir=psf_dir,
                                   filter_pattern=filter_pattern)

    # image
    with fits.open(dpsf.driz_image) as im:
        cutout = PSF.from_data(im[0].data, (ra * u.deg, dec * u.deg),
                               wcs=dpsf.driz_wcs,
                               search_boxsize=11,
                               size=size,
                               verbose=True)

    cra, cdec = cutout.wcs.all_pix2world([cutout.pos], 0)[0]

    psf_hdu = dpsf.get_psf(
        ra=cra,
        dec=cdec,
        filter=fkey,
        wcs_slice=cutout.wcs,
        kernel=dpsf.driz_header['KERNEL'],
        pixfrac=dpsf.driz_header['PIXFRAC'],
        verbose=False,
        npix=51,
    )

    mask = CircularAperture(cutout.pos,
                            r=18).to_mask(method='center').to_image(
                                (size, size)) > 0
    mask_large = CircularAperture(cutout.pos,
                                  r=30).to_mask(method='center').to_image(
                                      (size, size)) > 0
    mask_small = CircularAperture(cutout.pos,
                                  r=8).to_mask(method='center').to_image(
                                      (size, size)) > 0

    from skimage.morphology import dilation, square, disk

    psf = psf_hdu[1].data.copy()
    psf = _convolve2d(psf,  PSF.gaussian(11, 3., 3.).data)
    mask_star = dilation(psf > 0,
                         disk(1.))  # dilation to include diffraction spikes
    #   bg_off = np.median(
    #      psf[mask * ~mask_star])  # set level between diffraction spikes to zero
    #   bg_off = np.percentile(psf[mask * ~mask_star], 10)
    bg_off = 0.0
    plt.imshow(psf, vmin=-0.001, vmax=0.001, cmap='gray')
    plt.imshow(psf * mask, vmin=-0.001, vmax=0.001, cmap='gray')
    plt.imshow(psf * ~mask_star, vmin=-0.001, vmax=0.001, cmap='gray')
    print(f"Background offset: {bg_off:.3e}", len(psf[mask * ~mask_star]))

    psf -= bg_off
    #    psf[~mask] = 0.0

    scl = (cutout.data * psf)[mask].sum() / (psf[mask]**2).sum()
    #    scl = (cutout.data *
    #           psf)[mask * ~mask_small].sum() / (psf[mask * ~mask_small]**2).sum()
    psf *= scl

    #   fits.writeto('testpsf444.fits', psf, psf_hdu[0].header, overwrite=True)

    _ = PSF.from_data(psf,
                      cutout.pos,
                      search_boxsize=11,
                      size=201,
                      verbose=True)

    if 0:
        with fits.open(
                '/Users/ivo/Astro/PROJECTS/MINERVA/data/v1.0/f444w_psf_norm.fits'
        ) as hdul:
            psf_stack = PSF.from_data(hdul[0].data,
                                      ((hdul[0].data.shape[0] - 1) / 2,
                                       (hdul[0].data.shape[1] - 1) / 2),
                                      search_boxsize=11,
                                      size=201,
                                      verbose=True)

        scl_psf_stack = (cutout.data * psf_stack.data)[mask].sum() / (
            psf_stack.data[mask]**2).sum()
        scl_psf_stack = (cutout.data * psf_stack.data)[mask * ~mask_small].sum(
        ) / (psf_stack.data[mask * ~mask_small]**2).sum()
        psf_ext = psf_stack.data * scl_psf_stack

        from scipy.ndimage import rotate
        from scipy.optimize import minimize_scalar, minimize
        from scipy.ndimage import shift

        def loss_fn(theta, psf_large, psf_core, mask):
            theta = float(np.atleast_1d(theta)[0])
            psf_rot = rotate(psf_large, theta, reshape=False, order=3)
            diff = (psf_rot - psf_core)[mask]
            plt.imshow((psf_rot - psf_core) * mask,
                       vmin=-0.001,
                       vmax=0.001,
                       cmap='gray')
            print(theta, np.sum(diff**2))
            return np.sum(diff**2)

        res = minimize_scalar(loss_fn,
                              args=(psf_ext, psf, mask * ~mask_small),
                              bounds=(-5, 15),
                              method='bounded')

        # res = minimize_scalar(loss_fn, args=(psf_ext, psf, mask * ~mask_small), bounds=(-5, 15), method='bounded')

        print("Optimal theta:", res.x)
        psf_ext_rot = rotate(psf_ext, res.x, reshape=False, order=3)
        psf[~mask] = psf_ext_rot[~mask] * 0.7  # scale to match the core

    # from lmfit import minimize, Parameters
    # def loss_fn_shift_lmfit(params, image, theta, mask=None, plot=False):
    #     dx = params['dx'].value
    #     dy = params['dy'].value
    #     shifted = shift(image, shift=(dy, dx), order=3)
    #     rotated = rotate(shifted, theta, reshape=False, order=1)
    #     diff = shifted - rotated
    #     if mask is not None:
    #         diff = diff[mask]
    #     loss = np.sum(diff**2)
    #     if plot:
    #         plt.imshow(diff if mask is None else diff * mask, cmap='gray')
    #         plt.title(f'dx={dx:.2f}, dy={dy:.2f}, loss={loss:.2e}')
    #         plt.show()
    #     return diff.ravel()

    # params = Parameters()
    # params.add('dx', value=1.0, min=-10, max=10)
    # params.add('dy', value=1.0, min=-10, max=10)

    # res = minimize(loss_fn_shift_lmfit, params,
    #                args=(cutout.data, 180, None, False),
    #                method='leastsq')

    # print("Optimal shift (lmfit):", result.params['dx'].value, result.params['dy'].value)

    from matplotlib.colors import SymLogNorm
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=False, sharey=False)
    axes = axes.flatten()
    offset, rms = np.median(cutout.data), mad_std(cutout.data)
    norm = SymLogNorm(linthresh=offset + 3 * rms,
                      vmin=offset - 3 * rms,
                      vmax=cutout.data.max())
    kws = dict(norm=norm,
               cmap='bone_r',
               origin='lower',
               interpolation='nearest')

    axes[0].imshow(cutout.data, **kws)
    axes[1].imshow(psf, **kws)
    axes[2].imshow(psf_star.data, **kws)
    axes[4].imshow(cutout.data - psf, **kws)
    axes[5].imshow(cutout.data - _convolve2d(psf,
                                             PSF.gaussian(11, 3, 3).data),
                   **kws)
#    axes[5].imshow(cutout.data - psf_stack.data * scl_psf_stack, **kws)

# %%
