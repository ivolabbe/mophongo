#%%
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



def test_make_jwst_extended_grid(tmp_path):

    import matplotlib.pyplot as plt
    from photutils.psf import STDPSFGrid
    from pathlib import Path
    from mophongo import jwst_psf

    stdfile = '/Users/ivo/Astro/PROJECTS/JWST/PSF/PSF/JWST/NIRCam/LWC/STDPSF_NRCBL_F444W.fits'

    epsf = STDPSFGrid(stdfile)  # NRCAL shortwave -> adapt names
    epsf_ext = jwst_psf.make_extended_grid(epsf,
                                           Rmax=2.0,
                                           Rtaper=0.2,
                                           verbose=True)

    p = Path(stdfile)
    outdir = p.parent / 'EXTENDED'
    os.makedirs(p.parent / 'EXTENDED', exist_ok=True)

    jwst_psf.write_stdpsf(outdir / p.name.replace('.fits', '_EXTENDED.fits'),
                          epsf_ext,
                          overwrite=True,
                          verbose=True)

# do all NIRCam  
    psf_dir = '/Users/ivo/Astro/PROJECTS/JWST/PSF/PSF/JWST/NIRCam/'
    p = Path(psf_dir)
    files_dir = list(p.rglob('*.fits'))

    for f in files_dir:
        outdir = p.parent / 'EXTENDED'
        outname = outdir / f.name.replace('.fits', '_EXTENDED.fits')
        if outname.exists():
            print(f"Skipping {outname} (exists)")
            continue
        if 'EXTENDED' in f.name or 'NRC' not in f.name:
            continue

        print(f"Processing {f.name}")

        epsf = STDPSFGrid(f)
        epsf_ext = jwst_psf.make_extended_grid(epsf,
                                               Rmax=2.0,
                                               Rtaper=0.2,
                                               verbose=True)
        os.makedirs(outdir, exist_ok=True)

        if not outname.exists():
            print(f"Writing {outname}")
            jwst_psf.write_stdpsf(outname,
                                epsf_ext,
                                overwrite=True,
                                verbose=True)




    from mophongo.jwst_psf import , blend_psf, write_stdpsf

    scl_ext = 1.2
    blend_core2 = w * core_psf + (1 - w) * cutout_data * scl_ext

    N = blend_core.shape[0]//2
    r = np.hypot(*np.indices(blend_core.shape) - N)  
    bgmask = ~(core_psf > np.nanpercentile(core_psf[core_psf > 0.0],10) ) & (r < N-2)
    offset = np.nanmedian(core_psf[bgmask])

    print(np.median(core_psf[bgmask]))

    plt.imshow(core_psf * bgmask, vmin=-2e-4, vmax=2e-4)
    plt.imshow(core_psf, vmin=-2e-4, vmax=2e-4)
    plt.imshow(blend_psf, vmin=-2e-4, vmax=2e-4)
    plt.imshow(blend_core2, vmin=-2e-4, vmax=2e-4)
    plt.imshow(w, vmin=-0.01, vmax=0.01)
    plt.imshow(w * cutout_data, vmin=-0.1, vmax=0.2)
    plt.imshow((1 - w) * cutout_data, vmin=-0.0001, vmax=0.0001)
    tmp = ext_psf.copy() # * scl_ext
    #  tmp[slices] = blend_core
    tmp[slices] = (1 - w) * cutout_data
    tmp[slices] = (1 - w) * cutout_data * scl_ext + w * core_psf 
    plt.imshow(tmp, vmin=-1e-4, vmax=1e-4)

    #    plt.imshow(blend_core,vmin=-0.002,vmax=0.002)

    return

    #    Nemp, Ny, Nx = emp.data.shape                        # 25, 101, 101
    # Ensure the detector name is compatible with stpsf
    if emp.meta['detector'][-1] == 'L':
        emp.meta['detector'] = emp.meta['detector'][:-1] + '5'

    Rmax = 1.0
    emp_grid = emp

    import stpsf
    nrc = stpsf.NIRCam()
    nrc.filter   = emp.meta['filter']
    nrc.detector = emp.meta['detector']

    oversamp = emp_grid.oversampling
    grid_xy = emp_grid.grid_xypos
    det_name = emp_grid.meta.get("detector", "NRC")
    filt_name = emp_grid.meta.get("filter", "F200W")
    Nemp, Ny_emp, _ = emp_grid.data.shape
    Rcore_px = (Ny_emp - 1) // 2

    # th_raw = nrc.psf_grid(
    #         num_psfs=25,
    #         all_detectors=False,
    #         oversample=oversamp[0],
    #         fov_arcsec=2 * Rmax,
    #         verbose=True,
    # #        save=True,
    # #        overwrite=True,
    # #        outdir='/Users/ivo/Desktop/mophongo/mophongo/tests/',
    # #        outfile='test_psf_ext.fits'
    #     )

    th_raw = nrc.psf_grid(
        num_psfs=len(grid_xy),
        all_detectors=False,
        oversample=4,
        fov_pixels = 101,
    )

    from astropy.nddata import Cutout2D
    scl = emp.data[0].sum() / th_raw.data[0][152:253,152:253].sum()

    plt.imshow(emp.data[0] ,vmin=-1e-3,vmax=1e-3)
    plt.imshow(scl*th_raw.data[0][152:253,152:253],vmin=-1e-3,vmax=1e-3)
    plt.imshow(emp.data[0] - scl*th_raw.data[0][152:253,152:253],vmin=-1e-3,vmax=1e-3)


    #cutout = Cutout2D(emp.data[0],position=pos, size=th_raw.data[0].shape, mode='partial')
    #plt.imshow(cutout.data - scl*th_raw.data[0],vmin=-1e-3,vmax=1e-3)

    core_psf = emp.data[22]
    ext_psf = th_raw.data[22]

    core_shape = core_psf.shape
    pos = np.asarray(ext_psf.shape) // 2
    cutout = Cutout2D(ext_psf,position=pos, size=core_shape)
    #plt.imshow(core_psf ,vmin=-1e-3,vmax=1e-3)
    #plt.imshow(scl*cutout.data,vmin=-1e-3,vmax=1e-3)
    #plt.imshow(core_psf - scl*cutout.data,vmin=-1e-3,vmax=1e-3)

    Rtaper_px = 12
    buf_px = 2
    R_inner = Rcore_px - buf_px

    N = cutout.shape[0]//2
    yy, xx = np.indices(cutout.shape)
    r = np.hypot(yy - N, xx - N)
    w = np.zeros(cutout.shape)
    w[r <= R_inner - Rtaper_px] = 1.0
    m = (r > R_inner - Rtaper_px) & (r < R_inner)
    w[m] = 0.5 * (1 - np.cos(math.pi * (r[m] - R_inner) / Rtaper_px))

    mask_norm = r <= Rnorm_px
    scl_ext = core_psf[mask_norm].sum() / cutout.data[mask_norm].sum()

    blend_core = w * core_psf + (1 - w) * cutout.data * scl_ext

    blend_psf = ext_psf.copy()
    blend_psf[cutout.slices_original] = blend_core

    #plt.imshow(w)
    #plt.imshow(blend_core ,vmin=-1e-3,vmax=1e-3)
    #plt.imshow(blend_psf, vmin=-1e-3,vmax=1e-3)



    out_arr = np.empty((Nemp, Nfinal, Nfinal), dtype=float)
    for i in range(Nemp):
        out_arr[i] = blend_psf(
            emp_grid.data[i], th_raw.data[i], Rcore_px, Rmax_px, Rtaper_px
        )

    hdu = fits.PrimaryHDU(th_raw.render())
    hdu.header['PIXELSCL'] = nrc.pixelscale
    hdu.header['DETNAME'] = det_name
    hdu.header['FILTNAME'] = filt_name
    hdu.header['RMAX'] = Rmax

    stpsf.display_ee(fits.HDUList( [hdu]))
    jwst_psf.write_stdpsf(  'tests/test_psf_ext_nrca5.fits', th_raw,overwrite=True)


    test_emp = STDPSFGrid('tests/test_psf_ext_nrca5.fits')

    # %%
def drizzle_orig_psf():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.utils.data import download_file
    import grizli.galfit.psf
    from grizli import utils

    # DJA mosaic file
    # A log of the exposures that contribute to the mosaic is in drz_file.replace("_drz_sci.fits.gz", "_wcs.csv")
    drz_file = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/smacs0723-grizli-v7.4-f770w_drz_sci.fits.gz"

    # Cache downloads from remote URL
    cached_drz_file = download_file(drz_file, cache=True)
    cached_csv_file = download_file(drz_file.split('_dr')[0] + "_wcs.csv", cache=True)

    info = grizli.galfit.psf.DrizzlePSF._get_wcs_from_csv(
        cached_drz_file,
        csv_file=cached_csv_file,
    )

    # Initialize ePSF drizzler. 
    # The target mosaic and its output WCS doesn't necessarily have to be the same drz_file
    dpsf = grizli.galfit.psf.DrizzlePSF(
        flt_files=None,
        driz_image=cached_drz_file,
        info=info
    )

    #####################
    # Coordinates of a demo star
    ra, dec = 110.7962901, -73.4660494

    # Cutout size in mosaic pixels
    N = 100

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
