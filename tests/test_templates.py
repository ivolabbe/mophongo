#%%
import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates, Template
from utils import make_simple_data, save_template_diagnostic
import pytest
from astropy.wcs import WCS

def test_extract_templates_sizes_and_norm(tmp_path):
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(seed=5,nsrc=15, size=51, ndilate=2, peak_snr=3)
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpl = Templates()
    tmpl.extract_templates(images[0], segmap, list(zip(catalog["x"], catalog["y"])))
    templates_hires = tmpl._templates
    templates = tmpl.convolve_templates(kernel, inplace=False)

    assert len(templates) == len(catalog['flux_true'])

    hires = images[0]
    for tmpl_obj in templates:
        np.testing.assert_allclose(tmpl_obj.data.sum(), 1.0, rtol=1e-5)
        slo = tmpl_obj.slices_original
        label = segmap[tmpl_obj.position_original]
        hi_cut = hires[slo] * (segmap[slo] == label)
        assert np.all(hi_cut[segmap[slo] != label] == 0)

    fname = tmp_path / "templates.png"
    # Since templates are already PSF-matched, pass them as both arguments
    save_template_diagnostic(fname, templates_hires[:5], templates[:5], segmap=segmap, catalog=catalog)
    assert fname.exists()


def test_template_extension_methods(tmp_path):
    images, segmap, catalog, psfs, _, _ = make_simple_data(ndilate=3, nsrc=20, size=101, peak_snr=1)
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpl = Templates()
    tmpl.extract_templates(images[0], segmap, list(zip(catalog["x"], catalog["y"])))
    templates_hires = tmpl._templates
#    templates = tmpl.convolve_templates(kernel, inplace=False)

    templates_psf = tmpl.extend_with_psf_wings(psf_hi.array,
                                              radius_factor=1.5,
                                              inplace=False)

    fname_moff = tmp_path / "extension_psf.png"
    save_template_diagnostic(
        fname_moff,
        templates_hires[:5],
        templates_psf[:5],
        segmap=segmap,
        catalog=catalog,
    )
    assert fname_moff.exists()

    return

def test_kernel_padding():
import numpy as np
from mophongo.psf import PSF
from mophongo.astrometry import make_gradients
from mophongo.templates import Templates, Template
from utils import make_simple_data, save_template_diagnostic
import pytest
from astropy.wcs import WCS
from matplotlib import pyplot as plt

#    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=150, size=301, peak_snr=0.5, seed=11)
images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=100, size=301, peak_snr=0.5, seed=11, border_size=40)

dirac = lambda n: ((np.arange(n)[:,None] == n//2) & (np.arange(n) == n//2)).astype(float)

from scipy.ndimage import shift as nd_shift, map_coordinates
h, w = images[0].shape
y, x = np.mgrid[0:h, 0:w]

# @@ fitting is ok, but goes nuts when kernel is applied. 
# probably to do with enlarging the template and offsets of the positions 
positions = list(zip(catalog["x"], catalog["y"]))
tmpls = Templates.from_image(truth, segmap, positions, kernel=None)

tmpls_c = Templates.from_image(truth, segmap, positions, kernel=dirac(3))

i = 0
for (k1,v1),(d2,v2) in zip(tmpls[i].__dict__.items(), tmpls_c[i].__dict__.items()):
    if k1 != 'data':
        print(k1,v1,v2)


offset = 3e-5
scl = images[1].sum()
kws = dict(vmin=-5.3, vmax=-1.5, cmap='bone_r', origin='lower')
plt.imshow(np.log10(tmpls[i].data + offset), **kws)
plt.imshow(np.log10(tmpls_c[i].data+ offset), **kws)

gx,gy = make_gradients(tmpls[0:1])

gx_c,gy_c = make_gradients(tmpls_c[0:1])

plt.imshow(gx[i].data,vmin=-1,vmax=1)
plt.imshow(gx_c[i].data,vmin=-1,vmax=1)

# %%
