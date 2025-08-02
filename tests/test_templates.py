#%%
import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates, Template, per_source_chi2
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
    return


def test_add_component_clones_parent():
    tmpl = Template(np.ones((3, 3)), (1, 1), (3, 3), label=7)
    tlist = Templates()
    tlist._templates.append(tmpl)
    new_data = np.full((3, 3), 2.0)
    comp = tlist.add_component(tmpl, new_data, "psf_core")

    assert comp is tlist[-1]
    assert comp.component == "psf_core"
    assert comp.parent_id == tmpl.id
    np.testing.assert_array_equal(comp.data, new_data)


def test_per_source_chi2_simple():
    residual = np.ones((5, 5))
    sigma = np.ones((5, 5))
    t1 = Template(np.ones((3, 3)), (2, 2), (3, 3), label=1)
    t2 = Template(np.ones((3, 3)), (2, 2), (3, 3), label=2)
    chis = per_source_chi2(residual, sigma, [t1, t2])
    assert np.allclose(chis, 1.0)
