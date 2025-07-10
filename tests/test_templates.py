#%%
import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates, Template
from utils import make_simple_data, save_template_diagnostic
import pytest

def test_extract_templates_sizes_and_norm(tmp_path):
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(seed=5,nsrc=20, size=101, ndilate=2, peak_snr=3)
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
        np.testing.assert_allclose(tmpl_obj.array.sum(), 1.0, rtol=1e-5)
        slo = tmpl_obj.slices_original
        label = segmap[tmpl_obj.position_original]
        hi_cut = hires[slo] * (segmap[slo] == label)
        assert np.all(hi_cut[segmap[slo] != label] == 0)

    fname = tmp_path / "templates.png"
    # Since templates are already PSF-matched, pass them as both arguments
    save_template_diagnostic(fname, templates_hires[:5], templates[:5], segmap=segmap, catalog=catalog)
    assert fname.exists()


def test_template_extension_methods(tmp_path):
    images, segmap, catalog, psfs, _, _ = make_simple_data(ndilate=3, nsrc=80, size=201, peak_snr=1)
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

    tmpl.extract_templates_old(images[0], segmap,
                               list(zip(catalog["x"], catalog["y"])), kernel)
    orig_templates_hi = list(tmpl._templates_hires)
    tmpls_dil = tmpl.extend_with_psf_dilation(psf_hi.array,
                                              kernel,
                                              iterations=2)
    for t in tmpls_dil:
        np.testing.assert_allclose(t.array.sum(), 1.0, rtol=1e-5)

    fname_dil = tmp_path / "dilation_extension.png"
    save_template_diagnostic(
        fname_dil,
        orig_templates_hi[:5],
        tmpl._templates_hires[:5],
        segmap=segmap,
        catalog=catalog,
    )
    assert fname_dil.exists()

    # Test simple PSF dilation extension
    tmpl.extract_templates_old(images[0], segmap,
                               list(zip(catalog["x"], catalog["y"])), kernel)
    orig_templates_hi = list(tmpl._templates_hires)
    tmpls_simple = tmpl.extend_with_psf_dilation_simple(psf_hi.array,
                                                        kernel,
                                                        radius_factor=2.0)
    #    for t in tmpls_simple:
    #        np.testing.assert_allclose(t.array.sum(), 1.0, rtol=1e-5)

    fname_simple = tmp_path / "simple_dilation_extension.png"
    save_template_diagnostic(
        fname_simple,
        orig_templates_hi[:5],
        tmpl._templates_hires[:5],
        segmap=segmap,
        catalog=catalog,
    )
    assert fname_simple.exists()

    assert len(tmpls_moffat) == len(tmpls_dil) == len(tmpls_simple)

# %%
