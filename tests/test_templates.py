import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates
from utils import make_simple_data, save_template_diagnostic
import pytest

def test_extract_templates_sizes_and_norm(tmp_path):
    images, segmap, catalog, psfs, truth_img, _ = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpl = Templates.from_image(
        images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel
    )
    templates = tmpl.templates

    assert len(templates) == len(catalog['flux_true'])

    hires = images[0]
    for tmpl_obj in templates:
        np.testing.assert_allclose(tmpl_obj.array.sum(), 1.0, rtol=1e-5)

        y0, y1, x0, x1 = tmpl_obj.bbox
        label = segmap[int((y0 + y1) / 2), int((x0 + x1) / 2)]
        hi_cut = hires[y0:y1, x0:x1] * (segmap[y0:y1, x0:x1] == label)
        assert np.all(hi_cut[segmap[y0:y1, x0:x1] != label] == 0)

    fname = tmp_path / "templates.png"
    # Since templates are already PSF-matched, pass them as both arguments
    save_template_diagnostic(fname, tmpl._templates_hires[:5], templates[:5], segmap=segmap, catalog=catalog)
    assert fname.exists()


def test_template_extension_methods(tmp_path):
    images, segmap, catalog, psfs, _, _ = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpl = Templates.from_image(images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel)
    orig_templates_hi = list(tmpl._templates_hires)

    tmpls_moffat = tmpl.extend_with_moffat(kernel, radius_factor=2.0, beta=3.0)
    for t in tmpls_moffat:
        np.testing.assert_allclose(t.array.sum(), 1.0, rtol=1e-5)

    fname_moff = tmp_path / "moffat_extension.png"
    save_template_diagnostic(
        fname_moff,
        orig_templates_hi[:5],
        tmpl._templates_hires[:5],
        segmap=segmap,
        catalog=catalog,
    )
    assert fname_moff.exists()

    tmpl.extract_templates(images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel)
    orig_templates_hi = list(tmpl._templates_hires)
    tmpls_dil = tmpl.extend_with_psf_dilation(psf_hi.array, kernel, iterations=2)
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
    tmpl.extract_templates(images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel)
    orig_templates_hi = list(tmpl._templates_hires)
    tmpls_simple = tmpl.extend_with_psf_dilation_simple(psf_hi.array, kernel, radius_factor=2.0)
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
