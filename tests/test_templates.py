import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates
from utils import make_simple_data, save_template_diagnostic

def test_extract_templates_sizes_and_norm(tmp_path):
    images, segmap, catalog, psfs, truth_img = make_simple_data()
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpl = Templates()
    templates = tmpl.extract_templates(
        images[0], segmap, list(zip(catalog["y"], catalog["x"])), kernel
    )

    assert len(templates) == len(catalog['flux_true'])

    hires = images[0]
    for tmpl_obj in templates:
        np.testing.assert_allclose(tmpl_obj.array.sum(), 1.0, rtol=1e-5)

        y0, y1, x0, x1 = tmpl_obj.bbox
        label = segmap[int((y0 + y1) / 2), int((x0 + x1) / 2)]
        hi_cut = hires[y0:y1, x0:x1] * (segmap[y0:y1, x0:x1] == label)
        assert np.all(hi_cut[segmap[y0:y1, x0:x1] != label] == 0)

    fname = tmp_path / "templates.png"
    save_template_diagnostic(fname, images[0], templates[:5])
    assert fname.exists()
