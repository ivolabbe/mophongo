import numpy as np
from mophongo.templates import Templates
from utils import save_template_diagnostic


def test_extract_templates_sizes_and_norm(tmp_path):
    # simple 7x7 high-res image with two sources
    hires = np.zeros((7, 7))
    segmap = np.zeros_like(hires, dtype=int)

    # source 1: 3x3 square around center with flux 9
    segmap[2:5, 2:5] = 1
    hires[2:5, 2:5] = 1.0

    # source 2: 2x2 square bottom-right with flux 4
    segmap[5:7, 5:7] = 2
    hires[5:7, 5:7] = 2.0

    # kernel normalized to sum 1
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel /= kernel.sum()

    positions = [(3, 3), (5.5, 5.5)]
    tmpl = Templates()
    templates = tmpl.extract_templates(hires, segmap, positions, kernel)

    # check two templates
    assert len(templates) == 2

    # template 1 bounding box expected (1,6,1,6) -> size 5x5
    t1 = templates[0]
    assert t1.array.shape == (5, 5)
    assert t1.bbox == (1, 6, 1, 6)
    np.testing.assert_allclose(t1.array.sum(), 1.0, rtol=1e-6)

    # template 2 bounding box expected to be clipped -> (4,7,4,7) -> size 3x3
    t2 = templates[1]
    assert t2.array.shape == (3, 3)
    assert t2.bbox == (4, 7, 4, 7)
    np.testing.assert_allclose(t2.array.sum(), 0.875, rtol=1e-6)
    fname = tmp_path / "templates.png"
    save_template_diagnostic(fname, hires, templates)
    assert fname.exists()
