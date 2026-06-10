import numpy as np
from astropy.nddata import block_reduce

from mophongo.templates import Template, Templates
from mophongo.utils import fftconvolve


def _paste_template(tmpl: Template, shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros(shape, dtype=float)
    image[tmpl.slices_original] += tmpl.data[tmpl.slices_cutout]
    return image


def _wing_test_template(parent: np.ndarray) -> Template:
    tmpl = Template(parent, (40.0, 40.0), (8, 8), label=7)
    data = np.zeros_like(tmpl.data, dtype=float)
    data[3, 3] = 0.45
    data[3, 4] = 0.25
    data[4, 3] = 0.20
    data[4, 4] = 0.10
    tmpl.data = data
    return tmpl


def _asymmetric_template(parent: np.ndarray, position: tuple[float, float], size: tuple[int, int]) -> Template:
    tmpl = Template(parent, position, size, label=1)
    data = np.zeros_like(tmpl.data, dtype=float)
    cy, cx = np.array(data.shape) // 2
    data[cy, cx] = 1.0
    data[min(data.shape[0] - 1, cy + 1), max(0, cx - 2)] = 0.3
    tmpl.data = data
    return tmpl


def test_template_convolve_cutout_matches_global_convolution_for_odd_template_even_kernel():
    parent = np.zeros((80, 80), dtype=float)
    kernel = np.zeros((4, 4), dtype=float)
    kernel[2, 2] = 1.0
    kernel[1, 3] = 0.2
    tmpl = _asymmetric_template(parent, (20.0, 22.3), (9, 9))

    convolved = tmpl.convolve_cutout(kernel, parent_image=parent, preserve_dtype=False)
    local_model = _paste_template(convolved, parent.shape)
    global_model = fftconvolve(_paste_template(tmpl, parent.shape), kernel, mode="same")

    np.testing.assert_allclose(local_model, global_model, rtol=0, atol=1e-12)


def test_template_convolve_cutout_matches_global_convolution_for_all_parities():
    parent = np.zeros((80, 80), dtype=float)
    positions = [(20.0, 22.3), (20.25, 22.3), (20.5, 22.3), (20.75, 22.3)]
    template_sizes = [(8, 8), (9, 9), (8, 9), (9, 8)]
    kernel_sizes = [(4, 4), (5, 5)]

    for position in positions:
        for template_size in template_sizes:
            for kernel_size in kernel_sizes:
                kernel = np.zeros(kernel_size, dtype=float)
                kernel[kernel_size[0] // 2, kernel_size[1] // 2] = 1.0
                kernel[max(0, kernel_size[0] // 2 - 1), min(kernel_size[1] - 1, kernel_size[1] // 2 + 1)] = 0.2
                tmpl = _asymmetric_template(parent, position, template_size)

                convolved = tmpl.convolve_cutout(kernel, parent_image=parent, preserve_dtype=False)
                local_model = _paste_template(convolved, parent.shape)
                global_model = fftconvolve(_paste_template(tmpl, parent.shape), kernel, mode="same")

                np.testing.assert_allclose(local_model, global_model, rtol=0, atol=1e-12)


def test_template_block_projection_matches_global_native_pixel_basis():
    parent = np.zeros((80, 80), dtype=float)
    positions = [(20.0, 22.3), (20.25, 22.3), (20.5, 22.3), (20.75, 22.3)]
    template_sizes = [(8, 8), (9, 9), (8, 9), (9, 8)]
    factor = 2

    for position in positions:
        for template_size in template_sizes:
            tmpl = _asymmetric_template(parent, position, template_size)
            projected = tmpl.project_to_block_replicated_grid(
                factor,
                parent_image=parent,
                preserve_dtype=False,
            )

            local_model = _paste_template(projected, parent.shape)
            full_model = _paste_template(tmpl, parent.shape)
            native = block_reduce(full_model, (factor, factor), func=np.sum)
            expected = np.kron(native, np.ones((factor, factor), dtype=float) / factor**2)

            np.testing.assert_allclose(local_model, expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(local_model.sum(), full_model.sum(), rtol=0, atol=1e-12)


def test_psf_wing_completion_fills_zero_pixels_from_unit_psf_shape_convolution():
    parent = np.zeros((96, 96), dtype=float)
    psf = np.array(
        [
            [0.00, 0.08, 0.00, 0.02, 0.00],
            [0.04, 0.20, 0.35, 0.14, 0.03],
            [0.02, 0.44, 1.10, 0.30, 0.05],
            [0.01, 0.10, 0.26, 0.08, 0.01],
            [0.00, 0.02, 0.04, 0.01, 0.00],
        ],
        dtype=float,
    )
    tmpl = _wing_test_template(parent)
    templates = Templates()
    templates._templates = [tmpl]

    [completed] = templates.extend_with_psf_wings(psf, inplace=False)
    model = _paste_template(completed, parent.shape)

    core = _paste_template(tmpl, parent.shape)
    psf_shape = psf / psf.sum()
    smeared = fftconvolve(core, psf_shape, mode="same")
    expected = core.copy()
    expected[core == 0.0] = smeared[core == 0.0]
    expected /= expected.sum()

    np.testing.assert_allclose(model, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.sum(), 1.0, rtol=0, atol=1e-12)
    assert completed.extension_mode == "psf_wings"
    np.testing.assert_allclose(completed.extension_psf_sum, 1.0, rtol=0, atol=1e-12)
    assert completed.extension_psf_throughput == psf.sum()


def test_psf_wing_completion_uses_psf_shape_not_native_throughput():
    parent = np.zeros((96, 96), dtype=float)
    psf = np.zeros((5, 5), dtype=float)
    psf[2, 2] = 1.0
    psf[1, 2] = 0.2
    psf[2, 1] = 0.1

    tmpl_a = _wing_test_template(parent)
    tmpl_b = _wing_test_template(parent)
    tmpls_a = Templates()
    tmpls_b = Templates()
    tmpls_a._templates = [tmpl_a]
    tmpls_b._templates = [tmpl_b]

    [completed_a] = tmpls_a.extend_with_psf_wings(psf, inplace=False)
    [completed_b] = tmpls_b.extend_with_psf_wings(2.0 * psf, inplace=False)

    model_a = _paste_template(completed_a, parent.shape)
    model_b = _paste_template(completed_b, parent.shape)
    core = _paste_template(tmpl_a, parent.shape) != 0.0

    np.testing.assert_allclose(completed_a.extension_psf_sum, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(completed_b.extension_psf_sum, 1.0, rtol=0, atol=1e-12)
    assert completed_b.extension_psf_throughput == 2.0 * completed_a.extension_psf_throughput
    np.testing.assert_allclose(model_b[core].sum(), model_a[core].sum(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(model_a, model_b, rtol=0, atol=1e-12)


def test_psf_wing_completion_extends_deblended_templates_by_default():
    parent = np.zeros((96, 96), dtype=float)
    psf = np.zeros((5, 5), dtype=float)
    psf[2, 2] = 1.0
    psf[1, 2] = 0.25
    tmpl = _wing_test_template(parent)
    tmpl.is_deblended = True
    tmpl.deblend_parent_label = 99
    tmpl.deblend_nchildren = 3
    before = tmpl.data.copy()
    tmpls = Templates()
    tmpls._templates = [tmpl]

    [completed] = tmpls.extend_with_psf_wings(psf, inplace=False)

    assert completed.is_deblended
    assert completed.flag & Template.FLAG_DEBLENDED
    assert completed.deblend_parent_label == 99
    assert completed.deblend_nchildren == 3
    assert completed.extension_mode == "psf_wings"
    assert completed.extension_skip_reason == ""
    assert completed.data.shape != before.shape or not np.array_equal(completed.data, before)


def test_psf_wing_completion_can_skip_deblended_templates():
    parent = np.zeros((96, 96), dtype=float)
    psf = np.zeros((5, 5), dtype=float)
    psf[2, 2] = 1.0
    psf[1, 2] = 0.25
    tmpl = _wing_test_template(parent)
    tmpl.is_deblended = True
    tmpl.deblend_parent_label = 99
    tmpl.deblend_nchildren = 3
    before = tmpl.data.copy()
    tmpls = Templates()
    tmpls._templates = [tmpl]

    [completed] = tmpls.extend_with_psf_wings(psf, skip_deblended=True, inplace=False)

    np.testing.assert_allclose(completed.data, before, rtol=0, atol=0)
    assert completed.is_deblended
    assert completed.flag & Template.FLAG_DEBLENDED
    assert completed.deblend_parent_label == 99
    assert completed.deblend_nchildren == 3
    assert completed.extension_mode == "none"
    assert completed.extension_skip_reason == "is_deblended"


def test_convolved_template_projection_matches_global_projected_convolution():
    parent = np.zeros((80, 80), dtype=float)
    kernel = np.zeros((4, 4), dtype=float)
    kernel[2, 2] = 1.0
    kernel[1, 3] = 0.2
    factor = 2
    tmpl = _asymmetric_template(parent, (20.25, 22.75), (9, 8))

    convolved = tmpl.convolve_cutout(kernel, parent_image=parent, preserve_dtype=False)
    projected = convolved.project_to_block_replicated_grid(
        factor,
        parent_image=parent,
        preserve_dtype=False,
    )
    local_model = _paste_template(projected, parent.shape)

    global_convolved = fftconvolve(_paste_template(tmpl, parent.shape), kernel, mode="same")
    native = block_reduce(global_convolved, (factor, factor), func=np.sum)
    expected = np.kron(native, np.ones((factor, factor), dtype=float) / factor**2)

    np.testing.assert_allclose(local_model, expected, rtol=0, atol=1e-12)


def test_extract_templates_then_convolve_matches_global_convolution_for_origin_parities():
    """Exercise the production Template.extract_templates + convolve_templates path."""
    parent = np.zeros((96, 96), dtype=float)
    segmap = np.zeros(parent.shape, dtype=np.int32)
    positions = []
    flux_by_id = {}

    label = 1
    for ycen in (16, 29, 42, 55):
        for xcen in (15, 28, 43, 58):
            yy, xx = np.mgrid[-2:3, -3:4]
            mask = (yy * yy / 4.0 + xx * xx / 9.0) <= 1.0
            values = (1.0 + 0.2 * xx + 0.1 * yy)[mask]
            values = values - values.min() + 0.1
            ys = ycen + yy[mask]
            xs = xcen + xx[mask]
            parent[ys, xs] = values * label
            segmap[ys, xs] = label
            positions.append((xcen + 0.25, ycen + 0.75))
            flux_by_id[label] = float(parent[ys, xs].sum())
            label += 1

    kernel = np.zeros((10, 8), dtype=float)
    kernel[5, 4] = 0.7
    kernel[4, 6] = 0.2
    kernel[7, 3] = 0.1

    tmpls = Templates()
    extracted = tmpls.extract_templates(parent, segmap, positions, dilate_segmap=0)
    origin_parities = {
        (int(t._origin_original_true[0]) % 2, int(t._origin_original_true[1]) % 2)
        for t in extracted
    }
    assert origin_parities == {(0, 0), (0, 1), (1, 0), (1, 1)}

    convolved = tmpls.convolve_templates(kernel, inplace=False)
    local_model = np.zeros_like(parent)
    for tmpl in convolved:
        local_model[tmpl.slices_original] += (
            tmpl.data[tmpl.slices_cutout] * flux_by_id[int(tmpl.id)]
        )

    global_model = fftconvolve(parent, kernel, mode="same")
    np.testing.assert_allclose(local_model, global_model, rtol=0, atol=1e-12)


def test_psf_wing_completion_background_only_blocks_neighbor_segments():
    """Wing fill must not write into pixels owned by another segment."""
    image = np.zeros((96, 96), dtype=float)
    segmap = np.zeros((96, 96), dtype=np.int32)
    # source segment 1 around (40, 40), neighbor segment 2 directly east
    segmap[37:44, 37:44] = 1
    segmap[37:44, 45:52] = 2
    image[40, 40] = 1.0
    image[40, 46] = 0.5

    psf = np.zeros((19, 19), dtype=float)
    psf[9, 9] = 1.0
    psf += 0.05  # broad wings so the fill reaches the neighbor segment

    templates = Templates()
    templates.extract_templates(image, segmap, [(40.0, 40.0)], dilate_segmap=0)
    assert templates.segmap is not None

    [blocked] = templates.extend_with_psf_wings(psf, background_only=True, inplace=False)
    [filled] = templates.extend_with_psf_wings(psf, background_only=False, inplace=False)

    model_blocked = _paste_template(blocked, image.shape)
    model_filled = _paste_template(filled, image.shape)

    neighbor = segmap == 2
    background = segmap == 0
    assert model_blocked[neighbor].sum() == 0.0
    assert model_filled[neighbor].sum() > 0.0
    assert model_blocked[background].sum() > 0.0
    np.testing.assert_allclose(model_blocked.sum(), 1.0, rtol=0, atol=1e-12)
    assert blocked.extension_blocked_sum > 0.0
