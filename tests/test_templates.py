

def test_as_label_array_accepts_float_segmaps():
    """Segmaps stored as float must be usable as labels.

    MINERVA UDS and EGS ship int32 segmaps, COSMOS ships the same labels as
    float64, and SegmentationImage rejects non-integer input outright.
    """
    import numpy as np
    import pytest

    from mophongo.utils import as_label_array

    ints = np.array([[0, 1], [2, 2]], dtype=np.int32)
    assert as_label_array(ints) is ints

    floats = np.array([[0.0, 1.0], [2085561.0, 0.0]], dtype=np.float64)
    out = as_label_array(floats)
    assert np.issubdtype(out.dtype, np.integer)
    assert out.max() == 2085561

    with pytest.raises(ValueError, match="non-integer"):
        as_label_array(np.array([[0.0, 0.5]]))


def test_uncovered_sources_get_no_template(caplog):
    """A segment the detection image does not cover must not become a template.

    Combined catalogs and segmaps reach past one band's coverage: another band
    saw the source, this one did not. The segment is there and the data are
    not, so a template built there is noise given a shape -- it fits nothing,
    absorbs its neighbours' flux, and if it clears the anchor cuts it drags the
    scene's astrometry with it.
    """
    import logging

    import numpy as np

    from mophongo.templates import Template, Templates

    ny = nx = 60
    image = np.zeros((ny, nx), dtype=float)
    segmap = np.zeros((ny, nx), dtype=np.int32)
    positions = [(15.0, 30.0), (45.0, 30.0)]
    for label, (x, y) in enumerate(positions, start=1):
        xi, yi = int(x), int(y)
        segmap[yi - 3:yi + 4, xi - 3:xi + 4] = label
        image[yi - 3:yi + 4, xi - 3:xi + 4] = 1.0

    # the right half has no exposure
    weight = np.ones((ny, nx), dtype=float)
    weight[:, nx // 2:] = 0.0

    tmpls = Templates()
    with caplog.at_level(logging.INFO, logger="mophongo.templates"):
        out = tmpls.extract_templates(image, segmap, positions,
                                      detection_weight=weight)
    assert [int(t.id) for t in out] == [1], "the uncovered source must be dropped"
    assert "no detection-band coverage" in caplog.text

    assert not out[0].flag & Template.FLAG_NO_COVERAGE, "this one is fully covered"

    # a source inside the footprint whose segment reaches outside it keeps its
    # template and carries the flag: the flux is over the exposed pixels only
    edge = np.ones((ny, nx), dtype=float)
    edge[:, 17:] = 0.0
    flagged = Templates().extract_templates(image, segmap, positions[:1],
                                            detection_weight=edge)
    assert len(flagged) == 1
    assert flagged[0].flag & Template.FLAG_NO_COVERAGE

    # without a weight map nothing is dropped or flagged: the caller said
    # nothing about coverage, so the old behaviour stands
    plain = Templates().extract_templates(image, segmap, positions)
    assert len(plain) == 2
    assert not any(t.flag & Template.FLAG_NO_COVERAGE for t in plain)

    # a placeholder map of zeros carries no information and is ignored, rather
    # than silently returning nothing
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mophongo.templates"):
        both = Templates().extract_templates(image, segmap, positions,
                                             detection_weight=np.zeros((ny, nx)))
    assert len(both) == 2
    assert "zero everywhere" in caplog.text


def test_pruned_templates_carry_the_outside_weight_flag(caplog):
    """A source this band cannot measure is flagged, then dropped.

    The flag is the only trace it leaves: a pruned template has no row in that
    band's outputs, so without it a source that vanished from F1800W but
    survived in F770W is a difference in row counts and nothing else.
    """
    import logging

    import numpy as np

    from mophongo.templates import Template, Templates

    ny = nx = 40
    image = np.zeros((ny, nx), dtype=float)
    segmap = np.zeros((ny, nx), dtype=np.int32)
    positions = [(10.0, 20.0), (30.0, 20.0)]
    for label, (x, y) in enumerate(positions, start=1):
        xi, yi = int(x), int(y)
        segmap[yi - 2:yi + 3, xi - 2:xi + 3] = label
        image[yi - 2:yi + 3, xi - 2:xi + 3] = 1.0

    tmpls = Templates()
    tmpls.extract_templates(image, segmap, positions)
    kept_before = list(tmpls.templates)
    assert len(kept_before) == 2

    # this band sees the left source only
    weight = np.ones((ny, nx), dtype=float)
    weight[:, nx // 2:] = 0.0

    with caplog.at_level(logging.INFO, logger="mophongo.templates"):
        kept = tmpls.prune_outside_weight(weight)

    assert [int(t.id) for t in kept] == [1]
    assert "FLAG_OUTSIDE_WEIGHT" in caplog.text
    # the dropped object still exists for the caller that held it, and says why
    dropped = [t for t in kept_before if int(t.id) == 2][0]
    assert dropped.flag & Template.FLAG_OUTSIDE_WEIGHT
    assert not kept[0].flag & Template.FLAG_OUTSIDE_WEIGHT


def test_prune_outside_weight_is_subset_invariant():
    """A template's verdict must not depend on the company it keeps.

    The threshold used to be ``rtol * median(wnorm)`` over the set being
    pruned, so pruning a subset applied a different cut than pruning the whole
    field. Re-solving one scene of a COSMOS F770W run dropped 76 of its 260
    templates for that reason alone: the scene's members are brighter than the
    field median, the threshold rose with them, and the faint edge members went
    with it. Anything that re-extracts a subset -- ``Pipeline.refit_scene``
    above all -- was silently working on a different source set.
    """
    import numpy as np

    from mophongo.templates import Template, Templates

    ny = nx = 320
    half = 12

    def _tmpl(xc, yc, sigma, label, scale):
        n = 2 * half + 1
        x0, y0 = int(round(xc)) - half, int(round(yc)) - half
        yy, xx = np.mgrid[y0 : y0 + n, x0 : x0 + n]
        g = np.exp(-0.5 * (((xx - xc) / sigma) ** 2 + ((yy - yc) / sigma) ** 2))
        g = scale * g / g.sum()
        return Template.from_stamp(g, (x0, y0), (xc, yc), (ny, nx), label=label)

    weight = np.ones((ny, nx))

    def _at(i, scale):
        return _tmpl(40.0 + 9 * (i % 30), 40.0 + 9 * (i // 30), 2.0, i + 1, scale)

    bright = [_at(i, 1.0) for i in range(3)]        # the scene's bright members
    faint = [_at(100 + i, 3e-6) for i in range(3)]  # its faint edge members
    rest = [_at(200 + i, 1e-7) for i in range(20)]  # the rest of the field

    def survivors(templates):
        ts = Templates()
        ts.original_shape = (ny, nx)
        ts._templates = list(templates)
        return {int(t.id) for t in ts.prune_outside_weight(weight)}

    full = survivors(bright + faint + rest)
    subset = survivors(bright + faint)

    assert {int(t.id) for t in faint} <= full
    assert {int(t.id) for t in faint} <= subset
    # the verdict on the shared templates is identical either way
    assert full & {int(t.id) for t in bright + faint} == subset


def test_prune_outside_weight_drops_templates_off_the_weight_map():
    """The documented behaviour still holds: no usable pixels, no template."""
    import numpy as np

    from mophongo.templates import Template, Templates

    ny = nx = 200
    half = 12
    weight = np.ones((ny, nx))
    weight[:, 120:] = 0.0

    def _tmpl(xc, label):
        n = 2 * half + 1
        x0, y0 = int(round(xc)) - half, 100 - half
        yy, xx = np.mgrid[y0 : y0 + n, x0 : x0 + n]
        g = np.exp(-0.5 * (((xx - xc) / 2.0) ** 2 + ((yy - 100.0) / 2.0) ** 2))
        return Template.from_stamp(g / g.sum(), (x0, y0), (xc, 100.0), (ny, nx),
                                   label=label)

    inside, outside = _tmpl(60.0, 1), _tmpl(160.0, 2)
    ts = Templates()
    ts.original_shape = (ny, nx)
    ts._templates = [inside, outside]

    kept = ts.prune_outside_weight(weight)

    assert [int(t.id) for t in kept] == [1]
    assert outside.flag & Template.FLAG_OUTSIDE_WEIGHT
