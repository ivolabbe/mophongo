

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
