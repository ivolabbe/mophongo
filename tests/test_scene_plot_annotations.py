"""What the scene diagnostic PNG reports about its own fit, and how it scales.

The panels are annotated in place, not in their titles: how many templates the
scene fitted, and how many of those anchored its shift field, with the
systematic floor and reduced chi-square beside them. The residual also carries
the lo-res PSF region boundaries, which is where the PSF and the matching
kernel change under it.

The greyscale stretch is measured on this scene's own pixels. Sources
belonging to other scenes are dropped from it, and the noise-like panels clip
before taking a width, so one bright neighbour can no longer flatten the whole
figure to grey.
"""

from __future__ import annotations

import numpy as np
import pytest

from mophongo.scene import generate_scenes
from mophongo.templates import Templates


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib

    matplotlib.use("Agg")


def _make_image_segmap(positions, shape=(80, 80), src_radius=2):
    image = np.zeros(shape, dtype=np.float32)
    segmap = np.zeros(shape, dtype=np.int32)
    yy, xx = np.indices(shape)
    for k, (y, x) in enumerate(positions, start=1):
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= src_radius**2
        image[mask] += 1.0
        segmap[mask] = k
    return image, segmap


def _one_scene(positions=((20, 20), (22, 22), (24, 24))):
    rng = np.random.default_rng(7)
    image, segmap = _make_image_segmap(list(positions))
    image += rng.normal(0.0, 0.05, image.shape).astype(np.float32)
    weight = np.ones(image.shape, dtype=np.float32)
    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(x, y) for y, x in positions])
    scenes, _ = generate_scenes(
        tmpls.templates, image, weight, coupling_thresh=0.0, minimum_bright=1,
    )
    scene = max(scenes, key=lambda s: len(s.templates))
    scene.solve()
    return scene, image, segmap


def _titles(axes):
    return [a.get_title() for a in axes]


def _note(axis):
    """The annotation drawn inside a panel, or "" when it carries none."""
    return " ".join(t.get_text() for t in axis.texts)


def test_panels_report_the_template_and_anchor_counts():
    import matplotlib.pyplot as plt

    scene, image, segmap = _one_scene()
    fig, axes = scene.plot(image, segmap)
    titles, template_note, residual_note = _titles(axes), _note(axes[0]), _note(axes[4])
    plt.close(fig)

    # the counts go inside the panels; the titles stay bare
    assert titles == ["Template", "Image", "Model", "Segmap", "Residual", "Color"]
    assert f"{len(scene.templates)} templates" in template_note
    assert f"{scene.n_anchors()} anchor" in residual_note
    # the count the scene catalog writes as n_anchor, so the two cannot drift
    assert scene.n_anchors() == int(np.sum(scene.is_bright))


def test_residual_note_carries_chi2_and_the_astrometric_floor():
    """The same two scalars the scene catalog reports for this scene."""
    import matplotlib.pyplot as plt
    from mophongo.astrom_robust import AnchorWeights

    scene, image, segmap = _one_scene()

    fig, axes = scene.plot(image, segmap)
    note = _note(axes[4])
    plt.close(fig)
    chi2 = scene.chi2_dof(None)
    assert np.isfinite(chi2), "test setup: scene has no weights to form chi2 from"
    assert f"chi2/dof {chi2:.3g}" in note
    # no robust pass ran, so no floor is claimed
    assert "floor" not in note

    n = max(scene.n_anchors(), 1)
    scene.anchor_report = AnchorWeights(
        weight=np.ones(n), coeff=np.zeros(2), field=np.zeros((n, 2)),
        resid=np.zeros((n, 2)), sys_floor=0.0312, n_rejected=2,
        n_eff=float(n), applied=True,
    )
    fig, axes = scene.plot(image, segmap)
    note = _note(axes[4])
    plt.close(fig)
    assert "2 rejected" in note
    assert "floor 0.031 px" in note


def test_residual_note_uses_the_global_residual_when_given():
    """chi2 must be measured on what the panel shows, as the catalog does."""
    import matplotlib.pyplot as plt

    scene, image, segmap = _one_scene()
    inflated = np.full_like(image, 10.0)

    fig, axes = scene.plot(image, segmap, residual_image=inflated)
    note = _note(axes[4])
    plt.close(fig)

    assert f"chi2/dof {scene.chi2_dof(inflated):.3g}" in note
    assert scene.chi2_dof(inflated) > scene.chi2_dof(None)


def test_region_outlines_land_on_the_residual_panel_only():
    """One ring over the scene, one far outside it; only the first is drawn."""
    import matplotlib.pyplot as plt

    scene, image, segmap = _one_scene()
    y0, y1, x0, x1 = scene.bbox
    over = np.array(
        [(x0 - 2, y0 - 2), (x1 + 2, y0 - 2), (x1 + 2, y1 + 2), (x0 - 2, y1 + 2),
         (x0 - 2, y0 - 2)],
        dtype=float,
    )
    far = over + 10_000.0

    fig, axes = scene.plot(image, segmap, region_outlines=[over, far])
    drawn = [len(a.collections) for a in axes]
    residual_ax = axes[4]
    paths = residual_ax.collections[0].get_paths()
    xlim, ylim = residual_ax.get_xlim(), residual_ax.get_ylim()
    plt.close(fig)

    # only the residual panel gains a collection (the model panel's quiver
    # arrows are a collection too, so compare against the untouched panels)
    assert drawn[0] == drawn[1] == drawn[3] == drawn[5] == 0
    assert len(paths) == 1, "the far-away ring was drawn"
    # the ring must not stretch the panel off the cutout
    ny, nx = image[y0 : y1 + 1, x0 : x1 + 1].shape
    assert xlim[0] >= -1.0 and xlim[1] <= nx
    assert ylim[0] >= -1.0 and ylim[1] <= ny


def test_no_outlines_leaves_the_residual_panel_bare():
    import matplotlib.pyplot as plt

    scene, image, segmap = _one_scene()
    fig, axes = scene.plot(image, segmap, region_outlines=None)
    n = len(axes[4].collections)
    plt.close(fig)
    assert n == 0


def test_region_outlines_pixels_projects_every_ring():
    """The pipeline-side conversion: sky polygons -> fit-grid pixel paths."""
    import geopandas as gpd
    from astropy.wcs import WCS
    from shapely.geometry import box

    from mophongo.pipeline import _region_outlines_pixels
    from mophongo.psf_map import PSFRegionMap

    w = WCS(naxis=2)
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cdelt = [-0.04 / 3600.0, 0.04 / 3600.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    regions = gpd.GeoDataFrame(
        {"psf_key": [0, 1]},
        geometry=[box(149.999, 1.999, 150.0, 2.0), box(150.0, 2.0, 150.001, 2.001)],
        crs="EPSG:4326",
    )
    rings = _region_outlines_pixels(PSFRegionMap(regions=regions), w)

    assert len(rings) == 2
    assert all(r.ndim == 2 and r.shape[1] == 2 and np.isfinite(r).all() for r in rings)
    # the boxes meet at crval, which is crpix - 1 = (49, 49) in 0-based pixels
    for ring in rings:
        assert np.abs(ring - 49.0).sum(axis=1).min() < 1e-3
    # 0.001 deg at 0.04"/pix is 90 pixels, and RA runs against +x
    assert np.isclose(np.ptp(np.concatenate(rings)[:, 0]), 180.0, rtol=0.02)


def test_region_outlines_pixels_is_empty_without_a_map_or_wcs():
    from mophongo.pipeline import _region_outlines_pixels

    assert _region_outlines_pixels(None, None) == []


def _clims(axes):
    return {
        name: axes[i].get_images()[0].get_clim()
        for i, name in enumerate(
            ["Template", "Image", "Model", "Segmap", "Residual", "Color"]
        )
    }


def test_a_bright_neighbour_does_not_set_the_stretch():
    """A source in another scene used to flatten every panel to grey.

    The neighbour is 100x this scene's sources and sits inside the bbox, which
    is the ordinary case in a crowded field. Its pixels are out of the display
    scale now, so the panels are stretched to what the scene itself holds.
    """
    import matplotlib.pyplot as plt

    scene, image, segmap = _one_scene()
    y0, y1, x0, x1 = scene.bbox
    mine = {int(t.id) for t in scene.templates}

    # a bright foreign source, with its own segment, on free pixels inside the
    # scene bbox -- the ordinary crowded-field case
    inside = np.zeros(image.shape, bool)
    inside[y0 : y1 + 1, x0 : x1 + 1] = True
    hot = inside & (segmap == 0)
    ys, xs = np.nonzero(hot)
    assert ys.size >= 4, "test setup: no free pixels inside the bbox"
    hot = np.zeros(image.shape, bool)
    hot[ys[: ys.size // 4], xs[: xs.size // 4]] = True

    foreign_id = int(max(int(segmap.max()), max(mine))) + 1
    loud = image.copy()
    loud[hot] += 100.0
    seg2 = segmap.copy()
    seg2[hot] = foreign_id
    assert foreign_id not in mine, "test setup: the neighbour is one of ours"

    scene.image = loud
    scene.solve()
    fig, axes = scene.plot(loud, seg2)
    masked = _clims(axes)
    plt.close(fig)

    # ... versus treating the neighbour as one of ours, i.e. the old behaviour
    fig, axes = scene.plot(loud, np.where(hot, list(mine)[0], segmap))
    unmasked = _clims(axes)
    plt.close(fig)

    # Not the model panel: it holds only this scene's own templates, so the
    # neighbour's pixels are zero there and never entered its scale anyway.
    for panel in ("Template", "Image", "Residual"):
        assert masked[panel][1] < unmasked[panel][1], (
            f"{panel} stretch still set by the bright neighbour"
        )
    assert masked["Model"][1] == unmasked["Model"][1]
    # and the noise panels are stretched to the noise, not to a source
    assert masked["Image"][1] < 0.2 * float(np.abs(loud[hot]).max())


def test_template_panel_scale_does_not_collapse():
    """A robust noise scale on the template panel would burn it white.

    Its nonzero pixels are the template itself -- mostly near-zero wings --
    so a clipped or MAD scale there goes to zero. Pinned because the image and
    residual panels *do* use one.
    """
    import matplotlib.pyplot as plt

    scene, image, segmap = _one_scene()
    fig, axes = scene.plot(image, segmap)
    clims = _clims(axes)
    tmpl_max = float(np.abs(axes[0].get_images()[0].get_array()).max())
    plt.close(fig)

    assert clims["Template"][1] > 0.05 * tmpl_max
    assert clims["Model"][1] > 0.0
