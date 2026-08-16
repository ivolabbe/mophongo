"""Regenerate a finished run's stamps and scene figures without refitting.

The first full-field campaign wrote every band's fit table, residual and
kernels and then died rendering the scene figures, so the expensive part is
already on disk and only the figures are missing. :meth:`Pipeline.load_fit`
restores the post-run state from those products - rebuilding and writing the
stamps file when it is absent or truncated - and this script then draws the
figures. Nothing else is rewritten: the residual, fit table and template table
are left exactly as the run left them.

``load_fit`` deliberately does not restore ``all_scenes``; scene objects are
not persisted. Membership does survive, as ``id_scene`` on each template, so
the scenes are regrouped from that here, and only the fields
:meth:`Scene.plot` reads are populated - id, members, bounding box, the band's
image and weights, and the solved fluxes that :func:`solution_from` carries
over from the templates.

Usage::

    python scene_plots.py <run_config.json>
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mophongo.pipeline import Pipeline  # noqa: E402
from mophongo.scene import Scene, SceneFitter, _bbox_union  # noqa: E402
from mophongo.verification import save_scene_blobs, save_scene_overview  # noqa: E402

logger = logging.getLogger("scene_plots")

IFILT = 1


def solution_from(templates: list) -> SimpleNamespace:
    """The scene's solved state, rebuilt from the templates ``load_fit`` restored.

    ``Scene.model_image`` raises ``RuntimeError("No solution available")`` when
    ``solution`` is None, and a regrouped scene has never been solved, so
    without this every figure is skipped and the run reports "wrote 0 of N".
    That guard is the only thing on this path that reads ``solution``: the
    model is accumulated from ``t.flux`` and ``t.data``, both of which the load
    restores. Carrying the per-source fluxes across satisfies it with the same
    numbers the fit produced rather than by weakening the check.
    """
    return SimpleNamespace(
        flux=np.array([float(t.flux) for t in templates]),
        err=np.array([float(getattr(t, "err", np.nan)) for t in templates]),
        shifts=None,
        info={"source": "load_fit"},
    )


def rebuild_scenes(pipe: Pipeline, ifilt: int = IFILT) -> list[Scene]:
    """Group the loaded templates back into scenes by ``id_scene``."""
    templates = pipe.all_templates[0] if pipe.all_templates else []
    if not templates:
        raise SystemExit("load_fit produced no templates; nothing to draw")

    groups: dict[int, list] = defaultdict(list)
    for t in templates:
        groups[int(getattr(t, "id_scene", 1))].append(t)

    image = pipe.images[ifilt]
    weights = None
    if getattr(pipe, "weights", None) is not None:
        w = pipe.weights[ifilt]
        # Scene.plot slices the weights on the image grid; a lo-res weight map
        # would index wrongly, and the panel is legible without it.
        weights = w if w is not None and getattr(w, "shape", None) == image.shape else None
    if weights is None:
        logger.warning("no weight map on the image grid; residual panels are unmasked")

    scenes = [
        Scene(id=sid, templates=ts, fitter=SceneFitter(), bbox=_bbox_union(ts),
              image=image, weights=weights, config=pipe.config,
              solution=solution_from(ts))
        for sid, ts in sorted(groups.items())
    ]
    sizes = sorted(len(s.templates) for s in scenes)
    logger.info("regrouped %d templates into %d scenes (sizes %d-%d)",
                len(templates), len(scenes), sizes[0], sizes[-1])
    return scenes


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pipe = Pipeline.from_config(argv[1])
    with pipe.log_run():
        pipe.load_fit(ifilt=IFILT)
        scenes = rebuild_scenes(pipe)

        # Saturated stars are kept out of every other scene's display scale and
        # nulled in their own residual panel, exactly as write_outputs does.
        sat_ids = [int(t.id) for s in scenes for t in s.templates
                   if getattr(t, "is_saturated", False)]

        name = pipe.run_config.name
        scene_dir = Path(pipe.out_dir) / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)

        drawn = 0
        for s in scenes:
            try:
                fig, _ = s.plot(pipe.images[0], pipe.segmap, display_sig=5,
                                null_segments=sat_ids)
            except Exception as exc:  # noqa: BLE001 - one bad scene must not
                # cost the other several hundred figures
                logger.warning("scene %s: %s: %s", s.id, type(exc).__name__, exc)
                continue
            fig.savefig(scene_dir / f"{name}_scene_{s.id}.png", dpi=300)
            plt.close(fig)
            drawn += 1

        logger.info("wrote %d of %d scene figures to %s", drawn, len(scenes), scene_dir)

        # The two full-field views of the partition that write_outputs draws,
        # from the same helpers and under the same names, so a recovered band
        # carries the same products as one that finished on its own. Both need
        # only the mosaic, the segmap and the regrouped scenes, all of which
        # the load restores, so these are the real figures rather than
        # approximations of them.
        #
        # The scene catalog and the shift field are deliberately not rebuilt.
        # Both carry per-scene astrometry - dx, dy, astrom_niter, flag_astrom -
        # recorded during the solve and not persisted, so anything written here
        # would be a plausible-looking file with invented columns. A missing
        # product is better than a wrong one.
        stem = str(Path(pipe.out_dir) / name)
        save_scene_overview(pipe.images[0], pipe.segmap, scenes,
                            f"{stem}_scene_map.png")
        save_scene_blobs(scenes, pipe.images[0].shape, f"{stem}_scene_blobs.png")
        logger.info("wrote %s_scene_map.png and %s_scene_blobs.png", name, name)


if __name__ == "__main__":
    main(sys.argv)
