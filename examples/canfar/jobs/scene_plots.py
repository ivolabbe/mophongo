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
the scenes are regrouped from that here. Only the fields :meth:`Scene.plot`
reads are populated - id, members, bounding box, and the band's image and
weights - because nothing is solved.

Usage::

    python scene_plots.py <run_config.json>
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mophongo.pipeline import Pipeline  # noqa: E402
from mophongo.scene import Scene, SceneFitter, _bbox_union  # noqa: E402

logger = logging.getLogger("scene_plots")

IFILT = 1


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
              image=image, weights=weights, config=pipe.config)
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


if __name__ == "__main__":
    main(sys.argv)
