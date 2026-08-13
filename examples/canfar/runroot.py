"""Where a CANFAR campaign keeps its run tree.

The default is the shared project allocation, ``/arc/projects/minerva/ifl``.
It has to be: a run writes about 12 GB per band -- a 3.5 GB residual and 8-11
GB of stamps -- so a 17-band release lands in the region of 200 GB, and
``/arc/home`` carries a few hundred GB for everything a user owns. Home has
already been filled once this way, which is what the note in
``jobs/seed_cache.sh`` records.

``$CANFAR_RUN`` points the tree somewhere else. It may not point it under
``/arc/home``: that is refused rather than warned about, because the failure
it prevents is silent until a quota stops a campaign halfway through. Home is
the right place for a laptop-side checkout, not for a run tree -- the tree
holds the staged inputs, the ePSF grids and every product as well as the code.

Project space is shared with the collaboration, so treat it as somewhere to
write deliberately rather than by default.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: Shared project allocation. Big enough for a release, and the grids and
#: outputs in it are useful to the collaboration rather than to one user.
DEFAULT_RUN = "/arc/projects/minerva/ifl"


def canfar_user(repo: Path) -> str:
    """CADC username, from ``$CANFAR_USER`` or ``scratch/canfar/canfar.conf``."""
    user = os.environ.get("CANFAR_USER")
    if user:
        return user
    conf = repo / "scratch" / "canfar" / "canfar.conf"
    if conf.exists():
        match = re.search(r'^\s*CANFAR_USER\s*=\s*"?([^"\s]+)', conf.read_text(), re.M)
        if match and match.group(1) != "your_cadc_username":
            return match.group(1)
    raise SystemExit(
        "set CANFAR_USER to your CADC username, or fill in scratch/canfar/canfar.conf"
    )


def run_root(repo: Path) -> tuple[str, str]:
    """Return the run tree as ``(posix_path, vospace_uri)``.

    ``$CANFAR_RUN`` wins; otherwise :data:`DEFAULT_RUN`. A tree under
    ``/arc/home`` is refused: see the module docstring.
    """
    root = os.environ.get("CANFAR_RUN", "").rstrip("/") or DEFAULT_RUN
    if not root.startswith("/arc/"):
        raise SystemExit(f"CANFAR_RUN must be an absolute path under /arc (got {root})")
    if root.startswith("/arc/home/"):
        raise SystemExit(
            f"refusing a run tree under /arc/home (got {root}). A run writes "
            "~12 GB per band and home carries a few hundred GB in total, which "
            "a release exhausts partway through. Use the project allocation: "
            f"unset CANFAR_RUN for the default ({DEFAULT_RUN}), or set it to "
            "another path under /arc/projects."
        )
    return root, "arc:" + root[len("/arc/"):]
