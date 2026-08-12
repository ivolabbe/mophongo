"""Where a CANFAR campaign keeps its run tree.

The default is the user's own home, which is private but carries a quota of a
few hundred GB - enough for trial patches, not for full-field runs of a whole
release. Point ``CANFAR_RUN`` at a project directory to use the shared
allocation instead::

    export CANFAR_RUN=/arc/projects/minerva/ifl

Project space is shared with the collaboration, so treat it as somewhere to
write deliberately rather than by default.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


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

    ``$CANFAR_RUN`` wins; otherwise ``/arc/home/<user>/run``.
    """
    root = os.environ.get("CANFAR_RUN", "").rstrip("/")
    if not root:
        root = f"/arc/home/{canfar_user(repo)}/run"
    if not root.startswith("/arc/"):
        raise SystemExit(f"CANFAR_RUN must be an absolute path under /arc (got {root})")
    return root, "arc:" + root[len("/arc/"):]
