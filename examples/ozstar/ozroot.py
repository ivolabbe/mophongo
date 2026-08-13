#!/usr/bin/env python
"""Where an OzStar campaign lives, and how to reach the cluster.

OzStar (Ngarrgu Tindebeek) has two storage areas and only one of them is usable
for a campaign: ``/home`` has a 20 GB quota and is for configs, while ``/fred``
is the 10 TB group allocation everything actually runs from. The default run
root is therefore ``/fred/<project>/<user>/mophongo/run``.

Environment overrides, all optional except ``OZSTAR_USER``::

    OZSTAR_USER      cluster username (no default; there is nothing to guess)
    OZSTAR_HOST      login node, default nt.swin.edu.au
    OZSTAR_PROJECT   group allocation, default oz030
    OZSTAR_RUN       run tree, default /fred/<project>/<user>/mophongo/run
    OZSTAR_STPSF     STPSF reference data, default /fred/<project>/<user>/stpsf-data

``OZSTAR_RUN`` must be under ``/fred``: a run tree in ``/home`` fills the quota
partway through the first field.
"""
from __future__ import annotations

import os


def user() -> str:
    """Cluster username from ``$OZSTAR_USER``."""
    name = os.environ.get("OZSTAR_USER", "").strip()
    if not name:
        raise SystemExit("set OZSTAR_USER to your OzStar username")
    return name


def host() -> str:
    """Login node hostname."""
    return os.environ.get("OZSTAR_HOST", "nt.swin.edu.au")


def ssh_target() -> str:
    """``user@host`` for ssh, scp and rsync."""
    return f"{user()}@{host()}"


def project() -> str:
    """Group allocation directory under ``/fred``."""
    return os.environ.get("OZSTAR_PROJECT", "oz030")


def run_root() -> str:
    """The run tree on ``/fred``."""
    root = os.environ.get("OZSTAR_RUN", "").rstrip("/")
    if not root:
        root = f"/fred/{project()}/{user()}/mophongo/run"
    if not root.startswith("/fred/"):
        raise SystemExit(
            f"OZSTAR_RUN must be under /fred (got {root}); /home has a 20 GB quota"
        )
    return root


def stpsf_dir() -> str:
    """STPSF reference data, which compute nodes cannot download themselves."""
    path = os.environ.get("OZSTAR_STPSF", "").rstrip("/")
    return path or f"/fred/{project()}/{user()}/stpsf-data"
