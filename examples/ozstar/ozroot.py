#!/usr/bin/env python
"""Where an OzStar campaign lives, and how to reach the cluster.

The tree separates what is stable from what is per-run, mirroring the CANFAR
layout::

    /fred/<project>/<user>/mophongo/     $OZSTAR_BASE
    ├── bin/                  job scripts, shared by every run
    ├── PSF/                  ePSF grids, expensive and reusable
    ├── data/                 inputs staged from CANFAR arc
    └── run2/                 one catalog version   $OZSTAR_RUN
        ├── config/           rewritten configs, the venv, the mophongo clone
        ├── logs/             SLURM job output
        └── <field>/          uds, cosmos, egs
            ├── <field>_repair_cache.fits
            └── <field>_<band>/   the fit's out_dir

Data and PSF grids sit *above* the run directory because they are stable: a
release is re-fitted many times against the same 64 GB of mosaics and the same
grids, and only the configs and the outputs change. Putting them inside a run
would mean re-staging and rebuilding for every version, or symlink sprawl.

The per-field level matters too. ``RunConfig.repair_cache_path`` defaults to
``".."`` relative to ``out_dir``, so with ``<field>/<field>_<band>/`` a field's
bands share one saturation-repair cache and nothing else does.

Environment overrides, all optional except ``OZSTAR_USER``::

    OZSTAR_USER      cluster username (no default; there is nothing to guess)
    OZSTAR_HOST      login node, default nt.swin.edu.au
    OZSTAR_PROJECT   group allocation, default oz030
    OZSTAR_BASE      shared tree, default /fred/<project>/<user>/mophongo
    OZSTAR_RUN       run directory *name*, default run2
    OZSTAR_STPSF     STPSF reference data, default <base>/../stpsf-data

``OZSTAR_BASE`` must be under ``/fred``: a tree in ``/home`` fills the 20 GB
quota partway through the first field.
"""
from __future__ import annotations

import os

#: Default run directory. Bump this (or set $OZSTAR_RUN) for a new catalog
#: version; data/, PSF/ and bin/ are untouched by that.
DEFAULT_RUN = "run2"


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


def base_root() -> str:
    """The shared tree holding bin/, PSF/, data/ and the run directories."""
    root = os.environ.get("OZSTAR_BASE", "").rstrip("/")
    if not root:
        root = f"/fred/{project()}/{user()}/mophongo"
    if not root.startswith("/fred/"):
        raise SystemExit(
            f"OZSTAR_BASE must be under /fred (got {root}); /home has a 20 GB quota"
        )
    return root


def run_name() -> str:
    """Run directory name, e.g. ``run2``."""
    return os.environ.get("OZSTAR_RUN", "").strip("/ ") or DEFAULT_RUN


def run_root() -> str:
    """The current run directory: outputs, configs and the environment."""
    return f"{base_root()}/{run_name()}"


def config_dir() -> str:
    """Rewritten configs, staging lists, the venv and the mophongo clone."""
    return f"{run_root()}/config"


def data_dir() -> str:
    """Staged inputs, shared by every run."""
    return f"{base_root()}/data"


def psf_dir() -> str:
    """ePSF grids, shared by every run."""
    return f"{base_root()}/PSF"


def bin_dir() -> str:
    """Job scripts, shared by every run."""
    return f"{base_root()}/bin"


def field_of(name: str) -> str:
    """Field a run name belongs to: ``cosmos_f770w_v1.0b`` -> ``cosmos``."""
    return name.split("_")[0]


def out_dir(name: str) -> str:
    """Where one band's products go: ``<run>/<field>/<name>``."""
    return f"{run_root()}/{field_of(name)}/{name}"


def stpsf_dir() -> str:
    """STPSF reference data, which compute nodes cannot download themselves."""
    path = os.environ.get("OZSTAR_STPSF", "").rstrip("/")
    return path or f"/fred/{project()}/{user()}/stpsf-data"
