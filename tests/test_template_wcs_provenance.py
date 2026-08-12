"""World-coordinate invariance of template geometry operations.

A ``Template`` carries two WCSs: ``wcs_original``, the WCS of the parent
image it was cut from, and ``wcs``, that same WCS with CRPIX shifted onto the
stamp. Operations that build a new ``Template`` on a parent-sized image must
pass the *parent* WCS. Passing the cutout WCS instead shifts the new
template's world coordinates by the stamp origin, and shifts them again on
the next operation.

This is not cosmetic metadata: ``Templates.convolve_templates`` converts
``position_original`` through ``wcs_original`` to pick the PSF region and the
encircled-energy correction for each source.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from astropy.wcs import WCS

from mophongo.templates import Template

SIZE = 400
SRC = (260.0, 190.0)


def _tan_wcs(*, crval=(150.0, 2.0), crpix=(SIZE / 2, SIZE / 2), rot_deg=0.0,
             scale_arcsec=1.0, sip=False):
    w = WCS(naxis=2)
    w.wcs.crpix = list(crpix)
    w.wcs.crval = list(crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cd = scale_arcsec / 3600.0
    t = np.deg2rad(rot_deg)
    w.wcs.cd = np.array([[-cd * np.cos(t), cd * np.sin(t)],
                         [cd * np.sin(t), cd * np.cos(t)]])
    if sip:
        from astropy.wcs import Sip
        a = np.zeros((3, 3)); a[2, 0] = 1e-6
        b = np.zeros((3, 3)); b[0, 2] = -1e-6
        w.sip = Sip(a, b, None, None, np.array(crpix))
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    w.wcs.set()
    return w


def _template(wcs, size=(41, 41)):
    parent = np.zeros((SIZE, SIZE), dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    parent += np.exp(
        -0.5 * (((xx - SRC[0]) / 3.0) ** 2 + ((yy - SRC[1]) / 3.0) ** 2)
    ).astype(np.float32)
    return Template(parent, position=SRC, size=size, wcs=wcs, label=1)


def _sky(tmpl):
    """Sky position the PSF-region lookup would use for this template."""
    x, y = tmpl.position_original
    return np.asarray(tmpl.wcs_original.wcs_pix2world(x, y, 0), dtype=float)


def _sep_arcsec(a, b):
    dec = np.deg2rad(0.5 * (a[1] + b[1]))
    return float(np.hypot((a[0] - b[0]) * np.cos(dec), a[1] - b[1]) * 3600.0)


WCS_CASES = {
    "plain TAN": dict(),
    "translated CRPIX": dict(crpix=(37.0, 311.0)),
    "rotated 30 deg": dict(rot_deg=30.0),
    "fine scale": dict(scale_arcsec=0.03),
    "SIP distortion": dict(sip=True),
}


@pytest.mark.parametrize("case", list(WCS_CASES))
def test_convolution_preserves_world_coordinates(case):
    """``convolve_cutout`` must not move the source on the sky."""
    tmpl = _template(_tan_wcs(**WCS_CASES[case]))
    before = _sky(tmpl)

    kernel = np.zeros((7, 7), dtype=np.float32)
    kernel[3, 3] = 1.0  # identity: only geometry can change
    after = _sky(tmpl.convolve_cutout(kernel))

    assert _sep_arcsec(before, after) < 1e-6, (
        f"{case}: source moved {_sep_arcsec(before, after):.1f} arcsec"
    )


@pytest.mark.parametrize("case", list(WCS_CASES))
def test_block_projection_preserves_world_coordinates(case):
    """``project_to_block_replicated_grid`` must not move the source."""
    tmpl = _template(_tan_wcs(**WCS_CASES[case]))
    before = _sky(tmpl)
    after = _sky(tmpl.project_to_block_replicated_grid(2))

    assert _sep_arcsec(before, after) < 1e-6, (
        f"{case}: source moved {_sep_arcsec(before, after):.1f} arcsec"
    )


def test_repeated_operations_do_not_accumulate_drift():
    """The defect compounded: each operation shifted by the stamp origin."""
    tmpl = _template(_tan_wcs())
    before = _sky(tmpl)

    kernel = np.zeros((5, 5), dtype=np.float32)
    kernel[2, 2] = 1.0
    for _ in range(3):
        tmpl = tmpl.convolve_cutout(kernel)
        assert _sep_arcsec(before, _sky(tmpl)) < 1e-6


def test_padding_preserves_world_coordinates():
    """``pad`` grows the stamp and must keep the same sky position."""
    tmpl = _template(_tan_wcs())
    before = _sky(tmpl)
    after = _sky(tmpl.pad((8, 8), (SIZE, SIZE)))
    assert _sep_arcsec(before, after) < 1e-6


def test_cutout_wcs_stays_consistent_with_the_parent():
    """``wcs`` must remain the parent WCS shifted onto the stamp.

    Both WCSs have to agree about where the source is: ``wcs_original`` at
    its original-grid pixel, ``wcs`` at its cutout pixel.
    """
    tmpl = _template(_tan_wcs())
    conv = tmpl.convolve_cutout(np.pad([[1.0]], 3).astype(np.float32))

    for t in (tmpl, conv):
        from_parent = t.wcs_original.wcs_pix2world(*t.input_position_original, 0)
        from_cutout = t.wcs.wcs_pix2world(*t.input_position_cutout, 0)
        assert _sep_arcsec(np.asarray(from_parent, float),
                           np.asarray(from_cutout, float)) < 1e-6
