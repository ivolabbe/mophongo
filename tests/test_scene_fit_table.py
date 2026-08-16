"""The SCENES extension of the fit table, and what it is for.

A band that dies drawing its scene figures has already solved everything; the
point of this extension is that the solution survives that. The tests that
matter are therefore about reconstruction: the stored coefficients have to
reproduce the anchor-solved shift field exactly, not approximately, because
refitting a polynomial to the applied shifts is what they exist to replace.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from mophongo.astrometry import AstroCorrect, cheb_basis, n_terms
from mophongo.pipeline import Pipeline, _cheb_order_from_coeffs


# ----------------------------------------------------------------- helpers
class _Template:
    def __init__(self, x: float, y: float) -> None:
        self.position_original = (x, y)
        self.shifted = np.zeros(2, dtype=float)


class _Scene:
    """Only the attributes ``_scene_fit_table`` reads."""

    def __init__(self, sid, shifts, order, *, n=6, x0=100.0, y0=200.0,
                 sx=50.0, sy=50.0, report=None):
        rng = np.random.default_rng(sid)
        self.id = sid
        self.templates = [_Template(*p) for p in rng.uniform(0, 300, size=(n, 2))]
        self.shifts = shifts
        self.shift_basis = None if shifts is None else [None, (x0, y0), (sx, sy)]
        self.is_bright = np.zeros(n, dtype=bool)
        self.is_bright[: max(1, n // 3)] = True
        self.astrom_niter = 3
        self.astrom_converged = True
        self.astrom_step = 0.02
        self.anchor_report = report
        self.order = order

    def shift_error(self):
        """Real ``Scene`` method the table reads; no covariance on the stub."""
        return float("nan")

    def chi2_dof(self, residual=None):
        """Real ``Scene`` method the table reads; no pixels on the stub."""
        return float("nan")

    def shift_scatter(self):
        """Real ``Scene`` method the table reads; stubbed with the same rule."""
        sh = np.array([np.asarray(t.shifted, float)[:2] for t in self.templates])
        if len(sh) < 2:
            return 0.0
        d = sh - sh.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


class _WCS:
    @staticmethod
    def wcs_pix2world(xy, origin):
        (x, y), = xy
        return [(150.0 + x / 3600.0, 2.0 + y / 3600.0)]


def _pipeline(scenes, damping: float = 0.8) -> Pipeline:
    pipe = Pipeline.__new__(Pipeline)
    pipe.all_scenes = [scenes]      # `scenes` is a read-only view of band 0
    pipe.wcs = [_WCS()]
    pipe.config = SimpleNamespace(astrom_damping=damping)
    return pipe


def _coeffs(order: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.3, size=2 * n_terms(order))


# ------------------------------------------------------------------- order
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_order_round_trips_through_coefficient_length(order):
    assert _cheb_order_from_coeffs(_coeffs(order)) == order


@pytest.mark.parametrize("bad", [None, np.zeros(0), np.zeros(1), np.zeros(8)])
def test_uninterpretable_coefficients_are_none(bad):
    # 8 is 2x4, and 4 is not n_terms of any order
    assert _cheb_order_from_coeffs(bad) is None


# -------------------------------------------------------------- the table
def test_scene_fit_table_stores_the_solution_not_a_refit():
    """The stored field must equal the anchor solution at arbitrary points.

    This is the distinction the extension exists for: sampling the applied
    shifts of every template gives a similar but different field, so a test
    that only compared scene centres would pass on the wrong data.
    """
    order = 2
    coeffs = _coeffs(order, seed=7)
    scene = _Scene(1, coeffs, order, x0=120.0, y0=240.0, sx=64.0, sy=48.0)
    tab = _pipeline([scene])._scene_fit_table()

    row = tab[0]
    assert row["shift_order"] == order
    assert row["n_coeff"] == 2 * n_terms(order)

    want = AstroCorrect.build_poly_predictor(coeffs, 120.0, 240.0, order, 64.0, 48.0)
    got = AstroCorrect.build_poly_predictor(
        np.asarray(row["shift_coeff"][: row["n_coeff"]]),
        row["shift_x0"], row["shift_y0"], int(row["shift_order"]),
        row["shift_sx"], row["shift_sy"],
    )
    probe_x = np.array([0.0, 55.0, 120.0, 400.0])
    probe_y = np.array([10.0, 300.0, 240.0, 90.0])
    assert np.allclose(np.column_stack(got(probe_x, probe_y)),
                       np.column_stack(want(probe_x, probe_y)), rtol=0, atol=1e-12)


def test_mixed_orders_pad_to_the_widest_and_cut_back_by_n_coeff():
    """Order is per scene: a saturated scene is forced to 0 beside an order-2."""
    scenes = [_Scene(1, _coeffs(0, 1), 0), _Scene(2, _coeffs(2, 2), 2),
              _Scene(3, _coeffs(1, 3), 1)]
    tab = _pipeline(scenes)._scene_fit_table()

    assert list(tab["shift_order"]) == [0, 2, 1]
    assert tab["shift_coeff"].shape[1] == 2 * n_terms(2)
    for row, scene in zip(tab, scenes):
        kept = np.asarray(row["shift_coeff"][: row["n_coeff"]])
        assert np.allclose(kept, scene.shifts)
        # everything past n_coeff is padding, never a silent zero
        assert np.isnan(np.asarray(row["shift_coeff"][row["n_coeff"]:])).all()


def test_scene_without_a_shift_solution_is_flagged_not_faked():
    tab = _pipeline([_Scene(1, None, None)])._scene_fit_table()
    assert tab["shift_order"][0] == -1
    assert tab["n_coeff"][0] == 0
    assert np.isnan(tab["shift_x0"][0])
    assert np.isnan(np.asarray(tab["shift_coeff"][0])).all()


def test_robust_pass_columns_follow_the_report():
    rep = SimpleNamespace(applied=True, n_rejected=4, sys_floor=0.011, n_eff=17.5)
    tab = _pipeline([_Scene(1, _coeffs(1), 1, report=rep),
                     _Scene(2, _coeffs(1), 1, report=None)])._scene_fit_table()
    assert list(tab["astrom_robust"]) == [1, 0]
    assert list(tab["astrom_nreject"]) == [4, -1]
    assert tab["astrom_floor"][0] == pytest.approx(0.011)
    assert np.isnan(tab["astrom_floor"][1])


# ---------------------------------------------------------------- on disk
def test_extension_round_trips_beside_the_fit_table(tmp_path):
    """Written as a second HDU, read back by EXTNAME, fit table untouched."""
    scenes = [_Scene(1, _coeffs(0, 1), 0), _Scene(2, _coeffs(2, 2), 2)]
    path = tmp_path / "band_fit_table.fits"
    Table({"id": [1, 2, 3], "flux_1": [1.0, 2.0, 3.0]}).write(path)

    hdu = fits.BinTableHDU(_pipeline(scenes)._scene_fit_table().as_array(),
                           name="SCENES")
    fits.append(path, hdu.data, hdu.header)

    with fits.open(path) as hdul:
        assert [h.name for h in hdul] == ["PRIMARY", "", "SCENES"]
    assert Table.read(path, hdu=1).colnames == ["id", "flux_1"]

    back = Table.read(path, hdu="SCENES")
    assert list(back["shift_order"]) == [0, 2]
    for row, scene in zip(back, scenes):
        assert np.allclose(np.asarray(row["shift_coeff"][: row["n_coeff"]]),
                           scene.shifts)


def test_damping_travels_with_the_coefficients():
    """The stored field is undamped; the applied shift is ``damping`` times it.

    Without the factor beside them the coefficients cannot be turned back into
    what the sources actually moved, and the config in force at read time is
    not necessarily the one that produced the run.
    """
    tab = _pipeline([_Scene(1, _coeffs(1), 1), _Scene(2, None, None)],
                    damping=0.8)._scene_fit_table()
    assert tab["astrom_damping"][0] == pytest.approx(0.8)
    assert np.isnan(tab["astrom_damping"][1])   # nothing solved, nothing applied


def test_restore_scene_fit_puts_the_solution_back(tmp_path):
    """The read side: regrouped scenes get their solver state back by id."""
    scenes = [_Scene(1, _coeffs(0, 1), 0), _Scene(2, _coeffs(2, 2), 2),
              _Scene(3, None, None)]
    pipe = _pipeline(scenes)
    table = pipe._scene_fit_table()

    # fresh scenes, as regrouped from id_scene: no solver state at all
    bare = [_Scene(1, None, None), _Scene(2, None, None), _Scene(3, None, None)]
    reader = _pipeline(bare)
    reader.scene_fit = table
    assert reader.restore_scene_fit(bare) == 3

    assert np.allclose(bare[0].shifts, scenes[0].shifts)
    assert np.allclose(bare[1].shifts, scenes[1].shifts)
    assert bare[2].shifts is None            # never solved, stays unsolved
    _, (x0, y0), (sx, sy) = bare[1].shift_basis
    assert (x0, y0, sx, sy) == (100.0, 200.0, 50.0, 50.0)
    assert bare[0].astrom_niter == 3
    assert bare[0].astrom_converged is True


def test_restore_is_a_no_op_without_the_extension():
    """A run written before SCENES existed reloads exactly as it used to."""
    bare = [_Scene(1, None, None)]
    pipe = _pipeline(bare)
    pipe.scene_fit = None
    assert pipe.restore_scene_fit(bare) == 0
    assert bare[0].shifts is None


def test_stored_field_differs_from_a_refit_on_applied_shifts():
    """Why the coefficients are stored rather than recovered from the shifts.

    A curved field sampled at the sources and refitted does not come back the
    same, so a reconstruction that refits is not the solution the fit used.
    """
    order = 2
    coeffs = _coeffs(order, seed=11)
    x0, y0, sx, sy = 120.0, 240.0, 64.0, 48.0
    predict = AstroCorrect.build_poly_predictor(coeffs, x0, y0, order, sx, sy)

    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 300, size=(30, 2))
    applied = np.column_stack(predict(pos[:, 0], pos[:, 1]))
    # the sources only sample the field where they happen to sit, so a lower
    # order refit -- what a rebuild would guess without the stored order --
    # cannot reproduce it
    design = np.vstack([cheb_basis((px - x0) / sx, (py - y0) / sy, 1)
                        for px, py in pos])
    beta, *_ = np.linalg.lstsq(design, applied, rcond=None)
    refit = design @ beta
    assert not np.allclose(refit, applied, atol=1e-6)
