import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.ndimage import shift as nd_shift
from scipy.signal import fftconvolve

from pathlib import Path
from utils import make_simple_data
from mophongo.psf import PSF
from mophongo.templates import Templates, Template
from mophongo.fit import SparseFitter, FitConfig
from mophongo.astrometry import AstroCorrect, AstroMap, cheb_basis
from mophongo.scene import generate_scenes, make_scene_basis, assemble_scene_system_AB
from mophongo.scene_fitter import SceneFitter, build_normal
from utils import save_diagnostic_image


# ---------------------------------------------------------------------------
# Shared helpers for the scene-solver astrometry tests
# ---------------------------------------------------------------------------


def _assign_solution(fitter, sol) -> None:
    """Write a SceneFitter solution back onto the fitter and its templates."""
    fitter.solution = sol.flux
    for tmpl, flux, err in zip(fitter.templates, sol.flux, sol.err):
        tmpl.flux = float(flux)
        tmpl.err = float(err)


def _make_gaussian_psf_array(fwhm: float, size: int = 51) -> np.ndarray:
    """Return a normalised 2-D Gaussian PSF of given FWHM (in pixels)."""
    sig = fwhm / 2.355
    y, x = np.mgrid[:size, :size] - size // 2
    p = np.exp(-0.5 * (y ** 2 + x ** 2) / sig ** 2)
    return p / p.sum()


def _collect_to_shifts(scenes) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of (dx, dy) stored in tmpl.to_shift across all scenes."""
    dx_list, dy_list = [], []
    for scene in scenes:
        for tmpl in scene.templates:
            ts = getattr(tmpl, "to_shift", None)
            if ts is not None and (abs(ts[0]) > 1e-6 or abs(ts[1]) > 1e-6):
                dx_list.append(float(ts[0]))
                dy_list.append(float(ts[1]))
    return np.array(dx_list), np.array(dy_list)


def _run_scenes(templates, science, weight, cfg):
    """Generate and solve all scenes; return the scene list."""
    scenes, _ = generate_scenes(
        templates,
        science,
        weight,
        coupling_thresh=cfg.scene_coupling_thresh,
        snr_thresh_astrom=cfg.snr_thresh_astrom,
        minimum_bright=cfg.scene_minimum_bright,
    )
    for scene in scenes:
        scene.set_band(science, weight, config=cfg)
        scene.solve(config=cfg, apply_shifts=False)
    return scenes


def test_polynomial_astrometry_reduces_residual(tmp_path):
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=151, peak_snr=5, seed=42
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    shx, shy = 0.6, -0.5
    images[1] = nd_shift(images[1], (shy, shx))

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    fitter = SparseFitter(tmpls.templates, images[1], wht[1], FitConfig())
    fitter.build_normal()
    sol = SceneFitter.solve(fitter.ata, fitter.atb, config=FitConfig())
    _assign_solution(fitter, sol)
    res0 = fitter.residual()

    ac = AstroCorrect(FitConfig())
    ac.fit(tmpls.templates, res0, fitter.solution)

    rhx, rhy = ac(np.array([[50.0, 50.0]]))

    print(f"Astrometry correction shifts: {rhx}, {rhy}")
    print(f"input shifts: {shx}, {shy}")

    assert abs(rhx[0] - shx) < 0.3
    assert abs(rhy[0] - shy) < 0.3

    tmp_path = Path("../tmp")
    tmp_path.mkdir(exist_ok=True)
    fname = tmp_path / "diagnostic_poly_shift.png"
    model = images[1] - res0
    save_diagnostic_image(
        fname,
        truth,
        images[0],
        images[1],
        model,
        res0,
        segmap=segmap,
        catalog=catalog,
    )


def test_gp_astrometry_returns_models():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=101, peak_snr=5, seed=1
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    images[1] = nd_shift(images[1], (-0.5, 0.6))

    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)

    fitter = SparseFitter(tmpls.templates, images[1], wht[1], FitConfig())
    fitter.build_normal()
    sol = SceneFitter.solve(fitter.ata, fitter.atb, config=FitConfig())
    _assign_solution(fitter, sol)
    res = fitter.residual()

    cfg = FitConfig(astrom_model="gp", astrom_kwargs={"gp": {"length_scale": 30.0}})
    ac = AstroCorrect(cfg)
    ac.fit(tmpls.templates, res, fitter.solution)

    dx, dy = ac(np.array([[50.0, 50.0]]))
    assert isinstance(float(dx[0]), float)
    assert isinstance(float(dy[0]), float)


def test_astromap_recovers_shift():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=151, peak_snr=5, seed=7
    )
    shx, shy = 0.4, -0.3
    shifted = nd_shift(images[0], (shy, shx))
    amap = AstroMap(order=1, snr_threshold=3.0)
    catalog["snr"] = 100.0  # make_simple_data has no snr column; pass the cut
    amap.fit(images[0], shifted, catalog)
    dx, dy = amap(np.array([[75.0, 75.0]]))
    assert abs(dx[0] - shx) < 0.3
    assert abs(dy[0] - shy) < 0.3


def test_apply_template_shifts_uses_shift_field():
    data = np.zeros((7, 7))
    data[3, 3] = 1.0
    tmpl = Template(data, (3.0, 3.0), (7, 7), label=1)
    tmpl.to_shift = np.array([0.5, -0.25])

    Templates.apply_template_shifts([tmpl])

    expected = nd_shift(
        data,
        (-0.25, 0.5),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    assert np.allclose(tmpl.data, expected)
    # ours accumulates the applied shift on the template instead of moving
    # the catalog position
    assert np.allclose(tmpl.shifted, (0.5, -0.25))
    assert np.allclose(tmpl.to_shift, (0.0, 0.0))


def test_build_poly_predictor_returns_expected_shift():
    order = 1
    betax = np.array([1.0, 0.2, -0.3])
    betay = np.array([-0.5, 0.1, 0.05])
    coeffs = np.concatenate([betax, betay])
    x0, y0 = 10.0, 20.0
    predict = AstroCorrect.build_poly_predictor(coeffs, x0, y0, order)

    x, y = 11.0, 19.0
    dx, dy = predict(x, y)
    phi = cheb_basis(x - x0, y - y0, order)
    assert np.allclose(dx, phi @ betax)
    assert np.allclose(dy, phi @ betay)


def test_cheb_basis_handles_domain_edges():
    order = 2
    phi = cheb_basis(1.0, -1.0, order)
    tx = [chebval(1.0, [0] * i + [1]) for i in range(order + 1)]
    ty = [chebval(-1.0, [0] * j + [1]) for j in range(order + 1)]
    expected = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            expected.append(tx[i] * ty[j])
    assert np.allclose(phi, expected)


# ---------------------------------------------------------------------------
# Test 1: end-to-end scene-solver shift recovery
# ---------------------------------------------------------------------------

def test_scene_solver_recovers_known_shift():
    """Scene.solve() recovers a known global pixel shift applied to the science image.

    A global (dx, dy) shift is applied to the low-resolution science image.  After
    running generate_scenes + Scene.solve(apply_shifts=False) the per-template
    to_shift values should have a mean close to the injected shift.
    """
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=15, size=151, peak_snr=20, seed=42
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    true_dx, true_dy = 0.6, -0.5
    science = nd_shift(images[1], (true_dy, true_dx))
    weight = wht[1]

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    cfg = FitConfig(
        fit_astrometry_joint=True,
        snr_thresh_astrom=3.0,
        scene_minimum_bright=2,
        scene_coupling_thresh=0.01,
        astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
    )

    scenes = _run_scenes(tmpls.templates, science, weight, cfg)
    dx_arr, dy_arr = _collect_to_shifts(scenes)

    print(f"\n[test1] True shift: dx={true_dx:.3f}, dy={true_dy:.3f} px")
    print(f"[test1] Recovered (n={len(dx_arr)}): "
          f"dx={dx_arr.mean():.3f}±{dx_arr.std():.3f}, "
          f"dy={dy_arr.mean():.3f}±{dy_arr.std():.3f}")

    assert len(dx_arr) > 0, "No shifts were recovered — check bright-source threshold"
    assert abs(dx_arr.mean() - true_dx) < 0.3, (
        f"dx recovered {dx_arr.mean():.3f}, expected {true_dx:.3f}"
    )
    assert abs(dy_arr.mean() - true_dy) < 0.3, (
        f"dy recovered {dy_arr.mean():.3f}, expected {true_dy:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 2: shift recovery degrades with PSF width
# ---------------------------------------------------------------------------

def test_scene_shift_uncertainty_scales_with_psf_width():
    """Shift recovery is more accurate with a narrower PSF (stronger gradient signal).

    The same true shift is injected into two science images: one convolved with a
    narrow PSF (F770W-like, FWHM≈6 px) and one with a wide PSF (F1800W-like,
    FWHM≈15 px).  The narrow-PSF case should have a smaller shift-recovery error.
    """
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=20, size=151, peak_snr=50, seed=7
    )
    psf_hi = PSF.from_array(psfs[0])

    narrow_arr = _make_gaussian_psf_array(fwhm=6.0, size=51)
    wide_arr = _make_gaussian_psf_array(fwhm=15.0, size=51)

    science_narrow = fftconvolve(truth, narrow_arr, mode="same")
    science_wide = fftconvolve(truth, wide_arr, mode="same")

    rng = np.random.default_rng(7)
    noise_std = float(truth.max()) / 50.0
    science_narrow += rng.normal(0, noise_std, science_narrow.shape)
    science_wide += rng.normal(0, noise_std, science_wide.shape)
    weight = np.ones_like(truth) / noise_std ** 2

    true_dx, true_dy = 0.4, -0.35
    science_narrow = nd_shift(science_narrow, (true_dy, true_dx))
    science_wide = nd_shift(science_wide, (true_dy, true_dx))

    cfg = FitConfig(
        fit_astrometry_joint=True,
        snr_thresh_astrom=3.0,
        scene_minimum_bright=2,
        scene_coupling_thresh=0.01,
        astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
    )

    positions = list(zip(catalog["x"], catalog["y"]))

    kernel_narrow = psf_hi.matching_kernel(PSF.from_array(narrow_arr))
    tmpls_narrow = Templates.from_image(images[0], segmap, positions, kernel_narrow)
    scenes_narrow = _run_scenes(tmpls_narrow.templates, science_narrow, weight, cfg)
    dx_n, dy_n = _collect_to_shifts(scenes_narrow)

    kernel_wide = psf_hi.matching_kernel(PSF.from_array(wide_arr))
    tmpls_wide = Templates.from_image(images[0], segmap, positions, kernel_wide)
    scenes_wide = _run_scenes(tmpls_wide.templates, science_wide, weight, cfg)
    dx_w, dy_w = _collect_to_shifts(scenes_wide)

    err_narrow = float(np.sqrt((dx_n.mean() - true_dx) ** 2 + (dy_n.mean() - true_dy) ** 2))
    err_wide = float(np.sqrt((dx_w.mean() - true_dx) ** 2 + (dy_w.mean() - true_dy) ** 2))

    print(f"\n[test2] True shift: dx={true_dx:.3f}, dy={true_dy:.3f} px")
    print(f"[test2] Narrow PSF (FWHM=6): dx={dx_n.mean():.3f}, dy={dy_n.mean():.3f} — error={err_narrow:.4f} px")
    print(f"[test2] Wide PSF  (FWHM=15): dx={dx_w.mean():.3f}, dy={dy_w.mean():.3f} — error={err_wide:.4f} px")

    assert len(dx_n) > 0 and len(dx_w) > 0, "No shifts recovered in one of the PSF cases"
    assert err_narrow < err_wide, (
        f"Expected narrow PSF error ({err_narrow:.4f}) < wide PSF error ({err_wide:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 3: shift estimate scales as 1/alpha0 — biased flux biases shifts
# ---------------------------------------------------------------------------

def test_scene_shift_depends_on_alpha0_scale():
    """Shift estimate ∝ 1/alpha0_scale: underestimated fluxes inflate shifts.

    The shift signal is bB ∝ alpha0 × image_gradient and the gradient information
    matrix BB ∝ alpha0².  Their ratio shift ≈ bB/BB ∝ 1/alpha0.  When alpha0 is
    scaled by factor k, the recovered shift should scale by 1/k.  This confirms
    that the old reg_astrom flux bias (alpha0 ≈ 0.57×) was inflating shifts by ~1.75×.
    """
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=10, size=101, peak_snr=30, seed=55
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    true_dx, true_dy = 0.5, 0.4
    science = nd_shift(images[1], (true_dy, true_dx))
    weight = wht[1]

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    cfg = FitConfig(
        fit_astrometry_joint=True,
        snr_thresh_astrom=0.0,
        scene_minimum_bright=1,
        scene_coupling_thresh=0.005,
        astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
    )

    # Build a single scene from all templates so the has_shift >= 2 requirement is met.
    # This keeps the test focused on the alpha0 → shift dependency, not scene partitioning.
    from mophongo.scene import Scene
    all_templates = list(tmpls.templates)
    A, b, _ = build_normal(all_templates, science, weight)
    scene = Scene(
        id=1,
        templates=all_templates,
        fitter=SceneFitter(),
        image=science,
        weights=weight,
        config=cfg,
    )
    scene.A = A
    scene.b = b
    d = A.diagonal()
    alpha0_true = np.divide(b, d, out=np.zeros_like(b), where=d > 0)
    bright_mask = np.ones(len(scene.templates), dtype=bool)
    basis, (x0, y0), (Sx, Sy) = make_scene_basis(scene.templates, bright_mask, order=0)

    recovered = {}
    for scale in (0.5, 1.0, 2.0):
        alpha0 = alpha0_true * scale
        AB, BB, bB = assemble_scene_system_AB(
            scene.templates, science, weight, basis,
            alpha0=alpha0, order=0, include_y=True, ab_from_bright_only=False,
        )
        sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=cfg)
        if sol.shifts is not None and len(sol.shifts) >= 2:
            # order=0: shifts = [beta_x, beta_y], each length p=1
            recovered[scale] = (float(sol.shifts[0]), float(sol.shifts[1]))

    print(f"\n[test3] True shift: dx={true_dx:.3f}, dy={true_dy:.3f} px")
    for scale, (rdx, rdy) in recovered.items():
        print(f"[test3] alpha0 scale={scale:.1f}: dx={rdx:.4f}, dy={rdy:.4f}")

    assert set(recovered) >= {0.5, 1.0, 2.0}, "Not all alpha0 scales produced shift estimates"

    dx_half, _ = recovered[0.5]
    dx_one, _ = recovered[1.0]
    dx_two, _ = recovered[2.0]

    # shift ∝ 1/scale: half alpha0 → double shift, double alpha0 → half shift
    ratio_half = dx_half / dx_one if abs(dx_one) > 1e-6 else float("nan")
    ratio_two = dx_two / dx_one if abs(dx_one) > 1e-6 else float("nan")
    print(f"[test3] dx(0.5×)/dx(1×) = {ratio_half:.3f}  (expected ~2.0)")
    print(f"[test3] dx(2×) /dx(1×) = {ratio_two:.3f}  (expected ~0.5)")
    assert abs(ratio_half - 2.0) < 0.5, f"Expected ratio ≈ 2, got {ratio_half:.3f}"
    assert abs(ratio_two - 0.5) < 0.25, f"Expected ratio ≈ 0.5, got {ratio_two:.3f}"

    # The unbiased (scale=1.0) estimate should match the true shift
    assert abs(dx_one - true_dx) < 0.3, f"Unbiased dx={dx_one:.4f}, true={true_dx}"
    assert abs(recovered[1.0][1] - true_dy) < 0.3, (
        f"Unbiased dy={recovered[1.0][1]:.4f}, true={true_dy}"
    )


# ---------------------------------------------------------------------------
# Test 4: iterative shift application converges (residual shift shrinks)
# ---------------------------------------------------------------------------

def test_scene_shift_iteration_converges():
    """Residual shifts in the second iteration are smaller than in the first.

    After applying the first-pass shifts to the templates, the second pass should
    see only small residual offsets.  If the solver is diverging, second-pass
    |to_shift| would be larger than or equal to the first pass — a red flag for
    the iterative scheme used by fit_astrometry_niter=2.
    """
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=15, size=151, peak_snr=20, seed=77
    )

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    true_dx, true_dy = 0.5, -0.4
    science = nd_shift(images[1], (true_dy, true_dx))
    weight = wht[1]

    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    cfg = FitConfig(
        fit_astrometry_joint=True,
        snr_thresh_astrom=3.0,
        scene_minimum_bright=2,
        scene_coupling_thresh=0.01,
        astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
    )

    # --- Pass 1: solve, record to_shift, then apply ---
    scenes1 = _run_scenes(tmpls.templates, science, weight, cfg)
    dx1, dy1 = _collect_to_shifts(scenes1)
    mag1 = float(np.sqrt(dx1 ** 2 + dy1 ** 2).mean()) if len(dx1) > 0 else 0.0

    # Apply the first-pass shifts to the templates in-place
    all_templates = [t for s in scenes1 for t in s.templates]
    Templates.apply_template_shifts(all_templates)

    # --- Pass 2: rebuild normal equations from shifted templates, solve again ---
    scenes2 = _run_scenes(tmpls.templates, science, weight, cfg)
    dx2, dy2 = _collect_to_shifts(scenes2)
    mag2 = float(np.sqrt(dx2 ** 2 + dy2 ** 2).mean()) if len(dx2) > 0 else 0.0

    print(f"\n[test4] True shift: dx={true_dx:.3f}, dy={true_dy:.3f} px")
    print(f"[test4] Pass 1: mean |shift| = {mag1:.4f} px  "
          f"(dx={dx1.mean():.3f}, dy={dy1.mean():.3f})")
    print(f"[test4] Pass 2: mean |shift| = {mag2:.4f} px  "
          f"(dx={dx2.mean():.3f}, dy={dy2.mean():.3f})")

    assert len(dx1) > 0, "Pass 1 produced no shifts"
    assert mag1 > 0.05, f"Pass-1 shift too small ({mag1:.4f}), test may not be exercising the solver"
    assert mag2 < mag1, (
        f"Pass-2 residual shift ({mag2:.4f} px) should be smaller than pass-1 ({mag1:.4f} px)"
    )
