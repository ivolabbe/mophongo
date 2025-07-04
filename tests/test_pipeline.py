import numpy as np
import matplotlib.pyplot as plt
from tests.utils import moffat_psf, simulate_image, fit_fluxes


def test_pipeline_recovers_fluxes_and_noise(tmp_path):
    rng = np.random.default_rng(42)
    shape = (400, 400)
    nsrc = 20
    fwhm_hi = 2.0
    fwhm_lo = 5 * fwhm_hi
    psf_hi = moffat_psf((21, 21), fwhm_hi)
    psf_lo = moffat_psf((51, 51), fwhm_lo)
    margin = 30
    min_sep = 20
    positions = []
    while len(positions) < nsrc:
        y = int(rng.integers(margin, shape[0] - margin))
        x = int(rng.integers(margin, shape[1] - margin))
        if all(np.hypot(y - py, x - px) >= min_sep for py, px in positions):
            positions.append((y, x))
    fluxes_true = rng.uniform(50, 100, size=nsrc)
    noise_std = 0.1

    img_hi = simulate_image(shape, positions, fluxes_true, psf_hi, noise_std, rng)
    img_lo = simulate_image(shape, positions, fluxes_true, psf_lo, noise_std, rng)

    fluxes_rec, model, residual = fit_fluxes(img_lo, positions, psf_lo)

    assert np.allclose(fluxes_rec, fluxes_true, rtol=0.05)
    assert np.isclose(np.std(residual), noise_std, rtol=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(img_hi, origin="lower", cmap="viridis")
    axes[0, 0].set_title("hi res")
    axes[0, 1].imshow(img_lo, origin="lower", cmap="viridis")
    axes[0, 1].set_title("lo res")
    axes[1, 0].imshow(model, origin="lower", cmap="viridis")
    axes[1, 0].set_title("model")
    axes[1, 1].imshow(residual, origin="lower", cmap="viridis")
    axes[1, 1].set_title("residual")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(tmp_path / "diagnostic.png")
