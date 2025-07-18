import numpy as np
import matplotlib
matplotlib.use('Agg')
from mophongo.utils import gaussian, CircularApertureProfile


def test_circular_aperture_profile_normalization():
    img = gaussian((51, 51), 5.0, 5.0, flux=100.0)
    xycen = (25.0, 25.0)
    edges = np.linspace(0, 20, 21)
    radii = np.linspace(1, 20, 20)
    prof = CircularApertureProfile(img, xycen, edges, cog_radii=radii, norm_radius=5.0)

    from scipy.interpolate import PchipInterpolator

    rp_norm_val = PchipInterpolator(prof.radius, prof.profile)(5.0)
    cog_norm_val = PchipInterpolator(prof.cog.radius, prof.cog.profile)(5.0)
    assert np.isclose(rp_norm_val, 1.0, atol=1e-3)
    assert np.isclose(cog_norm_val, 1.0, atol=1e-3)

    assert np.isclose(prof.gaussian_fwhm, 5.0, atol=1.0)
    assert np.isclose(prof.moffat_fwhm, 5.0, atol=2.0)

    fig, _ = prof.plot(show=False)
    assert fig is not None
    
    prof2 = CircularApertureProfile(img, xycen, edges, cog_radii=radii, norm_radius=5.0)
    ratio = prof.cog_ratio(prof2)
    assert np.allclose(ratio, 1.0, atol=1e-6)

