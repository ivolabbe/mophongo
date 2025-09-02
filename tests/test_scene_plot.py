import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mophongo.scene import generate_scenes
from mophongo.templates import Templates
from mophongo.psf import PSF
from utils import make_simple_data
import types


def test_scene_plot():
    images, segmap, catalog, psfs, _, wht = make_simple_data(nsrc=3, size=51)
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    scenes, _ = generate_scenes(tmpls.templates, images[1], wht[1], minimum_bright=1)
    scene = scenes[0]
    scene.solution = types.SimpleNamespace(
        flux=np.zeros(len(scene.templates)), err=np.zeros(len(scene.templates))
    )
    for t in scene.templates:
        t.flux = 0.0
        t.err = 0.0
    fig, ax = scene.plot(images[0])
    assert fig is not None
    plt.close(fig)
