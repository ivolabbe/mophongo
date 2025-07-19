import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from mophongo.psf_map import PSFRegionMap


def test_psf_region_map(tmp_path):
    footprints = {
        ("f1", 1): Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ("f2", 1): Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        ("f3", 1): Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
    }

    prm = PSFRegionMap.from_footprints(footprints)

    assert prm.regions.psf_key.nunique() > 1

    key_a = prm.lookup_key(0.5, 0.5)
    key_b = prm.lookup_key(1.5, 1.5)
    key_c = prm.lookup_key(2.5, 1.5)

    assert key_a is not None
    assert key_b is not None
    assert key_c is not None
    assert len({key_a, key_b, key_c}) == 3

    fig, ax = plt.subplots()
    prm.regions.plot(column="psf_key", ax=ax, edgecolor="k", cmap="tab20")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    out = tmp_path / "psf_region_demo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    assert out.exists()

