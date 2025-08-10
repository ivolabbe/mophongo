"""Tests for template downsampling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mophongo.templates import Template
from mophongo.utils import bin2d_mean, downsample_psf


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("h,w", [(5, 7), (6, 6), (9, 12)])
@pytest.mark.parametrize("y0,x0", [(0, 0), (1, 2), (4, 5)])
def test_downsample_centroids(k: int, h: int, w: int, y0: int, x0: int) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(h, w))
    t_hi = Template.__new__(Template)
    t_hi.data = data
    t_hi.bbox_original = ((y0, y0 + h - 1), (x0, x0 + w - 1))
    t_hi.slices_original = (slice(y0, y0 + h), slice(x0, x0 + w))
    t_hi.slices_cutout = (slice(0, h), slice(0, w))
    t_hi.input_position_cutout = ((h - 1) / 2, (w - 1) / 2)
    t_hi.input_position_original = (y0 + (h - 1) / 2, x0 + (w - 1) / 2)

    t_lo = t_hi.downsample(k)

    shift = (k - 1) / 2.0
    y_expect = ((h - 1) / 2 - shift) / k
    x_expect = ((w - 1) / 2 - shift) / k
    y_lo, x_lo = t_lo.input_position_cutout
    assert np.allclose([y_lo, x_lo], [y_expect, x_expect])

    sy, sx = t_lo.slices_original
    assert (sy.start, sy.stop) == (t_lo.bbox[0], t_lo.bbox[1])
    assert (sx.start, sx.stop) == (t_lo.bbox[2], t_lo.bbox[3])

    ny_lo, nx_lo = t_lo.data.shape
    hi_trim = t_hi.data[: ny_lo * k, : nx_lo * k]
    assert np.allclose(
        hi_trim.sum(dtype=np.float64),
        t_lo.data.sum(dtype=np.float64) * k * k,
        rtol=0,
        atol=1e-10,
    )

def test_downsample_stamp():
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from astropy.nddata import block_reduce,block_replicate
    from mophongo.templates import Template

    data_lo = np.arange(20).reshape(4, 5)
    data_hi = block_replicate(data_lo, 2, conserve_sum=True)
    data_lo4 = block_reduce(data_lo, 2, func=np.sum)

    size = np.array([6,6])
    pos = np.array([-0.1, -0.1])
    k = 4
    idx_min = np.ceil(pos - size[::-1] / 2)

    new_size, idx_min_new = Template.block_aligned(pos,size,k)

    print(new_size, idx_min_new)
    print(idx_min_new, idx_min_new + new_size[::-1])
    print(data_lo4)
    print(data_lo)
    print(data_hi)

    t = Template(data_hi,pos,size,block_align=4)
    tlo = t.downsample_wcs(2)
    tlo22 = tlo.downsample_wcs(2)
    tlo4 = t.downsample_wcs(4)
    print(t.data)
    print(tlo.data)
    print(tlo22.data)
    print(tlo4.data)

# size or copy makes no difference 
# %timeit t = Template(np.zeros((2,2)),(0.1,0.1),(1,1),block_align=1)
# %timeit t = Template(np.zeros((300,200)),(0.1,0.1),(2,4),block_align=4 )
# %timeit t = Template(np.zeros((30_000,20_000)),(0.1,0.1),(2,4),block_align=4 )
# %timeit t = Template(np.zeros((300,200)),(0.1,0.1),(2,4),block_align=4,copy=False)
# %timeit t = Template(np.zeros((30_000,20_000)),(0.1,0.1),(2,4),block_align=4, copy=False)

@pytest.mark.parametrize("k", [2, 3])
def test_bin2d_mean_vs_numpy(k: int) -> None:
    arr = np.arange(7 * 11, dtype=float).reshape(7, 11)
    out = bin2d_mean(arr, k)
    ref = np.empty_like(out)
    for iy in range(out.shape[0]):
        for ix in range(out.shape[1]):
            block = arr[iy * k : (iy + 1) * k, ix * k : (ix + 1) * k]
            ref[iy, ix] = block.mean()
    assert np.allclose(out, ref)


@pytest.mark.parametrize("k", [2, 3])
def test_psf_downsample_center(k: int) -> None:
    size = 9
    x = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, x)
    psf = np.exp(-(X ** 2 + Y ** 2) / (2 * 1.5 ** 2))
    psf /= psf.sum()

    psf_lo = downsample_psf(psf, k)
    y, x = np.indices(psf_lo.shape)
    cy = (psf_lo * y).sum() / psf_lo.sum()
    cx = (psf_lo * x).sum() / psf_lo.sum()
    assert np.allclose([cy, cx], [(psf_lo.shape[0] - 1) / 2, (psf_lo.shape[1] - 1) / 2])


def test_cutout2d():
    """Test that cutout2d works as expected."""

    from astropy.nddata import Cutout2D
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    def plot_cutout(cut, data, ax):
        print('Plotting cutout...')
        x0, y0 = cut.input_position_original
        cx, cy = cut.input_position_cutout
        ax.imshow(data, origin='lower')
        ax.scatter(x0, y0, marker='x', color='lightgray', s=100)
        ax.text(x0, y0 - 0.4, f"({cx:.3f}, {cy:.3f})", ha='center', color='lightgray', fontsize=12)
        ax.text(x0, y0 + 0.4, f"({x0:.3f}, {y0:.3f})", ha='center', color='lightgray', fontsize=12)
        x, y = cut.bbox_original[1][0]-0.5, cut.bbox_original[0][0]-0.5
        h, w = cut.bbox_original[0][1] - cut.bbox_original[0][0] + 1, cut.bbox_original[1][1] - cut.bbox_original[1][0] + 1 
        patch = mpatches.Rectangle((x,y), w, h, angle=0.0, color='lightgray', fill=False, lw=1)
        ax.add_patch(patch)
        x, y = np.ceil( (cut.input_position_original - np.array(cut.shape)/2) ) - 0.5
        h, w = cut.shape
        patch = mpatches.Rectangle((x,y), w, h, angle=0.0, color='lightgray', fill=False, lw=2)
        ax.add_patch(patch)
        print(cut.data)
        print(f'{cut.input_position_original=}')
        print(f'{cut.input_position_cutout=}')
        print(f'{cut.position_original=}')
        print(f'{cut.position_cutout=}')
        print(f'{cut.bbox_original=}')

    data = np.arange(81).reshape(9, 9)
    x0, y0 = 0.49, 2.01
    size=2
    cut49 = Cutout2D(data, (0.49, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut50 = Cutout2D(data, (0.50, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut51 = Cutout2D(data, (0.51, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)

    print(data)
    fig, ax = plt.subplots(1,3,figsize=(9, 3))
    plot_cutout(cut49, data, ax[0])
    plot_cutout(cut50, data, ax[1])
    plot_cutout(cut51, data, ax[2])
    plt.tight_layout() 
    plt.show()

    data = np.arange(25).reshape(5, 5)
    size=3
    cut49 = Cutout2D(data, (0.49, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut50 = Cutout2D(data, (0.50, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut51 = Cutout2D(data, (0.51, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)

    fig, ax = plt.subplots(1,3,figsize=(9, 3))
    plot_cutout(cut49, data, ax[0])
    plot_cutout(cut50, data, ax[1])
    plot_cutout(cut51, data, ax[2])
    plt.tight_layout() 

    size=3
    cut49 = Cutout2D(data, (1.49, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut50 = Cutout2D(data, (1.50, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut51 = Cutout2D(data, (1.51, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)

    print(data)
    fig, ax = plt.subplots(1,3,figsize=(9, 3))
    plot_cutout(cut49, data, ax[0])
    plot_cutout(cut50, data, ax[1])
    plot_cutout(cut51, data, ax[2])
    plt.tight_layout() 


    data = np.arange(81).reshape(9, 9)
    x0, y0 = 1.5, 2.01
    size=6
    cut49 = Cutout2D(data, (x0-0.01, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut50 = Cutout2D(data, (x0-0.00, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)
    cut51 = Cutout2D(data, (x0+0.01, y0), size, mode='partial', fill_value=0, limit_rounding_method=np.ceil)

    print(data)
    fig, ax = plt.subplots(1,3,figsize=(9, 3))
    plot_cutout(cut49, data, ax[0])
    plot_cutout(cut50, data, ax[1])
    plot_cutout(cut51, data, ax[2])
    plt.tight_layout() 


    cut = cut50
    print(data)
    for k in cut.__dict__:
        print(f"{k}: {cut.__dict__[k]}")


    origin = np.ceil( (cut.input_position_original - np.array(cut.shape)/2) )

    # fix (otherwise local and original coordinates are rounded differently due to bankers rounding )
    cut.position_cutout = np.round(cut.input_position_original) - cut._origin_original_true

    orig = np.ceil( (cut.input_position_original - np.array(cut.shape)/2) )

    #   assert cutout.shape == (size, size)
    #   assert np.all(cutout == data[y0 - size // 2 : y0 + size // 2 + 1, x0 - size // 2 : x0 + size // 2 + 1])