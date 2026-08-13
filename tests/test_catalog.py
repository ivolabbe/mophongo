import numpy as np

from mophongo.catalog import Catalog, _deblend_label_info


def test_deblend_label_info_marks_children_split_from_parent():
    parent = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 2, 2],
        ],
        dtype=int,
    )
    final = np.array(
        [
            [3, 3, 0, 5, 5],
            [3, 4, 0, 5, 5],
            [0, 0, 0, 5, 5],
        ],
        dtype=int,
    )

    info = _deblend_label_info(final, parent)

    assert info[3] == (1, 2, True)
    assert info[4] == (1, 2, True)
    assert info[5] == (2, 1, False)


def _noisy_field(seed=7, shape=(200, 160)):
    """Science + weight pair with the awkward pixels: zero, NaN weight, NaN sci."""
    rng = np.random.default_rng(seed)
    sci = rng.normal(0.0, 2e-3, shape).astype(np.float32)
    sci[40:52, 40:52] += 0.08  # something for the source mask to find
    wht = np.full(shape, 3e5, np.float32)
    wht[:4] = 0.0
    wht[-3:] = np.nan
    sci[15, 15] = np.nan
    return sci, wht


def test_get_bg_and_ivar_does_not_copy_on_byte_order():
    """Big-endian input gives identical output and is not cast up front.

    FITS stores big-endian, so a memory-mapped mosaic arrives as '>f4'.
    Casting it to native float32 would copy the whole array -- 3.5 GB per
    input on a MINERVA detection grid -- for a difference nothing downstream
    needs.
    """
    from mophongo.catalog import _as_float, get_bg_and_ivar

    big = np.zeros((4, 4), dtype=">f4")
    assert _as_float(big) is big
    # a non-float input still has to be converted
    assert _as_float(np.zeros((2, 2), np.int32)).dtype == np.float32

    sci, wht = _noisy_field()
    for need_bg in (True, False):
        bg_n, ivar_n = get_bg_and_ivar(sci, wht, need_bg=need_bg)
        bg_b, ivar_b = get_bg_and_ivar(
            sci.astype(">f4"), wht.astype(">f4"), need_bg=need_bg
        )
        assert np.array_equal(ivar_n, ivar_b)
        assert ivar_n.dtype == ivar_b.dtype == np.float32
        if need_bg:
            assert np.array_equal(bg_n, bg_b)
            assert bg_n.dtype == np.float32
        else:
            assert bg_n is None and bg_b is None


def test_get_bg_and_ivar_masks_ivar_and_background_differently():
    """ivar needs finite science; the background only needs a good weight.

    A lone bad science pixel carries no information, so its inverse variance
    is zeroed -- but the background surface there is still defined, and
    punching a hole in a smooth fit would be worse than keeping it.
    """
    from mophongo.catalog import get_bg_and_ivar

    sci, wht = _noisy_field()
    bg, ivar = get_bg_and_ivar(sci, wht, need_bg=True)

    assert ivar[0, 0] == 0.0        # zero weight
    assert ivar[-1, 0] == 0.0       # NaN weight
    assert ivar[15, 15] == 0.0      # NaN science
    assert bg[0, 0] == 0.0          # zero weight
    assert bg[-1, 0] == 0.0         # NaN weight
    assert bg[15, 15] != 0.0        # NaN science: background still defined
