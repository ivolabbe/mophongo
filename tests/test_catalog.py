from pathlib import Path

import numpy as np
import pytest

from mophongo.catalog import Catalog, _deblend_label_info


def test_catalog_from_fits_smoke():
    sci = Path("data/uds-test-f444w_sci.fits")
    wht = Path("data/uds-test-f444w_wht.fits")
    if not sci.exists() or not wht.exists():
        pytest.skip("Test data not available")

    cat = Catalog.from_fits(
        sci,
        wht,
        params={
            "kernel_size": 4.0,
            "detect_threshold": 1.0,
            "dilate_segmap": 2,
            "detect_npixels": 5,
        },
    )

    assert cat.segmap.shape == cat.sci.shape
    assert cat.ivar.shape == cat.sci.shape
    assert len(cat.table) > 0
    assert np.all(np.isfinite(cat.ivar))
    assert "is_deblended" in cat.table.colnames


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
