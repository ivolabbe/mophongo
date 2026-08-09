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
