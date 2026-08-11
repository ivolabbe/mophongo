

def test_as_label_array_accepts_float_segmaps():
    """Segmaps stored as float must be usable as labels.

    MINERVA UDS and EGS ship int32 segmaps, COSMOS ships the same labels as
    float64, and SegmentationImage rejects non-integer input outright.
    """
    import numpy as np
    import pytest

    from mophongo.utils import as_label_array

    ints = np.array([[0, 1], [2, 2]], dtype=np.int32)
    assert as_label_array(ints) is ints

    floats = np.array([[0.0, 1.0], [2085561.0, 0.0]], dtype=np.float64)
    out = as_label_array(floats)
    assert np.issubdtype(out.dtype, np.integer)
    assert out.max() == 2085561

    with pytest.raises(ValueError, match="non-integer"):
        as_label_array(np.array([[0.0, 0.5]]))
