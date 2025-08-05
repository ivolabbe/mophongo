import numpy as np
import pytest

from mophongo.templates import Template, Templates


def test_prune_and_dedupe_sets_norm_and_removes_duplicates():
    data = np.zeros((5, 5))
    data[2, 2] = 1.0
    t1 = Template(data, (2, 2), (5, 5), label=1)
    t2 = Template(data, (2, 2), (5, 5), label=2)
    weights = np.ones_like(data)

    kept = Templates.prune_and_dedupe([t1, t2], weights)

    assert len(kept) == 1
    assert pytest.approx(1.0) == kept[0].norm
