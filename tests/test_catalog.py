import numpy as np
from pathlib import Path

from mophongo.catalog import CatalogBuilder


def test_catalog_builder(tmp_path):
    sci = Path('data/uds-test-f444w_sci.fits')
    wht = Path('data/uds-test-f444w_wht.fits')
    out = tmp_path / 'uds-test-f444w_ivar.fits'
    cat = CatalogBuilder.from_fits(sci, wht, ivar_outfile=out)

    assert cat.segmap.shape == cat.sci.shape
    assert cat.ivar.shape == cat.sci.shape
    assert len(cat.catalog) > 0
    assert np.all(np.isfinite(cat.ivar))
    assert out.exists()

