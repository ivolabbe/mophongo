import numpy as np
from pathlib import Path
from astropy.io import fits

from mophongo.catalog import Catalog
from photutils.segmentation import SegmentationImage, SourceCatalog
import matplotlib.pyplot as plt
from utils import lupton_norm, label_segmap


def test_catalog(tmp_path):
    sci = Path('data/uds-test-f444w_sci.fits')
    wht = Path('data/uds-test-f444w_wht.fits')
    out = tmp_path / 'uds-test-f444w_ivar.fits'
    cat = Catalog.from_fits(sci, wht, ivar_outfile=out)
    cat.catalog["x"] = cat.catalog["xcentroid"]
    cat.catalog["y"] = cat.catalog["ycentroid"]

    assert cat.segmap.shape == cat.sci.shape
    assert cat.ivar.shape == cat.sci.shape
    assert len(cat.catalog) > 0
    assert np.all(np.isfinite(cat.ivar))
    assert out.exists()
    hdr = fits.getheader(out)
    assert 'CRPIX1' in hdr

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cat.det_img, origin="lower", cmap="gray", norm=lupton_norm(cat.det_img))
    ax.imshow(cat.segmap, origin="lower", cmap="nipy_spectral", alpha=0.3)
    label_segmap(ax, cat.segmap, cat.catalog)

    seg = SegmentationImage(cat.segmap)
    scat = SourceCatalog(cat.sci, seg, error=np.sqrt(1.0 / cat.ivar))
    for aper in scat.kron_aperture:
        aper.plot(ax=ax, color="white", lw=0.5)

    diag = tmp_path / "catalog_diagnostic.png"
    fig.savefig(diag, dpi=150)
    plt.close(fig)
    assert diag.exists()
