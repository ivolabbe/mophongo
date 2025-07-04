# Mophongo

A lightweight photometry pipeline.

## Usage

```python
from astropy.io import fits
from astropy.table import Table

from mophongo.pipeline import run_photometry

images = [fits.getdata("image1.fits"), fits.getdata("image2.fits")]
segmap = fits.getdata("segmap.fits")
catalog = Table.read("catalog.fits")
psfs = [fits.getdata("psf1.fits"), fits.getdata("psf2.fits")]

flux_table, residual = run_photometry(images, segmap, catalog, psfs)
```

The function returns the catalog with new flux columns and the residual image
from the fit.
