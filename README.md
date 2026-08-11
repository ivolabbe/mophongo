# Mophongo

Template-fitting photometry for multi-band imaging: per-source templates from
a high-resolution detection image are PSF-matched to each low-resolution band
and fit simultaneously for fluxes.

Documentation: https://mophongo.readthedocs.io

## Installation

Install the project dependencies using [Poetry](https://python-poetry.org/) before running the code or tests:

```bash
poetry install
```

Alternatively, you can install in editable mode with pip:

```bash
pip install -e .
```

## Usage

```python
from astropy.io import fits
from astropy.table import Table

from mophongo import pipeline

images = [fits.getdata("detection.fits"), fits.getdata("image.fits")]
segmap = fits.getdata("segmap.fits")      # labels match catalog "id"
catalog = Table.read("catalog.fits")      # columns: id, x, y
psfs = [fits.getdata("psf_hi.fits"), fits.getdata("psf_lo.fits")]
weights = [None, fits.getdata("weight.fits")]  # inverse variance

table, residuals, pipe = pipeline.run(
    images, segmap, catalog=catalog, psfs=psfs, weights=weights
)
```

`images[0]` is the high-resolution detection image; every later entry is a
band to measure. The returned `table` carries `flux_<i>` and `err_<i>`
columns for image index `i`, `residuals` holds one residual image per fitted
band, and `pipe` is the `Pipeline` instance for further inspection. See the
[quickstart](https://mophongo.readthedocs.io/en/latest/quickstart.html) for
the full example including PSF-matching kernels and throughput handling.
