# Mophongo

A lightweight photometry pipeline.

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

from mophongo.pipeline import run

images = [fits.getdata("image1.fits"), fits.getdata("image2.fits")]
segmap = fits.getdata("segmap.fits")
catalog = Table.read("catalog.fits")
psfs = [fits.getdata("psf1.fits"), fits.getdata("psf2.fits")]

flux_table, residual = run_photometry(images, segmap, catalog, psfs)
```

The function returns the catalog with new flux columns and the residual image
from the fit.
