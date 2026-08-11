# mophongo

Mophongo is a scientific Python package for PSF modeling, image
registration, image simulation, and template-fitting photometry, aimed at
multi-band JWST/HST imaging.

The main photometry flow:

1. Template extraction from a high-resolution detection image
   (`mophongo.templates`).
2. PSF handling and spatial PSF lookup (`mophongo.psf`,
   `mophongo.psf_map`).
3. PSF matching and convolution kernels (`mophongo.utils`,
   `mophongo.psf`).
4. Sparse or scene-based fitting (`mophongo.fit`, `mophongo.scene`,
   `mophongo.scene_fitter`).
5. Pipeline orchestration and diagnostics (`mophongo.pipeline`).

## Installation

```bash
pip install -e .
# or
poetry install
```

```{toctree}
:maxdepth: 2

api
```
