# mophongo

Mophongo is a scientific Python package for template-fitting photometry on
multi-band JWST/HST imaging: PSF modeling, PSF-matched convolution kernels,
image simulation, and simultaneous sparse fitting of source fluxes across
images of heterogeneous depth and resolution.

The main photometry flow:

1. Template extraction from a high-resolution detection image
   ({doc}`templates`).
2. PSF handling and spatial PSF lookup ({doc}`psf`, {doc}`psf_maps`).
3. PSF matching and convolution kernels ({doc}`psf`).
4. Sparse or scene-based fitting with optional astrometric shifts
   ({doc}`fitting`).
5. Pipeline orchestration, outputs, and diagnostics ({doc}`pipeline`,
   {doc}`outputs`, {doc}`diagnostics`).

```{toctree}
:maxdepth: 1
:caption: Getting started

overview
quickstart
```

```{toctree}
:maxdepth: 1
:caption: User guide

pipeline
repair
outputs
diagnostics
```

```{toctree}
:maxdepth: 1
:caption: Components

psf
psf_maps
templates
fitting
catalog
preprocessing
simulation
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```
