from .templates import Template 
from .fit import FitConfig, SparseFitter
from .catalog import Catalog
from .deblender import deblend_sources_symmetry, deblend_sources_hybrid
from .jwst_psf import make_extended_grid
from .psf_map import PSFRegionMap
from . import psf_map as _psf_map

_psf_map.fwhm = 0.2 / 3600

try:
    from .photutils_deblend import deblend_sources
except ImportError:
    from photutils.segmentation import deblend_sources

__all__ = [
    "Template",
    "FitConfig",
    "SparseFitter",
    "Catalog",
    "deblend_sources_symmetry",
    "deblend_sources_hybrid",
    "deblend_sources",
    "make_extended_grid",
    "PSFRegionMap",
]
