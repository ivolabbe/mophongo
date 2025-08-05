#from .templates import Template 
from .fit import SparseFitter
from .local_astrometry import AstroCorrect
from .catalog import Catalog
#from .deblender import deblend_sources_symmetry, deblend_sources_hybrid
#from .jwst_psf import make_extended_grid
from .psf_map import PSFRegionMap
#from .sim_data import make_mosaic_dataset
#from . import psf_map as _psf_map

# try:
#     from .photutils_deblend import deblend_sources
# except ImportError:
#     from photutils.segmentation import deblend_sources

from photutils.segmentation import deblend_sources

__all__ = [
#    "Template",
#    "FitConfig",
    "SparseFitter",
    "AstroCorrect",
    "Catalog",
#    "deblend_sources_symmetry",
#    "deblend_sources_hybrid",
    "deblend_sources",
#    "make_extended_grid",
    "PSFRegionMap",
#    "KernelLookup",
#    "make_mosaic_dataset",
]
