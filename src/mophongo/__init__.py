import logging as _logging

# numexpr announces its thread count at INFO the first time it is imported,
# which it is indirectly, via astropy. The line says nothing a mophongo user
# acts on, and it lands at the top of every run and every CLI invocation. Set
# before the imports below, which are what pull numexpr in.
_logging.getLogger("numexpr").setLevel(_logging.WARNING)

from .templates import Templates, Template
from .fit import SparseFitter
from .scene_fitter import SceneFitter
from .scene import Scene
from .astrometry import AstroCorrect, AstroMap
from .catalog import Catalog
from .psf_map import PSFRegionMap

from photutils.segmentation import deblend_sources

__all__ = [
    "SparseFitter",
    "SceneFitter",
    "Scene",
    "AstroCorrect",
    "AstroMap",
    "Catalog",
    "deblend_sources",
    "PSFRegionMap",
]
