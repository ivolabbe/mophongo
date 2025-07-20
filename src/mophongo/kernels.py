from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .psf_map import PSFRegionMap


@dataclass
class KernelLookup:
    """Fast kernel recall using a PSF region map."""

    psf_map: PSFRegionMap
    kernels: np.ndarray
    _cache: Dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def get_kernel(self, ra: float, dec: float) -> np.ndarray:
        """Return kernel for a sky position."""
        key = self.psf_map.lookup_key(ra, dec)
        if key is None:
            raise ValueError("Position outside PSF region map")
        if key not in self._cache:
            self._cache[key] = self.kernels[key]
        return self._cache[key]
