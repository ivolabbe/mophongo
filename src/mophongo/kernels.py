from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .psf_map import PSFRegionMap


@dataclass
class KernelLookup:
    """Lookup spatially varying PSF matching kernels."""

    region_map: PSFRegionMap
    kernels: np.ndarray
    _cache: Dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def get_kernel(self, ra: float | None, dec: float | None) -> np.ndarray | None:
        if ra is None or dec is None or np.isnan(ra) or np.isnan(dec):
            key = 0
        else:
            key = self.region_map.lookup_key(ra, dec)
            if key is None:
                return None
        if key not in self._cache:
            self._cache[key] = self.kernels[key]
        return self._cache[key]
