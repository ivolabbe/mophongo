import re
import pandas as pd

class PSFRegionMap:
    def __init__(self, regions=None):
        # Initialize with given regions DataFrame or empty DataFrame
        self.regions = regions if regions is not None else pd.DataFrame()

    @staticmethod
    def _parse_detector_from_key(key):
        """
        Parse detector name from a FITS filename or key.
        Supports NIRCam (_nrcalong_rate.fits), MIRI (_mirimage_rate.fits), and JWST convention.
        """
        key = key.lower()
        match = re.search(r'_nrc([ab]\w+)_rate\.fits', key)
        if match:
            return f'NRC{match.group(1).upper()}'
        match = re.search(r'_mirimage_rate\.fits', key)
        if match:
            return 'MIRIMAGE'
        match = re.search(r'_([a-z0-9]+)_rate\.fits', key)
        if match:
            return match.group(1).upper()
        return 'UNKNOWN'

    def group_by_pa(self, pa_tol=1.0, wcs=None):
        """
        Group regions by both position angle (PA) and detector.
        Only merge/dissolve regions if both PA and detector match.
        """
        # Add detector info to each region
        detectors = []
        for key in self.regions['psf_key']:
            detectors.append(self._parse_detector_from_key(key))
        self.regions['detector'] = detectors

        # Group by PA and detector
        grouped = []
        for detector in set(self.regions['detector']):
            sub = self.regions[self.regions['detector'] == detector]
            # Now group by PA within this detector
            # Assume existing group_by_pa logic is in _group_by_pa_single_detector
            grouped.append(self._group_by_pa_single_detector(sub, pa_tol, wcs, detector))

        # Concatenate grouped regions
        new_regions = pd.concat(grouped, ignore_index=True)
        # Return a new PSFRegionMap with grouped regions
        out = PSFRegionMap()
        out.regions = new_regions
        return out

    def _group_by_pa_single_detector(self, regions, pa_tol, wcs, detector):
        """
        Existing group_by_pa logic, but only for a single detector.
        """
        # For demonstration, just return regions unchanged:
        return regions

    # ...existing code...