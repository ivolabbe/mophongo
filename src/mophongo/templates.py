from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Tuple
from copy import deepcopy

import logging
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.segmentation import SegmentationImage
from tqdm import tqdm
from scipy.signal import fftconvolve

from .utils import measure_shape, convolve2d
from .psf_map import PSFRegionMap

logger = logging.getLogger(__name__)

def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    pad_before = (pad_y, pad_x)
    pad_after = (ky - 1 - pad_y, kx - 1 - pad_x)
    padded = np.pad(image, (pad_before, pad_after), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)


class Template(Cutout2D):
    """Cutout-based template storing slice bookkeeping."""

    def __init__(
        self,
        data: np.ndarray,
        position: tuple[float, float],
        size: tuple[int, int],
        label: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data,
            position,
            size,
            mode="partial",
            fill_value=0.0,
            copy=True,
            **kwargs,
        )
        # @@@ bug in Cutout2D: shape_input is not set correctly
        self.shape_input = data.shape
        # record shift from original position here
        self.id = label
        self.parent_id = label
        self.component = "main"
        self.flux = 0.0
        self.err = 0.0
        self.shift = np.array([0.0, 0.0], dtype=float)

    @property
    def bbox(
            self
    ) -> tuple[int, int, int, int]:  # pragma: no cover - simple alias
        (ymin, ymax), (xmin, xmax) = self.bbox_original
        return int(ymin), int(ymax) + 1, int(xmin), int(xmax) + 1

    def pad(self,
            padding: Tuple[int, int],
            original_shape: Tuple[int, int],
            *,
            inplace=False) -> "Template":
        """Create a new Template with padding, maintaining correct original coordinates."""

        # force padding to be even, otherwise unpredictable behavior for cutout
        ony, onx = padding[0] // 2, padding[1] // 2
        pady, padx = ony * 2, onx * 2
        ny, nx = self.data.shape

        # Create new Template directly from the original array reference
        # This ensures all coordinates remain consistent with the true original
        new_template = Template(
            data=np.zeros(self.shape_input, dtype=self.data.dtype),
            position=self.input_position_original,
            size=(ny + pady, nx + padx),
            wcs=self.wcs,
            label=self.id,
        )

        # Now place the old data in our padded version
        new_template.data[ony:ony + ny, onx:onx + nx] = self.data

        # if inplace is True, update the current instance
        if inplace:
            # overwrite the current attributes with the new one
            self.__dict__.update(new_template.__dict__)
 
        return new_template


class Templates:
    """Container for source templates."""

    def __init__(self) -> None:
        self._templates: List[Template] = []

    def __len__(self) -> int:
        return len(self._templates)

    def __getitem__(self, idx: int) -> Template:
        return self._templates[idx]

    def __iter__(self) -> Iterator[Template]:
        return iter(self._templates)

    def add_component(
        self,
        parent: Template,
        data: np.ndarray,
        component: str,
        **kwargs: Any,
    ) -> Template | None:
        """Clone ``parent`` and append a new component template.

        Parameters
        ----------
        parent
            The template providing spatial metadata.
        data
            Pixel data for the new component. Must match the shape of
            ``parent.data``.
        component
            Informational tag describing the component type.
        **kwargs
            Additional attributes to set on the cloned template.

        Returns
        -------
        Template | None
            The newly created template or ``None`` if the component was
            discarded due to high similarity with ``parent``.
        """

        arr_parent = parent.data[parent.slices_cutout]
        arr_new = data[parent.slices_cutout]
        norm_p = np.linalg.norm(arr_parent.ravel())
        norm_n = np.linalg.norm(arr_new.ravel())
        if norm_p > 0 and norm_n > 0:
            corr = float(np.dot(arr_parent.ravel(), arr_new.ravel()) / (norm_p * norm_n))
            if corr > 0.999:
                logger.info(
                    "Skipping component %s for source %s due to high similarity (%.3f)",
                    component,
                    parent.id,
                    corr,
                )
                return None

        tmpl = deepcopy(parent)
        tmpl.data = data
        tmpl.component = component
        tmpl.parent_id = parent.parent_id or parent.id
        for key, val in kwargs.items():
            setattr(tmpl, key, val)

        self._templates.append(tmpl)
        return tmpl

    @classmethod
    def from_image(
        cls,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        kernel: np.ndarray,
        extension: np.ndarray | str | None = None,  # 'psf', 'wings', 'both', None
        wcs: WCS | None = None,
    ) -> "Templates":
        obj = cls()
        obj.wcs = wcs

        # Step 1: Extract raw cutouts
        obj.extract_templates(hires_image, segmap, positions, wcs=wcs)

        #if type(extension) == np.ndarray:
            # Extend templates with PSF wings
            #obj.extend_with_psf_wings(extension, inplace=True)

        # Step 2: Convolve with kernel (includes padding)
        if kernel is not None:
            obj.convolve_templates(kernel, inplace=True)

        return obj

    @staticmethod
    def quick_flux(templates: List[Template], image: np.ndarray) -> np.ndarray:
        """Return quick flux estimates based on template data and image."""
        flux = np.zeros(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            tt = tmpl.data[tmpl.slices_cutout]
            img = image[tmpl.slices_original]
            ttsqs = np.sum(tt**2)
            flux[i] = np.sum(img * tt) / ttsqs if ttsqs > 0 else 0.0
            tmpl.quick_flux = flux[i]  # Store quick flux in the template for later use
        return flux

    @staticmethod
    def predicted_errors(templates: List[Template], weights: np.ndarray) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        pred = np.empty(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            w = weights[tmpl.slices_original]
            pred[i] = 1.0 / np.sqrt(np.sum(w * tmpl.data[tmpl.slices_cutout]**2))
            tmpl.pred_err = pred[i]  # Store RMS in the template for later use
        return pred

    @property
    def templates(self) -> List[Template]:
        """Return the list of templates."""
        return self._templates

    def deduplicate(self, threshold: float = 0.999) -> List[Template]:
        """Remove nearly identical templates based on correlation."""
        if len(self._templates) < 2:
            return self._templates

        keep: list[int] = []
        data_arrays = [t.data[t.slices_cutout].ravel() for t in self._templates]
        norms = [np.sqrt(np.sum(d * d)) for d in data_arrays]
        for i, (arr_i, norm_i) in enumerate(zip(data_arrays, norms)):
            if norm_i == 0:
                continue
            duplicate = False
            for j in keep:
                arr_j = data_arrays[j]
                if arr_j.size != arr_i.size:
                    continue
                corr = np.dot(arr_i, arr_j) / (norm_i * norms[j])
                if corr > threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(i)

        dropped = len(self._templates) - len(keep)
        if dropped > 0:
            logger.warning("Dropped %d duplicate templates", dropped)
        self._templates = [self._templates[i] for i in keep]
        return self._templates

    def extract_templates(
        self,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        wcs: WCS | None = None,
    ) -> list[Template]:
        """Extract cutout templates around segmentation regions."""

        self.original_shape = hires_image.shape
        segm = SegmentationImage(segmap)
        templates: list[Template] = []
        ny, nx = hires_image.shape

        for pos in tqdm(positions, desc="Extracting templates"):
            # silently skip invalid positions
            if not np.isfinite(pos).all(): continue
            x, y = int(round(pos[0])), int(round(pos[1]))
            if y < 0 or y >= ny or x < 0 or x >= nx:
                continue
            label = segm.data[y, x]
            if label == 0:
                continue

            idx = segm.get_index(label)
            bbox = segm.bbox[idx]
            segm.slices[idx]

            # Make bbox symmetric around the center to ensure proper centering
            height = max(y - bbox.iymin, bbox.iymax - y) * 2
            width = max(x - bbox.ixmin, bbox.ixmax - x) * 2

            # Create template cutout
            cut = Template(hires_image, pos, (height, width), wcs=wcs, label=label)

            # zero out all non segment pixels
            cut.data[cut.slices_cutout] *= (
                segm.data[cut.slices_original] == label).astype(cut.data.dtype)

            templates.append(cut)

        self._templates = templates
        return templates

    def convolve_templates(
        self,
        kernel: np.ndarray | PSFRegionMap,
        inplace: bool = False,
    ) -> list[Template]:
        """Convolve all templates with ``kernel``.

        Parameters
        ----------
        kernel
            Either a fixed convolution kernel or a :class:`~mophongo.kernels.KernelLookup`
            instance providing spatially varying kernels.
        inplace
            If ``True`` update the stored templates. Otherwise return a new
            list of convolved templates.
        """

        if not self._templates:
            raise ValueError(
                "No templates to convolve. Run extract_templates first.")

        # Determine kernel shape from fixed kernel or first lookup entry
        if isinstance(kernel, PSFRegionMap):
            ky, kx = kernel.psfs[0].shape
        else:
            ky,kx = kernel.shape

        new_templates: list[Template] = []
        for tmpl in tqdm(self._templates, desc="Convolving templates"):
            new_tmpl = tmpl.pad((ky, kx), self.original_shape, inplace=inplace)

            # look up local kernel if using PSFRegionMap
            if isinstance(kernel, PSFRegionMap):
                x, y = tmpl.position_original
                if tmpl.wcs is not None:
                    ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
                else:
                    ra, dec = x, y
                kern = kernel.get_psf(ra, dec)
            else:
                # use fixed kernel
                kern = kernel

            conv = fftconvolve(new_tmpl.data, kern, mode='same')
            new_tmpl.data[:] = conv / conv.sum()     #  normalizing to 1.0 makes astrofitter bomb
                                                     #  needs to pre-whiten sparse matrix.
#            new_tmpl.data[:] = conv                 

            if not inplace:
                new_templates.append(new_tmpl)

        if not inplace:
            return new_templates
        else:
            return self._templates



    # put this in PSF class?
    @staticmethod
    def _sample_psf(psf: np.ndarray, position: Tuple[float, float],
                    height: int, width: int) -> np.ndarray:
        """Sample PSF at all positions in a grid centered at (center_x, center_y)."""

        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        dx = x_grid - position[0]
        dy = y_grid - position[1]

        # PSF center coordinates
        cy = (psf.shape[0] - 1) / 2
        cx = (psf.shape[1] - 1) / 2

        # Calculate PSF indices
        iy = np.round(cy + dy).astype(int)
        ix = np.round(cx + dx).astype(int)

        # Check bounds
        valid = ((iy >= 0)
                 & (iy < psf.shape[0])
                 & (ix >= 0)
                 & (ix < psf.shape[1]))

        # Sample PSF values
        vals = np.zeros((height, width), dtype=float)
        vals[valid] = psf[iy[valid], ix[valid]]

        return vals

    def extend_with_psf_wings(self,
                              psf: np.ndarray,
                              *,
                              radius_factor: float = 1.5,
                              inplace: bool = False) -> List[Template]:
        """Extend templates using PSF scaled to segment flux, placed where template is zero."""

        psf = psf / psf.sum()
        new_templates: list[Template] = []

        # Add progress bar here
        for i, tmpl in enumerate(tqdm(self._templates, desc="Extending with PSF wings")):
            data = tmpl.data
            ny, nx = data.shape

            # Measure shape to determine padding needed
            x_c, y_c, sigma_x, sigma_y, theta = measure_shape(data, data != 0)
            effective_radius = max(sigma_x, sigma_y)

            # Calculate padding based on radius factor
            pad_radius = int(np.ceil(effective_radius * radius_factor))
            pady, padx = int(ny * (radius_factor - 1)), int(nx * (radius_factor - 1))

            # Pad the template
            new_tmpl = tmpl.pad((pady,padx), self.original_shape, inplace=inplace)

            # Sample PSF at all template positions
            nh, nw = new_tmpl.data.shape
            psf_template = self._sample_psf(psf, new_tmpl.position_cutout, nh, nw)

            # Create mask for segment pixels in the padded template
            # Calculate scaling using only segment pixels
            segment_mask = new_tmpl.data > 0
            data_in_segment = np.sum(new_tmpl.data[segment_mask])
            psf_in_segment = np.sum(psf_template[segment_mask])

            if psf_in_segment > 0:
                psf_scale = data_in_segment / psf_in_segment
            else:
                psf_scale = 0.0

            # Add PSF flux only where the template is currently zero
            # if inplace, this will modify the original template
            new_tmpl.data[~segment_mask] += psf_template[~segment_mask] * psf_scale

            # Update the output templates list if not inplace
            if not inplace:
                new_templates.append(new_tmpl)

            # Store original flux for diagnostics
            flux_before = data.sum()
            flux_after = new_tmpl.data.sum()
            flux_added =  flux_after - flux_before

            # Print diagnostics
#            print(f"Source flux: {flux_before:.2f}, PSF scale: {psf_scale:.3f}, "
#                  f"Flux before: {flux_before:.2f}, after: {flux_after:.2f}, "
#                  f"added: {flux_added:.2f} ({100*flux_added/flux_before:.1f}%)")

        if not inplace:
            return new_templates
        else:
            return self._templates
