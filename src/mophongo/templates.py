from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.segmentation import SegmentationImage, SourceCatalog
from skimage.morphology import binary_erosion, dilation, disk, footprint_rectangle

from .utils import measure_shape


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
        )

        # Now place the old data in our padded version
        new_template.data[ony:ony + ny, onx:onx + nx] = self.data

        # if inplace is True, update the current instance
        if inplace:
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

    @classmethod
    def from_image(
        cls,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        kernel: np.ndarray,
        extension: np.ndarray | str | None = None,  # 'psf', 'wings', 'both', None
    ) -> "Templates":
        obj = cls()
        # Step 1: Extract raw cutouts
        obj.extract_templates(hires_image, segmap, positions)

        if type(extension) == np.ndarray:
            # Extend templates with PSF wings
            obj.extend_with_psf_wings(extension, inplace=True)

        # Step 2: Convolve with kernel (includes padding)
        obj.convolve_templates(kernel, inplace=True)

        return obj

    @classmethod
    def from_image_old(
        cls,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        kernel: np.ndarray,
    ) -> "Templates":
        obj = cls()
        obj.extract_templates_old(hires_image, segmap, positions, kernel)
        return obj

    @property
    def templates(self) -> List[Template]:
        """Return the list of templates."""
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

        for i, tmpl in enumerate(self._templates):
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

    def extract_templates(
        self,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
    ) -> list[Template]:
        """Extract cutout templates around segmentation regions."""

        self.original_shape = hires_image.shape
        segm = SegmentationImage(segmap)
        templates: list[Template] = []
        ny, nx = hires_image.shape

        for pos in positions:
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
            cut = Template(hires_image, pos, (height, width))

            # zero out all non segment pixels
            cut.data[cut.slices_cutout] *= (
                segm.data[cut.slices_original] == label).astype(cut.data.dtype)

            templates.append(cut)

        self._templates = templates
        return templates

    def convolve_templates(self,
                           kernel: np.ndarray,
                           inplace: bool = False) -> list[Template]:
        """Convolve all templates with kernel, operating in-place with padding."""

        if not self._templates:
            raise ValueError(
                "No templates to convolve. Run extract_templates first.")

        kernel = kernel / kernel.sum()
        ky, kx = kernel.shape

        # pad and convolve templates
        new_templates: list[Template] = []
        for i, tmpl in enumerate(self._templates):
            # Pad template for convolution in place
            new_tmpl = tmpl.pad((ky, kx), self.original_shape, inplace=inplace)
            conv = _convolve2d(new_tmpl.data / new_tmpl.data.sum(), kernel)
            new_tmpl.data[:] = conv / conv.sum()

            if not inplace:
                new_templates.append(new_tmpl)

        if not inplace:
            return new_templates
        else:
            return self._templates
