from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.segmentation import SegmentationImage
from skimage.morphology import binary_erosion, dilation, disk, footprint_rectangle

from .utils import elliptical_moffat, measure_shape


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


@dataclass
class Template:
    array: np.ndarray
    bbox: Tuple[int, int, int, int]


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
    ) -> "Templates":
        obj = cls()
        obj.extract_templates(hires_image, segmap, positions, kernel)
        return obj

    @property
    def templates(self) -> List[Template]:
        """Return the list of templates."""
        return self._templates
    
    def extract_templates(
        self,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        kernel: np.ndarray,
    ) -> List[Template]:
        """Extract PSF-matched templates for a list of source positions."""

        if hires_image.shape != segmap.shape:
            raise ValueError("hires_image and segmap must have the same shape")

        self.hires_shape = hires_image.shape
        self._templates = []
        self._templates_hires = []
        kernel = kernel / kernel.sum()

        segm = SegmentationImage(segmap)

        for pos in positions:
            y, x = int(round(pos[0])), int(round(pos[1]))
            if (
                y < 0
                or y >= segm.data.shape[0]
                or x < 0
                or x >= segm.data.shape[1]
            ):
                continue
            label = segm.data[y, x]
            if label == 0:
                continue

            idx = segm.get_index(label)
            bbox = segm.bbox[idx]
            slices = segm.slices[idx]

            # use cutout2d to extract just the bbox area without expanding kernel size
            cuthi = hires_image[slices]
            maskhi = (segm.data == label).astype(hires_image.dtype)[slices]
            self._templates_hires.append(Template(cuthi * maskhi, (bbox.iymin, bbox.iymax, bbox.ixmin, bbox.ixmax)))

            ky, kx = kernel.shape
            pad_y, pad_x = ky // 2, kx // 2

            y0_ext = bbox.iymin - pad_y
            y1_ext = bbox.iymax + pad_y
            x0_ext = bbox.ixmin - pad_x
            x1_ext = bbox.ixmax + pad_x

            ny, nx = hires_image.shape
            y0_ext = max(0, y0_ext)
            y1_ext = min(ny, y1_ext)
            x0_ext = max(0, x0_ext)
            x1_ext = min(nx, x1_ext)

            height = y1_ext - y0_ext
            width = x1_ext - x0_ext
            center = ((x0_ext + x1_ext) / 2.0, (y0_ext + y1_ext) / 2.0)

            cut = Cutout2D(
                hires_image, center, (height, width), mode="partial", fill_value=0.0
            )
            mask_cut = Cutout2D(
                (segm.data == label).astype(hires_image.dtype),
                center,
                (height, width),
                mode="partial",
                fill_value=0.0,
            )
            cutout = cut.data * mask_cut.data

            flux = float(cutout.sum())
            if flux == 0.0:
                continue

            cutout_norm = cutout / flux
            conv = _convolve2d(cutout_norm, kernel)
            s = conv.sum()
            if s != 0:
                conv = conv / s
            self._templates.append(Template(conv, (y0_ext, y1_ext, x0_ext, x1_ext)))

        return self._templates


    def extend_with_moffat(
        self,
        kernel: np.ndarray,
        *,
        radius_factor: float = 2.0,
        beta: float = 3.0,
    ) -> List[Template]:
        """Extend templates by fitting a 2-D Moffat profile."""

        new_templates: List[Template] = []
        new_templates_hires: List[Template] = []
        kernel = kernel / kernel.sum()
        
        # Get kernel padding
        ky, kx = kernel.shape
        pad_y, pad_x = ky // 2, kx // 2
        
        for tmpl_hi in self._templates_hires:
            data = tmpl_hi.array
            mask = data > 0
            if not np.any(mask):
                new_templates.append(tmpl_hi)
                continue

            flux = data[mask].sum()
            x_c, y_c, sigma_x, sigma_y, theta = measure_shape(data, mask)

            fwhm_x = 2.355 * sigma_x
            fwhm_y = 2.355 * sigma_y

            # bounding box size accounting for ellipticity and angle
            a = radius_factor * fwhm_x
            b = radius_factor * fwhm_y
            half_width = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
            half_height = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

            cy_global = tmpl_hi.bbox[0] + y_c
            cx_global = tmpl_hi.bbox[2] + x_c

            # Extended bbox for the Moffat profile
            y0_ext = int(np.floor(cy_global - half_height))
            x0_ext = int(np.floor(cx_global - half_width))
            y1_ext = int(np.ceil(cy_global + half_height))
            x1_ext = int(np.ceil(cx_global + half_width))

            # Further extend for convolution padding
            y0_padded = y0_ext - pad_y
            x0_padded = x0_ext - pad_x
            y1_padded = y1_ext + pad_y
            x1_padded = x1_ext + pad_x

            # Clip to image boundaries
            y0_clipped = max(0, y0_padded)
            x0_clipped = max(0, x0_padded)
            y1_clipped = min(self.hires_shape[0], y1_padded)
            x1_clipped = min(self.hires_shape[1], x1_padded)

            ny = y1_clipped - y0_clipped
            nx = x1_clipped - x0_clipped
            y_grid, x_grid = np.indices((ny, nx))

            moffat_unit = elliptical_moffat(
                y_grid,
                x_grid,
                1.0,
                fwhm_x,
                fwhm_y,
                beta,
                theta,
                cx_global - x0_clipped,
                cy_global - y0_clipped,
            )

            # map original template region into extended grid
            orig_y0 = tmpl_hi.bbox[0] - y0_clipped
            orig_x0 = tmpl_hi.bbox[2] - x0_clipped
            orig_y1 = orig_y0 + data.shape[0]
            orig_x1 = orig_x0 + data.shape[1]

            # Only use pixels within bounds for amplitude calculation
            orig_y0 = max(0, orig_y0)
            orig_y1 = min(ny, orig_y1)
            orig_x0 = max(0, orig_x0)
            orig_x1 = min(nx, orig_x1)

            if orig_y1 > orig_y0 and orig_x1 > orig_x0:
                amp = flux / moffat_unit[orig_y0:orig_y1, orig_x0:orig_x1][mask[:orig_y1-orig_y0, :orig_x1-orig_x0]].sum()
            else:
                amp = flux / moffat_unit.sum()

            moffat_ext = moffat_unit * amp

            # Create extended template with original data preserved
            extended_template = moffat_ext.copy()

            if orig_y1 > orig_y0 and orig_x1 > orig_x0:
                # Get the slice that corresponds to original data
                data_h, data_w = data.shape
                slice_h = min(data_h, orig_y1 - orig_y0)
                slice_w = min(data_w, orig_x1 - orig_x0)
                
                data_slice = extended_template[orig_y0:orig_y0+slice_h, orig_x0:orig_x0+slice_w]
                data_mask = mask[:slice_h, :slice_w]
                data_slice[data_mask] = data[:slice_h, :slice_w][data_mask]
                extended_template[orig_y0:orig_y0+slice_h, orig_x0:orig_x0+slice_w] = data_slice

            # Store high-res template
            new_templates_hires.append(Template(extended_template, (y0_clipped, y1_clipped, x0_clipped, x1_clipped)))

            # Convolve and normalize
            conv = _convolve2d(extended_template / extended_template.sum(), kernel)
            if conv.sum() != 0:
                conv = conv / conv.sum()
            new_templates.append(Template(conv, (y0_clipped, y1_clipped, x0_clipped, x1_clipped)))

        self._templates = new_templates
        self._templates_hires = new_templates_hires
        return new_templates

    @staticmethod
    def _sample_psf(psf: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
        cy = (psf.shape[0] - 1) / 2
        cx = (psf.shape[1] - 1) / 2
        iy = np.round(cy + dy).astype(int)
        ix = np.round(cx + dx).astype(int)
        valid = (
            (iy >= 0)
            & (iy < psf.shape[0])
            & (ix >= 0)
            & (ix < psf.shape[1])
        )
        vals = np.zeros_like(iy, dtype=float)
        vals[valid] = psf[iy[valid], ix[valid]]
        return vals

    def extend_with_psf_dilation(
        self,
        psf: np.ndarray,
        kernel: np.ndarray,
        *,
        iterations: int = 3,
        selem: np.ndarray = np.ones((3,3)),
    ) -> List[Template]:
        """Extend templates using PSF-weighted dilation."""

        psf = psf / psf.sum()
        new_templates: List[Template] = []
        new_templates_hires: List[Template] = []
        
        # Get kernel padding
        ky, kx = kernel.shape
        pad_y, pad_x = ky // 2, kx // 2

        for tmpl_hi in self._templates_hires:
            data = tmpl_hi.array
            mask = data > 0
            if not np.any(mask):
                new_templates.append(tmpl_hi)
                continue

            pad = iterations + 1
            arr = np.pad(data, pad)
            mask_curr = np.pad(mask, pad)

            y_idx, x_idx = np.indices(arr.shape)
            flux = arr[mask_curr].sum()
            y_c = (y_idx[mask_curr] * arr[mask_curr]).sum() / flux
            x_c = (x_idx[mask_curr] * arr[mask_curr]).sum() / flux

            prev_ring = mask_curr & ~binary_erosion(mask_curr, selem)
            prev_flux = arr[prev_ring].sum()

            coords_prev = np.argwhere(prev_ring)
            psf_prev = self._sample_psf(psf, coords_prev[:, 0] - y_c, coords_prev[:, 1] - x_c)
            psf_sum_prev = psf_prev.sum()
            if psf_sum_prev == 0:
                scale = 0.0
            else:
                scale = prev_flux / psf_sum_prev

            for iteration in range(iterations):
                dilated = dilation(mask_curr, selem)
                new_ring = dilated & ~mask_curr
                if not np.any(new_ring):
                    break

                coords_new = np.argwhere(new_ring)
                psf_new = self._sample_psf(psf, coords_new[:, 0] - y_c, coords_new[:, 1] - x_c)
                new_values = scale * psf_new
                
                arr[coords_new[:, 0], coords_new[:, 1]] = new_values
                mask_curr = dilated
                prev_ring = new_ring
                prev_flux = arr[prev_ring].sum()
                psf_prev = psf_new
                psf_sum_prev = psf_prev.sum()
                if psf_sum_prev == 0:
                    break
                scale = prev_flux / psf_sum_prev

            # Extract dilated template
            inds = np.argwhere(mask_curr)
            y0 = inds[:, 0].min()
            y1 = inds[:, 0].max() + 1
            x0 = inds[:, 1].min()
            x1 = inds[:, 1].max() + 1
            cut = arr[y0:y1, x0:x1]

            # Calculate global coordinates for dilated template
            global_y0_dil = tmpl_hi.bbox[0] - pad + y0
            global_y1_dil = tmpl_hi.bbox[0] - pad + y1
            global_x0_dil = tmpl_hi.bbox[2] - pad + x0
            global_x1_dil = tmpl_hi.bbox[2] - pad + x1

            # Extend further for convolution padding
            global_y0_padded = global_y0_dil - pad_y
            global_y1_padded = global_y1_dil + pad_y
            global_x0_padded = global_x0_dil - pad_x
            global_x1_padded = global_x1_dil + pad_x

            # Clip to image boundaries
            clipped_y0 = max(0, global_y0_padded)
            clipped_y1 = min(self.hires_shape[0], global_y1_padded)
            clipped_x0 = max(0, global_x0_padded)
            clipped_x1 = min(self.hires_shape[1], global_x1_padded)

            # Create padded array
            padded_height = clipped_y1 - clipped_y0
            padded_width = clipped_x1 - clipped_x0
            padded_template = np.zeros((padded_height, padded_width))

            # Calculate where to place the dilated template in the padded array
            offset_y = global_y0_dil - clipped_y0
            offset_x = global_x0_dil - clipped_x0
            
            # Ensure offsets are within bounds
            offset_y = max(0, offset_y)
            offset_x = max(0, offset_x)
            
            # Calculate how much of the cut array fits
            fit_h = min(cut.shape[0], padded_height - offset_y)
            fit_w = min(cut.shape[1], padded_width - offset_x)
            
            # Place the dilated template
            if fit_h > 0 and fit_w > 0:
                padded_template[offset_y:offset_y+fit_h, offset_x:offset_x+fit_w] = cut[:fit_h, :fit_w]

            # Store high-res template
            new_templates_hires.append(Template(padded_template, (clipped_y0, clipped_y1, clipped_x0, clipped_x1)))

            # Convolve and normalize
            conv = _convolve2d(padded_template / padded_template.sum(), kernel)
            if conv.sum() != 0:
                conv = conv / conv.sum()
            new_templates.append(Template(conv, (clipped_y0, clipped_y1, clipped_x0, clipped_x1)))

        self._templates = new_templates
        self._templates_hires = new_templates_hires
        return new_templates
