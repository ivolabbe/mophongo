from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.segmentation import SegmentationImage, SourceCatalog
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
    position_cutout: Tuple[float, float] | None = None


class TemplateNew(Cutout2D):
    """Cutout-based template storing slice bookkeeping."""

    def __init__(
        self,
        data: np.ndarray,
        position: tuple[float, float],
        size: tuple[int, int],
        *,
        source_xy: tuple[float, float] | None = None,
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
        self.source_xy = source_xy

    @property
    def array(self) -> np.ndarray:  # pragma: no cover - simple alias
        return self.data

    @property
    def bbox(self) -> tuple[int, int, int, int]:  # pragma: no cover - simple alias
        (ymin, ymax), (xmin, xmax) = self.bbox_original
        return int(ymin), int(ymax) + 1, int(xmin), int(xmax) + 1


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
            maskhi = (segm.data == label)[slices]
            xc, yc, _, _, _ = measure_shape(cuthi, maskhi)

            self._templates_hires.append(Template(cuthi * maskhi.astype(hires_image.dtype), (bbox.iymin, bbox.iymax, bbox.ixmin, bbox.ixmax), (xc, yc)))

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

            # Calculate local source position for the convolved template
            xc_local_conv = pos[1] - x0_ext
            yc_local_conv = pos[0] - y0_ext

            cutout_norm = cutout / flux
            conv = _convolve2d(cutout_norm, kernel)
            s = conv.sum()
            if s != 0:
                conv = conv / s
            self._templates.append(Template(conv, (y0_ext, y1_ext, x0_ext, x1_ext), (xc_local_conv, yc_local_conv)))

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
                new_templates.append(Template(tmpl_hi.array, tmpl_hi.bbox, tmpl_hi.source_xy))
                new_templates_hires.append(Template(tmpl_hi.array, tmpl_hi.bbox, tmpl_hi.source_xy))
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

            # This is the local center for the new extended template
            xc_new = cx_global - x0_clipped
            yc_new = cy_global - y0_clipped

            moffat_unit = elliptical_moffat(
                y_grid,
                x_grid,
                1.0,
                fwhm_x,
                fwhm_y,
                beta,
                theta,
                xc_new,
                yc_new,
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
            new_templates_hires.append(Template(extended_template, (y0_clipped, y1_clipped, x0_clipped, x1_clipped), (xc_new, yc_new)))

            # Convolve and normalize
            conv = _convolve2d(extended_template / extended_template.sum(), kernel)
            if conv.sum() != 0:
                conv = conv / conv.sum()
            new_templates.append(Template(conv, (y0_clipped, y1_clipped, x0_clipped, x1_clipped), (xc_new, yc_new)))

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
                new_templates.append(Template(tmpl_hi.array, tmpl_hi.bbox, tmpl_hi.source_xy))
                new_templates_hires.append(Template(tmpl_hi.array, tmpl_hi.bbox, tmpl_hi.source_xy))
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

            # Calculate local center in the final padded template
            xc_local_cut = x_c - x0
            yc_local_cut = y_c - y0

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
            
            xc_final = xc_local_cut + offset_x
            yc_final = yc_local_cut + offset_y
            
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
            new_templates_hires.append(Template(padded_template, (clipped_y0, clipped_y1, clipped_x0, clipped_x1), (xc_final, yc_final)))

            # Convolve and normalize
            conv = _convolve2d(padded_template / padded_template.sum(), kernel)
            if conv.sum() != 0:
                conv = conv / conv.sum()
            new_templates.append(Template(conv, (clipped_y0, clipped_y1, clipped_x0, clipped_x1), (xc_final, yc_final)))

        self._templates = new_templates
        self._templates_hires = new_templates_hires
        return new_templates

    def extend_with_psf_dilation_simple(
        self,
        psf: np.ndarray,
        kernel: np.ndarray,
        *,
        radius_factor: float = 1.5,
    ) -> List[Template]:
        """Extend templates using PSF scaled to segment flux, placed where template is zero."""

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
                new_templates.append(Template(tmpl_hi.array, tmpl_hi.bbox, tmpl_hi.source_xy))
                new_templates_hires.append(Template(tmpl_hi.array, tmpl_hi.bbox, tmpl_hi.source_xy))
                continue

            # Get total flux and centroid in original template coordinates
            flux = data[mask].sum()
            y_idx, x_idx = np.indices(data.shape)
            y_c_local = (y_idx[mask] * data[mask]).sum() / flux
            x_c_local = (x_idx[mask] * data[mask]).sum() / flux

            # Global coordinates of centroid
            cy_global = tmpl_hi.bbox[0] + y_c_local
            cx_global = tmpl_hi.bbox[2] + x_c_local

            # Create extended bbox based on PSF size and radius_factor
            psf_half_h = psf.shape[0] // 2
            psf_half_w = psf.shape[1] // 2
            
            # Extended bbox for the PSF placement
            y0_ext = int(np.floor(cy_global - radius_factor * psf_half_h))
            x0_ext = int(np.floor(cx_global - radius_factor * psf_half_w))
            y1_ext = int(np.ceil(cy_global + radius_factor * psf_half_h))
            x1_ext = int(np.ceil(cx_global + radius_factor * psf_half_w))

            # Further extend for convolution padding - THIS IS OUR WORKING COORDINATE SYSTEM
            y0_padded = y0_ext - pad_y
            x0_padded = x0_ext - pad_x
            y1_padded = y1_ext + pad_y
            x1_padded = x1_ext + pad_x

            # Size of padded template (our working array)
            ny_padded = y1_padded - y0_padded
            nx_padded = x1_padded - x0_padded

            # ===== ALL OPERATIONS NOW IN PADDED COORDINATE SYSTEM =====
            
            # Create extended template array
            extended_template = np.zeros((ny_padded, nx_padded))

            # Map original data into padded coordinates
            orig_y0_padded = tmpl_hi.bbox[0] - y0_padded
            orig_x0_padded = tmpl_hi.bbox[2] - x0_padded
            orig_y1_padded = orig_y0_padded + data.shape[0]
            orig_x1_padded = orig_x0_padded + data.shape[1]

            # Place original data in padded template
            extended_template[orig_y0_padded:orig_y1_padded, orig_x0_padded:orig_x1_padded] = data

            # Create mask of where template is non-zero
            template_mask = extended_template > 0

            # Centroid in padded coordinates
            yc_padded = cy_global - y0_padded
            xc_padded = cx_global - x0_padded

            # Create PSF centered at the centroid in padded coordinates
            y_grid, x_grid = np.indices((ny_padded, nx_padded))
            dy = y_grid - yc_padded
            dx = x_grid - xc_padded

            # Sample PSF at all grid positions
            psf_sampled = self._sample_psf(psf, dy, dx)
            
            # Calculate PSF scaling: map segment mask to padded coordinates
            segment_mask_padded = np.zeros((ny_padded, nx_padded), dtype=bool)
            segment_mask_padded[orig_y0_padded:orig_y1_padded, orig_x0_padded:orig_x1_padded] = mask

            # Get PSF and data values in the overlapping region
            psf_in_segment = psf_sampled[segment_mask_padded].sum()
            data_in_segment = data[mask].sum()

            if psf_in_segment > 0:
                psf_scale = data_in_segment / psf_in_segment
            else:
                # Fallback: scale based on total flux
                psf_scale = flux / psf_sampled.sum() if psf_sampled.sum() > 0 else 1.0

            # Scale the PSF
            psf_scaled = psf_sampled * psf_scale

            # Add PSF flux only where the template is currently zero
            zero_mask = ~template_mask
            flux_before = extended_template.sum()
            extended_template[zero_mask] += psf_scaled[zero_mask]
            flux_after = extended_template.sum()
            flux_added = flux_after - flux_before

            print(f"Source flux: {flux:.2f}, PSF scale: {psf_scale:.3f}, "
                  f"Flux before: {flux_before:.2f}, after: {flux_after:.2f}, "
                  f"added: {flux_added:.2f} ({100*flux_added/flux:.1f}%)")

            # Convolve in padded coordinates
            conv_padded = _convolve2d(extended_template / extended_template.sum(), kernel)
            if conv_padded.sum() != 0:
                conv_padded = conv_padded / conv_padded.sum()

            # ===== FINAL STEP: CLIP TO IMAGE BOUNDARIES =====
            
            # Clip padded coordinates to image boundaries
            y0_clipped = max(0, y0_padded)
            x0_clipped = max(0, x0_padded)
            y1_clipped = min(self.hires_shape[0], y1_padded)
            x1_clipped = min(self.hires_shape[1], x1_padded)

            # Extract clipped region from padded arrays
            clip_y0 = y0_clipped - y0_padded
            clip_x0 = x0_clipped - x0_padded
            clip_y1 = clip_y0 + (y1_clipped - y0_clipped)
            clip_x1 = clip_x0 + (x1_clipped - x0_clipped)

            extended_clipped = extended_template[clip_y0:clip_y1, clip_x0:clip_x1]
            conv_clipped = conv_padded[clip_y0:clip_y1, clip_x0:clip_x1]

            # Update source position for clipped coordinates
            xc_clipped = xc_padded - clip_x0
            yc_clipped = yc_padded - clip_y0

            # Store templates
            new_templates_hires.append(Template(extended_clipped, (y0_clipped, y1_clipped, x0_clipped, x1_clipped), (xc_clipped, yc_clipped)))
            new_templates.append(Template(conv_clipped, (y0_clipped, y1_clipped, x0_clipped, x1_clipped), (xc_clipped, yc_clipped)))

        self._templates = new_templates
        self._templates_hires = new_templates_hires
        return new_templates

def extract_templates_new(
    hires_image: np.ndarray,
    segmap: np.ndarray,
    positions: Iterable[Tuple[float, float]],
    kernel: np.ndarray,
) -> list[TemplateNew]:
    """Return PSF-matched templates using :class:`TemplateNew`."""

    if hires_image.shape != segmap.shape:
        raise ValueError("hires_image and segmap must have the same shape")

    kernel = kernel / kernel.sum()
    segm = SegmentationImage(segmap)
    templates: list[TemplateNew] = []

    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    ny, nx = hires_image.shape

    for pos in positions:
        y, x = int(round(pos[0])), int(round(pos[1]))
        if y < 0 or y >= ny or x < 0 or x >= nx:
            continue
        label = segm.data[y, x]
        if label == 0:
            continue

        idx = segm.get_index(label)
        bbox = segm.bbox[idx]

        y0_ext = bbox.iymin - pad_y
        y1_ext = bbox.iymax + pad_y
        x0_ext = bbox.ixmin - pad_x
        x1_ext = bbox.ixmax + pad_x

        height = y1_ext - y0_ext
        width = x1_ext - x0_ext
        center = ((x0_ext + x1_ext) / 2.0, (y0_ext + y1_ext) / 2.0)

        cut = TemplateNew(hires_image, center, (height, width), source_xy=pos)
        mask_cut = Cutout2D(
            (segm.data == label).astype(hires_image.dtype),
            center,
            (height, width),
            mode="partial",
            fill_value=0.0,
        )

        data = cut.data * mask_cut.data
        flux = float(data.sum())
        if flux == 0.0:
            continue

        data_valid = data[cut.slices_cutout] / flux
        conv = _convolve2d(data_valid, kernel)
        if conv.sum() != 0:
            conv = conv / conv.sum()
        padded = np.zeros_like(data)
        padded[cut.slices_cutout] = conv
        cut.data = padded
        templates.append(cut)

    return templates

