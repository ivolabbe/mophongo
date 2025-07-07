from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.segmentation import SegmentationImage
from skimage.morphology import binary_erosion, dilation, disk, footprint_rectangle


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

    @staticmethod
    def _elliptical_moffat(
        y: np.ndarray,
        x: np.ndarray,
        amplitude: float,
        fwhm_x: float,
        fwhm_y: float,
        beta: float,
        theta: float,
        x0: float,
        y0: float,
    ) -> np.ndarray:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xr = (x - x0) * cos_t + (y - y0) * sin_t
        yr = -(x - x0) * sin_t + (y - y0) * cos_t
        factor = 2 ** (1 / beta) - 1
        alpha_x = fwhm_x / (2 * np.sqrt(factor))
        alpha_y = fwhm_y / (2 * np.sqrt(factor))
        r2 = (xr / alpha_x) ** 2 + (yr / alpha_y) ** 2
        return amplitude * (1 + r2) ** (-beta)

    def extend_with_moffat(
        self,
        kernel: np.ndarray,
        *,
        radius_factor: float = 2.0,
        beta: float | None = None,
    ) -> List[Template]:
        """Extend templates by fitting a 2-D Moffat profile."""

        new_templates: List[Template] = []
        new_templates_hires: List[Template] = []
        kernel = kernel / kernel.sum()
        for tmpl_hi in self._templates_hires:
            data = tmpl_hi.array
            mask = data > 0
            if not np.any(mask):
                new_templates.append(tmpl_hi)
                continue

            y_idx, x_idx = np.indices(data.shape)
            flux = data[mask].sum()
            y_c = (y_idx[mask] * data[mask]).sum() / flux
            x_c = (x_idx[mask] * data[mask]).sum() / flux

            y_rel = y_idx - y_c
            x_rel = x_idx - x_c
            cov_xx = (data[mask] * x_rel[mask] ** 2).sum() / flux
            cov_yy = (data[mask] * y_rel[mask] ** 2).sum() / flux
            cov_xy = (data[mask] * x_rel[mask] * y_rel[mask]).sum() / flux
            cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            sigma_x = np.sqrt(vals[0])
            sigma_y = np.sqrt(vals[1])
            theta = np.arctan2(vecs[1, 0], vecs[0, 0])

            fwhm_x = 2.355 * sigma_x
            fwhm_y = 2.355 * sigma_y
            beta_val = beta if beta is not None else 2.5

            moffat_unit = self._elliptical_moffat(
                y_idx,
                x_idx,
                1.0,
                fwhm_x,
                fwhm_y,
                beta_val,
                theta,
                x_c,
                y_c,
            )
            amp = flux / moffat_unit[mask].sum()

            rmax = radius_factor * max(fwhm_x, fwhm_y)
            size = int(np.ceil(2 * rmax)) + 1

            cy_global = tmpl_hi.bbox[0] + y_c
            cx_global = tmpl_hi.bbox[2] + x_c

            y0 = int(round(cy_global - size / 2))
            x0 = int(round(cx_global - size / 2))
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(self.hires_shape[0], y0 + size)
            x1 = min(self.hires_shape[1], x0 + size)

            ny = y1 - y0
            nx = x1 - x0
            y_grid, x_grid = np.indices((ny, nx))

            moffat_ext = self._elliptical_moffat(
                y_grid,
                x_grid,
                amp,
                fwhm_x,
                fwhm_y,
                beta_val,
                theta,
                cx_global - x0,
                cy_global - y0,
            )

            # Create extended template with original data preserved
            extended_template = np.zeros_like(moffat_ext)
            
            # Map original template coordinates to extended template coordinates
            orig_y0 = tmpl_hi.bbox[0] - y0
            orig_y1 = tmpl_hi.bbox[1] - y0
            orig_x0 = tmpl_hi.bbox[2] - x0
            orig_x1 = tmpl_hi.bbox[3] - x0
            
            # Ensure coordinates are within bounds
            orig_y0 = max(0, orig_y0)
            orig_y1 = min(ny, orig_y1)
            orig_x0 = max(0, orig_x0)
            orig_x1 = min(nx, orig_x1)
            
            # Use original data where the segment exists
            if orig_y1 > orig_y0 and orig_x1 > orig_x0:
                # Calculate the slice of original data that fits in the extended template
                data_y0 = max(0, y0 - tmpl_hi.bbox[0])
                data_y1 = data_y0 + (orig_y1 - orig_y0)
                data_x0 = max(0, x0 - tmpl_hi.bbox[2])
                data_x1 = data_x0 + (orig_x1 - orig_x0)
                
                extended_template[orig_y0:orig_y1, orig_x0:orig_x1] = data[data_y0:data_y1, data_x0:data_x1]
                
                # Use Moffat profile only where original data is zero (outside segment)
                original_mask = extended_template > 0
                extended_template[~original_mask] = moffat_ext[~original_mask]
            else:
                # If no overlap, use full Moffat profile
                extended_template = moffat_ext

            conv = _convolve2d(extended_template, kernel)
            if conv.sum() != 0:
                conv = conv / conv.sum()
            new_templates.append(Template(conv, (y0, y1, x0, x1)))
            new_templates_hires.append(Template(extended_template, (y0, y1, x0, x1)))

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
        selem: np.ndarray = disk(2),
    ) -> List[Template]:
        """Extend templates using PSF-weighted dilation."""

        psf = psf / psf.sum()
        new_templates: List[Template] = []
        new_templates_hires: List[Template] = []

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

            print(f'\n=== Template {len(new_templates)} Debug ===')
            print(f'Original flux: {flux:.6f}, Original pixels: {np.count_nonzero(mask)}')
            print(f'Center: ({y_c:.1f}, {x_c:.1f}), Padded shape: {arr.shape}')
            print(f'Initial ring: {np.count_nonzero(prev_ring)} pixels')
            print(f'Ring flux: {prev_flux:.6f}, Mean ring level: {prev_flux/np.count_nonzero(prev_ring) if np.count_nonzero(prev_ring) > 0 else 0:.6f}')
            print(f'PSF sum at ring: {psf_sum_prev:.6f}, Scale: {scale:.6f}')

            total_added_flux = 0.0
            total_added_pixels = 0

            for iteration in range(iterations):
                dilated = dilation(mask_curr, selem)
                new_ring = dilated & ~mask_curr
                if not np.any(new_ring):
                    print(f'Iteration {iteration}: No new pixels to add')
                    break

                coords_new = np.argwhere(new_ring)
                psf_new = self._sample_psf(psf, coords_new[:, 0] - y_c, coords_new[:, 1] - x_c)
                
                # Calculate new pixel values
                new_values = scale * psf_new
                
                # Diagnostic: analyze what we're adding
                n_new_pixels = len(coords_new)
                new_flux_added = new_values.sum()
                mean_new_value = new_values.mean() if n_new_pixels > 0 else 0
                max_new_value = new_values.max() if n_new_pixels > 0 else 0
                
                print(f'Iteration {iteration}: Adding {n_new_pixels} pixels')
                print(f'  New flux added: {new_flux_added:.6f}')
                print(f'  Mean new pixel value: {mean_new_value:.6f}')
                print(f'  Max new pixel value: {max_new_value:.6f}')
                print(f'  PSF values range: [{psf_new.min():.6f}, {psf_new.max():.6f}]')
                
                # Set the new pixel values
                arr[coords_new[:, 0], coords_new[:, 1]] = new_values
                mask_curr = dilated
                prev_ring = new_ring
                prev_flux = arr[prev_ring].sum()
                psf_prev = psf_new
                psf_sum_prev = psf_prev.sum()
                if psf_sum_prev == 0:
                    print(f'  PSF sum became zero, stopping')
                    break
                scale = prev_flux / psf_sum_prev
                
                # Track totals
                total_added_flux += new_flux_added
                total_added_pixels += n_new_pixels
                
                print(f'  Updated scale for next iteration: {scale:.6f}')
                print(f'  Cumulative added flux: {total_added_flux:.6f}')
                print(f'  Cumulative added pixels: {total_added_pixels}')

            # Final diagnostics
            final_flux = arr[mask_curr].sum()
            final_pixels = np.count_nonzero(mask_curr)
            flux_ratio = final_flux / flux if flux > 0 else 0
            
            print(f'=== Final Results ===')
            print(f'Original: {flux:.6f} flux, {np.count_nonzero(mask)} pixels')
            print(f'Final: {final_flux:.6f} flux, {final_pixels} pixels')
            print(f'Added: {total_added_flux:.6f} flux, {total_added_pixels} pixels')
            print(f'Flux ratio (final/original): {flux_ratio:.3f}')
            print(f'Mean pixel level - original: {flux/np.count_nonzero(mask):.6f}')
            print(f'Mean pixel level - final: {final_flux/final_pixels:.6f}')

            inds = np.argwhere(mask_curr)
            y0 = inds[:, 0].min()
            y1 = inds[:, 0].max() + 1
            x0 = inds[:, 1].min()
            x1 = inds[:, 1].max() + 1
            cut = arr[y0:y1, x0:x1]

            conv = _convolve2d(cut / cut.sum(), kernel)
            if conv.sum() != 0:
                conv = conv / conv.sum()

            bbox = (
                tmpl_hi.bbox[0] - pad + y0,
                tmpl_hi.bbox[0] - pad + y1,
                tmpl_hi.bbox[2] - pad + x0,
                tmpl_hi.bbox[2] - pad + x1,
            )
            new_templates.append(Template(conv, bbox))
            new_templates_hires.append(Template(cut, bbox))

        self._templates = new_templates
        self._templates_hires = new_templates_hires
        return new_templates
