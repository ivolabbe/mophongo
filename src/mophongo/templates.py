from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Tuple
from copy import deepcopy

import logging
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.segmentation import SegmentationImage
from tqdm import tqdm
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from skimage.measure import block_reduce

from .utils import measure_shape, bin2d_mean
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
        self.ee_rlim: float = 0.0
        self.ee_fraction: float = 1.0

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

    def downsample(self, k: int) -> "Template":
        """Return a new template averaged by ``k×k`` blocks.

        Parameters
        ----------
        k : int
            Integer downsampling factor.

        Returns
        -------
        Template
            New downsampled template; the original is untouched.
        """
        if k == 1:
            return self

        hi = self
        lo = deepcopy(hi)

        # pixel data and bounding box
        lo.data = bin2d_mean(hi.data, k)
        ny_lo, nx_lo = lo.data.shape
        y0, _, x0, _ = hi.bbox
        bbox = (y0 // k, y0 // k + ny_lo, x0 // k, x0 // k + nx_lo)

        from .fit import SparseFitter  # local import to avoid circular dependency

        lo.bbox_original = ((bbox[0], bbox[1] - 1), (bbox[2], bbox[3] - 1))
        lo.slices_original = SparseFitter._bbox_to_slices(bbox)
        lo.slices_cutout = (slice(0, ny_lo), slice(0, nx_lo))
        hi_shape = getattr(hi, "shape_input", hi.data.shape)
        lo.shape_input = (hi_shape[0] // k, hi_shape[1] // k)

        # centroid-sensitive attributes
        shift = (k - 1) / 2.0

        def _map(coord: float) -> float:
            return (coord - shift) / k

        for attr in (
                "input_position_cutout",
                #            "input_position_original",
                "center_cutout",
                #            "center_original",
                "y",
                "x",
        ):
            if hasattr(lo, attr):
                y, x = getattr(lo, attr)
                setattr(lo, attr, (_map(y), _map(x)))

        return lo

    def downsample_wcs(self, image_lo: np.ndarray, wcs_lo,
                       k: int) -> "Template":
        """
        Downsample this template to a lower resolution using the target image and WCS.

        Parameters
        ----------
        image_lo : np.ndarray
            The low-resolution image to extract the template from.
        wcs_lo : astropy.wcs.WCS
            The WCS of the low-resolution image.
        k : int
            Integer downsampling factor.

        Returns
        -------
        Template
            New template extracted from the low-res image using the correct WCS.
        """
        # Get the original position in the high-res WCS
        pos = self.input_position_cutout  # needs to be cutout coordinates
        ra, dec = self.wcs.wcs_pix2world(*pos, 0)

        # Convert RA/Dec to pixel coordinates in the low-res WCS
        # note: x_lo, y_lo are now original coordinates in the low-res image
        x_lo, y_lo = wcs_lo.wcs_world2pix(ra, dec, 0)

        # Calculate new size (downsampled)
        height, width = self.data.shape[0] // k, self.data.shape[1] // k
        # print('Original position:', pos)
        # print(f"Downsampling {self.id} from {self.data.shape} to {height, width} at pos ({x_lo}, {y_lo})")
        # print('original data shape:', self.shape_input, image_lo.shape)
        # print(self.wcs)
        # print(wcs_lo)
#        Create the new template using the low-res image and WCS
        lowres_tmpl = Template(image_lo, (x_lo, y_lo), (height, width),
                               wcs=wcs_lo,
                               label=self.id)

        # Fill the data with block-reduced (averaged) values from the high-res template
        lowres_tmpl.data[:] = block_reduce(self.data, k, func=np.mean)
        return lowres_tmpl


class Templates:
    """Container for source templates."""
    min_size = 8  # minimum size of a template in pixels

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
    def _prepare_fft_fast(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray, interp1d]:
        """Return radial profile, EE curve and inverse profile interpolator."""
        y, x = np.indices(psf.shape)
        cy, cx = (np.array(psf.shape) - 1) / 2
        r = np.hypot(y - cy, x - cx)
        r_int = r.astype(int)
        prof_num = np.bincount(r_int.ravel(), psf.ravel())
        prof_den = np.bincount(r_int.ravel())
        prof = prof_num / np.maximum(prof_den, 1)
        rr = np.arange(len(prof))
        ee = np.cumsum(prof * 2 * np.pi * rr)
        if ee[-1] > 0:
            ee /= ee[-1]
        p2r = interp1d(
            prof[::-1],
            rr[::-1],
            bounds_error=False,
            fill_value=(rr.max(), rr.max()),
        )
        return prof, ee, p2r

    @staticmethod
    def _crop_kernel(kern: np.ndarray, rlim: float) -> tuple[np.ndarray, float]:
        """Crop ``kern`` around its centre to ``rlim`` pixels."""
        r = int(np.ceil(rlim))
        cy, cx = (np.array(kern.shape) - 1) / 2
        size_y = min(2 * r + (kern.shape[0] % 2), kern.shape[0])
        size_x = min(2 * r + (kern.shape[1] % 2), kern.shape[1])
        cut = Cutout2D(kern, (cx, cy), (size_y, size_x), mode="trim", copy=True)
        kc = cut.data
        return kc, float(kc.sum())

    @staticmethod
    def prepare_kernel_info(
        templates: list["Template"],
        psf_full: np.ndarray,
        image_770: np.ndarray,
        weight_770: np.ndarray | None,
        *,
        eta: float,
        r_min_pix: float = 1.0,
        r_max_pix: float | None = None,
    ) -> None:
        """Compute quick-flux based kernel crop radius and encircled energy."""
        if not eta:
            return

        prof, ee, p2r = Templates._prepare_fft_fast(psf_full)
        rr = np.arange(len(prof))

        if weight_770 is not None:
            sigma_pix = float(
                np.median(np.sqrt(1 / weight_770[weight_770 > 0]))
            )
        else:
            sigma_pix = float(np.std(image_770))

        qflux = Templates.quick_flux(templates, image_770)

        for tmpl, Fq in zip(templates, qflux):
            if not np.isfinite(Fq) or Fq <= 0:
                tmpl.ee_rlim = 0.0
                tmpl.ee_fraction = 1.0
                continue

            thresh = float(eta) * sigma_pix / Fq
            thresh = np.clip(thresh, prof.min(), prof.max())
            r_pix = float(p2r(thresh))
            r_pix = max(r_min_pix, r_pix)
            if r_max_pix is not None:
                r_pix = min(r_pix, r_max_pix)
            tmpl.ee_rlim = r_pix
            tmpl.ee_fraction = float(np.interp(r_pix, rr, ee))

    @staticmethod
    def quick_flux(templates: List[Template], image: np.ndarray) -> np.ndarray:
        """Return quick flux estimates based on template data and image."""
        flux = np.zeros(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            tt = tmpl.data[tmpl.slices_cutout]
            img = image[tmpl.slices_original]
            ttsqs = np.sum(tt**2)
            flux[i] = np.sum(img * tt) / ttsqs if ttsqs > 0 else 0.0
            tmpl.flux = flux[i]  # Store quick flux in the template for later use
        return flux

    @staticmethod
    def predicted_errors(templates: List[Template], weights: np.ndarray) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        pred = np.empty(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            w = weights[tmpl.slices_original]
            pred[i] = 1.0 / np.sqrt(np.sum(w * tmpl.data[tmpl.slices_cutout]**2))
            tmpl.err = pred[i]  # Store RMS in the template for later use
        return pred

    def prune_outside_weight(self, weight: np.ndarray) -> List[Template]:
        """Remove templates with no overlap with the provided ``weight`` map.

        A template is discarded if all pixels belonging to its segmentation
        footprint fall on non-positive weight values. The check is performed in
        the original image coordinates using ``tmpl.slices_original``.

        Parameters
        ----------
        weight : np.ndarray
            Weight map aligned with ``self.original_shape``.

        Returns
        -------
        list[Template]
            Remaining templates after pruning.
        """
        keep: list[Template] = []
        for tmpl in self._templates:
            sl = tmpl.slices_original
            seg = tmpl.data[tmpl.slices_cutout] > 0
            w = weight[sl] > 0
            if np.any(seg & w):
                keep.append(tmpl)

        dropped = len(self._templates) - len(keep)
        if dropped:
            print(f"Pruned {dropped} templates outside weight map")
        self._templates = keep
        return self._templates

    @property
    def templates(self) -> List[Template]:
        """Return the list of templates."""
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
            # enfore minimum size
            height = max(y - bbox.iymin, bbox.iymax - y, self.min_size//2) * 2 
            width =  max(x - bbox.ixmin, bbox.ixmax - x, self.min_size//2) * 2 

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
        kernel: np.ndarray | PSFRegionMap | None,
        inplace: bool = False,
    ) -> list[Template]:
        """Convolve all templates with ``kernel``.

        Parameters
        ----------
        kernel : np.ndarray or PSFRegionMap or None
            Convolution kernel matching the template resolution. If ``None``,
            templates are returned unchanged (aside from optional padding).
            If templates have ``ee_rlim`` set via :meth:`prepare_kernel_info`,
            kernels are cropped to this radius and their ``ee_fraction`` is
            stored on each template.
        inplace : bool, optional
            If ``True``, templates are modified in place and the internal list
            is returned. Otherwise a new list of convolved templates is
            produced.

        Returns
        -------
        list of Template
            Convolved templates.
        """

        if not self._templates:
            raise ValueError("No templates to convolve. Run extract_templates first.")

        tmpls = self._templates
        original_shape = self.original_shape

        ker = None
        if kernel is not None:
            if isinstance(kernel, PSFRegionMap):
                ker = kernel
            else:
                ker = kernel

        new_templates: list[Template] = []
        for tmpl in tqdm(tmpls, desc="Convolving templates"):

            # Obtain kernel for this template
            kern = None
            if ker is not None:
                if isinstance(ker, PSFRegionMap):
                    x, y = tmpl.position_original
                    if tmpl.wcs is not None:
                        ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
                    else:
                        ra, dec = x, y
                    kern = ker.get_psf(ra, dec)
                else:
                    kern = ker

            rlim = getattr(tmpl, "ee_rlim", 0.0)
            if kern is not None and rlim > 0:
                kern_use, ee = Templates._crop_kernel(kern, rlim)
            else:
                kern_use = kern
                ee = float(kern_use.sum()) if kern_use is not None else 1.0

            ky_pad, kx_pad = kern_use.shape if kern_use is not None else (0, 0)

            new_tmpl = tmpl.pad((ky_pad, kx_pad), original_shape, inplace=inplace)

            if kern_use is not None:
                tmpl_flux = new_tmpl.data.sum()
                conv = fftconvolve(new_tmpl.data, kern_use, mode="same")
                new_tmpl.data[:] = conv / max(tmpl_flux, 1e-12)

            new_tmpl.ee_rlim = rlim
            new_tmpl.ee_fraction = ee

            if not inplace:
                new_templates.append(new_tmpl)

        return new_templates if not inplace else self._templates



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
