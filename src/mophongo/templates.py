from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Tuple
from copy import deepcopy
from collections import defaultdict

import logging
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.segmentation import SegmentationImage
from tqdm import tqdm
from scipy.signal import fftconvolve
from skimage.measure import block_reduce

from .utils import measure_shape, bin2d_mean, intersection
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


def _key_window(cy: int, cx: int, radius: int = 2):
    """Yield integer coordinate pairs in a square neighbourhood."""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yield (cy + dy, cx + dx)


def _weighted_norm(tmpl: "Template", wht: np.ndarray) -> float:
    """Return weighted L2 norm of ``tmpl`` over its support."""
    y0, y1, x0, x1 = tmpl.bbox
    data = tmpl.data[tmpl.slices_cutout]
    w = wht[y0:y1, x0:x1]
    return float(np.sum(data * w * data))


def _weighted_dot(t1: "Template", t2: "Template", wht: np.ndarray) -> float:
    """Return weighted dot product between two templates."""
    inter = intersection(t1.bbox, t2.bbox)
    if inter is None:
        return 0.0
    y0, y1, x0, x1 = inter
    s1 = (
        slice(y0 - t1.bbox[0], y1 - t1.bbox[0]),
        slice(x0 - t1.bbox[2], x1 - t1.bbox[2]),
    )
    s2 = (
        slice(y0 - t2.bbox[0], y1 - t2.bbox[0]),
        slice(x0 - t2.bbox[2], x1 - t2.bbox[2]),
    )
    return float(np.sum(t1.data[s1] * t2.data[s2] * wht[y0:y1, x0:x1]))


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

    def downsample_wcs(self, image_lo: np.ndarray, wcs_lo, k: int) -> "Template":
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
        pos = self.input_position_original
        ra, dec = self.wcs.wcs_pix2world(*pos, 0)

        # Convert RA/Dec to pixel coordinates in the low-res WCS
        x_lo, y_lo = wcs_lo.wcs_world2pix(ra, dec, 0)

        # Calculate new size (downsampled)
        height, width = self.data.shape[0] // k, self.data.shape[1] // k

        # Create the new template using the low-res image and WCS
        lowres_tmpl = Template(image_lo, (x_lo, y_lo), (height, width), wcs=wcs_lo, label=self.id)

        # Fill the data with block-reduced (averaged) values from the high-res template
        lowres_tmpl.data[:] = block_reduce(self.data, k, func=np.mean)
        return lowres_tmpl


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

    @staticmethod
    def prune_and_dedupe(
        templates: List["Template"],
        weights: np.ndarray,
        *,
        radius: int = 2,
        norm_rel_tol: float = 1e-12,
        cos_tol: float = 0.999,
    ) -> List["Template"]:
        """Prune templates with low norm and remove near-duplicates.

        Parameters
        ----------
        templates
            List of templates to process.
        weights
            Weight map matching the image on which templates live.
        radius
            Integer radius of the key-window hash used for duplicate search.
        norm_rel_tol
            Relative tolerance below which templates are dropped.
        cos_tol
            Cosine similarity threshold for duplicate removal.

        Returns
        -------
        list[Template]
            Pruned list of templates; original order is preserved.
        """

        if not templates:
            return []

        norms = np.array([_weighted_norm(t, weights) for t in templates])
        tol = norm_rel_tol * norms.max() if norms.size else 0.0

        bucket: defaultdict[tuple[int, int], List[int]] = defaultdict(list)
        kept: List[Template] = []
        kept_norms: List[float] = []

        for tmpl, nrm in zip(templates, norms):
            if nrm < tol:
                continue

            cy, cx = map(int, tmpl.position_original)
            duplicate = False
            for key in _key_window(cy, cx, radius):
                for k in bucket.get(key, []):
                    cos = _weighted_dot(tmpl, kept[k], weights) / np.sqrt(nrm * kept_norms[k])
                    if cos > cos_tol:
                        duplicate = True
                        break
                if duplicate:
                    break

            if duplicate:
                continue

            tmpl.norm = nrm
            idx = len(kept)
            kept.append(tmpl)
            kept_norms.append(nrm)
            for key in _key_window(cy, cx, radius):
                bucket[key].append(idx)

        return kept

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
        kernel: np.ndarray | PSFRegionMap | None,
        inplace: bool = False,
    ) -> list[Template]:
        """Convolve all templates with ``kernel``.

        Parameters
        ----------
        kernel : np.ndarray or PSFRegionMap or None
            Convolution kernel matching the template resolution. If ``None``,
            templates are returned unchanged (aside from optional padding).
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
                ky, kx = ker.psfs[0].shape
            else:
                ker = kernel
                ky, kx = ker.shape
        else:
            ky = kx = 0

        new_templates: list[Template] = []
        for tmpl in tqdm(tmpls, desc="Convolving templates"):
            new_tmpl = tmpl.pad((ky, kx), original_shape, inplace=inplace)

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

                conv = fftconvolve(new_tmpl.data, kern, mode="same")
                new_tmpl.data[:] = conv / conv.sum()

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
