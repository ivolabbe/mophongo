from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Sequence, Tuple
from copy import deepcopy

import logging
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.segmentation import SegmentationImage
from tqdm import tqdm
from astropy.nddata import block_reduce

from .utils import bin_remap, fftconvolve
from .psf_map import PSFRegionMap

logger = logging.getLogger(__name__)

try:
    from astropy.wcs import WCS, Sip
except Exception:  # if astropy not available / SIP missing
    WCS = None
    Sip = None

__all__ = ["AlignedCutout", "Template", "Templates", "scale_wcs_pixel"]

# ───────────────────────── helpers ─────────────────────────


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _aligned_bounds_1d(pos: float, size_min: int, align: int) -> tuple[int, int]:
    """
    Return [imin, imax) bounds with:
      • lower bound divisible by `align` (align≥1)
      • length >= size_min
      • length is a multiple of `align`
      • center close to `pos`
    """
    size_min = int(size_min)
    align = int(max(1, align))

    if align == 1:
        imin = int(np.ceil(pos - size_min / 2.0))
        imax = imin + size_min
        return imin, imax

    size_min = max(size_min, align)

    imin0 = int(np.ceil(pos - size_min / 2.0))
    dx_min = imin0 % align
    imin = imin0 - dx_min

    imax_f = pos + max((pos - imin), size_min / 2.0)
    #    print(pos, size_min)
    dx_max = (-imax_f) % align
    imax = int(np.rint(imax_f + dx_max))

    L = imax - imin
    if L <= 0 or (L % align) != 0:
        imax = imin + ((L + align - 1) // align) * align
    return int(imin), int(imax)


def _bbox_from_slices(sl):
    return ((sl[0].start, sl[0].stop - 1), (sl[1].start, sl[1].stop - 1))


def _slices_from_bbox(bbox):
    return (slice(bbox[0], bbox[1] + 1), slice(bbox[2], bbox[3] + 1))


def _block_reduce(arr: np.ndarray, fact: int, func=np.sum) -> np.ndarray:
    """
    Fast 2-D block reduction by integer `fact` (flux-conserving with func=np.sum).
    """
    a = np.asarray(arr, dtype=np.float32, order="C")
    H, W = a.shape
    H2, W2 = (H // fact) * fact, (W // fact) * fact
    if H2 != H or W2 != W:
        a = a[:H2, :W2]
    a = a.reshape(H2 // fact, fact, W2 // fact, fact)
    return func(a, axis=(1, 3), dtype=np.float32)


def _block_replicate(arr: np.ndarray, fact: int, conserve_sum: bool = True) -> np.ndarray:
    """
    Fast 2-D nearest upsampling by integer `fact`. If `conserve_sum` True,
    each pixel is divided by fact**2 (so flux is preserved).
    """
    a = np.asarray(arr, dtype=np.float32, order="C")
    tile = np.ones((fact, fact), dtype=np.float32)
    if conserve_sum:
        tile /= fact * fact
    return np.kron(a, tile)


def _is_identity_kernel(kernel: np.ndarray | None) -> bool:
    """Return True for a centered delta-function convolution kernel."""
    if kernel is None:
        return True
    arr = np.asarray(kernel)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        return False
    expected = np.zeros_like(arr, dtype=float)
    expected[arr.shape[0] // 2, arr.shape[1] // 2] = 1.0
    return bool(np.allclose(arr, expected, rtol=0.0, atol=1e-12))


def _copy_template_overlap(src: "Template", dst: "Template") -> None:
    """Copy ``src.data`` into ``dst.data`` using true parent-image bounds."""
    sx0, sy0 = map(int, src._origin_original_true)
    dx0, dy0 = map(int, dst._origin_original_true)
    sx1 = sx0 + src.data.shape[1]
    sy1 = sy0 + src.data.shape[0]
    dx1 = dx0 + dst.data.shape[1]
    dy1 = dy0 + dst.data.shape[0]

    x0 = max(sx0, dx0)
    y0 = max(sy0, dy0)
    x1 = min(sx1, dx1)
    y1 = min(sy1, dy1)
    if x1 <= x0 or y1 <= y0:
        return

    dst.data[y0 - dy0 : y1 - dy0, x0 - dx0 : x1 - dx0] = src.data[y0 - sy0 : y1 - sy0, x0 - sx0 : x1 - sx0]


def scale_wcs_pixel(
    wcs: WCS | None, pixel_scale_factor: float, new_shape: tuple[int, int] | None = None
) -> WCS | None:
    """
    Scale a WCS by a pixel-size factor (>=0), **preserving sky coordinates**.
      pixel_scale_factor > 1  → pixels get larger (downsampling)
      pixel_scale_factor < 1  → pixels get smaller (upsampling)

    cd/cdelt ← cd/cdelt * pixel_scale_factor
    crpix   ← (crpix - 0.5)/pixel_scale_factor + 0.5
    """
    if wcs is None:
        return None
    w2 = deepcopy(wcs)

    f = float(pixel_scale_factor)
    if hasattr(w2.wcs, "cd") and w2.wcs.cd is not None and w2.wcs.cd.size:
        w2.wcs.cd = w2.wcs.cd * f
    else:
        w2.wcs.cdelt = w2.wcs.cdelt * f

    old_crpix = w2.wcs.crpix.copy()
    w2.wcs.crpix = (old_crpix - 0.5) / f + 0.5

    if new_shape is not None:
        try:
            w2.pixel_shape = (int(new_shape[0]), int(new_shape[1]))
        except Exception:
            pass

    if getattr(wcs, "sip", None) is not None and Sip is not None:
        # SIP polynomials evaluated relative to their CRPIX (in pixel units):
        # just shift SIP CRPIX the same way as WCS CRPIX
        off = old_crpix - w2.wcs.crpix
        w2.sip = Sip(wcs.sip.a, wcs.sip.b, wcs.sip.ap, wcs.sip.bp, wcs.sip.crpix - off)

    w2.wcs.set()
    return w2


# ──────────────────────── main class ─────────────────────────


class AlignedCutout:
    """
    Minimal 2-D cutout that:
      • uses *partial* mode only (zero outside the image)
      • `size` is a **minimum**; actual data may be enlarged by `align`
      • lower-left bound is aligned to a multiple of `align` (per axis)
      • shape is a multiple of `align`
      • stores an adjusted WCS (incl. SIP if present)

    Parameters
    ----------
    data : 2D ndarray
    position : (x, y) float — pixel-center coords
    size : (ny, nx) int or scalar
    align : int >= 1
    copy : bool
    fill_value : float
    wcs : astropy.wcs.WCS (optional)
    """

    def __init__(
        self,
        data: np.ndarray,
        position: tuple[float, float],
        size: tuple[int, int] | int,
        *,
        align: int = 1,
        copy: bool = False,
        fill_value: float | int = 0.0,
        wcs: WCS | None = None,
    ):
        arr = np.asarray(data)
        self.align = int(max(1, align))
        self.shape_input = arr.shape  # (ny, nx)

        x, y = float(position[0]), float(position[1])
        if np.isscalar(size):
            ny = nx = int(size)
        else:
            ny, nx = int(size[0]), int(size[1])

        # aligned bounds in ORIGINAL coords
        x0, x1 = _aligned_bounds_1d(x, nx, self.align)
        y0, y1 = _aligned_bounds_1d(y, ny, self.align)
        h = y1 - y0
        w = x1 - x0

        # overlap with source image
        Y0 = max(0, y0)
        X0 = max(0, x0)
        Y1 = min(arr.shape[0], y1)
        X1 = min(arr.shape[1], x1)

        dy = Y0 - y0
        dx = X0 - x0
        yslice_dst = slice(dy, dy + (Y1 - Y0))
        xslice_dst = slice(dx, dx + (X1 - X0))
        yslice_src = slice(Y0, Y1)
        xslice_src = slice(X0, X1)

        fully_inside = (y0 >= 0) and (x0 >= 0) and (y1 <= arr.shape[0]) and (x1 <= arr.shape[1])

        if not fully_inside or copy:
            out = np.zeros((h, w), dtype=arr.dtype)
            if fill_value != 0:
                out[...] = out.dtype.type(fill_value)
            if (Y1 > Y0) and (X1 > X0):
                out[yslice_dst, xslice_dst] = arr[yslice_src, xslice_src]
            self.data = out
        else:
            self.data = arr[y0:y1, x0:x1]

        self.shape = self.data.shape
        self.input_position_original = (x, y)
        self.input_position_cutout = (x - x0, y - y0)

        self.slices_original = (yslice_src, xslice_src)
        self.slices_cutout = (yslice_dst, xslice_dst)

        self.bbox_original = _bbox_from_slices(self.slices_original)
        self.bbox_cutout = _bbox_from_slices(self.slices_cutout)

        self.origin_original = (self.slices_original[1].start, self.slices_original[0].start)  # (x, y)
        self.origin_cutout = (self.slices_cutout[1].start, self.slices_cutout[0].start)  # (x, y)

        # “true” cutout origin relative to original, including any fill padding
        self._origin_original_true = (
            self.origin_original[0] - self.slices_cutout[1].start,
            self.origin_original[1] - self.slices_cutout[0].start,
        )

        self.position_original = (_round_half_up(x), _round_half_up(y))
        self.position_cutout = (
            _round_half_up(self.input_position_cutout[0]),
            _round_half_up(self.input_position_cutout[1]),
        )

        so, sc = self.slices_original, self.slices_cutout
        self.center_original = (0.5 * (so[1].start + so[1].stop - 1), 0.5 * (so[0].start + so[0].stop - 1))
        self.center_cutout = (0.5 * (sc[1].start + sc[1].stop - 1), 0.5 * (sc[0].start + sc[0].stop - 1))

        # WCS adjusted to the cutout (shift CRPIX, keep SIP consistent)
        if wcs is not None:
            off_xy = np.array(self._origin_original_true, dtype=float)  # (x, y)
            w2 = deepcopy(wcs)
            if getattr(w2, "wcs", None) is not None and getattr(w2.wcs, "crpix", None) is not None:
                w2.wcs.crpix -= off_xy
            try:
                w2.array_shape = self.data.shape
                w2.pixel_shape = self.data.shape
            except Exception:
                pass
            if getattr(wcs, "sip", None) is not None and Sip is not None:
                w2.sip = Sip(wcs.sip.a, wcs.sip.b, wcs.sip.ap, wcs.sip.bp, wcs.sip.crpix - off_xy)
            w2.wcs.set()
            self.wcs = w2
        else:
            self.wcs = None

    # ───────────── array-only helpers (no geometry changes) ─────────────

    def as_block_reduced(self, factor: int, func=np.sum) -> np.ndarray:
        """Return block-reduced self.data by `factor` (trims edges as needed)."""
        if factor < 1 or int(factor) != factor:
            raise ValueError("factor must be a positive integer")
        return _block_reduce(self.data, int(factor), func=func)

    def as_block_replicated(self, factor: int, conserve_sum: bool = True) -> np.ndarray:
        """Return block-replicated self.data by `factor` (nearest upsample)."""
        if factor < 1 or int(factor) != factor:
            raise ValueError("factor must be a positive integer")
        if factor == 1:
            return np.asarray(self.data, dtype=np.float32, order="C")
        return _block_replicate(self.data, int(factor), conserve_sum=conserve_sum)

    # ───────────── geometry-aware resampling (returns new cutouts) ────────────

    def downsample(self, factor: int) -> "AlignedCutout":
        """
        Return a new cutout binned by integer `factor`:
          • flux-conserving (sum)
          • correct position & WCS updates
          • exact only if origin and shape are divisible by `factor`
        """
        f = int(factor)
        if f < 1:
            raise ValueError("factor must be >= 1")
        if f == 1:
            return deepcopy(self)

        H, W = self.shape
        x0, y0 = self.origin_original

        if (x0 % f) or (y0 % f) or (H % f) or (W % f):
            raise ValueError(
                "Downsample requires origin and size divisible by factor "
                f"(origin=({x0},{y0}), shape=({H},{W}), factor={f})."
            )

        data_lo = _block_reduce(self.data, f, func=np.sum)  # float32

        pos_lo = bin_remap(self.input_position_original, f)  # (x, y)
        shape_input_lo = (self.shape_input[0] // f, self.shape_input[1] // f)
        wcs_lo = scale_wcs_pixel(self.wcs, pixel_scale_factor=f, new_shape=shape_input_lo)

        # alignment propagates: new origin = old_origin / f
        align_lo = max(1, self.align // f)

        # build a new cutout on a dummy parent (zeros), then insert data
        dummy = np.zeros(shape_input_lo, dtype=np.float32)
        out = AlignedCutout(dummy, tuple(pos_lo), data_lo.shape, align=align_lo, copy=True, wcs=wcs_lo)
        out.data[...] = data_lo
        return out

    def upsample(self, factor: int, conserve_sum: bool = True) -> "AlignedCutout":
        """
        Return a new cutout expanded by integer `factor`:
          • uses block replication (optionally flux-conserving)
          • correct position & WCS updates
        """
        f = int(factor)
        if f < 1:
            raise ValueError("factor must be >= 1")
        if f == 1:
            return deepcopy(self)

        data_hi = _block_replicate(self.data, f, conserve_sum=conserve_sum)

        pos_hi = expand_remap(self.input_position_original, f)  # (x, y)
        shape_input_hi = (self.shape_input[0] * f, self.shape_input[1] * f)
        wcs_hi = scale_wcs_pixel(self.wcs, pixel_scale_factor=1.0 / f, new_shape=shape_input_hi)

        align_hi = self.align * f

        dummy = np.zeros(shape_input_hi, dtype=np.float32)
        out = AlignedCutout(dummy, tuple(pos_hi), data_hi.shape, align=align_hi, copy=True, wcs=wcs_hi)
        out.data[...] = data_hi
        return out


class Template(Cutout2D):
    """Cutout-based template storing slice bookkeeping."""

    FLAG_VALID = 0x01  # 0001: Template is valid
    FLAG_CONVOLVED = 0x02  # 0010: Template has been convolved
    FLAG_SUM_ZERO = 0x04  # 0100: Template sum is zero
    FLAG_HAS_NAN = 0x08  # 1000: Template contains NaN values
    FLAG_OUTSIDE_WEIGHT = 0x10  # 1 0000: Template is outside weight map
    FLAG_SHIFTED = 0x20  # 10 0000: Template has been shifted
    FLAG_DEBLENDED = 0x40  # catalog ``is_deblended`` provenance flag
    FLAG_SATURATED = 0x80  # catalog ``FLAG_SATURATED_<FILTER>`` provenance

    def __init__(
        self,
        data: np.ndarray,
        position: tuple[float, float],
        size: tuple[int, int],
        label: int | None = None,
        copy: bool = True,
        wcs: WCS | None = None,
        **kwargs,
    ) -> None:
        super().__init__(data, position, size, mode="partial", fill_value=0.0, copy=copy, wcs=wcs, **kwargs)
        # do not allow writing into a view
        #        if not copy:
        #            self.data.flags.writeable = False

        # basic metadata
        # Store the original data reference
        #        self.base_data = data.copy()

        self.is_dirty = False  # flag to track if data needs to be updated
        # @@@ bug in Cutout2D: shape_input is not set correctly
        self.shape_input = data.shape
        self.shape_original = data.shape
        self.wcs_original = wcs

        # logical
        self.id = label
        self.id_parent = label  # @@@ this is redundant -> remove
        self.id_scene = 1
        self.deblend_parent_label: int | None = None
        self.deblend_nchildren: int = 1
        self.name = "main"  # component name
        # Diagnostic flags (bitwise)

        self.flag = 0  # bitwise flag for diagnostics
        self.flag |= Template.FLAG_VALID
        self.is_star: bool = False  # set by pipeline from catalog flag_star

        # encircled energy of the low-res PSF stamp at this position, set by
        # convolve_templates.  NaN until then, so an unset value fails loudly
        # rather than silently applying no correction.
        self.ee_psf_lo: float = np.nan
        # fraction of the normalised source model this template retains, i.e.
        # sum(data) once construction is finished.  Below one when wing flux has
        # been handed to a neighbour.  Diagnostic only: the fitted amplitude does
        # not scale with it, because the blanked pixels carry flux but almost no
        # fitting weight (docs/ENCIRCLED_ENERGY.pdf).
        self.ee_tmpl: float = np.nan

        # flux
        self.flux = 0.0
        self.err = 0.0
        self.err_pred = 0.0  # predicted error from weight map and profile
        self.wnorm = 0.0  # weighted norm of the template d * w * d

        # astrometry
        # record shift from original position here
        # this is the intended shift from base_data to data
        self.to_shift = np.array([0.0, 0.0], dtype=float)  # impending shift
        self.shifted = np.array([0.0, 0.0], dtype=float)  # accumulated shift

    @property
    def is_deblended(self) -> bool:
        """Whether this template comes from a deblended catalog child."""
        return bool(self.flag & Template.FLAG_DEBLENDED)

    @is_deblended.setter
    def is_deblended(self, value: bool) -> None:
        if value:
            self.flag |= Template.FLAG_DEBLENDED
        else:
            self.flag &= ~Template.FLAG_DEBLENDED

    @property
    def is_saturated(self) -> bool:
        """Whether this template represents a saturated/repaired source.

        Set from a catalog ``FLAG_SATURATED_<FILTER>`` column. The scene
        builder forces saturated templates into their own scene so the
        large PSF wings do not contaminate neighbouring flux solutions.
        """
        return bool(self.flag & Template.FLAG_SATURATED)

    @is_saturated.setter
    def is_saturated(self, value: bool) -> None:
        if value:
            self.flag |= Template.FLAG_SATURATED
        else:
            self.flag &= ~Template.FLAG_SATURATED

    @property
    def bbox(self) -> tuple[int, int, int, int]:  # pragma: no cover - simple alias
        (ymin, ymax), (xmin, xmax) = self.bbox_original
        return int(ymin), int(ymax), int(xmin), int(xmax)

    @classmethod
    def from_stamp(
        cls,
        data: np.ndarray,
        origin: tuple[int, int],
        input_position_original: tuple[float, float],
        shape_original: tuple[int, int],
        *,
        wcs: WCS | None = None,
        label: int | None = None,
        parent_image: np.ndarray | None = None,
    ) -> "Template":
        """Rebuild a template from stored stamp pixels and geometry.

        Inverse of the stamp serialization in ``pipeline.write_stamps``:
        ``origin`` is the original-grid pixel (x, y) of ``data[0, 0]``
        (``_origin_original_true``, may be negative for padded cutouts) and
        ``shape_original`` the full image shape the cutout belongs to, which
        sets the clipped ``slices_original``/``slices_cutout`` pair.

        Args:
            data: Stamp pixels, shape ``(ny, nx)``.
            origin: Original-grid pixel (x, y) of ``data[0, 0]``.
            input_position_original: Source position (x, y) on the original grid.
            shape_original: Shape of the full parent image.
            wcs: WCS of the full parent image.
            label: Source id.
            parent_image: Optional zero array of ``shape_original`` reused
                across calls to avoid per-source allocations.

        Returns:
            Template with the same geometry the stamp was written from.
        """
        ny, nx = data.shape
        x0, y0 = (int(v) for v in origin)
        if parent_image is None:
            parent_image = np.zeros(shape_original, dtype=data.dtype)
        # position chosen so the aligned bounds land exactly on origin:
        # ceil((x0 + nx/2) - nx/2) == x0
        tmpl = cls(
            parent_image,
            position=(x0 + nx / 2.0, y0 + ny / 2.0),
            size=(ny, nx),
            wcs=wcs,
            label=label,
            copy=False,
        )
        tmpl.data = np.array(data, copy=True)
        x, y = (float(v) for v in input_position_original)
        tmpl.input_position_original = (x, y)
        tmpl.position_original = (_round_half_up(x), _round_half_up(y))
        tmpl.input_position_cutout = (x - x0, y - y0)
        tmpl.position_cutout = (
            _round_half_up(tmpl.input_position_cutout[0]),
            _round_half_up(tmpl.input_position_cutout[1]),
        )
        return tmpl

    def pad(
        self,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int],
        *,
        image: np.ndarray | None = None,
        inplace=False,
    ) -> "Template":
        """Create a new Template with padding, maintaining correct original coordinates."""

        # force padding to be even, otherwise unpredictable behavior for cutout
        ony, onx = padding[0] // 2, padding[1] // 2
        ny, nx = self.data.shape

        # Create new Template directly from the original array reference
        # This ensures all coordinates remain consistent with the true original
        if image is None:
            image = np.zeros(self.shape_input, dtype=self.data.dtype)

        new_template = Template(
            data=image,
            position=self.input_position_original,
            size=(ny + ony * 2, nx + onx * 2),
            wcs=self.wcs_original,
            label=self.id,
        )

        # Now place the old data in our padded version
        new_template.data[...] = 0
        _copy_template_overlap(self, new_template)

        # if inplace is True, update the current instance
        if inplace:
            # overwrite the current attributes with the new one
            self.__dict__.update(new_template.__dict__)

        return new_template

    # ------------------------------------------------------------------
    # centred, even-padding convolution
    # ------------------------------------------------------------------
    def convolve_cutout(
        self, kernel: np.ndarray, *, parent_image: np.ndarray | None = None, preserve_dtype: bool = True
    ) -> "Template":
        """
        Convolve *this* template with a centred ``kernel`` **and return a new
        `Template` that already has the correct, larger geometry**.

        The routine guarantees that the padding applied to the original
        cut-out is **even** – i.e. an integer number of pixels *on both
        sides* – which avoids the odd-size artefacts you saw earlier.

        Parameters
        ----------
        kernel
            2-D, centred convolution kernel.
        parent_image
            Reference to the *full* parent image.  If ``None`` a tiny dummy
            array of zeros (same dtype) is created just to satisfy Cutout2D.
            It is **never** copied, so the memory cost is negligible.
        preserve_dtype
            Cast the result back to ``self.data.dtype`` (default) instead of
            keeping the float64 that `fftconvolve` returns.

        Returns
        -------
        Template
            A *new* template whose ``data`` attribute contains the full
            convolution result and whose spatial metadata (WCS, slices, …)
            is already consistent with the enlarged size.
        """
        # 1. --- full convolution -------------------------------------------------
        full = fftconvolve(self.data, kernel, mode="full")
        if preserve_dtype:
            full = full.astype(self.data.dtype, copy=False)

        ny, nx = full.shape

        if parent_image is None:
            # a 1-byte dummy is enough – Cutout2D only keeps a *view*
            parent_image = np.zeros(self.shape_input, dtype=self.data.dtype)

        # 2. make *sure* the new cut-out is large enough -----------------------
        #     If ny or nx is odd, add 1 so it becomes even (keeps later padding
        #     code happy) *and* ≥ full.shape.
        ny_even = ny if ny % 2 == 0 else ny + 1
        nx_even = nx if nx % 2 == 0 else nx + 1

        origin_x = int(self._origin_original_true[0]) - int(kernel.shape[1] // 2)
        origin_y = int(self._origin_original_true[1]) - int(kernel.shape[0] // 2)
        center_x = origin_x + 0.5 * (nx_even - 1)
        center_y = origin_y + 0.5 * (ny_even - 1)

        # # --------- 3. build a fresh Cutout2D --------------------------------
        new_cut = Template(
            parent_image,  # original full image reference
            position=(center_x, center_y),  # note (x, y)
            size=(ny_even, nx_even),  # (ny, nx)
            wcs=self.wcs,  # note wcs origin is wrong
            label=self.id,
            copy=False,  # do not copy the data, we are replacing later
        )

        # copy the convolution result into the enlarged cut-out
        # account for the extra pixel
        # 4.  centre `full` inside the (possibly larger) even array -------------
        true_origin_x, true_origin_y = map(int, new_cut._origin_original_true)
        x0 = origin_x - true_origin_x
        y0 = origin_y - true_origin_y
        data = np.zeros(new_cut.data.shape, dtype=self.data.dtype)
        data[y0 : y0 + ny, x0 : x0 + nx] = full
        new_cut.data = data
        new_cut.input_position_original = self.input_position_original
        new_cut.position_original = self.position_original
        new_cut.input_position_cutout = (
            float(self.input_position_original[0]) - true_origin_x,
            float(self.input_position_original[1]) - true_origin_y,
        )
        new_cut.position_cutout = (
            _round_half_up(new_cut.input_position_cutout[0]),
            _round_half_up(new_cut.input_position_cutout[1]),
        )
        #        new_cut.base_data = data  # also store it in base data
        new_cut.flag = self.flag | Template.FLAG_CONVOLVED
        new_cut.id_parent = self.id_parent
        new_cut.id_scene = self.id_scene
        new_cut.deblend_parent_label = self.deblend_parent_label
        new_cut.deblend_nchildren = self.deblend_nchildren
        new_cut.name = self.name
        new_cut.ee_tmpl = self.ee_tmpl
        new_cut.ee_psf_lo = self.ee_psf_lo

        return new_cut



    # verified for k=2,4 for sizes 4-16
    def downsample(self, k: int, image: np.ndarray | None = None, wcs_lo: WCS | None = None) -> "Template":
        """
        Flux-conserving k× downsample aligned to the global hi-res grid.
        Handles negative origins, preserves center-of-pixel convention.
        """
        from copy import deepcopy

        if k == 1:
            return deepcopy(self)

        H, W = self.data.shape

        # Global lower-left of this cutout (integer pixel indices, can be negative)
        # Cutout2D uses (x, y); ensure we keep that order consistent
        x0_hi, y0_hi = map(int, self._origin_original_true)

        # Phase to reach the next k-aligned boundary *inside* this cutout
        dx = (-x0_hi) % k
        dy = (-y0_hi) % k

        # Low-res size from the remaining pixels after phase adjustment
        hlo = H // k
        wlo = W // k
        if hlo <= 0 or wlo <= 0:
            raise ValueError("Cutout too small to downsample with current k/phase.")

        # A non-zero phase shortens the k-aligned block, so the trailing low-res
        # row/column stays zero and that flux is dropped.  Exact only when the
        # origin is k-aligned (see AlignedCutout.downsample, which refuses).
        if (H - dy) // k < hlo or (W - dx) // k < wlo:
            logger.warning(
                "template %s: origin (%d, %d) is not aligned to k=%d; trailing "
                "row/column of the downsampled cutout is zero-filled and its flux "
                "is lost. Use multi_resolution_method='upsample' for exact "
                "block alignment.",
                self.id,
                x0_hi,
                y0_hi,
                k,
            )

        # Hi-res block aligned to k×k boundaries
        hi_aligned = self.data[dy : dy + hlo * k, dx : dx + wlo * k]
        # Flux-conserving reduction
        lo_block = block_reduce(hi_aligned, k, func=np.sum)

        # print(hlo, wlo, k, lo_block.shape, hi_aligned.shape)

        # Map the *center* correctly
        x_lo, y_lo = bin_remap(self.input_position_original, k)
        shape_input = np.array(self.shape_input) // k

        if image is None:
            image = np.zeros(shape_input)
        # Build the low-res Template at the correct fractional center
        low = Template(image, (x_lo, y_lo), (hlo, wlo), wcs=wcs_lo, label=self.id)

        ly, lx = lo_block.shape
        # print(wlo, hlo, low.shape)
        # print(dx, dy, ly, lx)
        low.data[:ly, :lx] = lo_block

        return low

    def project_to_block_replicated_grid(
        self, factor: int, *, parent_image: np.ndarray | None = None, preserve_dtype: bool = True
    ) -> "Template":
        """Project this template onto a globally aligned block-replicated grid.

        The upsampled low-resolution fitting path represents each native image
        pixel as a constant ``factor x factor`` block.  A convolved template
        fitted on that grid must live in the same pixel basis; otherwise the
        residual depends on the source phase inside the native pixel.  This
        method integrates the template over global native-pixel blocks and
        then replicates those block sums back to the high-resolution grid.
        """
        f = int(factor)
        if f < 1:
            raise ValueError("factor must be >= 1")
        if f == 1:
            return deepcopy(self)

        x0, y0 = map(int, self._origin_original_true)
        h, w = self.data.shape
        x1 = x0 + w
        y1 = y0 + h
        ax0 = (x0 // f) * f
        ay0 = (y0 // f) * f
        ax1 = ((x1 + f - 1) // f) * f
        ay1 = ((y1 + f - 1) // f) * f
        aw = ax1 - ax0
        ah = ay1 - ay0

        padded = np.zeros((ah, aw), dtype=float)
        py = y0 - ay0
        px = x0 - ax0
        padded[py : py + h, px : px + w] = np.asarray(self.data, dtype=float)

        native = block_reduce(padded, f, func=np.sum)
        projected = np.repeat(np.repeat(native, f, axis=0), f, axis=1) / float(f * f)
        if preserve_dtype:
            projected = projected.astype(self.data.dtype, copy=False)

        if parent_image is None:
            parent_image = np.zeros(self.shape_input, dtype=self.data.dtype)

        center_x = ax0 + 0.5 * (aw - 1)
        center_y = ay0 + 0.5 * (ah - 1)
        out = Template(
            parent_image,
            position=(center_x, center_y),
            size=(ah, aw),
            wcs=self.wcs,
            label=self.id,
            copy=False,
        )
        out.data = projected
        out.input_position_original = self.input_position_original
        out.position_original = self.position_original
        true_origin_x, true_origin_y = map(int, out._origin_original_true)
        out.input_position_cutout = (
            float(self.input_position_original[0]) - true_origin_x,
            float(self.input_position_original[1]) - true_origin_y,
        )
        out.position_cutout = (
            _round_half_up(out.input_position_cutout[0]),
            _round_half_up(out.input_position_cutout[1]),
        )
        out.flag = self.flag
        out.deblend_parent_label = self.deblend_parent_label
        out.deblend_nchildren = self.deblend_nchildren
        return out


class Templates:
    """Container for source templates."""

    min_size = 8  # minimum size of a template in pixels

    def __init__(self) -> None:
        self._templates: List[Template] = []
        # dilated segmentation map recorded by extract_templates; used by
        # extend_with_psf_wings to restrict wing filling to background pixels
        self.segmap: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._templates)

    def __getitem__(self, idx: int) -> Template:
        return self._templates[idx]

    def __iter__(self) -> Iterator[Template]:
        return iter(self._templates)

    def add_component(
        self, parent: Template, data: np.ndarray, component: str, **kwargs: Any
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
        tmpl.id_parent = parent.id_parent or parent.id
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
        kernel: np.ndarray | None = None,
        extension: np.ndarray | str | None = None,  # 'psf', 'wings', 'both', None
        wcs: WCS | None = None,
    ) -> "Templates":
        obj = cls()
        obj.wcs = wcs

        # Step 1: Extract raw cutouts
        obj.extract_templates(hires_image, segmap, positions, wcs=wcs)

        # if type(extension) == np.ndarray:
        # Extend templates with PSF wings
        # obj.extend_with_psf_wings(extension, inplace=True)

        # Step 2: Convolve with kernel (includes padding)
        if kernel is not None:
            obj.convolve_templates(kernel, inplace=True)

        return obj

    @classmethod
    def from_cutout_models(
        cls,
        cutouts: Iterable[np.ndarray],
        positions: Iterable[Tuple[float, float]],
        ids: Iterable[int],
        *,
        original_shape: tuple[int, int],
        wcs: WCS | None = None,
        normalize: bool = False,
    ) -> "Templates":
        """Build templates from precomputed source-model cutouts.

        This bypasses segmentation/template extraction while preserving the
        normal :class:`Template` geometry used by the fitter. Cutouts are
        interpreted as per-unit-flux models unless ``normalize`` is requested.
        """
        obj = cls()
        obj.original_shape = tuple(original_shape)
        obj.wcs = wcs
        parent = np.zeros(obj.original_shape, dtype=np.float32)
        templates: list[Template] = []
        for cutout, pos, obj_id in zip(cutouts, positions, ids):
            arr = np.asarray(cutout, dtype=np.float32)
            tmpl = Template(parent, pos, arr.shape, wcs=wcs, label=int(obj_id), copy=True)
            tmpl.data = np.zeros_like(tmpl.data, dtype=np.float32)
            if tmpl.data.shape != arr.shape:
                raise ValueError(
                    f"cutout shape {arr.shape} does not match template shape {tmpl.data.shape}"
                )
            tmpl.data[:] = arr
            if normalize:
                total = float(tmpl.data.sum())
                if total != 0.0:
                    tmpl.data /= total
                else:
                    tmpl.flag |= Template.FLAG_SUM_ZERO
            elif float(np.sum(np.abs(tmpl.data))) == 0.0:
                tmpl.flag |= Template.FLAG_SUM_ZERO
            templates.append(tmpl)
        obj._templates = templates
        return obj

    # ------------------------------------------------------------
    # static helpers
    # ------------------------------------------------------------
    @staticmethod
    def apply_template_shifts(templates: Sequence[Template]) -> None:
        """Apply stored ``shift`` values to templates in-place.

        Parameters
        ----------
        templates:
            Sequence of :class:`~mophongo.templates.Template` objects whose
            ``shift`` attribute encodes the ``(dx, dy)`` offset to apply.
        Sign convention:
        Let (dx, dy) be the image→template correction predicted by astrometry,
        i.e. “shift the image by (dx,dy) to match the template.”
        When applied to template, we must shift the template by (-dx,-dy).
        And scipy.ndimage.shift takes shifts in (axis0, axis1) = (y, x) order.
        """
        from scipy.ndimage import shift as nd_shift

        for tmpl in templates:
            #            if not tmpl.is_dirty:  # skip if shift was already applied
            #                continue

            dx, dy = map(float, tmpl.to_shift)
            if abs(dx) < 1e-2 and abs(dy) < 1e-2:
                continue

            # Interpolate from the ORIGINAL template data with the accumulated
            # total shift. Re-shifting already-shifted data applies the cubic
            # spline smoothing once per iteration; scenes that need several
            # astrometric passes then fit progressively blurred templates,
            # which biases the linearized shift estimator and stalls
            # convergence short of the true offset.
            base = getattr(tmpl, "_data_unshifted", None)
            if base is None:
                base = tmpl.data
                tmpl._data_unshifted = base.copy()
            total_dx = float(tmpl.shifted[0]) + dx
            total_dy = float(tmpl.shifted[1]) + dy
            # sign convention: image is shifted, so we reverse shift the template
            tmpl.data = nd_shift(
                tmpl._data_unshifted,
                (total_dy, total_dx),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            tmpl.shifted[:] = [total_dx, total_dy]
            tmpl.to_shift[:] = 0.0
            tmpl.flag |= Template.FLAG_SHIFTED  # mark as shifted


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
    def _psf_for_template(tmpl: Template, psf: np.ndarray | PSFRegionMap) -> np.ndarray:
        """Return the PSF array appropriate for ``tmpl``."""
        if isinstance(psf, PSFRegionMap):
            x, y = tmpl.input_position_original
            w_lookup = tmpl.wcs_original if tmpl.wcs_original is not None else None
            if w_lookup is not None:
                ra, dec = w_lookup.wcs_pix2world(x, y, 0)
            elif tmpl.wcs is not None:
                ra, dec = tmpl.wcs.wcs_pix2world(*tmpl.input_position_cutout, 0)
            else:
                ra, dec = x, y
            arr = psf.get_psf(ra, dec)
            if arr is None:
                raise ValueError(f"No PSF found at position ({ra}, {dec})")
        else:
            arr = psf
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or not np.any(np.isfinite(arr)):
            raise ValueError("PSF must be a finite 2-D array")
        return arr

    @staticmethod
    def _psf_shape_for_template(tmpl: Template, psf: np.ndarray | PSFRegionMap) -> tuple[np.ndarray, float]:
        """Return a unit-sum PSF shape and its native finite-stamp throughput.

        The fitting pipeline uses PSFs as morphology bases because extracted
        templates are normalized to unit sum. Keep the native finite-stamp sum
        only as throughput metadata; do not let it change the template shape.
        """
        arr = Templates._psf_for_template(tmpl, psf)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        total = float(arr.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("PSF has zero or non-finite finite-stamp throughput")
        return arr / total, total

    @staticmethod
    def _local_centroid(arr: np.ndarray, radius: int = 5) -> tuple[float, float]:
        """Centroid a compact PSF core, returning ``(x, y)`` pixel coordinates."""
        from photutils.centroids import centroid_com

        py, px = np.unravel_index(np.nanargmax(arr), arr.shape)
        y0, y1 = max(0, py - radius), min(arr.shape[0], py + radius + 1)
        x0, x1 = max(0, px - radius), min(arr.shape[1], px + radius + 1)
        cx_sub, cy_sub = centroid_com(arr[y0:y1, x0:x1])
        if np.isfinite(cx_sub) and np.isfinite(cy_sub):
            return float(x0 + cx_sub), float(y0 + cy_sub)
        return float(px), float(py)

    @staticmethod
    def _sample_psf_on_template(tmpl: Template, psf: np.ndarray) -> np.ndarray:
        """Sample ``psf`` onto ``tmpl`` without changing the template geometry."""
        from scipy.ndimage import map_coordinates

        psf = np.asarray(psf, dtype=float)
        finite = np.isfinite(psf)
        if not np.all(finite):
            psf = np.where(finite, psf, 0.0)
        cx, cy = Templates._local_centroid(psf)
        yy, xx = np.indices(tmpl.data.shape, dtype=float)
        coords = np.array(
            [cy + yy - float(tmpl.input_position_cutout[1]), cx + xx - float(tmpl.input_position_cutout[0])]
        )
        stamp = map_coordinates(psf, coords, order=3, mode="constant", cval=0.0)
        stamp = np.where(np.isfinite(stamp), stamp, 0.0)
        stamp[stamp < 0] = 0.0
        total = float(stamp.sum())
        if total > 0:
            stamp /= total
        return stamp

    @staticmethod
    def _resize_template(tmpl: Template, shape: tuple[int, int]) -> Template:
        """Return ``tmpl`` on a larger same-centre cutout, preserving metadata."""
        ny, nx = int(shape[0]), int(shape[1])
        if ny < tmpl.data.shape[0] or nx < tmpl.data.shape[1]:
            raise ValueError("resized template shape must contain the input template")
        if ny == tmpl.data.shape[0] and nx == tmpl.data.shape[1]:
            return deepcopy(tmpl)

        parent_image = np.zeros(tmpl.shape_input, dtype=tmpl.data.dtype)
        out = Template(
            parent_image,
            tmpl.input_position_original,
            (ny, nx),
            wcs=tmpl.wcs_original,
            label=tmpl.id,
            copy=False,
        )
        out.data = np.zeros(out.data.shape, dtype=tmpl.data.dtype)
        _copy_template_overlap(tmpl, out)
        out.input_position_original = tmpl.input_position_original
        out.position_original = tmpl.position_original
        out.id_parent = tmpl.id_parent
        out.id_scene = tmpl.id_scene
        out.deblend_parent_label = tmpl.deblend_parent_label
        out.deblend_nchildren = tmpl.deblend_nchildren
        out.name = tmpl.name
        out.flag = tmpl.flag
        out.flux = tmpl.flux
        out.err = tmpl.err
        out.err_pred = tmpl.err_pred
        out.wnorm = tmpl.wnorm
        out.to_shift = tmpl.to_shift.copy()
        out.shifted = tmpl.shifted.copy()
        return out

    def extend_with_psf_model(
        self,
        psf: np.ndarray | PSFRegionMap,
        *,
        gaussian_sigmas: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        target_shape: tuple[int, int] | None = None,
        mode: str = "wings",
        skip_deblended: bool = False,
        inplace: bool = False,
    ) -> list[Template]:
        """Complete missing template support outside the segment footprint.

        The extracted template pixels are treated as the authoritative source
        morphology inside the segmentation footprint.  Outside that footprint,
        this method adds the best-fitting PSF-convolved circular Gaussian wing
        model, scaled only from the segment pixels.  ``mode='model'`` replaces
        the full template by the best-fitting model; ``mode='wings'`` keeps the
        extracted segment pixels and fills only the missing support.
        """
        if mode not in {"wings", "model"}:
            raise ValueError("mode must be 'wings' or 'model'")
        if not self._templates:
            raise ValueError("No templates to extend. Run extract_templates first.")

        sigma_grid = tuple(float(s) for s in gaussian_sigmas)
        if not sigma_grid:
            raise ValueError("gaussian_sigmas must contain at least one value")
        if any(s < 0 for s in sigma_grid):
            raise ValueError("gaussian_sigmas must be non-negative")

        from scipy.ndimage import gaussian_filter

        completed: list[Template] = []
        for tmpl in tqdm(self._templates, desc="Extending templates"):
            if skip_deblended and tmpl.is_deblended:
                skipped = deepcopy(tmpl)
                skipped.extension_mode = "none"
                skipped.extension_skip_reason = "is_deblended"
                completed.append(skipped)
                continue

            psf_arr, psf_throughput = Templates._psf_shape_for_template(tmpl, psf)
            ny = max(tmpl.data.shape[0], psf_arr.shape[0])
            nx = max(tmpl.data.shape[1], psf_arr.shape[1])
            if target_shape is not None:
                ny = max(ny, int(target_shape[0]))
                nx = max(nx, int(target_shape[1]))
            if ny % 2:
                ny += 1
            if nx % 2:
                nx += 1
            expanded = Templates._resize_template(tmpl, (ny, nx))
            segment_mask = expanded.data != 0
            if not np.any(segment_mask):
                expanded.flag |= Template.FLAG_SUM_ZERO
                completed.append(expanded)
                continue

            data_seg = np.asarray(expanded.data[segment_mask], dtype=float)
            best_model = None
            best_scale = 0.0
            best_sigma = np.nan
            best_score = np.inf
            for sigma in sigma_grid:
                model_psf = psf_arr
                if sigma > 0:
                    model_psf = gaussian_filter(
                        psf_arr, sigma=sigma, mode="constant", cval=0.0, truncate=6.0
                    )
                model = Templates._sample_psf_on_template(expanded, model_psf)
                model_seg = model[segment_mask]
                denom = float(np.dot(model_seg, model_seg))
                if denom <= 0:
                    continue
                scale = float(np.dot(data_seg, model_seg) / denom)
                if scale < 0:
                    scale = 0.0
                resid = data_seg - scale * model_seg
                score = float(np.dot(resid, resid))
                if score < best_score:
                    best_score = score
                    best_scale = scale
                    best_sigma = sigma
                    best_model = model

            if best_model is None:
                completed.append(expanded)
                continue

            if mode == "model":
                data = best_scale * best_model
            else:
                data = np.asarray(expanded.data, dtype=float).copy()
                data[~segment_mask] = best_scale * best_model[~segment_mask]

            total = float(data.sum())
            if total != 0.0 and np.isfinite(total):
                data = data / total
            else:
                expanded.flag |= Template.FLAG_SUM_ZERO
            expanded.data = data.astype(tmpl.data.dtype, copy=False)
            expanded.extension_mode = mode
            expanded.extension_skip_reason = ""
            expanded.extension_sigma_pix = float(best_sigma)
            expanded.extension_score = float(best_score)
            expanded.extension_segment_fraction = float(np.sum(best_model[segment_mask]))
            expanded.extension_psf_throughput = float(psf_throughput)
            completed.append(expanded)

        if inplace:
            self._templates = completed
            return self._templates
        return completed

    def extend_with_psf_wings(
        self,
        psf: np.ndarray | PSFRegionMap,
        *,
        skip_deblended: bool = False,
        background_only: bool = True,
        inplace: bool = False,
    ) -> list[Template]:
        """Fill zero-valued template pixels with the local high-resolution PSF response.

        This method operates on already-extracted templates.  The template data
        are assumed to encode the segmentation ownership implicitly: nonzero
        pixels are trusted measured source pixels, and zero pixels are treated
        as outside the segment footprint.  The method convolves that sparse
        template with the local high-resolution PSF, fills only the zero-valued
        pixels with the convolved values, and normalizes the completed stamp to
        unit sum.

        With ``background_only=True`` (default) the fill is further restricted
        to background pixels of the segmentation map stored by
        :meth:`extract_templates`: pixels owned by a *different* segment keep
        their zero value, so blended neighbours model their own light there
        instead of receiving this source's extrapolated wings.  Pixels outside
        the original image footprint count as background.  If no segmap was
        recorded (e.g. prebuilt templates), all zero pixels are filled.

        PSFs are normalized to unit-sum shapes for this morphology operation.
        The native finite-stamp sum is stored as throughput metadata only; it
        must not change the relative amount of wing light inserted into a
        unit-normalized fitting template.
        """
        if not self._templates:
            raise ValueError("No templates to extend. Run extract_templates first.")

        completed: list[Template] = []
        for tmpl in tqdm(self._templates, desc="Completing PSF wings"):
            if skip_deblended and tmpl.is_deblended:
                skipped = deepcopy(tmpl)
                skipped.extension_mode = "none"
                skipped.extension_skip_reason = "is_deblended"
                completed.append(skipped)
                continue

            psf_arr, psf_throughput = Templates._psf_shape_for_template(tmpl, psf)

            parent_image = np.zeros(tmpl.shape_input, dtype=tmpl.data.dtype)
            smeared = tmpl.convolve_cutout(
                psf_arr,
                parent_image=parent_image,
                preserve_dtype=False,
            )

            core = deepcopy(smeared)
            core.data = np.zeros_like(smeared.data, dtype=float)
            _copy_template_overlap(tmpl, core)

            core_data = np.asarray(core.data, dtype=float)
            wing_data = np.asarray(smeared.data, dtype=float)
            outside_segment = core_data == 0.0

            fill = outside_segment
            if background_only and self.segmap is not None and tmpl.id is not None:
                seg = self.segmap[smeared.slices_original]
                allowed = np.ones_like(core_data, dtype=bool)
                allowed[smeared.slices_cutout] = (seg == 0) | (seg == int(tmpl.id))
                fill = outside_segment & allowed

            data = core_data.copy()
            data[fill] = wing_data[fill]

            total = float(data.sum())
            if total != 0.0 and np.isfinite(total):
                data = data / total
            else:
                smeared.flag |= Template.FLAG_SUM_ZERO

            smeared.data = data.astype(tmpl.data.dtype, copy=False)
            smeared.ee_tmpl = float(np.nansum(smeared.data))
            smeared.flag = tmpl.flag
            if total == 0.0 or not np.isfinite(total):
                smeared.flag |= Template.FLAG_SUM_ZERO
            smeared.id_parent = tmpl.id_parent
            smeared.id_scene = tmpl.id_scene
            smeared.deblend_parent_label = tmpl.deblend_parent_label
            smeared.deblend_nchildren = tmpl.deblend_nchildren
            smeared.name = tmpl.name
            smeared.flux = tmpl.flux
            smeared.err = tmpl.err
            smeared.err_pred = tmpl.err_pred
            smeared.wnorm = tmpl.wnorm
            smeared.to_shift = tmpl.to_shift.copy()
            smeared.shifted = tmpl.shifted.copy()
            smeared.extension_mode = "psf_wings"
            smeared.extension_skip_reason = ""
            smeared.extension_psf_sum = float(np.sum(psf_arr))
            smeared.extension_psf_throughput = float(psf_throughput)
            smeared.extension_core_sum = float(core_data.sum())
            smeared.extension_pre_norm_sum = total
            smeared.extension_filled_sum = float(wing_data[fill].sum())
            smeared.extension_filled_fraction = (
                float(wing_data[fill].sum() / total)
                if total != 0.0 and np.isfinite(total)
                else np.nan
            )
            smeared.extension_blocked_sum = float(wing_data[outside_segment & ~fill].sum())
            smeared.extension_sigma_pix = np.nan
            smeared.extension_score = np.nan

            completed.append(smeared)

        if inplace:
            self._templates = completed
            return self._templates
        return completed

    @staticmethod
    def predicted_errors(templates: List[Template], weights: np.ndarray) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance.

        Stores the prediction on ``tmpl.err_pred`` only.  ``tmpl.err`` is the
        solver error and must never be overwritten by a prediction; callers
        that want predicted values as pre-fit seeds use the returned array.
        """
        pred = np.empty(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            w = weights[tmpl.slices_original]
            inverse_epred = np.sqrt(np.sum(w * tmpl.data[tmpl.slices_cutout] ** 2))
            if inverse_epred > 0:
                pred[i] = 1.0 / inverse_epred
            else:  # @@@ need to debug why this happens should never have zero weight
                logger.debug(
                    f"error for template {i}: {inverse_epred} FLAG_SUM_ZERO {tmpl.flag & Template.FLAG_SUM_ZERO}"
                )
                tmpl.flag |= Template.FLAG_SUM_ZERO
                # np.empty left this slot uninitialized; zero weight means
                # the template carries no information, so the error is infinite
                pred[i] = np.inf

            tmpl.err_pred = pred[i]
        return pred

    def prune_outside_weight(self, weight: np.ndarray, rtol: float = 1e-8) -> List[Template]:
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
        norms = []
        for tmpl in self._templates:
            sl = tmpl.slices_original
            data = tmpl.data[tmpl.slices_cutout]
            w = weight[sl]
            wnorm = float(np.sum(data * w * data))
            tmpl.wnorm = wnorm
            norms.append(wnorm)

        atol = rtol * np.median(norms)
        keep = [t for t in self._templates if t.wnorm > atol]

        dropped = len(self._templates) - len(keep)
        if dropped:
            print(f"Pruned {dropped} templates with low L2 norm on weight map.")
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
        dilate_segmap: int = 2,
    ) -> list[Template]:
        """Extract cutout templates around segmentation regions.

        Parameters
        ----------
        dilate_segmap
            Disk radius (in pixels) used to dilate each segment *into
            background only* before cutting. The input segmap is usually a
            detection map (2-σ + small dilation) that captures only the
            bright core of a point source; template photometry then loses
            the PSF wings and biases the fit low. Dilating per-segment into
            background (no overlap with neighbors) recovers the wings.
            Pass 0 to disable.
        """

        self.original_shape = hires_image.shape
        segm = SegmentationImage(segmap)
        if dilate_segmap > 0:
            from .catalog import safe_dilate_segmentation
            from skimage.morphology import disk as _disk

            segm = SegmentationImage(safe_dilate_segmentation(segm, _disk(int(dilate_segmap))))
        self.segmap = np.asarray(segm.data)
        templates: list[Template] = []
        ny, nx = hires_image.shape

        for pos in tqdm(positions, desc="Extracting templates"):
            # silently skip invalid positions
            if not np.isfinite(pos).all():
                continue
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
            height = max(y - bbox.iymin, bbox.iymax - y, self.min_size // 2) * 2
            width = max(x - bbox.ixmin, bbox.ixmax - x, self.min_size // 2) * 2

            # Create template cutout
            cut = Template(hires_image, pos, (height, width), wcs=wcs, label=label)

            # zero out all non segment pixels
            cut.data[cut.slices_cutout] *= (segm.data[cut.slices_original] == label).astype(cut.data.dtype)

            # sum data should never be zero. There should
            # there should also never be NaNs.
            # Normalize the template so its sum is 1 (if nonzero)
            total = cut.data.sum()
            if total != 0:
                cut.data /= total
            else:
                cut.flag |= Template.FLAG_SUM_ZERO
            cut.ee_tmpl = float(cut.data.sum())

            templates.append(cut)

        self._templates = templates
        return templates

    def convolve_templates(
        self,
        kernel: np.ndarray | PSFRegionMap | None,
        inplace: bool = False,
        psf_lo: PSFRegionMap | None = None,
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
        psf_lo : PSFRegionMap, optional
            Low-resolution PSF map for the band being convolved to. When given,
            each output template records ``ee_psf_lo``, the encircled energy of
            that band's PSF stamp at the source position, which is the factor
            that converts the fitted amplitude to a total flux.

        Returns
        -------
        list of Template
            Convolved templates.
        """

        if not self._templates:
            raise ValueError("No templates to convolve. Run extract_templates first.")

        tmpls = self._templates
        original_shape = self.original_shape
        dummy_image = np.zeros(original_shape, dtype=np.byte)

        new_templates: list[Template] = []
        for i, tmpl in enumerate(tqdm(tmpls, desc="Convolving templates")):

            # Obtain kernel for this template
            ra = dec = None
            if isinstance(kernel, PSFRegionMap) or psf_lo is not None:
                x, y = tmpl.position_original
                w_lookup = tmpl.wcs_original if tmpl.wcs_original is not None else tmpl.wcs
                if w_lookup is not None:
                    ra, dec = w_lookup.wcs_pix2world(x, y, 0)
                else:
                    ra, dec = x, y
            kern = kernel.get_psf(ra, dec) if isinstance(kernel, PSFRegionMap) else kernel
            ee_lo = psf_lo.get_ee_box(ra, dec) if psf_lo is not None else np.nan

            if _is_identity_kernel(kern):
                new_tmpl = tmpl if inplace else deepcopy(tmpl)
                new_tmpl.ee_psf_lo = ee_lo
                if not inplace:
                    new_templates.append(new_tmpl)
                continue


            new_tmpl = tmpl.convolve_cutout(kern, parent_image=dummy_image)
            new_tmpl.ee_psf_lo = ee_lo

            if not inplace:
                new_templates.append(new_tmpl)

        return new_templates if not inplace else self._templates


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve and return an image-sized result using the shared convention."""
    return fftconvolve(image, kernel, mode="same")
