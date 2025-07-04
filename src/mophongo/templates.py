from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from astropy.nddata import Cutout2D
from photutils.segmentation import SegmentationImage


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    pad_before = (pad_y, pad_x)
    pad_after = (ky - 1 - pad_y, kx - 1 - pad_x)
    padded = np.pad(image, (pad_before, pad_after), mode="constant")
    # Use sliding windows for convolution
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)


@dataclass
class Template:
    array: np.ndarray
    bbox: Tuple[int, int, int, int]

class Templates:
    """Container for PSF-matched source templates."""

    def __init__(self) -> None:
        self._templates: List[Template] = []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._templates)

    def __getitem__(self, idx: int) -> Template:  # pragma: no cover - trivial
        return self._templates[idx]

    def __iter__(self) -> Iterator[Template]:  # pragma: no cover - trivial
        return iter(self._templates)

    @classmethod
    def from_image(
        cls,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        kernel: np.ndarray,
    ) -> "Templates":
        """Create a :class:`Templates` instance and extract cutouts."""
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

        self._templates = []
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
            self._templates.append(Template(conv, (y0_ext, y1_ext, x0_ext, x1_ext)))

        return self._templates