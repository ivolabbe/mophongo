"""
fft.py ─ fast FFT-based N-D convolution with optional cached FFTs
=================================================================

Highlights
----------
*   Same results and speed as `scipy.signal.fftconvolve`.
*   Extra keywords
        fft1=, fft2=        – pre-computed spectra on the working grid
        in1_shape=          – shape of the first operand (only needed if you
                              omit `in1` and still request mode='same'/'valid').
*   Uses `scipy.fft.next_fast_len` padding and pocketFFT/MKL back-end.
*   Real inputs automatically use rFFT/irFFT.

currently seems not worth it, as the speedup is not that great

Image size: (101, 101), kernel size: (101, 101)
method                                      time [s] per second
------------------------------------------------------------
direct (_convolve2d)                         0.01470  68
mophongo.fftconvolve (normal)                0.00030  3354
mophongo.fftconvolve (cached fft1, fft2)     0.00016  6312
scipy.ndimage.convolve                       0.63587  2
scipy.signal.convolve2d                      0.08689  12
scipy.signal.fftconvolve                     0.00029  3405
scipy.signal.oaconvolve                      0.00029  3410
astropy.convolution.convolve                 0.12900  8
astropy.convolution.convolve_fft             0.00192  520

Image size: (31, 31), kernel size: (101, 101)
method                                      time [s] per second
------------------------------------------------------------
direct (_convolve2d)                         0.00164  609
mophongo.fftconvolve (normal)                0.00014  7306
mophongo.fftconvolve (cached fft1, fft2)     0.00012  8039
scipy.ndimage.convolve                       0.04872  21
scipy.signal.convolve2d                      0.00585  171
scipy.signal.fftconvolve                     0.00014  7177
scipy.signal.oaconvolve                      0.00014  7029
astropy.convolution.convolve                 0.01248  80
astropy.convolution.convolve_fft             0.00097  1033

Example
-------
>>> from scipy import fft
>>> import numpy as np
>>> from fft import fftconvolve
>>>
>>> img   = np.random.randn(512, 512)
>>> kern  = np.random.randn(64,  64)
>>> shape = tuple(fft.next_fast_len(n + k - 1)
...               for n, k in zip(img.shape, kern.shape))
>>>
>>> Fimg  = fft.rfftn(img,  shape, axes=(0, 1))
>>> Fker  = fft.rfftn(kern, shape, axes=(0, 1))
>>>
>>> blur  = fftconvolve(None, None,
...                     fft1=Fimg, fft2=Fker,
...                     in1_shape=img.shape,
...                     mode="same", axes=(0, 1))
"""
from __future__ import annotations
import numpy as _np
from scipy import fft as _spfft
from scipy.signal import fftconvolve as _sc_fftconv  # for the fast fall-back

__all__ = ["fftconvolve"]

_next_fast_len = _spfft.next_fast_len


# ───────────────────────── helper utilities ──────────────────────────
def _as_axes(ndim: int, axes) -> tuple[int, ...]:
    """Normalize `axes` to a positive, sorted tuple."""
    if axes is None:
        return tuple(range(ndim))
    if _np.isscalar(axes):
        axes = (axes,)
    return tuple(a + ndim if a < 0 else a for a in axes)


def _pad_shape(s1, s2, axes):
    """Return padded FFT shape (next_fast_len along `axes`)."""
    shape = list(_np.maximum(s1, s2))
    for ax in axes:
        shape[ax] = s1[ax] + s2[ax] - 1
    return tuple(_next_fast_len(n) for n in shape)


def _centered(arr: _np.ndarray, newshape):
    newshape = _np.asarray(newshape)
    currshape = _np.array(arr.shape)
    start = (currshape - newshape) // 2
    end = start + newshape
    return arr[tuple(slice(s, e) for s, e in zip(start, end))]


# ────────────────────────── public routine ───────────────────────────
def fftconvolve(
    in1: _np.ndarray | None,
    in2: _np.ndarray | None,
    *,
    fft1: _np.ndarray | None = None,
    fft2: _np.ndarray | None = None,
    mode: str = "full",
    axes=None,
    in1_shape: tuple[int, ...] | None = None,
):
    """
    Fast N-D convolution; accepts cached FFTs.

    Parameters
    ----------
    in1, in2 : array_like or None
        Spatial operands.  For each operand you may supply either the array
        itself **or** its FFT via `fft1` / `fft2`.
    fft1, fft2 : ndarray or None
        Spectra already padded to the working grid.  Real spectra must be the
        output of `rfftn`.
    mode : {'full', 'same', 'valid'}
    axes : int or sequence of int or None
    in1_shape : tuple or None
        Shape of the first operand.  Required only if you omit `in1` *and*
        request `mode='same'` or `'valid'`.

    Returns
    -------
    ndarray
        Convolution result sliced according to *mode*.
    """
    # ---------- fast path: no cached spectra ---------------------------------
    if fft1 is None and fft2 is None:
        return _sc_fftconv(in1, in2, mode=mode, axes=axes)

    # ---------- need reference ndim / axes -----------------------------------
    ref = fft1 if fft1 is not None else fft2 if fft2 is not None else in1
    ndim = ref.ndim
    axes = _as_axes(ndim, axes)

    # ---------- determine padded working shape ------------------------------
    if fft1 is None or fft2 is None:
        s1 = fft1.shape if fft1 is not None else in1.shape
        s2 = fft2.shape if fft2 is not None else in2.shape
        fshape = _pad_shape(s1, s2, axes)
    else:
        fshape = tuple(fft1.shape)

    # ---------- choose rFFT vs cFFT -----------------------------------------
    # complex_result = ((fft1 is not None and _np.iscomplexobj(fft1))
    #                   or (fft2 is not None and _np.iscomplexobj(fft2))
    #                   or (in1 is not None and _np.iscomplexobj(in1))
    #                   or (in2 is not None and _np.iscomplexobj(in2)))
    #
    complex_result = False
    fftn = _spfft.fftn if complex_result else _spfft.rfftn
    ifftn = _spfft.ifftn if complex_result else _spfft.irfftn

    # ---------- forward transforms (only if needed) -------------------------
    if fft1 is None:
        fft1 = fftn(_np.asarray(in1), fshape, axes=axes, workers=-1)
    if fft2 is None:
        fft2 = fftn(_np.asarray(in2), fshape, axes=axes, workers=-1)

    # ---------- multiply in frequency space & inverse FFT -------------------
    conv = ifftn(fft1 * fft2, fshape, axes=axes, workers=-1)

    # ---------- slice according to mode -------------------------------------
    if mode == "full":
        return conv

    # (s1 is needed for cropping)
    if in1 is not None:
        s1 = in1.shape
    elif in1_shape is not None:
        s1 = in1_shape
    else:
        raise ValueError("For mode 'same' or 'valid', pass `in1` or "
                         "`in1_shape=` so cropping is defined.")

    if mode == "same":
        return _centered(conv, s1)

    if mode == "valid":
        s2 = in2.shape if in2 is not None else fft2.shape
        valid_shape = [
            conv.shape[i] if i not in axes else s1[i] - s2[i] + 1
            for i in range(ndim)
        ]
        return _centered(conv, valid_shape)

    raise ValueError("mode must be 'full', 'same', or 'valid'.")
