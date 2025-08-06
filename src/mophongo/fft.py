"""
fft.py ― minimal FFT-convolution helper.

Features
--------
* Same public signature as SciPy’s `fftconvolve`, plus
  optional `fft1`, `fft2` keyword arguments for pre-computed
  complex spectra.
* Works on arbitrary‐dimensional arrays and any subset of axes.
* Pads to the next “fast” FFT length (uses SciPy’s
  `next_fast_len` if available, else falls back to powers of two).
* Chooses r2c / c2c transforms automatically.

Usage
-----
>>> from fft import fftconvolve
>>> out = fftconvolve(img, ker, mode="same")                         # normal
>>> F1  = np.fft.rfftn(img, s=shape, axes=axes)                      # cache
>>> out = fftconvolve(None, ker, fft1=F1, mode="same", axes=axes)    # reuse
"""
from __future__ import annotations
import numpy as _np
from numpy.fft import fftn as _fftn, ifftn as _ifftn, rfftn as _rfftn, irfftn as _irfftn

try:                # SciPy gives faster padding lengths
    from scipy.fft import next_fast_len as _NFL
except ModuleNotFoundError:
    _NFL = lambda n: 1 << (n - 1).bit_length()          # power-of-two fallback


# --------------------------------------------------------------------
def _as_axes(na: int, axes) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(na))
    axes = (axes,) if _np.isscalar(axes) else tuple(axes)
    return tuple(a if a >= 0 else na + a for a in axes)


def _pad_shape(s1, s2, axes):
    shp = list(_np.maximum(s1, s2))
    for ax in axes:
        shp[ax] = s1[ax] + s2[ax] - 1
    return tuple(_NFL(n) for n in shp)


# --------------------------------------------------------------------
def fftconvolve(
    in1: _np.ndarray | None,
    in2: _np.ndarray | None,
    *,
    fft1: _np.ndarray | None = None,
    fft2: _np.ndarray | None = None,
    mode: str = "full",
    axes: tuple[int, ...] | int | None = None,
):
    """
    Fast N-D convolution with optional cached FFTs.

    Any of `in1`, `in2` may be `None` **iff** you supply the
    corresponding `fft1`, `fft2` already transformed to the working
    shape along `axes`.
    """
    if (in1 is None) == (fft1 is None) is False or (in2 is None) == (fft2 is None) is False:
        raise ValueError("Supply either spatial input OR its FFT, for each operand.")

    # --- establish axes & dtype -------------------------------------------------
    if fft1 is not None:
        axes = tuple(range(fft1.ndim)) if axes is None else _as_axes(fft1.ndim, axes)
        work_dtype = fft1.dtype
    elif fft2 is not None:
        axes = tuple(range(fft2.ndim)) if axes is None else _as_axes(fft2.ndim, axes)
        work_dtype = fft2.dtype
    else:
        in1 = _np.asarray(in1)
        in2 = _np.asarray(in2)
        axes = _as_axes(in1.ndim, axes)
        work_dtype = _np.result_type(in1.dtype, in2.dtype, _np.float32)

    # --- shapes -----------------------------------------------------------------
    if fft1 is None or fft2 is None:
        shape = _pad_shape(in1.shape if in1 is not None else fft1.shape,
                           in2.shape if in2 is not None else fft2.shape,
                           axes)
    else:
        shape = tuple(fft1.shape)

    # --- transform helpers ------------------------------------------------------
    is_complex = _np.issubdtype(work_dtype, _np.complexfloating)

    def _fwd(x):
        return _fftn(x, shape, axes=axes) if is_complex else _rfftn(x, shape, axes=axes)

    def _inv(X):
        out = _ifftn(X, shape, axes=axes) if is_complex else _irfftn(X, shape, axes=axes)
        return out

    # --- obtain spectra ---------------------------------------------------------
    if fft1 is None:
        fft1 = _fwd(in1.astype(work_dtype, copy=False))
    if fft2 is None:
        fft2 = _fwd(in2.astype(work_dtype, copy=False))

    # --- multiply & back-transform ---------------------------------------------
    conv = _inv(fft1 * fft2)

    # --- slice according to mode ------------------------------------------------
    if mode == "full":
        return conv
    elif mode == "same":
        idx = tuple(slice((s - o) // 2, (s - o) // 2 + o)
                    for s, o in zip(conv.shape, in1.shape if in1 is not None else fft1.shape))
        return conv[idx]
    elif mode == "valid":
        out_shape = [(o1 - o2 + 1) if ax in axes else o1
                     for ax, (o1, o2) in enumerate(zip(in1.shape if in1 is not None else fft1.shape,
                                                        in2.shape if in2 is not None else fft2.shape))]
        start = [(conv.shape[k] - out_shape[k]) // 2 for k in range(conv.ndim)]
        idx = tuple(slice(st, st + sz) for st, sz in zip(start, out_shape))
        return conv[idx]
    else:
        raise ValueError("mode must be 'full', 'same' or 'valid'")
