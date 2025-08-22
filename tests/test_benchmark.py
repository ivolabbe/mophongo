# %%
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mophongo.psf import PSF
from mophongo.templates import Templates
from mophongo.fit import SparseFitter
from utils import make_simple_data
import pytest


def test_benchmark_pipeline_steps():
    images, segmap, catalog, psfs, _, _ = make_simple_data(nsrc=20, size=101, ndilate=2)
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    start = time.perf_counter()
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)
    extract_time = time.perf_counter() - start

    start = time.perf_counter()
    fitter = SparseFitter(tmpls.templates, images[1])
    fitter.solve()
    fit_time = time.perf_counter() - start

    print(f"Extraction time: {extract_time:.4f} s")
    #    print(f"Extension time: {extend_time:.4f} s")
    print(f"Fitting time: {fit_time:.4f} s")

    assert extract_time > 0
    #    assert extend_time > 0
    assert fit_time > 0


# pure‐NumPy direct sliding‐window
def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    padded = np.pad(image, ((pad_y, ky - 1 - pad_y), (pad_x, kx - 1 - pad_x)), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, kernel.shape)
    # windows shape = (H, W, ky, kx)
    return np.einsum("ijkl,kl->ij", windows, kernel)


# scipy / astropy imports
from scipy.ndimage import convolve as nd_convolve
from scipy.signal import convolve2d, fftconvolve, oaconvolve
from astropy.convolution import convolve as astro_convolve, convolve_fft
from scipy import fft as spfft


def test_benchmark_convolution():
    def run_benchmark(image, kernel, niter=5):
        # Precompute FFTs for caching test
        from numpy.fft import rfftn

        shape = [spfft.next_fast_len(s1 + s2 - 1) for s1, s2 in zip(image.shape, kernel.shape)]
        fft1 = rfftn(image, shape)
        fft2 = rfftn(kernel, shape)

        methods = {
            "direct (_convolve2d)": lambda: _convolve2d(image, kernel),
            "mophongo.fftconvolve (normal)": lambda: mophongo_fftconvolve(
                image, kernel, mode="same"
            ),
            #            "mophongo.fftconvolve (cached fft1, fft2)": lambda: mophongo_fftconvolve(
            #                None, None, fft1=fft1, fft2=fft2, mode="same", in1_shape=image.shape
            #            ),
            #    "scipy.ndimage.convolve"     : lambda: nd_convolve(image, kernel, mode="constant", cval=0.0),
            "scipy.signal.convolve2d": lambda: convolve2d(
                image, kernel, mode="same", boundary="fill", fillvalue=0.0
            ),
            "scipy.signal.fftconvolve": lambda: fftconvolve(image, kernel, mode="same"),
            "scipy.signal.oaconvolve": lambda: oaconvolve(image, kernel, mode="same"),
            #    "astropy.convolution.convolve": lambda: astro_convolve(image, kernel, boundary="fill", fill_value=0.0, normalize_kernel=False),
            "astropy.convolution.convolve_fft": lambda: convolve_fft(
                image, kernel, normalize_kernel=False, boundary="fill", fill_value=0.0
            ),
        }

        print(f"{'method':40s}  {'time [s]':>10s} {'per second':>10s}")
        print("-" * 60)
        for name, func in methods.items():
            # warm up
            func()
            t0 = time.perf_counter()
            for _ in range(niter):
                out = func()
            dt = (time.perf_counter() - t0) / niter
            print(f"{name:40s}  {dt:10.5f}  {1/dt:.0f}")

    IMG_SIZE = 11
    KERNEL_SIZE = 31
    np.random.seed(0)
    image = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
    x = np.linspace(-1, 1, KERNEL_SIZE)
    y = x[:, None]
    gauss = np.exp(-(x[None, :] ** 2 + y**2) / (2 * (0.2**2)))
    kernel = (gauss / gauss.sum()).astype(np.float32)

    print(f"Image size: {image.shape}, kernel size: {kernel.shape}")
    run_benchmark(image, kernel, niter=50)


# %%
import os, time, platform, subprocess
import numpy as np


try:
    import pandas as pd
except Exception:
    pd = None

# --------------------------- utils / I/O --------------------------------------


def gib(nbytes):
    return nbytes / (1024**3)


def make_or_load_memmap(
    path="parent_1gib_float32.npy", shape=(16384, 16384), dtype=np.float32, seed=0
):
    """Create (~1 GiB) if missing, else open read-only memmap."""
    H, W = shape
    nbytes = np.dtype(dtype).itemsize * H * W
    if not os.path.exists(path):
        print(f"[init] creating {path} (shape={shape}, dtype={dtype}, ~{gib(nbytes):.2f} GiB)")
        mm = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        rng = np.random.default_rng(seed)
        slab = 1024
        for y0 in range(0, H, slab):
            y1 = min(H, y0 + slab)
            mm[y0:y1, :] = rng.random((y1 - y0, W), dtype=dtype)
        mm.flush()
        del mm
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def load_to_ram(arr_or_path, shape=None, dtype=None):
    """Return a C-contiguous RAM ndarray from memmap or path."""
    if isinstance(arr_or_path, np.memmap):
        return np.array(arr_or_path, copy=True, order="C")
    if isinstance(arr_or_path, str):
        assert shape is not None and dtype is not None
        mm = np.memmap(arr_or_path, mode="r", dtype=dtype, shape=shape)
        return np.array(mm, copy=True, order="C")
    return np.ascontiguousarray(arr_or_path)


def attempt_drop_os_cache(verbose=True):
    """
    Best-effort OS page cache drop. Requires privileges.
    Linux:  sync; echo 3 > /proc/sys/vm/drop_caches
    macOS:  purge
    Returns True if command reported success.
    """
    sys = platform.system().lower()
    try:
        if "linux" in sys:
            # try non-interactive sudo; if not permitted, will fail quickly
            cmd = ["bash", "-lc", "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'"]
        elif "darwin" in sys:
            cmd = ["bash", "-lc", "sync; sudo -n purge"]
        else:
            if verbose:
                print("[cold-cache] unsupported platform for auto cache drop.")
            return False
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ok = r.returncode == 0
        if verbose:
            print(
                "[cold-cache]",
                "success" if ok else f"failed: {r.stderr.strip() or 'non-zero exit'}",
            )
        return ok
    except Exception as e:
        if verbose:
            print("[cold-cache] exception:", e)
        return False


# --------------------------- index helpers ------------------------------------


def random_positions(H, W, h, w, n, seed=0):
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, H - h + 1, size=n, dtype=np.int64)
    xs = rng.integers(0, W - w + 1, size=n, dtype=np.int64)
    return ys, xs


def random_positions_var(H, W, hs, ws, seed=0):
    """
    Per-tile positions for variable sizes, ensuring tiles fit in-bounds.
    hs, ws: int arrays, same length M.
    """
    rng = np.random.default_rng(seed)
    hs = np.asarray(hs, dtype=np.int64)
    ws = np.asarray(ws, dtype=np.int64)
    M = hs.size
    ys = (rng.random(M) * (H - hs)).astype(np.int64)
    xs = (rng.random(M) * (W - ws)).astype(np.int64)
    return ys, xs


# --------------------------- bytes accounting ---------------------------------


def _bytes_moved_fixed(n_tiles, h, w, itemsize):
    # read + write tiles
    return 2 * n_tiles * h * w * itemsize


def _bytes_moved_var(hs, ws, itemsize):
    # read + write tiles
    return 2 * int(np.sum(hs.astype(np.int64) * ws.astype(np.int64))) * itemsize


# --------------------------- fixed-size benchmarks ----------------------------


def bench_loop_copy(arr, ys, xs, h, w):
    out = np.empty((len(ys), h, w), dtype=arr.dtype)
    t0 = time.perf_counter()
    for i, (y0, x0) in enumerate(zip(ys, xs)):
        for r in range(h):
            out[i, r, :] = arr[y0 + r, x0 : x0 + w]
    t = time.perf_counter() - t0
    return {
        "method": "loop_copy",
        "sec": t,
        "GiBps": (_bytes_moved_fixed(len(ys), h, w, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
    }


def bench_gather_chunked(arr, ys, xs, h, w, batch=4096):
    H, W = arr.shape
    flat = arr.ravel()
    ro = (np.arange(h, dtype=np.intp) * W)[None, :, None]
    co = np.arange(w, dtype=np.intp)[None, None, :]
    out = np.empty((len(ys), h, w), dtype=arr.dtype)
    t0 = time.perf_counter()
    for i0 in range(0, len(ys), batch):
        i1 = min(len(ys), i0 + batch)
        base = (ys[i0:i1].astype(np.intp) * W + xs[i0:i1].astype(np.intp))[:, None, None]
        idx = base + ro + co
        out[i0:i1] = flat[idx]
    t = time.perf_counter() - t0
    return {
        "method": "gather_chunked",
        "sec": t,
        "GiBps": (_bytes_moved_fixed(len(ys), h, w, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
    }


def retile_blocked(arr, B=64):
    """Return a blocked copy with shape (nby, nbx, B, B), plus timings."""
    H, W = arr.shape
    HH, WW = (H // B) * B, (W // B) * B
    t0 = time.perf_counter()
    blocked = arr[:HH, :WW].reshape(HH // B, B, WW // B, B).swapaxes(1, 2).copy()
    t = time.perf_counter() - t0
    bytes_moved = blocked.nbytes * 2  # read + write
    return blocked, HH, WW, t, bytes_moved


def bench_blocked_aligned(arr, ys, xs, h, w, B=64):
    """Upper bound for blocked layout: only tiles already aligned to block grid."""
    blocked, HH, WW, tprep, prep_bytes = retile_blocked(arr, B)
    aligned = (
        ((ys % B) == 0) & ((xs % B) == 0) & (ys + h <= HH) & (xs + w <= WW) & (h <= B) & (w <= B)
    )
    ys_al = ys[aligned]
    xs_al = xs[aligned]
    out = np.empty((len(ys_al), h, w), dtype=arr.dtype)
    t0 = time.perf_counter()
    for i, (y0, x0) in enumerate(zip(ys_al, xs_al)):
        by, bx = y0 // B, x0 // B
        out[i] = blocked[by, bx, :h, :w]  # oy=ox=0 by construction
    t = time.perf_counter() - t0
    used = int(aligned.sum())
    return {
        "method": "blocked_aligned",
        "sec": t,
        "GiBps": (
            ((_bytes_moved_fixed(used, h, w, arr.itemsize)) / t) / (1024**3) if used else np.nan
        ),
        "sig": float(out.sum(dtype=np.float64)) if used else np.nan,
        "prep_sec": tprep,
        "prep_GiBps": (prep_bytes / tprep) / (1024**3),
        "aligned_frac": used / len(ys),
    }


def bench_blocked_stitched(arr, ys, xs, h, w, B=64):
    """Python (non-Numba) stitched blocked fetch (<=4 blocks)."""
    blocked, HH, WW, tprep, prep_bytes = retile_blocked(arr, B)
    use = (ys + h <= HH) & (xs + w <= WW)
    ys2 = ys[use]
    xs2 = xs[use]
    out = np.empty((len(ys2), h, w), dtype=arr.dtype)
    t0 = time.perf_counter()
    for i, (y0, x0) in enumerate(zip(ys2, xs2)):
        by0, bx0 = y0 // B, x0 // B
        oy, ox = y0 % B, x0 % B
        h0 = min(h, B - oy)
        w0 = min(w, B - ox)
        out[i, :h0, :w0] = blocked[by0, bx0, oy : oy + h0, ox : ox + w0]
        if w0 < w:
            out[i, :h0, w0:w] = blocked[by0, bx0 + 1, oy : oy + h0, 0 : w - w0]
        if h0 < h:
            out[i, h0:h, :w0] = blocked[by0 + 1, bx0, 0 : h - h0, ox : ox + w0]
            if w0 < w:
                out[i, h0:h, w0:w] = blocked[by0 + 1, bx0 + 1, 0 : h - h0, 0 : w - w0]
    t = time.perf_counter() - t0
    used = int(use.sum())
    return {
        "method": "blocked_stitched",
        "sec": t,
        "GiBps": ((_bytes_moved_fixed(used, h, w, arr.itemsize)) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
        "prep_sec": tprep,
        "prep_GiBps": (prep_bytes / tprep) / (1024**3),
        "aligned_frac": used / len(ys),
    }


# --------------------------- Numba (fixed) ------------------------------------

HAVE_NUMBA = False
try:
    from numba import njit, prange, set_num_threads, get_num_threads

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _numba_copy_fixed(arr, ys, xs, out):
        M, h, w = out.shape
        for i in prange(M):
            y0 = ys[i]
            x0 = xs[i]
            for r in range(h):
                out[i, r, :] = arr[y0 + r, x0 : x0 + w]


def bench_numba_copy(arr, ys, xs, h, w):
    if not HAVE_NUMBA:
        return None
    out = np.empty((len(ys), h, w), dtype=arr.dtype)
    _numba_copy_fixed(arr, ys[:1], xs[:1], out[:1])  # compile
    t0 = time.perf_counter()
    _numba_copy_fixed(arr, ys, xs, out)
    t = time.perf_counter() - t0
    return {
        "method": "numba_copy",
        "sec": t,
        "GiBps": (_bytes_moved_fixed(len(ys), h, w, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
    }


# --------------------------- VARIABLE-SIZE paths ------------------------------


def bench_loop_copy_var(arr, ys, xs, hs, ws, Hpad, Wpad, padval=0.0):
    M = len(ys)
    out = np.full((M, Hpad, Wpad), padval, dtype=arr.dtype)
    t0 = time.perf_counter()
    for i in range(M):
        y0, x0, h, w = int(ys[i]), int(xs[i]), int(hs[i]), int(ws[i])
        for r in range(h):
            out[i, r, :w] = arr[y0 + r, x0 : x0 + w]
    t = time.perf_counter() - t0
    return {
        "method": "loop_copy_var",
        "sec": t,
        "GiBps": (_bytes_moved_var(hs, ws, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
    }


def bench_gather_var_grouped(arr, ys, xs, hs, ws, Hpad, Wpad, padval=0.0):
    """
    Group by (h,w) and do advanced-index gathers per group; pad into (Hpad,Wpad).
    """
    H, W = arr.shape
    flat = arr.ravel()
    M = len(ys)
    out = np.full((M, Hpad, Wpad), padval, dtype=arr.dtype)
    sizes = np.stack([hs, ws], axis=1)
    uniq, inv = np.unique(sizes, axis=0, return_inverse=True)
    t0 = time.perf_counter()
    for k, (h, w) in enumerate(uniq):
        idxs = np.nonzero(inv == k)[0]
        if idxs.size == 0:
            continue
        ro = (np.arange(h, dtype=np.intp) * W)[None, :, None]
        co = np.arange(w, dtype=np.intp)[None, None, :]
        base = (ys[idxs].astype(np.intp) * W + xs[idxs].astype(np.intp))[:, None, None]
        idx = base + ro + co  # (B,h,w)
        tiles = flat[idx]
        out[idxs, :h, :w] = tiles
    t = time.perf_counter() - t0
    return {
        "method": "gather_var",
        "sec": t,
        "GiBps": (_bytes_moved_var(hs, ws, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
    }


if HAVE_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _numba_copy_var(arr, ys, xs, hs, ws, out):
        M, Hpad, Wpad = out.shape
        for i in prange(M):
            y0 = ys[i]
            x0 = xs[i]
            h = hs[i]
            w = ws[i]
            for r in range(h):
                # copy each row
                out[i, r, :w] = arr[y0 + r, x0 : x0 + w]


def bench_numba_copy_var(arr, ys, xs, hs, ws, Hpad, Wpad, padval=0.0):
    if not HAVE_NUMBA:
        return None
    out = np.full((len(ys), Hpad, Wpad), padval, dtype=arr.dtype)
    _numba_copy_var(arr, ys[:1], xs[:1], hs[:1], ws[:1], out[:1])  # compile
    t0 = time.perf_counter()
    #    _numba_copy_var(arr, ys, xs, hs, ws, out)
    #    for i in range(len(ys)):
    #        _numba_copy_var(arr, ys[i], xs[i], hs[i], ws[i], out)
    t = time.perf_counter() - t0
    return {
        "method": "numba_copy_var",
        "sec": t,
        "GiBps": (_bytes_moved_var(hs, ws, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
    }


# --------- Numba blocked-stitched (variable-size, up to 4 blocks) -------------

if HAVE_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _numba_blocked_stitched_var(blocked, B, ys, xs, hs, ws, out):
        """
        blocked: (nby, nbx, B, B), contiguous
        out: (M, Hpad, Wpad) prefilled with padval
        Copies each tile (h,w) from up to 4 blocks into out[i, :h, :w].
        """
        M = ys.shape[0]
        nby = blocked.shape[0]
        nbx = blocked.shape[1]
        for i in prange(M):
            y0 = ys[i]
            x0 = xs[i]
            h = hs[i]
            w = ws[i]
            by0 = y0 // B
            bx0 = x0 // B
            oy = y0 - by0 * B
            ox = x0 - bx0 * B
            # top-left chunk
            h0 = h if (oy + h) <= B else (B - oy)
            w0 = w if (ox + w) <= B else (B - ox)
            for r in range(h0):
                out[i, r, 0:w0] = blocked[by0, bx0, oy + r, ox : ox + w0]
            # top-right
            if w0 < w:
                bx1 = bx0 + 1
                w1 = w - w0
                for r in range(h0):
                    out[i, r, w0:w] = blocked[by0, bx1, oy + r, 0:w1]
            # bottom-left
            if h0 < h:
                by1 = by0 + 1
                h1 = h - h0
                for r in range(h1):
                    out[i, h0 + r, 0:w0] = blocked[by1, bx0, r, ox : ox + w0]
                # bottom-right
                if w0 < w:
                    bx1 = bx0 + 1
                    w1 = w - w0
                    for r in range(h1):
                        out[i, h0 + r, w0:w] = blocked[by1, bx1, r, 0:w1]


def bench_numba_blocked_stitched_var(arr, ys, xs, hs, ws, Hpad, Wpad, B=64, padval=0.0):
    if not HAVE_NUMBA:
        return None
    blocked, HH, WW, tprep, prep_bytes = retile_blocked(arr, B)
    use = (ys + hs <= HH) & (xs + ws <= WW)
    ys2, xs2, hs2, ws2 = ys[use], xs[use], hs[use], ws[use]
    out = np.full((len(ys2), Hpad, Wpad), padval, dtype=arr.dtype)
    # warm-up compile
    if len(ys2) == 0:
        return {
            "method": "numba_blocked_var",
            "sec": np.nan,
            "GiBps": np.nan,
            "sig": np.nan,
            "prep_sec": tprep,
            "prep_GiBps": (prep_bytes / tprep) / (1024**3),
            "aligned_frac": 0.0,
        }
    _numba_blocked_stitched_var(blocked, B, ys2[:1], xs2[:1], hs2[:1], ws2[:1], out[:1])
    t0 = time.perf_counter()
    _numba_blocked_stitched_var(blocked, B, ys2, xs2, hs2, ws2, out)
    t = time.perf_counter() - t0
    return {
        "method": "numba_blocked_var",
        "sec": t,
        "GiBps": (_bytes_moved_var(hs2, ws2, arr.itemsize) / t) / (1024**3),
        "sig": float(out.sum(dtype=np.float64)),
        "prep_sec": tprep,
        "prep_GiBps": (prep_bytes / tprep) / (1024**3),
        "aligned_frac": len(ys2) / len(ys),
    }


# --------------------------- runners / summaries ------------------------------


def run_bench_fixed(
    arr,
    ys,
    xs,
    h=32,
    w=32,
    methods=("loop_copy", "gather_chunked", "numba_copy", "blocked_aligned", "blocked_stitched"),
    batch=4096,
    block=64,
    check=True,
):
    res = []
    if "loop_copy" in methods:
        res.append(bench_loop_copy(arr, ys, xs, h, w))
    if "gather_chunked" in methods:
        res.append(bench_gather_chunked(arr, ys, xs, h, w, batch=batch))
    if "numba_copy" in methods:
        r = bench_numba_copy(arr, ys, xs, h, w)
        if r is not None:
            res.append(r)
    if "blocked_aligned" in methods:
        res.append(bench_blocked_aligned(arr, ys, xs, h, w, B=block))
    if "blocked_stitched" in methods:
        res.append(bench_blocked_stitched(arr, ys, xs, h, w, B=block))

    if check:
        native = [r for r in res if r["method"] in {"loop_copy", "gather_chunked", "numba_copy"}]
        if len(native) >= 2:
            s0 = native[0]["sig"]
            for r in native[1:]:
                if not np.allclose(r["sig"], s0, rtol=1e-5, atol=1e-6):
                    print(
                        f"[warn] signature mismatch: {native[0]['method']} vs {r['method']} -> {s0} vs {r['sig']}"
                    )
    return pd.DataFrame(res) if pd is not None else res


def run_bench_var(
    arr,
    ys,
    xs,
    hs,
    ws,
    Hpad,
    Wpad,
    methods=("loop_copy_var", "gather_var", "numba_copy_var", "numba_blocked_var"),
    block=64,
    padval=0.0,
    check=True,
):
    res = []
    if "loop_copy_var" in methods:
        res.append(bench_loop_copy_var(arr, ys, xs, hs, ws, Hpad, Wpad, padval=padval))
    if "gather_var" in methods:
        res.append(bench_gather_var_grouped(arr, ys, xs, hs, ws, Hpad, Wpad, padval=padval))
    if "numba_copy_var" in methods:
        r = bench_numba_copy_var(arr, ys, xs, hs, ws, Hpad, Wpad, padval=padval)
        if r is not None:
            res.append(r)
    if "numba_blocked_var" in methods:
        r = bench_numba_blocked_stitched_var(
            arr, ys, xs, hs, ws, Hpad, Wpad, B=block, padval=padval
        )
        if r is not None:
            res.append(r)

    if check:
        native = [
            r for r in res if r["method"] in {"loop_copy_var", "gather_var", "numba_copy_var"}
        ]
        if len(native) >= 2:
            s0 = native[0]["sig"]
            for r in native[1:]:
                if not np.allclose(r["sig"], s0, rtol=1e-5, atol=1e-6):
                    print(
                        f"[warn] signature mismatch: {native[0]['method']} vs {r['method']} -> {s0} vs {r['sig']}"
                    )
    return pd.DataFrame(res) if pd is not None else res


# =============================== EXAMPLES =====================================


def run_tile_benchmarks():
    # ---- Build/open ~1 GiB memmap & RAM copy
    mm = make_or_load_memmap("parent_1gib_float32.npy", shape=(16384, 16384), dtype=np.float32)
    ram = load_to_ram(mm)

    # ---- (Optional) attempt cold-cache run for memmap
    # attempt_drop_os_cache()  # uncomment; needs privileges; then reopen mm if desired

    # ---- Fixed-size: 100k tiles of 32x32
    n_tiles, Htile, Wtile = 100_000, 32, 32
    ys, xs = random_positions(*mm.shape, h=Htile, w=Wtile, n=n_tiles, seed=0)

    res_mm_fixed = run_bench_fixed(
        mm,
        ys,
        xs,
        h=Htile,
        w=Wtile,
        methods=(
            "loop_copy",
            "gather_chunked",
            "numba_copy",
            "blocked_aligned",
            "blocked_stitched",
        ),
        batch=4096,
        block=64,
        check=True,
    )
    res_ram_fixed = run_bench_fixed(
        ram,
        ys,
        xs,
        h=Htile,
        w=Wtile,
        methods=(
            "loop_copy",
            "gather_chunked",
            "numba_copy",
            "blocked_aligned",
            "blocked_stitched",
        ),
        batch=4096,
        block=64,
        check=True,
    )

    print("Memmap parent (fixed)\n", res_mm_fixed)
    print("\nRAM parent (fixed)\n", res_ram_fixed)

    # ---- Variable-size: 100k tiles with sizes in [24..40], pad to 40x40
    rng = np.random.default_rng(42)
    hs = rng.integers(24, 41, size=n_tiles, dtype=np.int64)
    ws = rng.integers(24, 41, size=n_tiles, dtype=np.int64)
    ys_var, xs_var = random_positions_var(*mm.shape, hs, ws, seed=1)
    Hpad, Wpad = 40, 40  # common padded size; choose >= max(hs), max(ws)
    padval = 0.0

    res_mm_var = run_bench_var(
        mm,
        ys_var,
        xs_var,
        hs,
        ws,
        Hpad,
        Wpad,
        methods=("loop_copy_var", "gather_var", "numba_copy_var", "numba_blocked_var"),
        block=64,
        padval=padval,
        check=True,
    )
    res_ram_var = run_bench_var(
        ram,
        ys_var,
        xs_var,
        hs,
        ws,
        Hpad,
        Wpad,
        methods=("loop_copy_var", "gather_var", "numba_copy_var", "numba_blocked_var"),
        block=64,
        padval=padval,
        check=True,
    )

    print("\nMemmap parent (variable)\n", res_mm_var)
    print("\nRAM parent (variable)\n", res_ram_var)

    # ---- Side-by-side summaries (if pandas available)
    if pd is not None:
        fixed = pd.concat(
            {"memmap": res_mm_fixed.set_index("method"), "ram": res_ram_fixed.set_index("method")},
            axis=1,
        )
        fixed["speedup_ram_vs_mm"] = fixed[("ram", "GiBps")] / fixed[("memmap", "GiBps")]
        var = pd.concat(
            {"memmap": res_mm_var.set_index("method"), "ram": res_ram_var.set_index("method")},
            axis=1,
        )
        var["speedup_ram_vs_mm"] = var[("ram", "GiBps")] / var[("memmap", "GiBps")]
        print("\nSummary (fixed-size)\n", fixed)
        print("\nSummary (variable-size)\n", var)
    else:
        print("\n(pandas not installed; skipping side-by-side summaries)")

    # ---- (Optional) control Numba threads
    if HAVE_NUMBA:
        # from numba import set_num_threads, get_num_threads
        # set_num_threads(8)  # e.g., set to number of physical cores
        pass


# === FITS variable-size tile benchmark: var_copy vs gather_var =================
# Sources: uncompressed memmap, uncompressed RAM, RICE_1 tile-compressed, NOCOMPRESS tiled
# Requires: astropy, numpy (pandas optional for nicer tables)

import os, time, platform, subprocess
import numpy as np
from astropy.io import fits

try:
    import pandas as pd
except Exception:
    pd = None

# --------------------------- helpers ------------------------------------------


def gib(nbytes):
    return nbytes / (1024**3)


def attempt_drop_os_cache(verbose=True):
    """
    Best-effort page cache drop (needs sudo -n privileges).
    Linux:  sync; echo 3 > /proc/sys/vm/drop_caches
    macOS:  sync; purge
    Returns True if the command ran successfully, else False.
    """
    sys = platform.system().lower()
    try:
        if "linux" in sys:
            cmd = ["bash", "-lc", "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'"]
        elif "darwin" in sys:
            cmd = ["bash", "-lc", "sync; sudo -n purge"]
        else:
            if verbose:
                print("[cold-cache] unsupported platform")
            return False
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ok = r.returncode == 0
        if verbose:
            print(
                "[cold-cache]",
                "success" if ok else f"failed: {r.stderr.strip() or 'non-zero exit'}",
            )
        return ok
    except Exception as e:
        if verbose:
            print("[cold-cache] exception:", e)
        return False


def random_positions_var(H, W, hs, ws, seed=0):
    """Per-tile top-left positions, ensuring tiles fit in-bounds."""
    rng = np.random.default_rng(seed)
    hs = np.asarray(hs, dtype=np.int64)
    ws = np.asarray(ws, dtype=np.int64)
    M = hs.size
    ys = (rng.random(M) * (H - hs)).astype(np.int64)
    xs = (rng.random(M) * (W - ws)).astype(np.int64)
    return ys, xs


def _bytes_moved_var(hs, ws, itemsize):
    """Approximate read+write traffic for throughput accounting."""
    return 2 * int(np.sum(hs.astype(np.int64) * ws.astype(np.int64))) * itemsize


# --------------------------- FITS writers -------------------------------------


def write_uncompressed_fits(path, shape=(16384, 16384), dtype=np.float32, seed=0):
    """Write a single-HDU uncompressed image (PrimaryHDU)."""
    if os.path.exists(path):
        return
    H, W = shape
    rng = np.random.default_rng(seed)
    data = rng.random((H, W), dtype=dtype)  # ~1 GiB @ float32 for 16384^2
    fits.HDUList([fits.PrimaryHDU(data=data)]).writeto(path, overwrite=True)


def write_compressed_fits(
    path,
    shape=(16384, 16384),
    dtype=np.float32,
    compression="RICE_1",  # 'RICE_1' or 'NOCOMPRESS' here
    tile_shape=(64, 64),
    seed=0,
    quantize_level=None,  # only for RICE_1 (optional)
    hcomp_scale=None,  # only for HCOMPRESS_1 (not used here)
):
    """
    Write PrimaryHDU + CompImageHDU. Only pass codec-appropriate kwargs.
    """
    if os.path.exists(path):
        return
    H, W = shape
    rng = np.random.default_rng(seed)
    data = rng.random((H, W), dtype=dtype)

    prim = fits.PrimaryHDU()
    kw = dict(compression_type=compression, tile_shape=tile_shape)

    if compression == "RICE_1":
        if quantize_level is not None:
            kw["quantize_level"] = quantize_level
    elif compression == "HCOMPRESS_1":
        if hcomp_scale is not None:
            kw["hcomp_scale"] = hcomp_scale
    # For 'NOCOMPRESS' (and 'GZIP_1', 'PLIO_1'), do not pass RICE/HCOMPRESS args.

    comp = fits.CompImageHDU(data=data, **kw)
    comp.header["EXTNAME"] = "COMPRESSED"
    fits.HDUList([prim, comp]).writeto(path, overwrite=True)


# --------------------------- FITS open paths ----------------------------------


def open_uncompressed_memmap(path):
    """
    Return (hdul, arr_memmap). Keep hdul alive to retain mapping.
    """
    hdul = fits.open(path, memmap=True)
    arr = hdul[0].data  # numpy.memmap
    return hdul, arr


def _ensure_2d_from_header(arr, hdr):
    """
    Some compressed paths can yield a 1-D buffer; reshape using header dims.
    For compressed images, dims are in ZNAXISn; fallback to NAXISn.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.ascontiguousarray(a)
    ny = hdr.get("ZNAXIS2", hdr.get("NAXIS2"))
    nx = hdr.get("ZNAXIS1", hdr.get("NAXIS1"))
    if ny is None or nx is None:
        raise ValueError("Cannot infer 2-D shape from header")
    ny, nx = int(ny), int(nx)
    if a.size != ny * nx:
        raise ValueError(f"Data size {a.size} != {ny}*{nx}")
    return np.ascontiguousarray(a.reshape(ny, nx))


def open_uncompressed_ram(path):
    with fits.open(path, memmap=False) as hdul:
        prim = hdul[0]
        return _ensure_2d_from_header(prim.data, prim.header)


def open_compressed_ram(path):
    """
    Return decompressed 2-D ndarray from the first CompImageHDU.
    Astropy materializes compressed tiles into RAM on access.
    """
    with fits.open(path, memmap=False) as hdul:
        comp = next((h for h in hdul if isinstance(h, fits.CompImageHDU)), None)
        if comp is None:
            raise RuntimeError("No CompImageHDU found.")
        return _ensure_2d_from_header(comp.data, comp.header)


# --------------------------- methods (variable-size) ---------------------------


def var_copy(arr, ys, xs, hs, ws, Hpad, Wpad, padval=0.0):
    """
    Python loop: per-tile slice copy into a padded (Hpad x Wpad) output.
    """
    M = len(ys)
    out = np.full((M, Hpad, Wpad), padval, dtype=arr.dtype)
    t0 = time.perf_counter()
    for i in range(M):
        y0, x0, h, w = int(ys[i]), int(xs[i]), int(hs[i]), int(ws[i])
        out[i, :h, :w] = arr[y0 : y0 + h, x0 : x0 + w]
    t = time.perf_counter() - t0
    return t, float(out.sum(dtype=np.float64))


def gather_var(arr, ys, xs, hs, ws, Hpad, Wpad, padval=0.0):
    """
    Group by (h,w); for each group do a single advanced-index gather, then pad.
    """
    H, W = arr.shape
    M = len(ys)
    out = np.full((M, Hpad, Wpad), padval, dtype=arr.dtype)
    sizes = np.stack([hs, ws], axis=1)
    uniq, inv = np.unique(sizes, axis=0, return_inverse=True)
    flat = arr.ravel()
    t0 = time.perf_counter()
    for k, (h, w) in enumerate(uniq):
        idxs = np.nonzero(inv == k)[0]
        if idxs.size == 0:
            continue
        ro = (np.arange(h, dtype=np.intp) * W)[None, :, None]
        co = np.arange(w, dtype=np.intp)[None, None, :]
        base = (ys[idxs].astype(np.intp) * W + xs[idxs].astype(np.intp))[:, None, None]
        tiles = flat[base + ro + co]  # (B,h,w)
        out[idxs, :h, :w] = tiles
    t = time.perf_counter() - t0
    return t, float(out.sum(dtype=np.float64))


# --------------------------- runner / API -------------------------------------


def _run_one_source(name, arr, ys, xs, hs, ws, Hpad, Wpad):
    itemsize = arr.dtype.itemsize
    bytes_mv = _bytes_moved_var(hs, ws, itemsize)
    rows = []
    for mname, fn in (("var_copy", var_copy), ("gather_var", gather_var)):
        t, sig = fn(arr, ys, xs, hs, ws, Hpad, Wpad)
        rows.append(
            {
                "source": name,
                "method": mname,
                "sec": t,
                "GiBps": (bytes_mv / t) / (1024**3),
                "sig": sig,
            }
        )
    return rows


def _to_df(rows):
    if pd is None:
        return rows
    df = pd.DataFrame(rows)
    return (
        df[["source", "method", "sec", "GiBps", "sig"]]
        .sort_values(["source", "method"])
        .reset_index(drop=True)
    )


def test_fits_variable_size_benchmarks(
    shape=(16384, 16384),  # ~1.0 GiB @ float32
    dtype=np.float32,
    n_tiles=100_000,
    size_min=24,
    size_max=40,
    tile_shape=(64, 64),  # for RICE_1 / NOCOMPRESS
    padval=0.0,
    seed=123,
    cold_cache_memmap=False,  # try to drop OS cache before memmap run
    paths=("img_uncompressed.fits", "img_rice_tile.fits", "img_nocompress_tile.fits"),
):
    """
    Build/overwrite three FITS files; benchmark var_copy vs gather_var across four sources:
      - fits_uncompressed_memmap (disk-backed memmap)
      - fits_uncompressed_RAM
      - fits_rice1_compressed (RAM after decode)
      - fits_nocompress_tiled (RAM after "no-compress" tile read)
    Returns a DataFrame (if pandas) or list of dicts.
    """
    unc_path, rice_path, noc_path = paths

    # Fresh files (remove old)
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

    # Write files
    write_uncompressed_fits(unc_path, shape=shape, dtype=dtype, seed=seed)
    write_compressed_fits(
        rice_path, shape=shape, dtype=dtype, compression="RICE_1", tile_shape=tile_shape, seed=seed
    )
    write_compressed_fits(
        noc_path,
        shape=shape,
        dtype=dtype,
        compression="NOCOMPRESS",
        tile_shape=tile_shape,
        seed=seed,
    )

    # Variable sizes and positions
    rng = np.random.default_rng(seed + 1)
    hs = rng.integers(size_min, size_max + 1, size=n_tiles, dtype=np.int64)
    ws = rng.integers(size_min, size_max + 1, size=n_tiles, dtype=np.int64)
    Hpad, Wpad = int(hs.max()), int(ws.max())
    H, W = shape
    ys, xs = random_positions_var(H, W, hs, ws, seed=seed + 2)

    # Open sources
    hdul_mm, arr_mm = open_uncompressed_memmap(unc_path)  # memmap
    if cold_cache_memmap and attempt_drop_os_cache():
        # Reopen to avoid stale mapping after cache drop
        hdul_mm.close()
        hdul_mm, arr_mm = open_uncompressed_memmap(unc_path)

    arr_ram = open_uncompressed_ram(unc_path)
    arr_rice = open_compressed_ram(rice_path)
    arr_noc = open_compressed_ram(noc_path)

    # Sanity: ensure 2-D arrays
    for name, a in [
        ("memmap", arr_mm),
        ("RAM", arr_ram),
        ("rice", arr_rice),
        ("nocompress", arr_noc),
    ]:
        if a.ndim != 2:
            raise RuntimeError(f"{name} array not 2-D; got {a.ndim}D")

    # Run benches
    rows = []
    rows += _run_one_source("fits_uncompressed_memmap", arr_mm, ys, xs, hs, ws, Hpad, Wpad)
    rows += _run_one_source("fits_uncompressed_RAM", arr_ram, ys, xs, hs, ws, Hpad, Wpad)
    rows += _run_one_source("fits_rice1_compressed", arr_rice, ys, xs, hs, ws, Hpad, Wpad)
    rows += _run_one_source("fits_nocompress_tiled", arr_noc, ys, xs, hs, ws, Hpad, Wpad)

    # Close memmap handle to release file
    hdul_mm.close()

    df = _to_df(rows)
    print(df)
    return df


# --------------------------- Example call -------------------------------------


def test_fits_tile():
    # Run with defaults (~1 GiB image, 100k tiles, 24..40 varied sizes)
    df = test_fits_variable_size_benchmarks()


# Cold-cache the memmap first (requires sudo -n for cache drop; will gracefully skip otherwise)
# df = test_fits_variable_size_benchmarks(cold_cache_memmap=True)

# Smaller/faster smoke test (uncomment to try):
# df = test_fits_variable_size_benchmarks(shape=(8192,8192), n_tiles=20_000)


# %%
