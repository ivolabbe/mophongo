# Flux Recovery Bug Report

Date: 2026-05-01

## Summary

The Moffat mock validation showed an apparent 10 percent flux deficit:

```text
no_extension  hi=0.9023  lo=0.8796
dilated3      hi=0.9324  lo=0.8990
```

This was not caused by PSF matching, the split-cosine-bell window, the Moffat
PSF wing shape, or the source-size distribution. The high-resolution
self-consistency fit was already biased low, where no PSF-matching kernel is
used. That made the kernel hypothesis impossible as the primary cause.

The actual cause was an implementation error in the newer stateless scene
solver. `SceneFitter.solve()` was using `config.astrom_reg` as a flux
regularization term. `astrom_reg` is intended only for astrometric shift
parameters, but it was being added to the photometric normal matrix. With the
default `astrom_reg=1e-4`, broad, low-norm Moffat templates had their fitted
fluxes suppressed.

## Observable Symptom

The failure appeared in `tests/test_moffat_recovery.py` using the existing
pipeline validation path:

```python
table, resid, _ = pipeline.run(
    fit_images,
    segmap,
    catalog=catalog,
    weights=fit_wht,
    kernels=fit_kernels,
    extend_templates=None,
)
```

The diagnostic plot showed positive source-like residuals even for extended
objects. That was a key clue: modest PSF-matching errors should not produce a
coherent 10 percent flux deficit in the high-resolution self-fit.

## Why PSF Matching Was Not The Cause

A PSF-only check showed that the default matching window was imperfect but not
responsible for the photometric bias:

```text
default SCB(0.4,0.1):
  PSF residual RMS = 8.53e-05

optimized C^2 window:
  alpha = 0.04
  beta  = 0.90
  PSF residual RMS = 1.44e-06
```

However, using the optimized window did not remove the photometric offset:

```text
default window:
  no_extension  hi=0.9023  lo=0.8796
  dilated3      hi=0.9324  lo=0.8990

optimized window:
  no_extension  hi=0.9023  lo=0.8756
  dilated3      hi=0.9324  lo=0.8952
```

The high-resolution channel stayed biased at the same level. Since that channel
uses a delta kernel, the bias had to be in the fitting path, not in the PSF
matching kernel.

## Controlled Test That Isolated The Bug

The decisive test used the same Moffat mock, same segmentation map, same
catalog, same kernel, and same template extraction. Only the solver path was
changed.

Before the fix:

```text
scene solver, default config:
  dilated3 hi=0.9324  lo=0.8990

legacy solver, flux only:
  dilated3 hi=0.9746  lo=0.9864
```

Then `astrom_reg` was scanned in the scene solver flux-only path:

```text
scene flux-only, astrom_reg=1e-04:
  hi=0.9324  lo=0.8991

scene flux-only, astrom_reg=1e-08:
  hi=0.9746  lo=0.9864

scene flux-only, astrom_reg=0:
  hi=0.9746  lo=0.9864

scene flux-only, astrom_reg=1e-02:
  hi=0.1853  lo=0.0977
```

That establishes causality: `astrom_reg` was regularizing the flux solve.

## Implementation Error

The erroneous implementation was in `src/mophongo/scene_fitter.py`.

The buggy logic was effectively:

```python
ridge = getattr(config, "astrom_reg", 0)
if ridge <= 0:
    ridge = 1e-6 * np.median(A.diagonal())
Areg = A + sp.eye(A.shape[0], format="csr") * ridge
```

Here `A` is the photometric normal matrix. Adding `astrom_reg` to this matrix
mixes two different concepts:

- `reg_flux`: photometric flux ridge regularization
- `astrom_reg`: astrometric shift-block regularization

The shift-block regularization belongs only on the `BB` astrometric block.
Applying it to `A` shrinks fitted fluxes. The shrinkage becomes visible when
template normal-matrix diagonals are small, which happens naturally for broad,
unit-sum templates.

There was also a secondary implementation bug:

```python
@staticmethod
def solve_flux(self, A, b, config=None):
```

The extra `self` parameter made the scene flux-only branch fail when exercised
directly. That path was not being tested correctly before this investigation.

Finally, `fit_astrometry_niter=0` did not actually force the `Scene.solve()`
path to skip the joint astrometric solve. The scene code checked
`fit_astrometry_joint` but not `fit_astrometry_niter`.

## Correct Fix

### 1. Use a strictly positive flux ridge for the flux matrix

In `src/mophongo/scene_fitter.py`, the photometric normal matrix must use a
photometric ridge, not `config.astrom_reg`. The default can be tiny, but it
should be strictly positive when used as numerical regularization:

```python
diag_A = np.asarray(A.diagonal(), dtype=float)
pos_A = diag_A[np.isfinite(diag_A) & (diag_A > 0)]
scale_A = np.median(pos_A) if pos_A.size else 1.0

lam_A = _finite_positive(getattr(config, "reg_flux", 1e-4), default=1e-4) * scale_A
Areg = A + sp.eye(A.shape[0], format="csr") * lam_A
```

For a weighted normal matrix `A = T.T @ W @ T` with nonnegative weights, `A`
is positive semidefinite. Adding `lam_A * I` with a positive
factor makes the regularized matrix positive definite, unless the matrix contains non-finite
values. Filtering to finite, positive diagonal values avoids zero or NaN scale
factors when many templates have zero support. The configured `config.reg_flux`
is a dimensionless factor, analogous to `config.astrom_reg`, and must also be
checked explicitly: zero, negative, NaN, and infinity should all fall back to
the finite positive default. Here `_finite_positive(value, default)` denotes a
small helper that returns `float(value)` only when it is finite and positive,
otherwise `default`.

Keep `astrom_reg` only for the astrometric block. The existing pattern is the
right idea for the shift block, but it should also filter non-finite diagonal
values:

```python
diag_BB = np.asarray(BB.diagonal(), dtype=float)
pos_BB = diag_BB[np.isfinite(diag_BB) & (diag_BB > 0)]
scale_BB = np.median(pos_BB) if pos_BB.size else 1.0

lam_b = getattr(config, "astrom_reg", 1e-4) * scale_BB
BBreg = BB + sp.eye(BB.shape[0], format="csr") * lam_b
```

The important separation is:

- `lam_A = reg_flux * scale_A` regularizes the photometric flux block `A`.
- `astrom_reg * scale_BB` regularizes the astrometric shift block `BB`.

The line `scale_BB = np.median(diag_BB[diag_BB > 0]) ...` only protects the
astrometric block scale. It does not provide any protection for the flux block
and should not be reused as the flux ridge.

### 2. Fix the static flux-only solver signature

The static method should be:

```python
@staticmethod
def solve_flux(
    A: sp.spmatrix, b: np.ndarray, config: Optional[FitConfig] = None
) -> tuple[np.ndarray, np.ndarray, dict]:
```

It should use `sp.diags` or import `diags` explicitly:

```python
Dinv = sp.diags(1.0 / d, 0, format="csr")
```

### 3. Make `fit_astrometry_niter=0` mean flux-only in `Scene.solve`

In `src/mophongo/scene.py`, the flux-only path should be selected when either
joint astrometry is disabled or the requested number of astrometric iterations
is zero:

```python
if (not cfg.fit_astrometry_joint) or int(getattr(cfg, "fit_astrometry_niter", 0)) <= 0:
    sol = SceneFitter.solve(A, b, config=cfg, **kwargs)
else:
    ...
```

## Regression Test

Add a focused unit test so this cannot recur:

```python
def test_scene_fitter_reg_astrom_does_not_regularize_flux():
    A = sp.csr_matrix([[1e-3]])
    b = np.array([1e-3])
    cfg = FitConfig(reg_flux=0.0, astrom_reg=1e-2)
    sol = SceneFitter.solve(A, b, config=cfg)
    np.testing.assert_allclose(sol.flux, [1.0], rtol=1e-5)
```

This test is intentionally simple. If `astrom_reg` leaks into the flux matrix,
the solved flux is strongly suppressed. If the implementation is correct, the
answer remains one.

## Validation After Fix

After applying the fix, the exact existing Moffat test code gives:

```text
no_extension  hi=0.9424  lo=0.9613
dilated3      hi=0.9746  lo=0.9864
```

The no-extension case remains lower because the segmentation footprint is
smaller. The mildly dilated case returns to the expected near-unity recovery.

The original pipeline recovery test also remains healthy:

```text
flux_1/flux_true median = 1.00
flux_2/flux_true median = 1.01
```

## Verification Commands

The fix was verified with:

```bash
MPLCONFIGDIR=/tmp/mophongo-mplconfig poetry run pytest tests/test_scene_fitter.py -q
MPLCONFIGDIR=/tmp/mophongo-mplconfig poetry run pytest \
  'tests/test_moffat_recovery.py::test_moffat_flux_recovery[no_extension-0-None]' \
  'tests/test_moffat_recovery.py::test_moffat_flux_recovery[dilated3-3-None]' \
  -q -s --basetemp=scratch/moffat_fixed_validation
MPLCONFIGDIR=/tmp/mophongo-mplconfig poetry run pytest \
  tests/test_pipeline.py::test_pipeline_flux_recovery -q -s \
  --basetemp=scratch/pipeline_after_scene_reg_fix
MPLCONFIGDIR=/tmp/mophongo-mplconfig poetry run python -m py_compile \
  src/mophongo/scene.py \
  src/mophongo/scene_fitter.py \
  tests/test_scene_fitter.py \
  tests/test_moffat_recovery.py
```

Results:

```text
tests/test_scene_fitter.py: 9 passed
Moffat no_extension + dilated3: 2 passed
tests/test_pipeline.py::test_pipeline_flux_recovery: 1 passed
py_compile touched files: passed
```

## Practical Takeaway

When a validation shows a broad, coherent flux deficit, first test whether the
high-resolution self-fit is also biased. If it is, PSF matching cannot be the
primary cause. Compare scene and legacy solvers on identical templates and
images, then scan regularization terms. In this case, that isolated the bug to
a single wrong regularization parameter in the scene fitter.
