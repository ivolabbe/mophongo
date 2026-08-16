# WREN.md — Reference for fixes on `origin/main` (PR #97)

These six commits live on `origin/main` ahead of our local `main` (now at `2c21e18`).
We are NOT pulling them — we will re-implement them ourselves on branch `flux-bug`.
This document captures the intent, diffs, and key reasoning for each.

Range: `2c21e18..origin/main` (6 commits, all merged via PR #97 `wrensuess/dev-wren`)

```
9d2ed2d Fix flux bias from absolute regularization in SceneFitter
38c9917 Switch SceneFitter solvers from CG to spsolve
bdf9fcb Add scene_max_merge_radius to FitConfig
2893f43 Improve astrometry source selection: exclude stars and blended sources
eb6b922 Fix aperture correction: use post-conv template total instead of pre-conv aperture sum
80cad37 Merge PR #97
```

---

## 1. `9d2ed2d` — Fix flux bias from absolute regularization in SceneFitter

**File:** `src/mophongo/scene_fitter.py` (+6 −7)

**Problem.** `astrom_reg` (default `1e-4`) was being applied directly to the flux
block diagonal of `ATA`. For F1800W drizzled data `ATA[i,i] ≈ 1.33e-4`, so the
regularizer was the *same order* as the diagonal — causing **~43% flux suppression**.

**Fix.** Reserve `astrom_reg` for the *shift block only*. The flux block uses an
adaptive regularizer scaled to ATA's magnitude:
```python
flux_reg = 1e-6 * np.median(A.diagonal())
Areg = A + sp.eye(A.shape[0], format="csr") * flux_reg
```

**Latent bugs fixed in same commit (in `solve_flux`):**
- Stray `self` argument on a `@staticmethod` (would crash if called statically).
- Bare `diags(...)` reference → must be `sp.diags(...)` (NameError when path is hit).

**Key diff:**
```python
# OLD
reg = getattr(config, "astrom_reg", 0)
if reg <= 0:
    reg = 1e-6 * np.median(A.diagonal())
Areg = A + sp.eye(A.shape[0], format="csr") * reg

# NEW
flux_reg = 1e-6 * np.median(A.diagonal())
Areg = A + sp.eye(A.shape[0], format="csr") * flux_reg
```

---

## 2. `38c9917` — Switch SceneFitter solvers from CG to spsolve

**File:** `src/mophongo/scene_fitter.py` (+5 −9)

**Problem.** Scene systems are small (10–100 sources) and frequently
ill-conditioned, so iterative CG was unreliable (convergence failures, partial
solutions).

**Fix.** Use direct LU (`scipy.sparse.linalg.spsolve`) — exact to machine
precision, no convergence failure, faster at this problem size. Replaces CG
in two places:
- `solve_flux`: whitened flux-only system.
- `solve` joint flux+shift: bordered system `K = [[A_w, AB_w], [AB_w.T, BB_wI]]`.

`cg` import is retained (still used elsewhere in module).

**Key diff:**
```python
from scipy.sparse.linalg import cg, spsolve   # add spsolve

# replaces both cg(A_w, b_w, ...) and cg(K, rhs, ...)
x_w  = spsolve(A_w, b_w);  info = 0
sol  = spsolve(K, rhs);    info = 0
```

---

## 3. `bdf9fcb` — Add `scene_max_merge_radius` to FitConfig

**Files:** `src/mophongo/fit.py` (+1), `CHECKLIST.md` (+15)

**Why.** `pipeline.py` already reads `config.scene_max_merge_radius` via
`getattr(..., np.inf)`. Adding it as a real `FitConfig` field makes it
discoverable / documented.

**Diff:**
```python
# fit.py FitConfig
scene_max_merge_radius: float = np.inf  # Max distance (px) to merge underfilled scenes
```

CHECKLIST.md gains TODOs about scene-size guards and `max_template_radius`
validation (see commit text for details).

---

## 4. `2893f43` — Improve astrometry source selection: exclude stars and blended sources

**Files:** `fit.py` (+1), `pipeline.py` (+7), `scene.py` (+74 −34),
`templates.py` (+1), `tests/test_astrometry.py` (+313 new).

**Problem.** Astrometric shifts use bright sources, but bright sources can be
(a) stars (PSF-driven, not representative of galaxy templates) or
(b) heavily blended (large neighbor contamination biases shifts since
`shift ∝ 1/alpha0` and `alpha0` is inflated by neighbours).

**Three-part fix:**

### 4a. `Template.is_star` flag
```python
# templates.py — Template.__init__
self.is_star: bool = False  # set by pipeline from catalog flag_star
```

### 4b. Pipeline marks stars from catalog
```python
# pipeline.py
if catalog is not None and "flag_star" in catalog.colnames:
    star_ids = set(int(r["id"]) for r in catalog if r["flag_star"] == 1)
    for t in templates:
        if int(t.id) in star_ids:
            t.is_star = True
    logger.info("Marked %d templates as stars (excluded from astrometry)",
                sum(t.is_star for t in templates))
```

### 4c. New `astrom_isolation_thresh` + `_astrom_isolation_mask` helper
```python
# fit.py
astrom_isolation_thresh: float = 0.5  # min flux dominance for astrometry (0–1; 0 = no cut)
```

```python
# scene.py — new helper
def _astrom_isolation_mask(A, b, thresh):
    """dominance[i] = (alpha0[i]*ATA[i,i]) / (alpha0[i]*ATA[i,i] + sum_j alpha0[j]*|ATA[i,j]|)
    ATA[i,j] is the integral of T_i*T_j, so alpha0[j]*ATA[i,j] is the neighbour
    flux falling within source i's footprint."""
    n = A.shape[0]
    diag = np.maximum(A.diagonal(), 1e-12)
    alpha0 = np.abs(b) / diag
    Au = sp.triu(A, k=1).tocoo()
    if Au.nnz == 0:
        return np.ones(n, dtype=bool)
    i, j, aij = Au.row, Au.col, np.abs(Au.data)
    neighbor_flux = np.zeros(n)
    np.add.at(neighbor_flux, i, alpha0[j] * aij)
    np.add.at(neighbor_flux, j, alpha0[i] * aij)
    self_flux = alpha0 * diag
    dominance = self_flux / np.maximum(self_flux + neighbor_flux, 1e-12)
    return dominance >= thresh
```

### 4d. `generate_scenes` excludes stars from bright mask (so star-dominated scenes get merged into neighbours)
```python
not_star = ~np.array([t.is_star for t in templates], dtype=bool)
bright_mask = np.asarray(snr_proxy > astrom_minimum_snr, dtype=bool) & not_star
```

### 4e. `Scene.solve` triple-cut bright mask + `has_shifts` guard
```python
not_star = ~np.array([t.is_star for t in self.templates], dtype=bool)
isolated = _astrom_isolation_mask(A, b, cfg.astrom_isolation_thresh)
self.is_bright = (snr_proxy > cfg.astrom_minimum_snr) & not_star & isolated
```
Wraps the entire shift-application block in `if self.shifts is not None and len(self.shifts) > 0:`,
falling through to a warning for pathological all-blended scenes (with a TODO
about merging-with-neighbour as a future improvement, since isolation can't be
applied at merge time the way the star mask is).

---

## 5. `eb6b922` — Fix aperture correction: post-conv total / post-conv aperture

**File:** `src/mophongo/pipeline.py` (+10 −14)

**Problem.** Aperture correction numerator was `aperture_sum(pre_conv_template, r_cat_pix)`,
which equals `EE_F444W(r) ~ 0.3`. This produced a **~1.2 mag systematic offset**.

**Conceptually correct form:** `corr = post_conv_total / post_conv_aperture = 1 / EE_source_MIRI(r)`.
This converts the partial aperture flux `ap_raw` into total source flux.

**Fix.** Use `tmpl.data.sum()` (post-conv total) for the numerator. Drops the
`r_cat_pix_by_id` lookup and the `ref_tmpls` parent-id mapping entirely.

**Key diff:**
```python
# OLD
r_cat_pix_by_id = self._resolve_catalog_ap_radius_pix(cat, cfg, r_default=r_img_pix)
ref_tmpls = {int(t.id): t for t in self.tmpls.templates}
...
tmpl_ref = ref_tmpls.get(int(pid))
r_cat_pix = r_cat_pix_by_id.get(int(pid), np.nan)
num = (self._aperture_sum_on_template(tmpl_ref, r_cat_pix)
       if (tmpl_ref and np.isfinite(r_cat_pix)) else np.nan)

# NEW
num = float(tmpl.data.sum())   # post-conv total flux
```
Denominator (`self._aperture_sum_on_template(tmpl, r_img_pix)`) unchanged.

**Known residual.** ~10% offset vs IDL mophongo remains: IDL uses *per-source
Kron radius* (r_Kron >> r_aper) in the correction (Kron-total convention).
Future step: denominator becomes `aperture_sum(tmpl, r_Kron)` using detection-
catalog Kron radii.

---

## 6. `80cad37` — Merge commit (PR #97)

Title: *"long-wavelength residual bug fix + astrometry source selection improvements"*

Composes commits 1–5 above. Net touch: 7 files, +437 / −65.

---

## Implementation order suggestion (for redoing on `flux-bug`)

The commits build on each other; cleanest order to re-implement:

1. **`9d2ed2d`** — flux regularization fix + the two latent `self`/`diags` bugs in `solve_flux`. Smallest, most isolated.
2. **`38c9917`** — CG → spsolve. One-line API switch in two call sites.
3. **`bdf9fcb`** — promote `scene_max_merge_radius` into `FitConfig`. Trivial.
4. **`2893f43`** — astrometry source selection (stars + isolation). The largest semantic change; needs the test suite (`tests/test_astrometry.py`, +313 lines new) — worth porting.
5. **`eb6b922`** — aperture correction fix. Independent of the others.

Our local WIP on `flux-bug` already touches all of these files
(`scene.py`, `scene_fitter.py`, `fit.py`, `pipeline.py`, `templates.py`,
`CHECKLIST.md`), so check for overlap before applying — some of our edits may
already cover the same ground or conflict with these fixes.
