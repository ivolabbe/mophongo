# Array lifetimes in a pipeline run

Audit note, 2026-08-13, written against `b7bec1e`; sections 4 to 6 revised
after the five changes described in section 6 landed.

Companion to `docs/SCALING_FIXED_MEMORY.md`, which proposes a decomposition that
bounds the peak by the largest scene rather than by the field. This note is
narrower: it inventories every full-field array a run allocates, records where
each one is born and where it is last read, and separates the arrays that are
still needed from the ones that are merely still referenced.

Sizes are for the full-field MINERVA UDS grid: 34560 x 25344 = 876 Mpx, so
3.5 GB per float32 plane and the same for an int32 segmentation map. Arrays on
the fitted band's native grid are `3.5/k^2` GB, where
`k = bin_factor_from_wcs(wcs[0], wcs[ifilt])` (`pipeline.py:3715`); for MIRI
F770W against the NIRCam detection grid that is around 0.4 GB.

## 1. Anonymous versus file-backed

The distinction runs through the whole table, so it is worth stating once.

A **memory-mapped** array is a window onto a file. Pages fault in on first
touch, they are clean (identical to what is on disk), and the kernel can evict
them at any time and re-read them later. Two processes mapping the same file
share one set of pages. Such an array inflates RSS while resident but is not a
hard claim on memory.

An **anonymous** array — anything from `np.zeros`, `np.empty`, or arithmetic —
has no file behind it. Its pages are dirty by definition, so the only relief
under pressure is swap, and without swap the process is killed. It is not
shared across processes.

Two consequences matter here. First, a peak figure is not one number: the
memory-mapped share is soft and the anonymous share is the ceiling. Second,
`np.zeros` on a large shape is anonymous but not yet resident, because the
kernel maps every page copy-on-write to a single shared zero page; reading is
free and only writing faults a real page. That is why `_read_image` can
allocate a full-mosaic shape and cost only the box it fills
(`pipeline.py:381`), and why the code insists on `np.zeros` over
`np.zeros_like` at `pipeline.py:1643` and `4046` — `zeros_like` is
`empty_like` plus a memset, which writes every page and materialises the whole
grid.

Full-field reads hand over memory maps: `_read_image` with no box returns
`np.asarray(hdu.data)` over an open `memmap=True` HDU (`pipeline.py:378`), and
`as_label_array` returns an integer segmentation map untouched for exactly this
reason (`utils.py:44-54`).

## 2. Lifetime table

| array | size (GB) | backing | born | last real read | released |
|---|---|---|---|---|---|
| `sci_hi` -> `images[0]` | 3.5 | memmap (copy-on-write over the repair patches) | `load_data:1524` | scene plots, `write_outputs` | no |
| `segmap` | 3.5 | memmap (`>i4`); anon int32 (COSMOS float64) | `load_data:1530` | as above | no |
| `wht_hi` (raw) | 3.5 | memmap | `_load_detection_ivar:1471` | inside `get_bg_and_ivar` | yes, on return |
| `wht0` (raw, repair path) | 3.5 | memmap | `load_data:1565` | repair + cache write | yes, `del wht0:1609` |
| `ivar_hi` = `weights[0]` | 3.5 | anon | `load_data:1656` | template build | yes, `run:3862` |
| `wht_lo` (raw) | 3.5/k^2 | memmap | `load_data:1526` | bg/ivar, footprint cut | yes, scope exit |
| `bg` (lo) | 3.5/k^2 | anon | `load_data:1633` | `load_data:1647` | yes, scope exit |
| `ivar` = `weights[1]` | 3.5/k^2 | anon | `load_data:1633` | `_convolved_templates:3730` | **no** |
| `sci_fit` = `images[1]` | 3.5/k^2 | anon | `load_data:1646` | `_convolved_templates:3730` | yes, rebound |
| `weights_i` (upsampled ivar) | 3.5 | anon | `_convolved_templates` | `predicted_errors`, then the scene figures | yes, `_release_scene_weights` |
| `images[1]` (upsampled sci) | 3.5 | anon | `_convolved_templates:3730` | `run:4057` | no |
| `res` | 3.5 | **memmap over `f_residual`** | `_allocate_residual` | `write_outputs` | flushed; stays mapped |
| hi-res template set | ~6 | anon | `_prepare_hi_templates` | `write_stamps` | no |
| convolved template set | ~6 | anon | `_convolved_templates` | `write_stamps` | no |
| `_data_unshifted` | ~6 | anon | astrometric passes | last shift applied | yes, in `run` |
| model image | 3.5 | anon, cached | `_ModelImages` | diagnostics only | no |

## 3. What the weight-map names actually are

The four names that look like four weight maps for one band are two bands, and
two of them are the same object.

Detection band:

```
wht_hi      raw on disk, memmap, transient
   |  get_bg_and_ivar:  ivar_new = w * scale, zeroed where invalid
ivar_hi     anon, 3.5 GB
   |  Pipeline.__init__(weights=[ivar_hi, ivar])
weights[0]  the same array; a list slot, not a copy
```

Fitted band:

```
wht_lo      raw on disk, memmap, transient
   |  _bg_and_ivar_boxed
ivar  ==  weights[1]       anon, 3.5/k^2
   |  _upsample_boxed (block replicate, weights x k^2)
weights_i                  anon, 3.5
```

So one persistent weight array for the detection band and two for the fitted
band. The calibration itself is a scalar and a mask (`catalog.py:369-372`):

```python
scale    = np.float32(1.0) / (sigma_true * sigma_true + np.float32(1e-30))
ivar_new = np.multiply(w, scale, dtype=np.float32)
np.copyto(ivar_new, np.float32(0.0), where=~valid)
```

Raw weights are read directly in only two places, both appropriate.
`get_bg_and_ivar` reads them because they are what it calibrates.
`repair_in_memory` reads them (`load_data:1590`) because `saturate.py` uses a
weight map only as a validity mask, `wht > 0` (`saturate.py:108,222,281,394,462`),
and as relative weights inside a stamp ring (`saturate.py:298,413`). Both are
invariant under the global scale factor that calibration supplies, so an
uncalibrated map is not a defect there. The ordering is forced the same way
round: repair fills saturated cores and restores their weights, changing which
pixels are valid, so the calibration must run on the repaired map. It does —
`_load_detection_ivar(tmpl_hi, wht_hi=wht_hi_repaired)` at `load_data:1656`.

Nothing on the fit side ever sees an uncalibrated weight.

## 4. Dead but still referenced

**`weights[1]`, 3.5/k^2 GB.** Dead once `_convolved_templates` has upsampled
it. Retained deliberately: `source_products` handles a native-grid weight map
as well as a reference-grid one, and rebins when it gets the former. Low
priority at 0.4 GB, but it is the only true information duplicate left in the
set — the same weights on two grids.

**`images[1]` and `res`, 3.5 GB each.** Once the residual has been formed the
pair is redundant with the model, which `_ModelImages` reconstructs as
`images[i+1] - residuals[i]` on demand. Only one of the three is independent,
and touching the model allocates and caches a third full plane. `res` is now
file-backed, so the standing anonymous cost here is one plane, not two.

**`weights_i`, 3.5 GB — fixed.** It used to survive to the end of the process:
last read by `Templates.predicted_errors`, but held by every `Scene` through
`self.all_scenes`, so 3.5 GB of dead weights sat through the stamp write. See
section 6.

## 5. Two costs that are not in the table

**Byte-order copies inside `get_bg_and_ivar`.** FITS stores big-endian, so a
memory-mapped float32 image arrives as `>f4`. The function opens with
(`catalog.py:267-268`):

```python
s = np.asarray(sci, dtype=np.float32)
w = np.asarray(wht, dtype=np.float32)
```

`>f4` and `float32` differ in byte order, so `asarray` copies rather than
viewing. Confirmed directly: `np.asarray(np.zeros(4, dtype='>f4'),
dtype=np.float32)` does not share memory with its input. The detection-band
call on a full field therefore allocates 3.5 GB for `s`, 3.5 GB for `w`,
0.88 GB each for `valid_w` and `valid`, and 3.5 GB for `ivar_new` — about
12 GB transient inside one call, and every memory-map saving upstream is undone
for its duration.

The full-resolution arrays are used for `_valid_block_means`, a strided median
sample, and the final `w * scale`. All three work on `>f4` directly, at a small
arithmetic cost, so the copies are avoidable. Still open.

**Template stamps are the largest anonymous term left.** Two full sets, ~6 GB
each, alive together from the convolution to the stamp write. Bounding them is
what `SCALING_FIXED_MEMORY.md` sections 3 and 8 are about; nothing here changes
it.

## 6. Changes made

Five, in the order they run. All five are in `b7bec1e`'s successor; the suite
is 362 passing.

1. **The band weight map is released once nothing reads it.**
   `_release_scene_weights` clears `Scene.weights` across a band. `run` calls it
   straight after `predicted_errors` when the run draws no scene figures;
   `write_outputs` calls it for everything else, after the figures and before
   the stamps. `Scene.residual` and `Scene.plot` now null zero-weight pixels
   only when the weights are still attached, and `Scene.solve` still refuses to
   run without them — the intended failure, since the fit is over.

2. **`write_outputs` writes the stamps last.** The stamps were written before
   the scene figures, which is what forced the weights to stay alive through
   the run's other memory peak. Reordering costs nothing (the products are
   independent) and has a second benefit: a run that dies in the stamp write
   now has its figures already on disk.

3. **The residual accumulates into its own output file.** `_residual_memmap`
   writes the FITS header, extends the file with `truncate` — leaving it
   sparse, so a trial patch costs the patch — and maps the data section big
   endian. `run` accumulates scene models into that map and subtracts in place;
   `write_outputs` flushes instead of writing. API-driven runs and bands past
   the first fall back to anonymous memory, as does any `OSError` on the map.

4. **The repair replays its patch table onto a fresh map.**
   `repair_saturated_holes` returns full-field copies of sci and segmap
   (`saturate.py:733`), which the run then held for its whole length even
   though they differ from the inputs only over the saturated cores.
   `load_data` now re-reads both inputs and applies the patch table that
   `_save_repair_cache` just computed, which is exactly what the cache-reuse
   path already did — the two paths now produce the same representation
   through the same helper, `_apply_repair_patches`. This is safe because
   astropy maps a read-only HDU copy-on-write: the patched pages go private and
   the input mosaic on disk is untouched, which
   `test_repair_patches_do_not_write_through_to_the_input` pins.

5. **`write_stamps` streams.** Offsets are computed from the shapes recorded in
   the first pass, each dataset is created at its final size, and every stamp
   is written straight into its slot. The old form collected every flattened
   stamp in a list and concatenated — a full extra copy of every stamp, 12 GB
   on a MINERVA field, at the very end of the run.

Two smaller things went with them: the `isfinite` sweep at the top of `run` is
gone (its image branch was inverted and unreachable, and its weight branch
re-checked a guard `load_data` had already applied), and the stamps test now
asserts pixel equality after a round trip, which is what would catch an offset
error in the streaming write.

## 7. Still open

- The byte-order copies in `get_bg_and_ivar` (section 5).
- ivar as `(memmapped wht, scale, invalid mask)` rather than a materialised
  array, for both bands. Composes with the on-demand upsampled band in
  `SCALING_FIXED_MEMORY.md` section 5 — with both, neither ivar array exists.
- `weights[1]` after the upsample (section 4), which needs `source_products`
  pointed at the reference-grid array.
- The two template sets, the largest anonymous term left.
