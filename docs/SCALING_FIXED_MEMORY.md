# Scaling to arbitrarily large images at a fixed memory budget

Research note, 2026-08-13. No code changes. Written against `9fc52a6`.

The question is whether mophongo can be made to run a field of any size inside
a memory budget fixed in advance — the property that lets SExtractor process a
mosaic it could never hold — and whether the scene solves, currently a
sequential loop, can be run in parallel without giving that property back.

The short answer is yes, and the reason is structural rather than incidental:
the fit is already exactly block-diagonal over scenes, and the scene partition
is a function of the sparse normal matrix alone, not of the pixels. That
separation is what a fixed budget needs. What follows is the accounting, the
proposed decomposition, and the measurements behind the parallelism
recommendation.

## 1. What sets the peak today

Measured, from the full-field UDS F770W run recorded in `STATUS.md`
(138,610 templates, 591 scenes, `trial: null`, repair reloaded from cache):
87 minutes, **46.5 GB peak physical footprint**, 32.2 GB max RSS. Checkpoints
`(start)` 10.4 GB, `(templates)` 28.2, weight release 24.7, `(convolved)` 27.7,
`(end)` 11.0. COSMOS (151,778 templates) needs about 72 GB and dies at 48 right
at the upsample; EGS extrapolates to 110–130 GB. CANFAR reports
`memoryGB.defaultLimit = 32`, and jobs asking 48 GB and above have queued for
hours.

The peak decomposes into two families, both linear in field area.

**Full-grid rasters.** The detection grid is 34560 × 25344 = 876 Mpx, so
3.50 GB per float32 plane and the same for the int32 segmentation map. A run
holds, concurrently: detection science, segmentation map, the band science
image upsampled onto the reference grid, its inverse variance, and the
residual. The detection-band inverse variance is a sixth, released after
template construction (`pipeline.py:3653`). Call it 17–21 GB.

**Template stamps.** 138,610 stamps with a 100 px floor and a mean side of
104 px is 6.0 GB for one complete set. Up to three sets are live at once: the
hi-res set on `self.tmpls`, the convolved set returned by
`_convolved_templates` (`convolve_templates(inplace=False)` deep-copies each
stamp, `templates.py:1960`, and the hi-res set stays alive because
`write_stamps` reads it at `pipeline.py:2254`), and `_data_unshifted`, the
pre-shift pixels each shifted template retains so successive astrometric passes
resample the original rather than compounding the cubic smoothing
(`templates.py:1110`). That is 12–18 GB.

Two smaller terms spike rather than persist, and both scale with a scene's
*spatial extent* rather than its template count, so `scene_max_size` does not
bound either: `assemble_scene_system_AB` allocates `nB` float64 planes over the
bright anchors' bounding box, doubled when the leverage cap clips
(`scene.py:618`) — 0.61 GB for the widest full-field scene at `nB = 2`, and the
`Bq[k] * wbuf` temporaries in the `BB` loop add another 150 MB each; and
`Scene.model_image` builds a float64 plane over the full scene bbox
(`scene.py:1533`). The widest such buffer came from a 25-template scene whose
14 anchors spanned 18.8 Mpx.

Everything in the first family is read-only or write-once. Everything in the
second is needed only while the scene that owns it is being solved. That is the
whole opportunity.

## 2. The property that makes a fixed budget possible

Three facts, all already true of the code:

1. **The solve is exactly block-diagonal over scenes.** `generate_scenes`
   partitions templates by normal-equation coupling and each `Scene.solve`
   reads only its own templates and slices of the shared image and weights
   (`scene.py:961`). Nothing in a scene solve touches another scene's state.
   The block structure is exact by construction, not an approximation: the
   partition is what defines which couplings are carried.

2. **The partition depends on `ATA` alone, not on pixels.**
   `build_scene_tree_from_normal` and `merge_small_scenes` take the sparse
   normal matrix, the right-hand side, template positions and a bright mask.
   For 138k templates `ATA` has of order 5 M stored values — about 60 MB in
   CSR, against 6 GB of the stamps that produced it.

3. **`ATA` is an assembly of local integrals.** Entry `(i, j)` is
   `sum(T_i · T_j · w)` over the intersection of two stamps. Every contributing
   pixel lies inside the union of two bounding boxes, and a stamp is ~104 px on
   a side.

Together these say: the coupling graph can be assembled a tile at a time and
still be bit-exact, the partition then costs nothing, and the solve can be
streamed one block at a time. Peak memory is then set by the largest *block*,
not by the image.

That is the honest analogue of SExtractor's rolling row buffer. It is not the
same mechanism — SExtractor can stream in `y` because detection, deblending
and measurement are all local in `y`, whereas a mophongo scene can be six
arcminutes wide and its astrometric shift field is fitted across the whole of
it — but it delivers the same property.

## 3. Proposed decomposition

### Phase 1 — tiled template build and graph assembly

Iterate over tiles of the detection grid with a halo of one maximum stamp
side. Per tile: read detection science, segmentation map and detection inverse
variance for tile + halo; extract, extend, convolve and project the templates
whose *centre* falls in the tile core; append the stamps to an on-disk store;
accumulate COO contributions to `ATA` and `ATb`.

Reading a box is already implemented — `_read_image(path, box)` uses
`hdu.section` and places the result into a full-shape array so every pixel
coordinate, slice and WCS keeps its full-frame meaning (`pipeline.py:278`).
The trial-patch machinery (`trial_box_hi` / `trial_box_lo`) already scopes the
whole-array passes the same way.

Exactness of the tiled assembly: with a halo of at least one stamp side, any
pair whose stamps intersect has both stamps present in at least one tile.
Attribute each pair to the single tile whose core contains the centroid of the
pair's intersection, and every off-diagonal is accumulated exactly once. The
resulting matrix is identical to the global build, not an approximation of it.

This also disposes of a known defect: `build_normal` assembles a
138k × 138k `lil_matrix` entry by entry from Python (`scene_fitter.py:64`,
already in `TODO.md`) — about 400 MB of Python objects. Per-tile COO
accumulation into `int32`/`float64` arrays replaces it, and the tiles are
independent, so this phase parallelises trivially.

### Phase 2 — global partition

`build_scene_tree_from_normal` then `merge_small_scenes` on `ATA`/`ATb` alone.
No pixels, no stamps. A few hundred MB, seconds of runtime, and the partition
is identical to today's. Keeping this global is what preserves equivalence with
a full-field run — the reason the cheap alternative in §7 does not.

### Phase 3 — scene-streaming solve

Process scenes one at a time (or `W` at a time, §6), batched in spatial order
so the page cache stays warm. Per scene: load its stamps from the store, slice
the band science and inverse variance over its bbox, run the astrometric
iteration to convergence and the final flux-only re-solve, add its model into
the output residual, write its catalog rows, release the stamps.

One inversion was needed here, and it has since landed. The loop used to be
`for pass in 1..niter: for scene in pending:`, a barrier per pass for no
reason — scenes are independent *across* passes as well as within one. The unit
of work is now "one scene, run to convergence, including its final flux-only
pass", which removes every barrier and bounds the live template set to one
scene. Results are bit-identical; see `STATUS.md` for the equivalence check and
`tests/test_pipeline.py::test_scene_results_do_not_depend_on_scene_order` for
the invariant it rests on.

### The stamp store

A store already exists, and `8ca21f5` has since put it in the right shape:
`write_stamps` writes one flat float32 buffer per band into HDF5 with a
per-source offset table, and `read_stamps` / `_templates_from_stamps` restore
it. The offsets are exactly what a scene-local read needs, so one change is
left to make it working storage rather than an output — write incrementally as
tiles complete, instead of concatenating the whole buffer in memory first
(today that is a full extra copy of every stamp, 12 GB at full-field scale, at
the very end of the run).

## 4. Budget arithmetic

Phase 1, per worker, 4096 px tiles: three raster planes at 67 MB each is
0.20 GB; templates whose centres fall in the tile are ~2660 at full-field
density, 0.12 GB hi-res plus 0.12 GB convolved; COO accumulators grow to about
120 MB shared. Roughly **0.5 GB per worker**, and 0.15 GB at 2048 px tiles.

Phase 2: `ATA` in CSR ~60 MB, labels and catalog ~100 MB, working copies during
the merge a few hundred MB. Under **1 GB**, single-threaded.

Phase 3, per worker: one scene's stamps at `scene_max_size = 1000` is 43 MB,
doubled by `_data_unshifted` to 86 MB; the dense Schur complement is
1000² float64 = 8 MB; the band pixels come from a memory map (§5) and cost page
cache, which is reclaimable; `AB`/`BB` accumulated over row bands (§8) is a few
MB. Roughly **150 MB per worker** — against ~700 MB if the `Bq`/`Bl` buffers
are left whole-scene.

With a 1 GB floor for interpreter, libraries and PSF maps, eight workers land
near **2–5 GB total**, independent of field size. The 8 GB CANFAR jobs that
schedule instantly become sufficient, and the tile size and worker count are
the two knobs that trade runtime for the budget.

The cost is disk: two stamp sets (~12 GB) plus the residual (3.5 GB) as
scratch, and one extra full read of each input.

## 5. Raster discipline: memory-map what is read-only

Independently of the phase structure, most of the 17–21 GB of rasters need not
be anonymous memory at all.

`_read_image` currently reads into a full-shape anonymous array. Pages never
touched are never faulted in, which is what makes a trial patch cheap, but a
tile *sequence* eventually touches all of them, and dirty anonymous pages
cannot be evicted. Mapping the file instead (`np.memmap`, or `fits.open(...,
memmap=True)` kept open) makes those pages clean page cache: the kernel
reclaims them under pressure, and forked workers share them. `as_label_array`
already returns a memmap view when the segmentation map ships as an integer
type — UDS and EGS do; COSMOS ships float64 and needs a one-time conversion to
an int32 file, which the existing banded validation path can write streaming.

Two arrays need more than a change of allocator:

**The upsampled band.** `_upsample_boxed` materialises the band science and
inverse variance full-size on the reference grid — 7 GB for the pair, and the
exact point at which COSMOS died. But the upsampled image is a pure function
of the low-resolution pixels (block replicate, divide by `k²`; weights copied
and multiplied by `k²`). A scene solve only ever indexes it as
`image[slices_original]`. A thin object exposing `__getitem__` for 2-D slices
and upsampling the requested box on demand removes the array entirely, at the
cost of re-replicating a few hundred pixels per access. This is worth doing on
its own merits: it is the difference between COSMOS running and not.

**The residual.** `res` is allocated full-size, accumulated into, then
subtracted in place (`pipeline.py:3828`). Making it a `np.memmap` over the
output file's data section turns 3.5 GB of dirty anonymous memory into
file-backed pages that flush as they go, and `fits.writeto` at the end becomes
a header write.

## 6. Parallelising the scene solves

### Threads are the wrong tool — measured

Two microbenchmarks on this machine (10 cores, Accelerate BLAS, BLAS pinned to
one thread), in `scratch`-style throwaway scripts:

| work | K = 4 | K = 8 |
|---|---|---|
| `spsolve` + dense `inv` of the Schur complement | **2.14×** | **2.23×** |
| `build_normal`-shaped: Python loop over stamps, small `np.sum` per pair | **0.66×** | **0.47×** |

The linear-algebra tail does release the GIL and scales, sub-linearly. The
assembly does not: threading it makes it *slower*, because the per-operation
Python and GIL overhead of thousands of small `np.sum` calls dominates and the
threads contend. Since `Scene.solve` rebuilds `A`/`b` on every astrometric pass
(`scene.py:1000`, after `self.A, self.b = None, None` at line 1072), assembly
is not a small share of the scene loop. Threads are out.

For reference on the same synthetic 1000-template scene: `spsolve` 2.5 ms,
`splu` 2.4 ms, the sparse `_flux_errors` path (`n` unit back-solves) 45 ms, the
dense path 18 ms — consistent with the dispatch fix already recorded in
`STATUS.md`.

### Processes, over memory-mapped inputs

Each worker takes a scene id, maps the same input files, reads that scene's
stamps from the store, and returns fluxes, errors, shift coefficients,
accumulated per-template shifts, and its model patch with its bbox. The parent
adds model patches into the residual — scene bounding boxes overlap even though
their template sets do not, so accumulation must not race, and having the
parent own that write is simpler than locking.

Memory maps matter more than they look here. On Linux, `fork` gives
copy-on-write and workers see the parent's arrays for free; on macOS the
default start method is `spawn` and they do not. Mapping files makes the design
identical on both, and makes the page cache shared rather than duplicated.

Set `OMP_NUM_THREADS=1` in workers, or `W` workers each spawning a BLAS pool
will oversubscribe.

Schedule largest scene first. Scene sizes span 2 to ~1000 with a median near
200, and cost is superlinear in size (the joint path forms a dense `n × n`
Schur complement), so a small-first order strands the largest scene in the
tail.

`joblib` is already a dependency, and `PSFRegionMap` was made picklable in an
earlier pass explicitly for user-side multiprocessing (`TODO.md`), so the
plumbing is mostly in place.

### Memory impact of parallelism

`W` workers multiply the per-scene peak, so **the `Bq`/`Bl` buffers must be
bounded before parallelism is turned on**. At 0.61 GB worst case, eight workers
can spike 4.9 GB on top of everything else — enough to defeat the budget the
rest of this note buys. Section 8 lists the fix; it is small and independent.

### Expected return

Unquantified, and it should be measured before it is promised. 591 scenes over
8 workers is plenty of parallelism, but the 87-minute full-field run predates
any timing instrumentation. `56530f2` has since added the outer breakdown —
template build, convolution, scene generation, the astrometry passes, the final
flux solve, the residual — which is half of what is needed. The other half is
the split *within* a scene solve, across the three calls `build_normal`,
`assemble_scene_system_AB` and the factorisation, since that is what says how
much of the loop is the GIL-bound assembly measured below. Re-run a full field
with both and read the answer off the log before building any of this.

## 7. The cheap alternative, and why it is not equivalent

Everything above could be skipped by tiling the *whole* pipeline: run the
existing code on each tile with a halo, keep the sources whose centres fall in
the core, and concatenate the catalogs. The `trial` geometry already does
exactly this for one patch, so it is nearly free.

It is not equivalent, for two reasons the code already documents. Background
and inverse-variance calibration is per-box: `load_data` warns that a trial
patch's fluxes and errors will *not* match a full-field run, because the
robust baseline in `get_bg_and_ivar` is measured over the box. And a scene
straddling a tile boundary is truncated in both tiles, so its astrometric shift
field — fitted jointly across all the scene's anchors — is fitted on a
fragment. Fluxes would differ by more than the coupling threshold bounds.

Two mitigations make it defensible if the exact route is too much work:
calibrate background and weights once, globally (that pass is already banded
and streaming — `_valid_block_means`, `catalog.py:571`), and pass the scalars
into every tile; and choose tile boundaries from the *scene partition* rather
than a fixed grid, which requires phases 1 and 2 anyway. At which point phase 3
is the smaller remaining step, and it is exact.

## 8. Prerequisites, independent of the rest

Each of these is small, is worth doing on its own, and blocks something above.

- **Chunk `assemble_scene_system_AB` over row bands.** `Bq`/`Bl` and the
  `Bq[k] * wbuf` temporaries are the only per-scene term that scales with
  spatial extent. Accumulate `BB`, `bB` and `AB` band by band and the buffers
  become `nB × band_height × width`. Already flagged in `STATUS.md` as the
  candidate for the unexplained spike between checkpoints; blocks §6.
- **`Scene.model_image` in float32**, or accumulated straight into the residual
  in the band's dtype. 150 MB per wide scene, ×`W` under parallelism.
- **Bound scene *extent*, not only scene size.** `scene_max_size` binds
  template count; the widest buffer came from a 25-template scene spanning
  18.8 Mpx. A bbox-area cap in the split, or the row-chunking above, closes it.
- **Incremental stamp writing.** `write_stamps` builds the complete `vla` list
  before writing — a full extra copy of every stamp, at the end of the run.
- **`ATA` assembly in COO, not `lil`.** Already in `TODO.md`; prerequisite for
  the tiled build.
- **Phase timers.** In `run()` and in `Scene.solve`. Prerequisite for deciding
  any of this on evidence rather than on the arithmetic above.

## 9. Open questions

- Does the tiled `ATA` assembly reproduce the whole-field matrix bit for bit
  in practice? The argument in §3 says it must; it needs a test on a real field
  (compare CSR structure and values, then compare the scene partition).
- What is the phase split of a full-field run, now that `run()` reports one?
  Everything about the parallelism payoff depends on it, and on the
  within-scene split that is still missing.
- Is the on-demand upsampled-band view fast enough, or does re-replicating on
  every stamp access cost more than it saves? A cache of the last few boxes
  probably settles it, but it should be measured against the existing
  `multi_resolution_method: downsample` path, which avoids the upsample
  entirely and is 4× cheaper in template memory but diverges from the path v8
  and v9 validated.
- Where do saturated-star scenes fit? Their templates carry PSF wings far
  beyond the segment, so their bboxes are the widest in the field and they are
  the most likely to blow a per-worker budget.
