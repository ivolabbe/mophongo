# Scene partition

## Current algorithm (original two-step + optional local size cap)

1. **Components at the coupling threshold.** `build_scene_tree_from_normal`
   scores template pairs by predicted cross-leakage of their diagonal-only
   fits, `score_ij = max(|A_ij a_j|/(A_ii|a_i|), |A_ij a_i|/(A_jj|a_j|))`,
   draws an edge where `score >= scene_coupling_thresh`, and takes connected
   components.
   With `scene_max_size` set (default None = off), a component over the cap
   is split by bisecting over *its own* edge scores until its pieces fit —
   the threshold is raised only inside that component, the accepted local
   leakage is logged, and strong couplings elsewhere are never touched.
2. **Merge small scenes.** `merge_small_scenes` merges scenes with fewer than
   `scene_minimum_anchors` bright members (SNR proxy above
   `astrom_minimum_snr`) into their nearest scene by centroid, iterating until
   all scenes clear the floor. (Unchanged original; it has no cap, so merged
   scenes can exceed `scene_max_size` where bright sources are scarce.)

Knobs: `scene_coupling_thresh` (floor), `scene_minimum_anchors`, and the
optional `scene_max_size`.

## Template support sets the coupling graph (the wren discrepancy)

Coupling range = template support after convolution = segment (+) kernel
stamp, and the kernel stamp is `psf_size`. This repo's `psf_size: null`
gives 8" native stamps: every template overlaps everything within 8", the
graph percolates, and the UDS F770W 0.6' trial patch is one 2241-template
component at the 1e-3 floor. At wren's `psf_size = 3.0` the same code and
threshold reproduce wren's structure — including the giant: the trial patch
is the deepest region of the mosaic, and wren's own scene there is 1738
templates (its ~200-source scenes are typical-field, not dense-region).
Measured on the same patch, cost of breaking the giant with `scene_max_size
= 300`:

| stamps | local threshold needed | note |
|---|---|---|
| 8" (`psf_size: null`) | 0.16 | bright fluxes move 1-3% (nmad 0.45 sigma) |
| 3" (`psf_size: 3.0`) | 0.0289 | below 0.044, where partition changes moved fluxes < 0.01 median, nmad 0.13 sigma |

`uds_770_dr0.1_ps3.json` (3" stamps, cap 300, floor 1e-3, min bright 5,
r_trial 0.6') yields 9 compact scenes, 75-628 templates, 5-8 bright each.
Caveat before adopting 3" stamps for science: the `flux_<i>_total`
stamp-sum bug in `TODO.md` biases totals +4.9% at 3.0" (vs -0.7% at null) —
fix that first.

## 2026-08-08 experiments: what was tried and why it was reverted

A session attempted to replace the per-band threshold with size-driven
partitioning (`scene_max_size`). Sequence of designs, each fixing the
previous one's failure on the UDS F770W dr0.1 trial patch (1549 templates,
0.5' circle, 39 bright at `astrom_minimum_snr = 15`, `psf_size: null` 8"
stamps):

| design | failure |
|---|---|
| global threshold bisection to cap | at cap 300 raised the threshold to 0.16 field-wide, severing bright pairs everywhere (766 tree scenes) |
| merge: nearest-neighbour, cap-refusing | stranded a 95-template scene at 0 bright when its one candidate breached the cap |
| merge: (bright, distance) ranked anchors | bright count dominated, distance only tie-break: long-range flights, elongated scenes, one scene nested inside another |
| local per-component bisection | confined the cut to the giant (0.106-0.16 inside, 1e-3 elsewhere) but its subcomponents are percolation clusters — inherently ragged/dendritic |
| merge: strongest-cut-coupling | single-linkage clustering: straggly chains |
| (not run) sum-coupling merge, k-d spatial split | reverted before trial — corrections upon corrections |

Durable lessons, independent of any design:

- The hand-tuned per-band threshold ladder collapses to a single constant
  0.030 +/- 0.005 once divided by median source SNR and PSF area: the
  threshold is a proxy for graph mean degree (percolation) with a 1/SNR
  factor because the score is fractional-in-flux rather than in sigma.
- The coupling graph's density is set by template support. This repo's
  `psf_size: null` (8" stamps) percolates at any usable threshold — the
  F770W giant only breaks apart above ~0.1, where cut couplings bias bright
  fluxes by 1-3% (measured: nmad 0.45 sigma, max 12 sigma vs a single-scene
  solve). wren's 3" stamps gave a graph sparse enough for a fixed 1e-3 over
  the full field with scenes of 100-1800 templates. Template support, not
  the partition algorithm, is the lever for small scenes.
- Partition changes at low cut levels do not move photometry: 1 vs 3 vs 2
  scene runs at thresh <= 0.044 agree to nmad 0.13 sigma at SNR > 10.
- Compactness matters physically: astrometric offsets vary on ~arcmin
  scales, so an elongated scene is internally misaligned where a compact
  blob of the same membership is not.
- Scenes below `minimum_bright` cannot constrain their shift block
  (`build_scene_ab_blocks` returns an empty block below 2 bright; the scene
  solves flux-only).

The experimental code is preserved in `git stash` ("session scene-partition
experiments (reverted)").

## Known inconsistency

`fit.py:1134` (the `SparseFitter.solve` path, not the pipeline
`generate_scenes` path) calls its own copy of `build_scene_tree_from_normal`
with `coupling_thresh=1e-4` hardcoded, ignoring `cfg.scene_coupling_thresh`.
Scene partitions from that path do not honour the config. Tracked in
`TODO.md`.
