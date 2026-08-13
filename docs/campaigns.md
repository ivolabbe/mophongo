# Running a campaign: CANFAR and OzStar

A *campaign* is one command that fits every band of every MINERVA field on a
cluster. Two toolkits do this — `examples/canfar/` for CANFAR and
`examples/ozstar/` for Swinburne's Ngarrgu Tindebeek — and they are deliberately
similar: the same configs, the same pipeline, the same two-phase submission.
What differs is the platform underneath, and this page is mostly about where
those differences force the design.

Written against `7784f99`.

## The shape of a campaign

Both toolkits run the same chain of steps, and both let you enter it partway
with `--from` or drop a step with `--skip`:

| step | what it does |
|---|---|
| `arcify` / `ozify` | rewrite the local run configs into cluster paths |
| `push` | upload source, PSF grids, configs |
| `setup` | build the virtualenv on the cluster |
| `seed` | link PSF and kernel maps from a previous run's directory |
| `stage` | copy or decompress the inputs into the run tree |
| `prep` | **build what a field's bands share, once per field** |
| `run` | one fit per band, all bands at once |

`python campaign.py --dry-run` prints the whole plan and submits nothing. Use
it. Every example below is real `--dry-run` output.

## Why there is a prep phase

A field's bands are not independent, even though the fits are. Three things
belong to the field rather than to any band:

- the **F444W ePSF grids**, matched by `pattern_hi` — every band of a field
  fits against the same detection PSF;
- the **30" halo grids**, when `repair_saturated` is on — their pattern is
  derived from `pattern_hi`, so they are shared the same way;
- the **saturation repair**. `Pipeline._repair_provenance` keys the cache on
  the detection image, its weight, `pattern_hi`, the halo pattern,
  `repair_kwargs` and the trial box. Nothing in that varies between bands.

Submit a field's bands together without preparing any of it and every band
rebuilds the same grids and re-runs the same repair. Worse, they do it at the
same time: several jobs write the same grid filenames into one `psf_dir`, and
several write the same repair cache file. A half-written cache read by another
job used to be fatal — `_load_repair_cache` now treats an unreadable file as
absent and recomputes, but the wasted work remains.

So a campaign submits in two phases:

1. **prep** — one short job per field, every field at once. Nothing else may
   start until it lands.
2. **run** — every band of every field, fired off together, not waited on.

Phase 1 costs one job per field and buys a clean parallel fan-out.

## The F444W race, and why prep is one band per field

Everything about the two-phase shape follows from one fact: **a field's bands
all derive the same `pattern_hi`**, so they all want to build the same F444W
grids into the same `psf_dir`, under the same filenames.

The grid build itself parallelises cleanly. One `(detector, date)` pair is one
independent job writing one uniquely named file, and each worker costs the
stpsf wavefront propagation — tens to low hundreds of MB, nothing that scales
with the field. `PSFFactory(workers=N)` fans those out, and
`RunConfig.psf_workers` reaches it from a config.

What does **not** parallelise is two *bands of the same field* building at
once. `uds_f770w` and `uds_f1000w` both resolve `pattern_hi` to
`UDS_NRC.._F444W_MJD\d+_GRID25_OS4`, so both compute the same grids and both
write the same paths. The same applies to the 30" halo grids, whose pattern is
derived from `pattern_hi`. Interleaved writes to one FITS file give a torn
file, and the loser's work is wasted either way.

So the rule is: **serialise across patterns, parallelise within one.** That is
exactly what prep does — one band per field, alone, building the shared
products — and it is why `jobs/build_psfs.sh` walks its configs one at a time
while `--workers` fans out inside each.

Prep runs **F770W** where a field has it (`campaign.py`'s `prep_leader`). It is
the shortest MIRI band, so it is the cheapest job that still builds everything
shared. The band's own `pattern_lo` grids have per-band names and never
collide, so those are safe to build concurrently across bands afterwards.

## The pipeline steps behind it

`python -m mophongo.pipeline <config> <step>` takes any key of
`mophongo.pipeline.STEPS`. Two exist for prep:

- **`prep`** — `build_psfs()` then `build_repair_cache()`. One job.
- **`repair`** — `build_repair_cache()` alone: `load_data(kernels=False)`,
  which runs the repair and writes the cache. `kernels=False` stops it building
  a matching-kernel map, which is per-band and not shared.

Both stop before the fit. `repair` is a no-op when `repair_saturated` is off.

The split exists because of a platform difference, below.

## CANFAR

Compute containers have internet, so `stpsf` can query MAST for each exposure's
wavefront OPD and grids can be built wherever the job lands. Prep is therefore
one job:

```
$ python campaign.py --from prep --dry-run
campaign over 17 config(s): cosmos_f1000w, ..., uds_f770w
prep: 3 field(s), cosmos_f1000w, egs_f1000w, uds_f1280w
+ submit.py run cosmos_f1000w egs_f1000w uds_f1280w --step prep --ram 48
+ submit.py run cosmos_f1000w cosmos_f1280w ... cosmos_f770w --ram 48 --no-wait
+ submit.py run egs_f1000w egs_f1280w ... egs_f770w --ram 48 --no-wait
+ submit.py run uds_f1280w uds_f1500w uds_f1800w uds_f770w --ram 48 --no-wait
```

The first `submit.py run` has no `--no-wait`, so the laptop blocks until all
three prep jobs finish; the three that follow return immediately. This is the
one place a campaign waits. Fits never do: jobs live on CANFAR once submitted,
and a campaign must not depend on a laptop-side process staying alive.

Compute is the skaha REST API, not ssh. `submit.py status` lists sessions,
`submit.py logs <id>` tails one, `submit.py fetch <name>` pulls the small
outputs down.

## OzStar

Three node classes, and no single one can do everything:

| node | internet | `/apps` modules |
|---|---|---|
| login | yes | yes |
| datamover | yes | **no** |
| compute | **no DNS, no route** | yes |

Building an ePSF grid needs both: `stpsf` resolves each exposure date to an OPD
by querying MAST, and mophongo needs the module stack. Only the login node has
both, so grid building cannot be a SLURM job at all. `submit.py psfs` runs
`jobs/build_psfs.sh` there, detached with `setsid nohup` because it runs for
hours and an earlier attempt died with the ssh connection that started it.

That is why the pipeline has a `repair` step separate from `prep`: on OzStar
the two halves of preparation run on different machines. Grids come from the
login node beforehand; the repair is an ordinary SLURM job.

Waiting is also different. Where CANFAR blocks the laptop, OzStar expresses the
dependency to the scheduler — each band job is submitted with
`--dependency=afterok:<the field's repair job>`, so nothing runs early and the
laptop is free:

```
$ python campaign.py --fields uds --from prep --dry-run
run tree /fred/oz030/ilabbe/run on ilabbe@nt.swin.edu.au
+ repair uds (uds_f1280w) after nothing
+ run uds_f1280w uds_f1500w uds_f1800w uds_f770w after <uds-repair>
```

`--prep-time` sets the repair job's walltime (default 4 h) separately from
`--time` for the fits (default 24 h).

## The repair cache

`arcify.py` and `ozify.py` write `repair_cache_path` next to `out_dir`, named
for the field and the patch geometry:

```
uds_f770w   (full field)     ../uds_full_repair_cache.fits
uds_f1800w  (full field)     ../uds_full_repair_cache.fits   same file
cosmos_f770w(full field)     ../cosmos_full_repair_cache.fits
uds_f770w   (r = 3')         ../uds_r3_34.38792-5.30102_repair_cache.fits
uds_f770w_test (r = 0.25')   ../uds_r0.25_34.38792-5.30102_repair_cache.fits
```

Bands of a field at one geometry share a cache, which is the point. Nothing
else does, which matters just as much: `RunConfig.repair_cache_path` defaults
to `'..'`, one unnamed file for the whole run tree, and a release campaign has
three fields and a dozen patch geometries whose provenance never matches. Each
would find the others' cache stale, recompute, and overwrite it.

## These directories hold no outputs

`examples/canfar/` and `examples/ozstar/` are tracked in full — launch scripts,
docs, and the generated per-band configs and staging lists. They stay
lightweight: all text, well under a megabyte, so a campaign's exact inputs are
in the history.

Nothing a command produces is written there. `submit.py fetch` writes to
`scratch/<toolkit>/out/<name>/`, and upload scratch to
`scratch/<toolkit>/_upload/`, both gitignored. The `out/` and `_upload/` names
are ignored inside the staging directories too, so an output cannot creep back
in by accident.

## Reading the results

`submit.py fetch <name>` brings back the fit table, the scene catalog and the
log. Residuals and stamps are multi-GB and stay on the cluster; pull them by
hand if you need them.

For what the outputs mean, see {doc}`outputs`. For the fit itself, see
{doc}`pipeline`. For the saturation repair, see {doc}`repair`.
