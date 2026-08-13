# Running mophongo on CANFAR

The MINERVA release lives on `arc:projects/minerva` and the CANFAR Science
Platform mounts `/arc` inside every container, so a run there needs no data
transfer at all: mosaics, segmaps and catalogs are read in place or decompressed
on arc itself. Only the mophongo source and the job scripts have to be uploaded.

`push` also ships the STPSF PSF grids (about 290 MB) as a convenience, since
they already exist locally. That is an optimisation, not a requirement:
`psf_autobuild` is on by default, so a run with an empty `psf_dir` generates the
grids from the exposure lists on arc and caches them there. Uploading just skips
a slow first run.

## Where runs live

The run root defaults to `/arc/projects/minerva/ifl` and nothing needs to be
exported. A run writes about 12 GB per band — a 3.5 GB residual and 8–11 GB of
stamps — so a 17-band release lands near 200 GB, and staged inputs alone are
61 GB. `/arc/home` carries a few hundred GB for everything a user owns, and
has already been filled this way once.

A tree under `/arc/home` is therefore **refused**, not warned about: the
failure it causes is silent until a quota stops a campaign halfway through.
Point the tree elsewhere under `/arc/projects` if you want to:

```bash
export CANFAR_RUN=/arc/projects/minerva/ifl_test
```

Project space is shared with the collaboration and has no comparable limit, so
that is where campaigns belong. Both
`submit.py` and `arcify.py` read `CANFAR_RUN`, and the arc paths baked into a
rewritten config come from it — so if you change it, re-run `arcify.py`.

## Everything at once

```bash
P=~/.venvs/canfar/bin/python
$P campaign.py                      # every config in ../minerva, trial patches
$P campaign.py --fields uds cosmos  # only those fields
$P campaign.py --from stage         # already pushed and built
$P campaign.py --dry-run            # print the plan, run nothing
```

A full-field campaign over the whole release, reusing the PSF and kernel maps
the patch runs already built rather than spending half an hour per band
rebuilding them:

```bash
$P campaign.py --r-trial 0 --suffix _full --seed-from ""
```

`--suffix` keeps the outputs in their own directories, `--seed-from ""` links
the caches across from the unsuffixed runs, and `--r-trial 0` disables the
trial-patch cut so every source in the footprint is fitted.

It runs upload, environment, config rewrite, staging and submission in order,
and encodes three things worth not rediscovering: runs are submitted without
waiting, so nothing depends on a local process surviving; staging does one band
per field first, since bands share their field's mosaic and segmap; and a field
with no F444W grids sends one band alone to build them, because bands of a field
would otherwise build the same grids concurrently into one `psf_dir`.

Then watch with `submit.py status` / `logs` / `fetch`. The steps below are the
same thing done by hand.

## Prerequisites

A CADC proxy certificate, valid 10 days:

```bash
../../scratch/canfar/canfar-cert.sh          # prompts for the CADC password
```

`skaha` and the `vos` tools live in `~/.venvs/canfar`. Everything below runs
with that interpreter:

```bash
P=~/.venvs/canfar/bin/python
```

## One-time setup

```bash
$P submit.py push        # mophongo source, job scripts, PSF grids
$P submit.py setup       # build the venv on /arc from pyproject.toml
```

The venv persists at `$CANFAR_RUN/venv`, so this is paid once. The
stock image's astropy and numpy are too old for mophongo, hence a clean venv
rather than `--system-site-packages`.

## Per run

```bash
$P arcify.py ../minerva/uds_f770w.json        # rewrite paths for arc
$P submit.py stage uds_f770w                  # decompress its inputs on arc
$P submit.py run   uds_f770w
$P submit.py fetch uds_f770w                  # pull the small outputs down
```

Several bands at once: pass them all. `stage` runs the first alone and the rest
concurrently, because bands of a field share the F444W mosaic and the segmap and
there is no point decompressing several GB more than once.

```bash
$P arcify.py ../minerva/uds_f*.json
$P submit.py stage uds_f770w uds_f1280w uds_f1500w uds_f1800w
$P submit.py run   uds_f770w uds_f1280w uds_f1500w uds_f1800w
```

A cheap smoke run on a small patch, without touching the source config:

```bash
$P arcify.py ../minerva/uds_f770w.json --r-trial 0.25 --suffix _test
$P submit.py run uds_f770w_test
```

`submit.py status` lists sessions, `submit.py logs <id>` prints one job's output
(also while it is running).

## What arcify.py does

It indexes the relevant `arc:projects/minerva` subtrees, then rewrites every
input path in a local `RunConfig` to its arc equivalent, resolving three
mismatches the pipeline would otherwise trip over:

- most files on arc are gzipped and the pipeline wants plain FITS, so those are
  listed in `<name>_stage.tsv` and decompressed once into `run/data`;
- uncompressed inputs (the SUPER catalog, the F444W frame table) are pointed at
  in place and never copied;
- MIRI frame tables ship as `*_f770_wcs.csv` while the filter parser expects the
  `f770w` spelling, so they are copied under the expected name.

It also repoints `psf_dir` at the uploaded grids and `out_dir` into the run tree.

## Layout on arc

```
$CANFAR_RUN/                        e.g. /arc/projects/minerva/ifl
├── venv/                  mophongo and its dependencies
├── mophongo/              the uploaded source
├── PSF/                   STPSF MJD-tagged grids, shared by every run
├── jobs/                  setup_env.sh, stage.sh, run.sh, seed_cache.sh, ...
├── data/                  decompressed inputs, shared between bands
├── <name>_canfar.json     rewritten configs
├── <name>_stage.tsv       per-config copy lists
└── out/<name>/            run outputs
```

`/arc/home/<user>` keeps only `.ssh`, `.ssl` and a README pointing here: its
quota is far too small for a campaign's outputs, which is why `runroot.py`
refuses a run tree there.

## Notes

- Compute is the skaha REST API, not ssh. The transfer endpoint on port 64022 is
  SFTP only and cannot execute anything.
- The installed skaha client defaults to API `v0`, which 404s; `submit.py` pins
  `v1`.
- Job `args` are whitespace-split into a YAML sequence server side and quotes
  cause a 500, so the command must be a single token. Parameters go through
  environment variables (`RUN`, `CFG`) instead.
- Runs request 4 cores and 48 GB by default. Measured CPU use is about 0.2 of
  a core — the runs wait on `/arc` and the fitting path has no thread pool — so
  the extra cores are headroom rather than throughput, and a bigger request
  takes longer to schedule when the platform is busy. `--cores` and `--ram`
  override; up to 16 cores and 192 GB are available.
- Outputs are large. A 3 arcmin patch of UDS writes 8.4 GB, of which 4.2 GB is
  `stamps.fits` and 3.5 GB the residual; a full field scales with the source
  count, which is roughly 8x. Set `save_stamps` or `scene_plots` false in the
  config when the diagnostics are not wanted.
- `seed` links rather than copies the PSF and kernel maps. Copying them once
  duplicated 10 GB and exhausted a home quota.
- Session names come back with a `-1` replica index appended, so the job named
  `mophongo-uds-f770w-v1-0` lists as `mophongo-uds-f770w-v1-0-1`.
- A submitted session is not listed until the service registers it, which takes
  minutes. `status` right after a `--no-wait` campaign under-reports, and
  destroying "everything" once leaves stragglers that surface later carrying
  the same run names as the next campaign and writing into the same `out/`
  directories. `submit.py kill` sweeps until several consecutive passes come
  back empty; it spares the `sync` job by default.
- Queue latency dominates small work, and the sshfs mount is *writable*, so
  file movement should not be a container job at all. A 1-core `sync` has sat
  Pending for half an hour to do seconds of copying; through the mount the same
  unpack takes about twenty seconds. `sync` therefore uses the mount when it
  finds one — `$CANFAR_RUN_LOCAL`, or `~/canfar_projects`/`~/canfar_home` — and
  `--job` forces a container. Reserve jobs for work that needs one. The caveat
  is the same either way: rewriting source under a running job is only safe
  because already-running jobs keep the code they imported.
- CANFAR always runs a commit. `push` ships `git archive` of `main` by
  default, never the working tree, so another session's half-finished edit or
  an editor mid-save cannot reach 17 jobs. `--ref` picks a different commit and
  `--worktree` ships the uncommitted tree for debugging, saying so loudly.
- The version is recorded end to end. `push` uploads `SRC_VERSION.pending`;
  `setup_env.sh`/`update_src.sh` promote it to `SRC_VERSION` *after* unpacking,
  so the file means "this is what is installed" rather than "this is what was
  uploaded"; `run.sh` prints it at the top of every job log; and `run` refuses
  to submit unless it matches the local ref (`--force-stale` overrides). The
  case this catches is a `push` with no `sync`: nothing else unpacks the
  tarball, so the jobs quietly import the previous campaign's code and the
  outputs look entirely normal.
- Only `setup_env.sh` unpacks `psf.tar`. Pushing it before a `sync`, which
  replaces the source alone, uploads several hundred MB that never reach
  `$RUN/PSF` — use `push --src-only` for a code change. Grids on arc are never
  at risk regardless: `PSFFactory` skips any grid file that already exists
  unless `overwrite` is set, so they are built once and reused.
- `campaign.py` serialises one band of a field ahead of the rest when the
  shared F444W or 30" halo grids are missing, since concurrent bands would race
  on one `psf_dir`. It counts grids already on arc, not just local ones: a
  field whose grids an earlier job built needs no leader, and serialising one
  at full-field scale costs hours.
- The quota page reports a 32 GB memory default, but that is a default and not
  a cap. A 16 GB request is OOM-killed with no traceback, which reads as a
  mysterious silent failure rather than an error.
- Full-field memory tracks the *source count*, not the field's pixel count,
  and 48 GB is below the line for any deep band. Measured on EGS full field
  (`scene_plots` off), peak at the end of the fit against the stamps written
  afterwards:

  | band | peak | sources | stamps |
  |---|---|---|---|
  | f560w | 29.7 GB | 23,125 | 1.9 GB |
  | f1800w | 33.3 GB | 48,296 | 4.0 GB |
  | f2100w | 46.1 GB | 131,416 | 10.8 GB |
  | f770w | 47.9 GB | 140,412 | 11.6 GB |
  | f1000w | 47.9 GB | 142,299 | 11.7 GB |

  The heavy bands reach 46-48 GB *before* writing 11 GB of stamps, so a 48 GB
  request dies in the output stage with no traceback. `ram_for` therefore asks
  for 64 GB, and 82 for EGS. Larger requests do schedule, but 48 GB and up have
  queued for hours when the platform is busy, so raising the request everywhere
  costs wall clock.
- `scene_plots` is the other half of that budget. Rendering Lupton RGB
  composites for several hundred scenes, on top of everything the fit still
  holds, killed all ten cosmos/uds bands of the first full-field campaign
  after their fit tables and stamps were safely written. Turning it off saves
  twice: the plots themselves, and the band's inverse-variance map, which
  `run()` releases early when `_scene_pixels_needed()` is false - worth 1.5 GB
  on EGS F1000W (49.4 -> 47.9 GB).
- Importing mophongo from the NFS-backed venv costs about three minutes before
  any work starts. That is not a hang.
- For a single trial patch CANFAR is not faster than a laptop; the gain is that
  the data are already there and bands can run as concurrent jobs.

Background, and the traps found getting the first run working, are in
`scratch/canfar/RUNNING_ON_CANFAR.md`. Data access from a laptop (scp, vsync,
sshfs) is in `MINERVA/data/00CANFAR`.
