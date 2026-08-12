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

Set the run root before anything else. The default is `/arc/home/<user>/run`,
which is fine for trial patches but has a quota of a few hundred GB — one
full-field campaign over a release needs several times that, and staged inputs
alone are 61 GB.

```bash
export CANFAR_RUN=/arc/projects/minerva/ifl
```

Project space is shared with the collaboration and has no comparable limit, so
that is where campaigns belong; a home run tree is for experiments. Both
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
$P campaign.py --r-trial 0 --suffix _full --seed-from "" --cores 4 --ram 64
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
$P submit.py run   uds_f770w --ram 64
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
$P submit.py run uds_f770w_test --ram 64
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
quota is far too small for a campaign's outputs.

## Notes

- Compute is the skaha REST API, not ssh. The transfer endpoint on port 64022 is
  SFTP only and cannot execute anything.
- The installed skaha client defaults to API `v0`, which 404s; `submit.py` pins
  `v1`.
- Job `args` are whitespace-split into a YAML sequence server side and quotes
  cause a 500, so the command must be a single token. Parameters go through
  environment variables (`RUN`, `CFG`) instead.
- Cores are chosen per config: 4 for a full field, 1 for a trial patch, since
  measured CPU use is about 0.2 of a core and the runs wait on `/arc`. Pass
  `--cores` to override. Up to 16 are available.
- Outputs are large. A 3 arcmin patch of UDS writes 8.4 GB, of which 4.2 GB is
  `stamps.fits` and 3.5 GB the residual; a full field scales with the source
  count, which is roughly 8x. Set `save_stamps` or `scene_plots` false in the
  config when the diagnostics are not wanted.
- `seed` links rather than copies the PSF and kernel maps. Copying them once
  duplicated 10 GB and exhausted a home quota.
- The quota page reports a 32 GB memory default, but that is a default and not a
  cap: 64 GB is an allowed request and is what these scripts use.
- Importing mophongo from the NFS-backed venv costs about three minutes before
  any work starts. That is not a hang.
- For a single trial patch CANFAR is not faster than a laptop; the gain is that
  the data are already there and bands can run as concurrent jobs.

Background, and the traps found getting the first run working, are in
`scratch/canfar/RUNNING_ON_CANFAR.md`. Data access from a laptop (scp, vsync,
sshfs) is in `MINERVA/data/00CANFAR`.
