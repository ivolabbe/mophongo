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

The venv persists on `/arc/home/<user>/run/venv`, so this is paid once. The
stock image's astropy and numpy are too old for mophongo, hence a clean venv
rather than `--system-site-packages`.

## Per run

```bash
$P arcify.py ../minerva/uds_f770w.json        # rewrite paths for arc
$P submit.py stage uds_f770w                  # decompress its inputs on arc
$P submit.py run   uds_f770w --cores 8 --ram 64
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
$P submit.py run uds_f770w_test --cores 4 --ram 64
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
/arc/home/<user>/run/
├── venv/                  mophongo and its dependencies
├── mophongo/              the uploaded source
├── PSF/                   STPSF MJD-tagged grids
├── jobs/                  setup_env.sh, stage.sh, run.sh
├── data/                  decompressed inputs, shared between bands
├── <name>_canfar.json     rewritten configs
├── <name>_stage.tsv       per-config copy lists
└── out/<name>/            run outputs
```

It is all under the user's own home rather than the shared project space, which
is read-mostly for the collaboration.

## Notes

- Compute is the skaha REST API, not ssh. The transfer endpoint on port 64022 is
  SFTP only and cannot execute anything.
- The installed skaha client defaults to API `v0`, which 404s; `submit.py` pins
  `v1`.
- Job `args` are whitespace-split into a YAML sequence server side and quotes
  cause a 500, so the command must be a single token. Parameters go through
  environment variables (`RUN`, `CFG`) instead.
- Up to 16 cores. The quota page reports a 32 GB default, but that is a default
  and not a cap: 64 GB is an allowed request and is what these scripts use.
- Importing mophongo from the NFS-backed venv costs about three minutes before
  any work starts. That is not a hang.
- For a single trial patch CANFAR is not faster than a laptop; the gain is that
  the data are already there and bands can run as concurrent jobs.

Background, and the traps found getting the first run working, are in
`scratch/canfar/RUNNING_ON_CANFAR.md`. Data access from a laptop (scp, vsync,
sshfs) is in `MINERVA/data/00CANFAR`.
