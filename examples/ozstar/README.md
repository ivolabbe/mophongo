# Running mophongo on OzStar

The counterpart of `../canfar`, for Swinburne's OzStar / Ngarrgu Tindebeek.
The difference that shapes everything here is that OzStar has no view of the
MINERVA release: on CANFAR `/arc` is mounted inside the container and most
inputs are read in place, while here every file a config names is copied from
CANFAR onto `/fred` first. What you get in exchange is a real batch system —
runs are `sbatch` jobs chained with `--dependency`, so a campaign is one
dependency graph submitted in a single command and nothing afterwards depends
on the laptop staying awake.

## Layout

Everything stable sits *above* the run directory; a run holds only what that
version of the catalog produced.

```
$OZSTAR_BASE/               default /fred/<project>/<user>/mophongo
├── bin/                job scripts, shared by every run
│   └── venv-vos/         CADC transfer tools (OS python)
├── PSF/                MJD-tagged ePSF grids
├── data/               inputs staged from CANFAR arc, including the wcs csvs
└── run3/               one catalog version      $OZSTAR_RUN
    ├── README.md         what this run is and how it differs from the last
    ├── config/           <name>_ozstar.json, <name>_stage.tsv
    │   ├── mophongo/       this run's clone — git here is the version
    │   └── venv/           this run's environment (module python)
    ├── logs/             SLURM logs of jobs with no single output dir
    └── <field>/          uds, cosmos, egs
        ├── <field>_repair_cache.fits
        └── <field>_<band>/   the fit's out_dir, and its own SLURM log
```

**The version is the run directory, never a name suffix.** Outputs are
`run3/uds/uds_f770w`, not `run3/uds/uds_f770w_v3`: a suffix would repeat the
version in every path and every output filename, and make two attempts at one
release impossible to compare without renaming files. `ozify.py` refuses a
`--suffix` that looks like a version; bump `$OZSTAR_RUN` instead. `--suffix`
remains for genuine variants, such as a `_trial` patch beside the full field.

Data and grids are shared because a release is re-fitted many times against the
same 64 GB of inputs and the same ~500 grids, and those do not change between
runs. It also means deleting a run destroys only that run — which was learned
the hard way, when consolidating two runs left the entire campaign in one
directory and a single `rm -rf` took all of it.

**The clone and the science venv are per run**, under `run<N>/config/`, matching
CANFAR. A run pins one mophongo version, and a shared clone meant a `sync` for
the run being worked on silently changed the code under every other run in the
tree: a finished run's outputs could not be tied to the source that produced
them, and its config — which does live per run — could disagree with the code
that read it. The clone sits beside the configs it was used with and the
outputs it produced, so `git` in that directory answers which source made them
— no stamp file, unlike CANFAR, where the laptop can only transfer files from
arc and cannot run `git` on it. The cost is a clone and a venv per run,
minutes and a few hundred MB, against a class of irreproducibility that is
silent. `campaign.py` runs `setup` as its third step, so a new `run<N>` gets its
own checkout without anyone remembering to ask.

`venv-vos` stays shared under `bin/`, because the two kinds of node do not
share a software stack and the CADC tools are not pinned to a mophongo version.
The
datamover nodes, which are the only ones that can reach CANFAR, have no `/apps`
module tree at all, so the module python the science venv is built against does
not exist there; `venv-vos` is built from the OS python, which is on every node.
Note a venv is not relocatable — its shebangs are absolute — so `setup` rebuilds
rather than moves one.

Everything lives on `/fred`. `/home` has a 20 GB quota and a single field's
staged inputs are about 15 GB.

## Prerequisites

```bash
export OZSTAR_USER=<your cluster username>     # ssh key access to nt.swin.edu.au
~/bin/remote/canfar-cert.sh                    # CADC proxy certificate, 10 days
```

`ozify.py` needs the `vos` client locally, which lives in `~/.venvs/canfar`:

```bash
P=~/.venvs/canfar/bin/python
```

## Everything at once

```bash
OZSTAR_RUN=run3 ./release.sh --skip-stage   # the full-field release
$P campaign.py --fields uds                 # one field
$P campaign.py --bands f770w                # one band of every field
$P campaign.py --skip psf repair            # grids and caches already in place
$P campaign.py --check-versions             # say what arc has moved on to
$P campaign.py --dry-run                    # print the plan, submit nothing
```

It rewrites the configs, uploads them, builds the environment, and submits the
work in three phases:

1. **`psf`** — every ePSF grid the configs name, built on the login node and
   *waited on*. Skipped when they are already on `/fred`, which is the usual
   case once a release has been fitted once.
2. **`repair`** — one saturation-repair job per field, each behind that field's
   staging. The repair depends on the detection band alone, so one job fills the
   cache every band of the field then reloads.
3. **`run`** — that field's band fits, chained behind its repair with
   `--dependency=afterok`.

Only the first phase blocks, and only because it cannot be a SLURM job: the
build queries MAST for each exposure's OPD and compute nodes have no route to
it, so nothing can be made to depend on it. Phases 2 and 3 go up as one
dependency graph and the laptop is free.

The step names and the flags are the same as `../canfar/campaign.py`, so one
campaign reads the same way on either platform; what differs is that CANFAR has
to block between all three phases, having no dependencies at all.

`release.sh` is the release recipe with the arguments filled in; set
`$OZSTAR_RUN` to choose which run directory it lands in. Each campaign writes
`$OZSTAR_RUN/README.md` before submitting anything — the mophongo commit, the
release versions each field is pinned to, whatever you pass as `--note`, and
what changed against the previous run.

Then watch with `submit.py status` / `logs` / `fetch`.

The source is pulled once, before the first job is submitted, and checked
against `--ref` (default `main`). A campaign has to run one version of the code
rather than whatever `main` happened to be at each dispatch, and the check
catches the case the pull cannot fix: a commit that was never pushed is absent
from the run while every output looks entirely normal. `--force-stale` submits
anyway.

## Step by step

The same thing done by hand:

```bash
$P submit.py cert                              # certificate -> OzStar
$P submit.py setup                             # clone mophongo, build the venvs
$P ozify.py ../minerva/uds_f770w.json          # rewrite paths, list the inputs
$P submit.py push uds_f770w                    # upload config and job scripts
$P submit.py psf  uds_f770w                    # login-node grid build; waits
$P submit.py stage uds_f770w                   # datamover job: arc -> /fred/data
$P submit.py run   uds_f770w --after <jobid>   # the fit
$P submit.py fetch uds_f770w                   # bring the small outputs home
$P submit.py push-arc <dir> arc:<uri>          # datamover job: /fred -> CANFAR
```

A cheap smoke run on a small patch, without touching the source config:

```bash
$P ozify.py ../minerva/uds_f770w.json --r-trial 1.5 --suffix _trial
$P submit.py push uds_f770w_trial
$P submit.py run  uds_f770w_trial --time 06:00:00
```

To ship a code change into a campaign that is already set up, `submit.py sync`
does a `git pull` in the run tree and leaves the venv alone. mophongo is
installed editable, so that is enough. Jobs already running keep the code they
imported; queued ones pick it up when they start.

## What ozify.py does

It indexes the relevant `arc:projects/minerva` subtrees — reusing
`../canfar/arcify.py`, since finding the files is the same problem on both
platforms — and then rewrites every input path in a local `RunConfig` to
`$OZSTAR_BASE/data/<basename>`, writing the copy list to `<name>_stage.tsv`,
and points `out_dir` at `$OZSTAR_RUN/<field>/<name>` with a per-field
`repair_cache_path` one level above it.
Unlike the CANFAR version it lists *every* input, compressed or not: nothing
can be read in place.

Two mismatches it resolves on the way: most files on arc are gzipped and the
pipeline wants plain FITS, so `stage.sh` decompresses them; and MIRI frame
tables ship as `*_f770_wcs.csv` while the filter parser expects `f770w`, so the
staged copy carries the expected name.

### Checking for a newer release

```bash
$P campaign.py --check-versions --dry-run     # as a pre-flight
$P ozify.py ../minerva/*.json --check-versions
```

Lists the release directories on arc, reads back the version each config is
pinned to from its own paths, and reports the pairs that are behind — the same
flags `../canfar` takes, answered the same way. Note this runs on the laptop and
needs only the CADC certificate: the datamover partition exists to *copy* inputs
to `/fred`, not to see what is on arc.

It rewrites nothing, and on `campaign.py` it warns and carries on rather than
refusing. Moving onto a new release changes the photometry, so it belongs to a
new run directory with a note saying why — a deliberate act, not something a
launch should do, or block on, because a directory appeared upstream. On OzStar
it also means re-staging: `/fred` holds a copy of each input, so a new release is
new files rather than an in-place update.

## PSF grids must be built on the login node

This is the one structural difference from CANFAR, where `psf.autobuild` just
works inside a run. `PSFFactory` generates MJD-tagged grids, and for each
exposure date `stpsf` resolves the wavefront OPD by querying MAST
(`load_wss_opd_by_date` → `mast_wss_opds_around_date_query`). There is no
offline path for that query, and it happens before any PSF is computed.

So a run that has to build a grid dies seconds after starting, with
`NameResolutionError: Failed to resolve 'mast.stsci.edu'`, on any node without
DNS — which is every compute node. Datamover nodes do have internet, but no
`/apps` module tree, so mophongo cannot run there either. The login node is the
only machine with both.

Two ways to get the grids in place, and they compose — the build skips any grid
whose file already exists:

```bash
$P submit.py push --psf                  # ship grids that exist on the laptop
$P submit.py psf uds_f770w cosmos_f770w  # build the rest on the login node
```

Copying is much cheaper than rebuilding when the grids are already local, so
`push --psf` first and let `psf` fill the gaps. Do not run the two at once:
scp writes straight to the destination name, the build skips a filename that
exists, and a half-written grid then reaches a fit as a truncated FITS.

`campaign.py` does both in order when given `--push-psf`, and otherwise runs
only the build. Either way it first lists `$OZSTAR_BASE/PSF` and skips the step
when every family the configs name is already there. `psf` blocks until the
build writes `PSF_BUILD_DONE`, judging liveness by the log growing rather than
by the process existing — `nt.swin.edu.au` round-robins over four login nodes,
so a second ssh usually lands somewhere the build's pid means nothing.
`--no-wait` returns as soon as it is detached.

Once `$OZSTAR_BASE/PSF` is populated the fits need no network at all, which is
the point — the grids sit above the run directory and every later run reads
them.

The build is parallel and deduplicated: it enumerates `(pattern, csv)` across
all configs, so a field's F444W set is built once rather than once per band,
and fans the epochs of each pattern over the cores the session may use. That
matters because the shared grids dominate — F444W is 25 PSFs per grid across
every epoch plus the 30" halo grids, against 9 per grid for a band's own MIRI
set. Measured: 301 grids in 17.9 min against 3h44m for the serial version.

Two things to know. A login session is cgroup-capped (four cores here) and
`nproc` under-reports it because the site sets `OMP_NUM_THREADS=1`, so the
worker count comes from `sched_getaffinity`. And the workers share one stpsf
OPD cache: on a cold cache two can read an OPD while a third is still
downloading it, which fails that epoch with `Empty or corrupt FITS file`. It
fails loudly per epoch and a re-run repairs it, but a first build against an
empty `$STPSF_PATH` is safest done serially.

The grids are one per epoch: `PSFFactory.date_mode` defaults to `"all"` (one
per unique integer MJD) and the configs state `psf.date_mode` explicitly. The
old default, `"modal"`, gave a single date for an exposure list spanning years,
which silently defeats the MJD-tagged lookup the grids exist for. See
`../canfar/README.md` for the full account; it applies to both platforms.

## Sending results back to CANFAR

`push-arc` is a `datamover` job that copies a directory on `/fred` into
`arc:`. Being a SLURM job is the point: a transfer measured in hours outlives
the laptop, the ssh session and the terminal.

```bash
$P submit.py push-arc /fred/.../mophongo/PSF arc:projects/minerva/ifl/mophongo/PSF
$P submit.py push-arc /fred/.../mophongo/run3 arc:.../mophongo/run3 --compress
```

It diffs the destination first and sends only what is missing, so a job that
hits the 24 h wall is resubmitted rather than restarted.

Measured throughput to arc: **1.25 MB/s on one stream, 14 MB/s on six**. The
bottleneck is per-connection latency to Canada, not bandwidth, which is why
this fans out rather than calling `vsync`. Many small files run slower than
that ceiling — per-file round trips dominate.

Decide what to send before deciding how. A full campaign is ~520 GB, of which
the photometry is ~350 MB:

| product | share | regenerable |
|---|---|---|
| `*_stamps.h5` | ~30% | yes |
| `*_residual.fits` | ~11% | yes |
| `scenes/` (10k PNGs) | ~9% | yes, `scene_plots.py` redraws without refitting |
| **`*_fit_table.fits`** | **0.07%** | **no** |

`--compress` gzips before sending, which pays on the zero-heavy products and
not on ePSF grids (measured 1.1x — dense float arrays). Two footnotes worth
having: `vls -R` returns nothing here, silently, so the diff lists per
directory instead; and `vcp` leaves files `0600`, useless in shared project
space, so the job opens read permission at the end.

## Resources

Thirty-two cores and 96 GB per fit, for every field. `--cores` and `--mem`
override, and `submit.MEM_GB_BY_FIELD` still exists for a per-field exception
if one is ever needed.

Measured CPU is ~6% of 16 cores, so the cores are not doing arithmetic. What
the request buys is a **share of one milan node** (64 cores, 250 GB), and
whichever of cores or memory binds first decides how many fits land on it:

| request | binds on | fits per node | measured |
|---|---|---|---|
| 16 cores / 96 GB | memory | 2 | 41–69 min (v1.0b) |
| 8 cores / 72 GB | memory | 3 | 1h27–1h35 (run3) |
| 32 cores / 96 GB | either | 2 | current default (run4) |

Three memory-bound fits sharing one node's bandwidth is the slow case, and
halving the cores nearly doubled the wall clock without the CPU number moving.
The default now uses both levers: 32 cores and 96 GB each bind at two fits per
node, and the memory one also buys OOM headroom — `cosmos_f770w` peaked at
67.5 GB, which is 94% of the old 72 GB request, and a run that exceeds its
request is killed with no Python traceback.

EGS turned out to need *less* memory than the others (29–59 GB), not more as
its 1.4x detection grid suggested, so it no longer carries a special case.

Measured on the v1.0b campaign (UDS bands at 16 cores / 96 GB; the defaults
above came from these numbers):

| run | wall | peak RSS |
|---|---|---|
| 1.5' trial patch | 9m51s | 20.0 GB |
| F770W full field | 56m19s | 57.4 GB |
| F1280W full field | 69m25s | 53.3 GB |
| F1500W full field | 41m15s | 55.6 GB |

Two things follow. Peak memory tracks the *field* — the detection grid and the
segmap are the same for every band of it, and the four UDS bands span only
4 GB — so the request belongs per field rather than per run. And CPU sat at
6.1% of 16 cores throughout, about one core: the fit is serial, so cores buy
threaded BLAS in the dense scene solves and queue position everywhere else.
Eight is headroom for the former without paying 16 cores of fair-share for idle
allocation.

72 GB against the 57.4 GB worst measured peak is 80%, about 15 GB of headroom.
That is deliberate: a run that exceeds its request is killed with no Python
traceback, so the failure reads as a mystery rather than an error — if a band
dies without one, check `sacct -j <id>.batch --format=MaxRSS` before anything
else. 72 still fits every milan and skylake node, so the headroom costs no
scheduling reach.

No partition is requested, so the scheduler picks. milan (64 cores / 256 GB,
147 nodes) and skylake (36 / 191, 118 nodes) both hold this, and constraining
the choice only lengthens the queue. `largemem` (`dave301-311`, 1 TB) is not
needed for any MINERVA field and requesting it would queue behind 11 contended
nodes.

Staging is different: it must be `--partition=datamover`, and that has to be
given on the `sbatch` command line. A site plugin reassigns the partition of a
job that names it in a `#SBATCH` directive alone, and the job then lands on a
compute node with no route to CANFAR — the symptom is `vcp` failing on every
file at once. Those nodes advertise 4000 MB, so `--mem=4g` (4096) is refused as
an impossible node configuration; the script asks for 3 GB.

## Notes

- Compute nodes have no internet. Anything that must be downloaded — the
  mophongo clone, pip wheels, the STPSF reference data — is fetched on the
  login node by `setup`, and the inputs by a datamover job.
- The venv's `bin/python` is a symlink into the module tree, so the same
  modules must be loaded to use it. Every job script loads the same
  `PYMODULES`; changing them means rebuilding the venv. Lmod here is
  hierarchical, so `python/3.12.3` is invisible until `gcccore/13.3.0` is
  loaded — the error names the parent, which is the only clue you get.
- The python module puts an EasyBuild shim ahead of the venv on `sys.path`.
  On some nodes that resolved `cryptography` to a build for another python and
  failed on a missing `libssl.so.1.1`, which surfaces as every `vcp` dying in
  an import. The job scripts `unset PYTHONPATH` after loading modules; the venv
  is self-contained and does not need it.
- `--dependency=afterok` means a failed stage leaves its fits queued as
  `DependencyNeverSatisfied` until SLURM cancels them. That is the wanted
  behaviour — no fit starts on half-copied data — but it reads as jobs silently
  disappearing if you do not know it.
- Staging is per field, not per band: the bands of a field share the F444W
  mosaic, its weight map and the segmap. Fields are independent, so their
  staging jobs run concurrently, one per datamover node.
- A cancelled staging job leaves `.<name>.<pid>` temporaries of several GB.
  The next staging job for that field sweeps the ones over six hours old;
  fresher ones are left alone because a concurrent job's temporaries look the
  same.
- `submit.py cancel` cancels by job-name prefix (`moph`), not the whole
  account, so it will not touch your other work on the cluster.
- Outputs are large. A 3 arcmin patch of UDS writes 8.4 GB, of which 4.2 GB is
  `stamps.fits` and 3.5 GB the residual; a full field scales with the source
  count, roughly 8x. Set `save_stamps` or `scene_plots` false in the config
  when the diagnostics are not wanted, and watch `lfs quota -g <project> /fred`.
- The PSF and kernel maps are cached in `out_dir` and do not depend on the trial
  patch, so `submit.py seed <patch-run>:<full-run>` links them across and saves
  about half an hour per band.

For the CANFAR equivalent and the reasoning behind the shared parts, see
`../canfar/README.md`. Cluster-level reference (partitions, modules, quotas) is
in `~/.claude/ozstar.md`.
