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

```
$OZSTAR_RUN/                        default /fred/<project>/<user>/mophongo/run
├── venv/                  mophongo, built with the module python
├── venv-vos/              CADC transfer tools, built with /usr/bin/python3
├── mophongo/              the GitHub clone
├── PSF/                   MJD-tagged ePSF grids, shared by every run
├── jobs/                  setup_env.sh, stage.sh, run.slurm, ...
├── data/                  staged inputs, shared between the bands of a field
├── logs/                  SLURM job output
├── <name>_ozstar.json     rewritten configs
├── <name>_stage.tsv       per-config copy lists
└── out/<name>/            run outputs
```

Two venvs, because the two kinds of node do not share a software stack. The
datamover nodes, which are the only ones that can reach CANFAR, have no `/apps`
module tree at all, so the module python the science venv is built against does
not exist there; `venv-vos` is built from the OS python, which is on every node.

Everything lives on `/fred`. `/home` has a 20 GB quota and a single field's
staged inputs are about 15 GB.

## Prerequisites

```bash
export OZSTAR_USER=<your cluster username>     # ssh key access to nt.swin.edu.au
../../scratch/canfar/canfar-cert.sh            # CADC proxy certificate, 10 days
```

`ozify.py` needs the `vos` client locally, which lives in `~/.venvs/canfar`:

```bash
P=~/.venvs/canfar/bin/python
```

## Everything at once

```bash
$P campaign.py                        # every config in ../minerva, trial patches
$P campaign.py --fields uds           # one field
$P campaign.py --r-trial 0 --suffix _v1.0b     # the full-field release
$P campaign.py --dry-run              # print the plan, submit nothing
```

It rewrites the configs, uploads them, builds the environment, submits one
staging job per field and then one fit per config, wiring the dependencies:
each field's fits wait on that field's staging, and a field with no PSF grids
yet sends one band ahead of the others to build them. `release_v1.0b.sh` is the
release recipe with the arguments already filled in.

Then watch with `submit.py status` / `logs` / `fetch`.

## Step by step

The same thing done by hand:

```bash
$P submit.py cert                              # certificate -> OzStar
$P submit.py setup                             # clone mophongo, build the venvs
$P ozify.py ../minerva/uds_f770w.json          # rewrite paths, list the inputs
$P submit.py push uds_f770w                    # upload config and job scripts
$P submit.py stage uds_f770w                   # datamover job: arc -> /fred/data
$P submit.py run   uds_f770w --after <jobid>   # the fit
$P submit.py fetch uds_f770w                   # bring the small outputs home
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
`$OZSTAR_RUN/data/<basename>`, writing the copy list to `<name>_stage.tsv`.
Unlike the CANFAR version it lists *every* input, compressed or not: nothing
can be read in place.

Two mismatches it resolves on the way: most files on arc are gzipped and the
pipeline wants plain FITS, so `stage.sh` decompresses them; and MIRI frame
tables ship as `*_f770_wcs.csv` while the filter parser expects `f770w`, so the
staged copy carries the expected name.

## PSF grids must be built on the login node

This is the one structural difference from CANFAR, where `psf_autobuild` just
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

Once `$OZSTAR_RUN/PSF` is populated the fits need no network at all, which is
the point — the grids are cached there and every later run reads them.

## Resources

16 cores and 64 GB per fit, the default. The fitting path is largely serial, so
the cores buy threaded BLAS and FFTs rather than a linear speed-up; the memory
is what matters. No partition is requested, so the scheduler picks — skylake
(36 cores / 191 GB) and milan (64 / 256) both hold this comfortably, and
constraining the choice only lengthens the queue.

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
