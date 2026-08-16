# Running mophongo on OzStar, from zero

A start-to-finish guide, ending in a completed `uds_f770w` run on the current
MINERVA release. It assumes an OzStar account and a CADC account; the data live
on CANFAR and the compute is at Swinburne, so both are needed.

Placeholders: `<user>` is your OzStar username, `<cadc>` your CADC username,
`<project>` your group allocation (`oz030` here).

---

## Part 1 — Accounts and access

### 1.1 OzStar

Access is through Swinburne's supercomputing team; you need a login on
`nt.swin.edu.au` and membership of a project with an allocation under `/fred`.
Check both:

```bash
ssh <user>@nt.swin.edu.au
ls -d /fred/<project>/<user>          # your directory in the allocation
lfs quota -g <project> /fred          # what is left of the 10 TB
```

Put an ssh key in place (`ssh-copy-id`) — the toolkit here runs many short ssh
commands and a password prompt on each is unworkable.

### 1.2 CADC and the MINERVA group

The inputs are copied from `arc:projects/minerva` on CANFAR, which needs a CADC
account that is a member of the `minerva` group. `../canfar/MANUAL.md` Part 1
covers getting both. You do not need Science Portal access for this — only read
access to the data.

Verify locally:

```bash
~/.venvs/canfar/bin/vls arc:projects/minerva/uds/mosaics/miri/m3.1
```

If that lists files, you are set. If `vls` is missing:

```bash
python -m venv ~/.venvs/canfar
~/.venvs/canfar/bin/pip install vos cadcutils
```

---

## Part 2 — One-time setup

### 2.1 Environment

```bash
cd mophongo/examples/ozstar
export OZSTAR_USER=<user>
export OZSTAR_PROJECT=<project>          # default oz030
P=~/.venvs/canfar/bin/python
```

The run tree defaults to `/fred/<project>/<user>/mophongo/run`; `OZSTAR_RUN`
overrides it. It must be under `/fred` — `/home` has a 20 GB quota and one
field's staged inputs are about 15 GB.

### 2.2 Certificate

```bash
~/bin/remote/canfar-cert.sh              # prompts for the CADC password
$P submit.py cert                        # copies it to OzStar
```

The certificate is valid for ten days and the staging job authenticates with it
and nothing else. A campaign that outlives one needs both commands again; the
symptom otherwise is a staging job failing immediately on every file.

### 2.3 The run tree

```bash
$P submit.py push        # job scripts
$P submit.py setup       # clone mophongo, build the two venvs
```

`setup` runs on the login node, which is the only place with both internet and
the module system, and is idempotent. It builds:

- `venv` — mophongo and its dependencies, from the `python/3.12.3` module. The
  fits use this. Its `bin/python` is a symlink into the module tree, so the
  same modules have to be loaded to use it.
- `venv-vos` — the CADC transfer tools, from `/usr/bin/python3`. Staging uses
  this, because it runs on the datamover partition and those nodes have no
  `/apps` module tree at all.

It also checks the STPSF reference data (about 250 MB, at
`/fred/<project>/<user>/stpsf-data` by default, `OZSTAR_STPSF` to override).
Compute nodes have no internet, so a run that has to build a PSF grid cannot
fetch this itself: it has to be there first. `setup` prints the download
command if it is missing.

### 2.4 PSF grids

Grids cannot be built inside a fit here, the way they are on CANFAR. Building
one makes `stpsf` query MAST for the wavefront OPD at each exposure's date, and
compute nodes have no DNS — the fit dies in half a minute with
`NameResolutionError: Failed to resolve 'mast.stsci.edu'`. The login node is
the only machine with both internet and the module stack.

```bash
$P submit.py push --psf                     # grids that already exist locally
$P submit.py psf uds_f770w cosmos_f770w     # build the rest on the login node
```

The two compose — the build skips any grid whose file already exists — so ship
what you have and build the gaps. Do not run them concurrently: scp writes
straight to the destination name, and a half-written grid would be skipped by
the build and later read as a truncated FITS.

Once `$OZSTAR_RUN/PSF` is populated, fits need no network at all.

---

## Part 3 — One band, end to end

### 3.1 Rewrite the config

```bash
$P ozify.py ../minerva/uds_f770w.json
```

writes `uds_f770w_ozstar.json` — the same config with every input path pointed
at `$OZSTAR_RUN/data` — and `uds_f770w_stage.tsv`, the list of eight files to
copy from arc.

For a first pass use a small patch, which takes minutes rather than the better
part of an hour and reads only its own pixels off disk:

```bash
$P ozify.py ../minerva/uds_f770w.json --r-trial 1.5 --suffix _trial
```

### 3.2 Stage the inputs

```bash
$P submit.py push  uds_f770w_trial
$P submit.py stage uds_f770w_trial
```

This submits a job to the `datamover` partition, which copies each file from
arc and decompresses the gzipped ones. About 15 GB per field; UDS takes tens of
minutes. Files already present are skipped, so the job can be resubmitted after
a timeout and will resume.

Pass several bands of a field in one call and they share a single job — bands
of a field share the F444W mosaic, its weight map and the segmap, and there is
no point copying several GB more than once:

```bash
$P submit.py stage uds_f770w uds_f1280w uds_f1500w uds_f1800w
```

### 3.3 Run

`stage` prints its job id; hang the fit off it so it cannot start on
half-copied data:

```bash
$P submit.py run uds_f770w_trial --after <jobid> --time 06:00:00
```

or without `--after` if the inputs are already there. The defaults are 16 cores,
64 GB and 24 hours.

The first run of a band with no PSF grids builds them, which takes an extra
half hour or so; they are cached in `$OZSTAR_RUN/PSF` and every later run of
that field reuses them.

### 3.4 Watch it

```bash
$P submit.py status                  # the queue
$P submit.py status --done 2026-08-13   # finished jobs, with MaxRSS
$P submit.py logs uds_f770w_trial    # tail the newest matching log
```

### 3.5 Results

```bash
$P submit.py fetch uds_f770w_trial   # fit table, scene catalog, log
```

The rest stays on `/fred` — a full field writes tens of GB of residual and
stamps:

| file | what |
|---|---|
| `<name>_fit_table.fits` | the photometry: `flux_<i>`, `flux_<i>_total`, `err_<i>` |
| `<name>_residual.fits` | data minus model, the first thing to check |
| `scenes/<name>_scene_*.png` | per-scene diagnostics |
| `<name>_kernel.fits`, `<name>_psf_*.fits` | PSF and kernel maps |
| `<name>.log` | the run log |

---

## Part 4 — A whole campaign

```bash
$P campaign.py --dry-run                        # the plan
$P campaign.py --check-versions --dry-run       # ... and what arc moved on to
$P campaign.py                                  # trial patches, every config
$P campaign.py --fields uds --bands f770w       # one band of one field
$P campaign.py --r-trial 0 --note "n3.0 UDS"    # the full-field release
./release.sh                                    # the same, arguments filled in
```

One command rewrites all 17 configs, uploads them, builds the environment, and
submits the work in the same three phases `../canfar/campaign.py` uses — `psf`,
`repair`, `run` — with the same step names and flags, so a campaign reads the
same way on either platform. Each even accepts the other's name for the config
rewrite, so `--from arcify` works here and `--from ozify` works there.
`--check-versions` reports configs pinned to an older release than arc now
holds, then carries on with the pinned ones.

Only `psf` blocks. It cannot be a SLURM job: the build resolves each exposure's
wavefront by querying MAST, and compute nodes have no route to it, so it is a
detached login-node process the laptop polls. Everything after it is submitted
with `--dependency=afterok` and returns immediately: staging per field, then
that field's repair, then its band fits.

Three things are encoded and worth knowing:

- each field's repair waits on that field's staging with `afterok`, and its fits
  on the repair. A failed stage therefore leaves everything behind it queued as
  `DependencyNeverSatisfied` until SLURM cancels them, which is what you want
  and looks like jobs vanishing if you do not expect it;
- with both preparation phases skipped, a field whose PSF grids do not exist yet
  sends one band ahead of the rest, because with no grid matching the config
  pattern the pipeline builds one, and several bands of a field would otherwise
  build the same grids concurrently into a single `psf_dir`;
- the source is pulled once for the whole campaign and checked against `--ref`,
  so every job runs one version of the code. The check refuses to submit when
  the clone on `/fred` is not that ref — including when the difference is a
  local commit you never pushed, since the cluster pulls GitHub. `--force-stale`
  overrides.

Each campaign writes `$OZSTAR_RUN/README.md` before submitting: the mophongo
commit, the release version each field is pinned to, your `--note`, and what
changed against the previous run directory.

To stop everything:

```bash
$P submit.py cancel        # by job-name prefix, not the whole account
```

To ship a code change mid-campaign:

```bash
$P submit.py sync          # git pull; the venv and running jobs are untouched
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Lmod has detected the following error: Parent modules not loaded` | Lmod is hierarchical: `python/3.12.3` needs `gcccore/13.3.0` first. The error names the parent |
| Every `vcp` dies in `ImportError: libssl.so.1.1` | the EasyBuild shim on `PYTHONPATH` shadowed the venv. The job scripts `unset PYTHONPATH`; if you run something by hand, do the same |
| Staging job runs on a compute node and fails on every file | `--partition=datamover` was only in a `#SBATCH` directive, which a site plugin overrides. It has to be on the `sbatch` command line |
| `Memory specification can not be satisfied` for a stage job | datamover nodes advertise 4000 MB, so `--mem=4g` (4096) is impossible. Ask for 3 GB |
| `bad interpreter: .../venv/bin/python: No such file` on a datamover node | those nodes have no `/apps`, so the module python is not there. Use `venv-vos` |
| Fits stuck in `PENDING (DependencyNeverSatisfied)` | their staging job failed. Read its log, fix, resubmit staging, resubmit the fits |
| Job dies with no Python traceback | out of memory. Resubmit with `--mem 96` or more; the nodes hold 191-256 GB |
| `stpsf` cannot find its data | `STPSF_PATH`. Compute nodes have no internet, so it must already be on `/fred` |
| Fit dies in 30 s with `NameResolutionError: Failed to resolve 'mast.stsci.edu'` | it tried to build a PSF grid, and stpsf resolves each exposure's OPD by querying MAST. Build the grids first: `submit.py push --psf` and/or `submit.py psf <configs>` (Part 2.4) |
| `refusing to run: /fred has [...], main is ...` | the clone was never moved to that ref, or the difference is a local commit you did not push. `submit.py sync`, or `git push`, then resubmit. `--force-stale` submits anyway |
| `PSF build log has not grown for N min` | the detached login-node build died. Read the log path it names; a first build against an empty `$STPSF_PATH` is the usual cause |
| `No such file or directory` for an input | the staging job has not finished, or the config was not re-pushed after `ozify.py` |

## Reference

- Cluster reference (partitions, modules, quotas, SLURM): `~/.claude/ozstar.md`
- The CANFAR equivalent of this toolkit: `../canfar/README.md`, `../canfar/MANUAL.md`
- What the MINERVA products are and where: `MINERVA/data/00WHERE`
- Getting data off CANFAR: `MINERVA/data/00CANFAR`
