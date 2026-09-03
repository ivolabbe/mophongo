# Running mophongo on OzStar, from zero

Start to finish, ending in a completed `uds_f770w` run on the current MINERVA
release. The data live on CANFAR and the compute is at Swinburne, so you need an
account on both.

Placeholders: `<user>` is your OzStar username, `<cadc>` your CADC username,
`<project>` your group allocation (`oz030` here).

After that, [README.md](./README.md) is the reference: layout, every flag,
resources, and the failure modes worth knowing in advance.

---

## Part 1. Accounts and access

### 1.1 OzStar

Access goes through Swinburne's supercomputing team. You need a login on
`nt.swin.edu.au` and membership of a project with an allocation under `/fred`.
Check both:

```bash
ssh <user>@nt.swin.edu.au
ls -d /fred/<project>/<user>          # your directory in the allocation
lfs quota -g <project> /fred          # what is left of the 10 TB
```

Put an ssh key in place (`ssh-copy-id`). The toolkit runs many short ssh
commands and a password prompt on each is unworkable.

### 1.2 CADC and the MINERVA group

Inputs are copied from `arc:projects/minerva`, which needs a CADC account that
is a member of the `minerva` group. `../canfar/MANUAL.md` Part 1 covers getting
both. You do not need Science Portal access, only read access to the data.

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

## Part 2. One-time setup

### 2.1 Environment

```bash
cd mophongo/examples/ozstar
export OZSTAR_USER=<user>
export OZSTAR_PROJECT=<project>          # default oz030
export OZSTAR_RUN=run3                   # default run2
P=~/.venvs/canfar/bin/python
```

The shared tree is `/fred/<project>/<user>/mophongo` (`$OZSTAR_BASE`), and the
run directory is `$OZSTAR_BASE/$OZSTAR_RUN`. It must be under `/fred`, because
`/home` has a 20 GB quota and one field's staged inputs are about 15 GB.

`$OZSTAR_RUN` is how versions are kept apart. Bump it for a new attempt; the
staged inputs, the PSF grids and the job scripts above it are reused.

### 2.2 Certificate

```bash
../canfar/remote/canfar-cert.sh          # prompts for the CADC password
$P submit.py cert                        # copies it to OzStar
```

`canfar-cert.sh` is one of the shell helpers in `../canfar/remote/`; it wants a
`canfar.conf` naming your CADC username, which `../canfar/MANUAL.md` Part 5
sets up. It writes `~/.ssl/cadcproxy.pem`, and refreshes it only when it is
close to expiring.

Valid for ten days. The staging job authenticates with this and nothing else, so
a campaign that has to stage after one expires needs both commands again. The
symptom otherwise is a staging job failing immediately on every file.

Only two things use it, and both ask first whether they have to. `ozify.py`
reads arc to find an input no manifest here already names; the staging job
copies an input that is not on `/fred` yet. For a release the toolkit has been
through once, neither has anything to do, so a re-fit needs no certificate and
does not notice that one has expired.

A certificate that exists but has expired used to be the confusing case:
`ozify.py` reported the release "not found on arc" and `submit.py cert`
installed the dead certificate on the cluster for the staging job to fail on.
Both now check the expiry rather than the file, and say so.

### 2.3 The run tree

```bash
$P submit.py push        # job scripts
$P submit.py setup       # clone mophongo, build the two venvs
```

`setup` runs on the login node, the only place with both internet and the module
system, and is idempotent. It builds:

- `$OZSTAR_RUN/config/venv`: mophongo and its dependencies, from the
  `python/3.12.3` module. The fits use this. It is per run, so a run's outputs
  can always be tied to the source that produced them. Its `bin/python` is a
  symlink into the module tree, so the same modules have to be loaded to use it.
- `bin/venv-vos`: the CADC transfer tools, from `/usr/bin/python3`. Shared
  across runs. Staging uses it, because staging runs on the datamover partition
  and those nodes have no `/apps` module tree at all.

It also checks the STPSF reference data (about 250 MB, at
`/fred/<project>/<user>/stpsf-data` by default, `OZSTAR_STPSF` to override).
Compute nodes have no internet, so a run that has to build a PSF grid cannot
fetch this itself; it has to be there first. `setup` prints the download
command if it is missing.

### 2.4 PSF grids

Grids cannot be built inside a fit here, the way they are on CANFAR. Building
one makes `stpsf` query MAST for the wavefront OPD at each exposure's date, and
compute nodes have no DNS, so the fit dies in half a minute with
`NameResolutionError: Failed to resolve 'mast.stsci.edu'`. The login node is the
only machine with both internet and the module stack.

```bash
$P submit.py push --psf                     # grids that already exist locally
$P submit.py psf uds_f770w cosmos_f770w     # build the rest on the login node
```

The two compose, because the build skips any grid whose file already exists, so
ship what you have and build the gaps. Do not run them concurrently: scp writes
straight to the destination name, and a half-written grid would be skipped by
the build and later read as a truncated FITS.

Grids live in `$OZSTAR_BASE/PSF`, above the run directory, so every later run
reads them. Once it is populated, fits need no network at all.

---

## Part 3. One band, end to end

### 3.1 Rewrite the config

```bash
$P ozify.py ../minerva/uds_f770w.json
```

writes two files: `uds_f770w_ozstar.json`, the same config with every input path
pointed at `$OZSTAR_BASE/data`, and `uds_f770w_stage.tsv`, the list of eight
files to copy from arc.

The first run of this lists `arc:projects/minerva` to find where those eight
files live, and needs the certificate from 2.2. Later ones re-read the arc
sources out of the manifests already in the directory and go nowhere: only a
basename no manifest names sends it back to arc, and a basename carries its
release version, so that means a genuinely new release. `--reindex` forces the
listing if a file has moved on arc without being renamed.

For a first pass use a small patch. It takes minutes rather than the better part
of an hour, and reads only its own pixels off disk:

```bash
$P ozify.py ../minerva/uds_f770w.json --r-trial 1.5 --suffix _trial
```

### 3.2 Stage the inputs

```bash
$P submit.py push  uds_f770w_trial
$P submit.py stage uds_f770w_trial
```

This submits a job to the `datamover` partition, which copies each file from arc
and decompresses the gzipped ones. About 15 GB per field; UDS takes tens of
minutes. Files already present are skipped, so the job can be resubmitted after
a timeout and will resume.

`stage` first lists `$OZSTAR_BASE/data` and submits nothing for a field whose
inputs are all there — the usual case for every run after the first, since
`data/` sits above the run directory and is shared. Nothing then contacts
CANFAR, which is why a re-fit works with an expired certificate. The job itself
makes the same check before reaching for `vcp`.

Pass several bands of a field in one call and they share a single job, since the
bands of a field share the F444W mosaic, its weight map and the segmap:

```bash
$P submit.py stage uds_f770w uds_f1280w uds_f1500w uds_f1800w
```

### 3.3 Run

`stage` prints its job id. Hang the fit off it so it cannot start on half-copied
data:

```bash
$P submit.py run uds_f770w_trial --after <jobid> --time 06:00:00
```

or without `--after` if the inputs are already there. Defaults are 32 cores,
96 GB and 24 hours.

### 3.4 Watch it

```bash
$P submit.py status                     # the queue
$P submit.py status --done 2026-08-17   # finished jobs, with MaxRSS
$P submit.py logs uds_f770w_trial       # tail the newest matching log
```

### 3.5 Results

```bash
$P submit.py fetch uds_f770w_trial   # fit table, scene catalog, log
```

The rest stays on `/fred`, since a full field writes tens of GB of residual and
stamps:

| file | what |
|---|---|
| `<name>_fit_table.fits` | the photometry: `flux_<i>`, `flux_<i>_total`, `err_<i>` |
| `<name>_residual.fits` | data minus model, the first thing to check |
| `scenes/<name>_scene_*.png` | per-scene diagnostics |
| `<name>_kernel.fits`, `<name>_psf_*.fits` | PSF and kernel maps |
| `<name>.log` | the run log |

---

## Part 4. A whole campaign

```bash
$P campaign.py --dry-run                        # the plan
$P campaign.py --check-versions --dry-run       # ... and what arc moved on to
$P campaign.py                                  # trial patches, every config
$P campaign.py --fields uds --bands f770w       # one band of one field
$P campaign.py --r-trial 0 --note "n3.0 UDS"    # the full-field release
./release.sh                                    # the same, arguments filled in
```

One command rewrites all 17 configs, uploads them, builds the environment, and
submits the work in the same three phases `../canfar/campaign.py` uses (`psf`,
`repair`, `run`) with the same step names and flags. Each even accepts the
other's name for the config rewrite, so `--from arcify` works here and
`--from ozify` works there. `--check-versions` reports configs pinned to an older
release than arc now holds, then carries on with the pinned ones.

Run it detached. The `psf` phase blocks, and the process is a child of your
shell:

```bash
OZSTAR_RUN=run3 nohup ./release.sh > ../../scratch/ozstar/run3.log 2>&1 &
```

If it does die mid-launch, nothing is lost: `push`, `setup` and `psf` are
idempotent, and a re-launch continues from where it stopped. Check
`submit.py status` first to see what was already submitted.

Only `psf` blocks. It cannot be a SLURM job, because the build resolves each
exposure's wavefront by querying MAST and compute nodes have no route to it, so
it is a detached login-node process the laptop polls. Everything after it is
submitted with `--dependency=afterok` and returns immediately: staging per
field, then that field's repair, then its band fits.

Three behaviours to expect:

- each field's repair waits on that field's staging with `afterok`, and its fits
  on the repair. A failed stage therefore leaves everything behind it queued as
  `DependencyNeverSatisfied` until SLURM cancels them, which is what you want and
  looks like jobs vanishing if you do not expect it;
- with both preparation phases skipped, a field whose PSF grids do not exist yet
  sends one band ahead of the rest. With no grid matching the config pattern the
  pipeline builds one, and several bands of a field would otherwise build the
  same grids concurrently into a single `psf_dir`;
- the source is pulled once for the whole campaign and checked against `--ref`,
  so every job runs one version of the code. The check refuses to submit when the
  clone on `/fred` is not that ref, including when the difference is a local
  commit you never pushed, since the cluster pulls GitHub. `--force-stale`
  overrides.

Each campaign writes `$OZSTAR_RUN/README.md` before submitting: the mophongo
commit, the release version each field is pinned to, your `--note`, and what
changed against the previous run directory. The commit is read from
`run<N>/config/mophongo`, not from your laptop, so it is the source that
actually ran.

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
| `ozify.py` says an input is `not found on arc`, or `none of the N arc subtrees listed` | usually an expired certificate rather than a missing file. `../canfar/remote/canfar-cert.sh --force`, then `submit.py cert` |
| Every `vcp` dies in `ImportError: libssl.so.1.1` | the EasyBuild shim on `PYTHONPATH` shadowed the venv. The job scripts `unset PYTHONPATH`; if you run something by hand, do the same |
| Staging job runs on a compute node and fails on every file | `--partition=datamover` was only in a `#SBATCH` directive, which a site plugin overrides. It has to be on the `sbatch` command line |
| `Memory specification can not be satisfied` for a stage job | datamover nodes advertise 4000 MB, so `--mem=4g` (4096) is impossible. Ask for 3 GB |
| `bad interpreter: .../venv/bin/python: No such file` on a datamover node | those nodes have no `/apps`, so the module python is not there. Use `venv-vos` |
| Fits stuck in `PENDING (DependencyNeverSatisfied)` | their staging job failed. Read its log, fix, resubmit staging, resubmit the fits |
| Job dies with no Python traceback | out of memory. Check `sacct -j <id>.batch --format=MaxRSS`, resubmit with more `--mem` |
| `stpsf` cannot find its data | `STPSF_PATH`. Compute nodes have no internet, so it must already be on `/fred` |
| Fit dies in 30 s with `NameResolutionError: Failed to resolve 'mast.stsci.edu'` | it tried to build a PSF grid, and stpsf resolves each exposure's OPD by querying MAST. Build the grids first: `submit.py push --psf` and/or `submit.py psf <configs>` (Part 2.4) |
| `refusing to run: /fred has [...], main is ...` | the clone was never moved to that ref, or the difference is a local commit you did not push. `submit.py sync`, or `git push`, then resubmit. `--force-stale` submits anyway |
| `PSF build log has not grown for N min` | the detached login-node build died. Read the log path it names; a first build against an empty `$STPSF_PATH` is the usual cause |
| `No such file or directory` for an input | the staging job has not finished, or the config was not re-pushed after `ozify.py` |
| Campaign exits during `setup` or `psf` with no error | the shell it was a child of closed. Re-launch under `nohup`; the completed steps are idempotent |

## Reference

- Cluster reference (partitions, modules, quotas, SLURM): `~/.claude/ozstar.md`
- The CANFAR equivalent of this toolkit: `../canfar/README.md`, `../canfar/MANUAL.md`
- What the MINERVA products are and where: `MINERVA/data/00WHERE`
- CADC certificate, sshfs mounts, `vsync`: `../canfar/remote/`
- Getting data off CANFAR: `MINERVA/data/00CANFAR`
