# Running mophongo on CANFAR, from zero

A start-to-finish guide for someone with no CANFAR account, ending in a
completed `uds_770` run on the current MINERVA release.

The main path needs **nothing installed locally** — a browser is enough. All the
work happens in a container on CANFAR, where the MINERVA data are already
mounted. Part 5 covers driving runs from a laptop instead, which is nicer for
launching many bands but requires installing two Python packages.

Placeholders: `<user>` is your CADC username, `<adam>` is the MINERVA group
administrator.

---

## Part 1 — Get an account

### 1.1 Request a CADC account

CADC accounts are the login for everything CANFAR.

1. Go to https://www.canfar.net and follow the login/registration link, or
   register directly with CADC at
   https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/auth/request.html
2. Fill in name, institution and email. Use your institutional address.
3. Approval is manual and typically takes a business day or two.

You will end up with a CADC username and password. The username is `<user>`
below, and it is also the name of your directory on CANFAR storage.

### 1.2 Ask to be added to the MINERVA group

The account alone does not give access to the data. The MINERVA products live in
`arc:projects/minerva`, readable only by members of the `minerva` group.

Send `<adam>` a message along these lines:

> I have a CADC account, username `<user>`. Could you add me to the CANFAR
> `minerva` group so I can read `arc:projects/minerva`, and confirm I have
> access to the Science Portal so I can launch sessions?

Two distinct things are being requested, and both are needed:

- **group membership** — read access to the data;
- **Science Portal access** — permission to launch compute sessions. On CANFAR
  this is tied to being in a project group that has an allocation.

Group membership can be checked at https://www.canfar.net/groups/ once you are
in.

### 1.3 Verify access

Log in at https://www.canfar.net/storage/arc/list/projects/minerva. If you can
list the directory, the group membership is live. If you get a permission error,
membership has not been granted yet.

---

## Part 2 — Your home directory

`/arc/home/<user>` does **not** exist until something creates it. A brand-new
account can already read `arc:projects/minerva` while having no home directory
at all, which produces confusing "No such file or directory" errors later.

The simplest fix is to launch a Science Portal session (Part 3) — it will create
and land you in your home. If it does not appear, it can be created explicitly
from the storage web UI at https://www.canfar.net/storage/arc/list/home, or from
a session terminal with `mkdir -p /arc/home/<user>`.

Check it exists and is yours:

```bash
ls -la /arc/home/<user>
```

Everything in this guide is written into `/arc/home/<user>/run`. Keep work
there rather than in `arc:projects/minerva`, which is shared with the whole
collaboration and should stay read-mostly.

---

## Part 3 — Launch a session

1. Go to https://www.canfar.net/science-portal and sign in.
2. Click the **+** to create a new session.
3. Choose:
   - **type**: `notebook`
   - **image**: `images.canfar.net/skaha/jwst-notebook:25.07.25`
   - **name**: anything, e.g. `mophongo`
   - **cores**: 2
   - **memory**: 48 GB
4. Launch, wait for it to turn green, and open it.
5. In JupyterLab, **File → New → Terminal**.

That terminal is a normal shell on a machine with `/arc` mounted and outbound
internet. Everything below is typed there.

On cores: 2 is the standard, for a full field or a trial patch alike. Measured
CPU use is about 0.2 of a core — the run waits on `/arc` rather than computing,
and the fitting path has no thread pool — so asking for 8 or 16 only idles
allocation someone else could use, and a large request waits longer to be
scheduled when the platform is busy.

On memory: ask for 48 GB and do not economise. The pipeline loads the full
3.3 GB F444W mosaic and the 3.3 GB segmap regardless of how small a trial patch
you fit, and the UDS runs peak near 34 GB. A 16 GB session is killed partway
through with no Python traceback, which looks like a mysterious silent failure,
and 32 GB leaves no headroom for a wider patch or a redder band. The portal
offers a 32 GB default, but that is a default and not a cap — the menu goes far
higher.

---

## Part 4 — Install and run

### 4.1 Get mophongo

```bash
mkdir -p /arc/home/$USER/run
cd /arc/home/$USER/run
git clone https://github.com/ivolabbe/mophongo.git
```

### 4.2 Build the environment

The image ships astropy 6.1 and numpy 1.26, both too old for mophongo, so build
a clean virtualenv rather than reusing the image's packages. It lives on `/arc`
and persists after the session ends, so this is a one-time cost of a few
minutes.

```bash
cd /arc/home/$USER/run
python -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -e mophongo
```

Check it:

```bash
./venv/bin/python -c "from mophongo.pipeline import RunConfig; print('ok')"
```

Set a writable matplotlib cache, or every command re-warns and rebuilds its font
cache (the container's home is read-only):

```bash
export MPLCONFIGDIR=/arc/home/$USER/run/.mplconfig
export MPLBACKEND=Agg
mkdir -p $MPLCONFIGDIR
```

### 4.3 PSF grids — nothing to do

The MJD-tagged ePSF grids are **built on the fly**. `data/PSF/*` is gitignored,
so a fresh clone has none of them, but `psf_autobuild` defaults to on: when no
file under `psf_dir` matches the config's pattern, `Pipeline._load_epsf` runs
`PSFFactory` over the band's exposure list (the `_wcs.csv` already on arc, which
is where the MJD tags come from), writes the grids to `psf_dir`, and carries on.

So just make the directory and let the first run populate it:

```bash
mkdir -p /arc/home/$USER/run/PSF
```

Two things worth knowing:

- it is slow — the code says so when it starts — but the grids are cached on
  `/arc`, so it is a one-time cost that later runs and other bands reuse;
- `stpsf` downloads about 1 GB of reference data on first use, automatically,
  into `/arc/home/<user>/data/stpsf-data`. Verified on CANFAR; no manual
  install needed.

If a run ever reports that no grids match the pattern *after* PSFFactory ran,
the config's `pattern_hi`/`pattern_lo` and the generated filenames disagree —
that is a config problem, not a missing-data problem.

### 4.4 Write the run config

The MINERVA data are all on arc; the run config just has to point at them.
Almost everything there is gzipped and the pipeline wants plain FITS, so
decompress the four large inputs once into a working directory:

```bash
cd /arc/home/$USER/run
mkdir -p data
M=/arc/projects/minerva/uds

gunzip -c $M/mosaics/nircam/n3.0/bkgsub/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_sci_bkgsub.fits.gz \
        > data/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_sci_bkgsub.fits
gunzip -c $M/catalogs/n3.0_v1.2/ACS+WEBB_chi-mean/ancillary/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits.gz \
        > data/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits
gunzip -c $M/mosaics/miri/m3.1/uds-sbkgsub-v3.1-80mas-f770w_drz_sci_extrabkg.fits.gz \
        > data/uds-sbkgsub-v3.1-80mas-f770w_drz_sci_extrabkg.fits
gunzip -c $M/mosaics/miri/m3.1/uds-sbkgsub-v3.1-80mas-f770w_drz_wht.fits.gz \
        > data/uds-sbkgsub-v3.1-80mas-f770w_drz_wht.fits

# the MIRI frame table ships as _f770_wcs.csv; the filter parser wants f770w
cp $M/mosaics/miri/m3.1/uds-v3.1_f770_wcs.csv data/uds-v3.1_f770w_wcs.csv
```

That is about 8 GB in `data/`. The SUPER catalog and the F444W frame table are
not gzipped, so the config reads those straight out of `arc:projects/minerva`
with no copy.

Now write the config. This is `examples/minerva/uds_f770w.json` with its paths
pointed at arc:

```bash
cd /arc/home/$USER/run
sed -e "s|/Users/ivo/Astro/PROJECTS/MINERVA/data/UDS/n3.0/|$PWD/data/|" \
    -e "s|/Users/ivo/Astro/PROJECTS/MINERVA/data/UDS/m3.1/|$PWD/data/|" \
    mophongo/examples/minerva/uds_f770w.json > uds_f770w_canfar.json
```

then open `uds_f770w_canfar.json` in the JupyterLab editor and fix the remaining
paths by hand so they read:

| key | value |
|---|---|
| `sci_hi` | `/arc/home/<user>/run/data/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_sci_bkgsub.fits` |
| `segmap` | `/arc/home/<user>/run/data/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits` |
| `catalog` | `/arc/projects/minerva/uds/catalogs/n3.0_m3.1_v1.2.1/ACS+WEBB_chi-mean/MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits` |
| `csv_hi` | `/arc/projects/minerva/uds/mosaics/nircam/n3.0/grizli/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_wcs.csv` |
| `sci_lo` | `/arc/home/<user>/run/data/uds-sbkgsub-v3.1-80mas-f770w_drz_sci_extrabkg.fits` |
| `wht_lo` | `/arc/home/<user>/run/data/uds-sbkgsub-v3.1-80mas-f770w_drz_wht.fits` |
| `csv_lo` | `/arc/home/<user>/run/data/uds-v3.1_f770w_wcs.csv` |
| `psf_dir` | `/arc/home/<user>/run/PSF` |
| `out_dir` | `/arc/home/<user>/run/out/uds_f770w` |

If you can run the toolkit locally (Part 5), `arcify.py` does all of this
automatically and is less error-prone.

### 4.5 Run it

Start small. `r_trial` is the fitted patch radius in arcmin; edit it to `0.25`
in the config for a first pass, which takes minutes instead of the better part
of an hour.

```bash
cd /arc/home/$USER/run
./venv/bin/python -m mophongo.pipeline uds_f770w_canfar.json all
```

Expect roughly three minutes of silence at the start while Python imports
mophongo off the network filesystem. That is normal, not a hang.

When it works, set `r_trial` back to `1.0` (or `0` for the full field) and run
again.

Measured on 8 cores, for reference:

| patch | sources fitted | wall time | peak memory |
|---|---|---|---|
| `r_trial` 0.25 | 810 | 8 min | 33.6 GB |
| `r_trial` 0.6 | 2242 | 14 min | 34 GB |

Most of the small run is fixed cost — loading the mosaics, building or loading
the PSF and kernel maps — which is why a quarter-size patch is not four times
faster. The PSF and kernel maps are cached in `out_dir` and reused by later runs
of the same configuration, so repeat runs skip that part.

### 4.6 Look at the results

```bash
ls -lh /arc/home/$USER/run/out/uds_f770w
```

| file | what |
|---|---|
| `uds_770_fit_table.fits` | the photometry: `flux_<i>`, `flux_<i>_total`, `err_<i>` |
| `uds_770_residual.fits` | data minus model, the first thing to check |
| `scenes/uds_770_scene_*.png` | per-scene diagnostics |
| `uds_770_shift_field.png` | the fitted astrometric shift field over the whole field |
| `uds_770_kernel.fits`, `uds_770_psf_*.fits` | PSF and kernel maps |
| `uds_770.log` | the run log |

The FITS files can be opened in the same session with astropy, or in the
JupyterLab file browser. Nothing needs downloading.

---

## Part 5 — Optional: drive runs from your laptop

Worth it once you are running several bands, since it launches jobs in parallel
without keeping a browser session open. It needs two packages installed locally:

```bash
python -m venv ~/.venvs/canfar
~/.venvs/canfar/bin/pip install vos cadcutils skaha
```

Then, from `mophongo/examples/canfar/`:

```bash
export CANFAR_USER=<user>
~/.venvs/canfar/bin/cadc-get-cert -u <user>    # 10-day certificate

P=~/.venvs/canfar/bin/python
$P submit.py push                              # source, job scripts, PSF grids
$P submit.py setup                             # build the venv on /arc
$P arcify.py ../minerva/uds_f770w.json         # rewrite paths for arc
$P submit.py stage uds_f770w
$P submit.py run   uds_f770w
$P submit.py fetch uds_f770w                   # bring the small outputs home
```

All four UDS bands at once:

```bash
$P arcify.py ../minerva/uds_f770w.json ../minerva/uds_f1280w.json \
             ../minerva/uds_f1500w.json ../minerva/uds_f1800w.json
$P submit.py stage uds_f770w uds_f1280w uds_f1500w uds_f1800w
$P submit.py run   uds_f770w uds_f1280w uds_f1500w uds_f1800w
```

See `README.md` here for the details of what each step does.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Cannot list `arc:projects/minerva` | not in the `minerva` group yet — ask `<adam>` |
| `No such file or directory: /arc/home/<user>` | home not created yet; see Part 2 |
| Job or session dies with no Python traceback | out of memory. Use 48 GB; memory scales with the mosaic, not `r_trial` |
| Nothing happens for three minutes at startup | importing mophongo off `/arc`; normal |
| `ModuleNotFoundError` after install | you used the image's `python` instead of `./venv/bin/python` |
| matplotlib font-cache warnings on every command | set `MPLCONFIGDIR` (Part 4.2) |
| `invalid choice: []` | old mophongo; the `all` argument is now optional, `git pull` |
| PSF pattern matches nothing *after* PSFFactory ran | the config pattern and the generated filenames disagree; see Part 4.3 |
| First run spends a long time before fitting | building the ePSF grids; cached on `/arc` afterwards |
| `ssh` to CANFAR says "sftp connections only" | expected. That endpoint moves files; compute is the Science Portal or the skaha API |

## Reference

- CANFAR docs: https://www.opencadc.org/canfar/
- Science Portal: https://www.canfar.net/science-portal
- Storage browser: https://www.canfar.net/storage/arc/list
- What the MINERVA products are and where: `MINERVA/data/00WHERE`
- Getting data on and off CANFAR from a laptop: `MINERVA/data/00CANFAR`
- Background and the traps found setting this up:
  `mophongo/scratch/canfar/RUNNING_ON_CANFAR.md`
