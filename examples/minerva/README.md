# MINERVA runs — every field, every MIRI band

Generated configs, one per field and MIRI band, in the mould of
`../uds_770_dr0.json`: the background-subtracted F444W mosaic on the 40 mas
NIRCam grid as the template image, the release segmap and SUPER catalog, and
one 80 mas MIRI band (`_drz_sci_extrabkg` + `_drz_wht`) as the image to fit.

Regenerate after staging new data:

```
cd examples && python make_minerva_configs.py [field ...]
```

The generator reads everything it can off the staged files — frame counts from
the two `_wcs.csv` tables, the trial-patch centre from a scan of the MIRI
weight map — so the configs carry no hand-copied numbers. Staging itself is
`MINERVA/data/stage/` (see its README).

## Running

```
cd examples/minerva
python -m mophongo.pipeline uds_f770w.json                  # all steps
python -m mophongo.pipeline uds_f770w.json psfs kernels     # selected steps
```

Outputs go to `<field>_<band>/` next to the config, with a full run log at
`<field>_<band>/<name>.log`.

Configs ship with `r_trial = 0.6`, i.e. a 0.6 arcmin trial patch at the deepest
well-covered position of that band's weight map. Set `r_trial` to 0 for a
full-field run.

How deep that patch actually is varies, and the generator logs the percentile
it reached within the band's footprint:

| | patch depth percentile |
|---|---|
| UDS, 4 bands | 94-99 |
| COSMOS, 6 bands | 83-98 |
| EGS, 5 of 7 bands | 82-97 |
| EGS F1280W | 21 |
| EGS F1800W | 15, coverage relaxed to 0.95 |

The two EGS outliers are geometry, not a bug: those footprints are too
fragmented for a 1.2 arcmin box to fit anywhere deep. Shrink `r_trial` or run
the full field for those bands.

## Release versions

| Field | NIRCam | MIRI | Segmap | SUPER catalog |
|---|---|---|---|---|
| UDS | v3.0 | m3.1 | n3.0_v1.2 | n3.0_m3.1_v1.2.1 |
| COSMOS | v3.0 | m3.0 | n3.0_v1.0 | n3.0_m3.0_v1.0.1 |
| EGS | v2.0 | m2.1 | n2.0_v1.3 | n2.0_m2.1_v1.3.1 |

## Caveats

- `psf_size` is 4.0 arcsec in every config because the same value sets the
  high-resolution support and the F444W ePSF grids are only 4.09 arcsec across.
  That is generous at F770W and tight at F1800W/F2100W; see `TODO.md`.
- EGS has no ePSF grids yet, so the first run of each band generates them
  (`psf_autobuild`, on by default). Expect the PSF step to take much longer
  there than in UDS or COSMOS.
- EGS coverage is highly uneven across its seven MIRI bands, so the trial patch
  of one band is not the trial patch of another.
- The F2100W broadening (`DEFAULT_PSF_GAUSSIAN_FWHM_ARCSEC`) is extrapolated,
  not measured.
