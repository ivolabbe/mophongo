# %%
# MINERVA-UDS DR0 F770W template-fitting photometry.
#
# All inputs and settings live in uds_770_dr0.json (mophongo.pipeline.RunConfig).
# Equivalent command line:  python -m mophongo.pipeline uds_770_dr0.json
# Run from examples/ (config paths are relative to it).
from mophongo.pipeline import Pipeline

pipe = Pipeline.from_config("uds_770_dr0.json")

# %%
pipe.build_psfs()      # per-band PSF region maps (geojson-cached in out_dir)
pipe.build_kernels()   # matching-kernel map (geojson-cached)

# %%
pipe.run()             # load data (+ footprint/trial cuts, bg/ivar, NaN guard) and fit
pipe.write_outputs()   # residual FITS, fit table, scene PNGs + scene catalog

# %%
# interactive inspection examples
# pipe.table                                       # fitted catalog
# pipe.prm_kern.psfs.shape                         # kernel cube
# pipe.scenes[0].plot(pipe.images[0], pipe.segmap) # scene diagnostics
