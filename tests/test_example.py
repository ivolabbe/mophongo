#%%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the source directory (relative to current directory)
current_dir = os.getcwd()
src_path = os.path.join(current_dir, '../src')
print(src_path)

if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(sys.path)

# Now import
from mophongo.psf import PSF
from mophongo.fit import SparseFitter
from mophongo.templates import Templates, Template
from utils import make_simple_data, lupton_norm 
from mophongo import pipeline  
from photutils.segmentation import SegmentationImage, SourceCatalog

#%%
# test_example.py
def test_addition():
    assert 1 + 1 == 2


if __name__ == "__main__":
    exit
    #from mophongo.psf import PSF
    #PSF.gaussian(5,0.1,0.1).array.round()

    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(nsrc=1, size=39, ndilate=2, peak_snr=100)

    plt.imshow(images[0],origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    plt.imshow(images[1],origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    plt.imshow(segmap,origin='lower',cmap='gray',norm=lupton_norm(images[0]))

    # create templates
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])

    kernel = psf_hi.matching_kernel(psf_lo)

    positions = list(zip(catalog["x"], catalog["y"]))

    tmpls = Templates.from_image(images[0], segmap, positions, kernel)

    plt.imshow(tmpls.templates[0].data ,origin='lower',cmap='gray',norm=lupton_norm(tmpls.templates[0].data))
    plt.imshow(images[0],origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    plt.show()
    plt.imshow(tmpls._templates_hires[0].data,origin='lower',cmap='gray',norm=lupton_norm(tmpls._templates_hires[0].data))
    plt.show()

    print(positions)
    print(tmpls.templates[0].data.sum())
    print(tmpls._templates_hires[0].data.sum())
    print(tmpls._templates_hires[0].data.shape)

    obj = tmpls._templates_hires[0]
    for attr in dir(obj):
        print(attr, getattr(obj, attr))

    tmpls._templates_hires[0].data += 5

    images[0][obj.slices_original] = obj.data[obj.slices_cutout]
    plt.imshow(images[0],origin='lower',cmap='gray',norm=lupton_norm(images[0]))

    # %%
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(nsrc=30, size=151, ndilate=0, peak_snr=3)
    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms)
    # %%
    sig = np.max(np.abs(resid[1])) * 3
    plt.imshow(resid[0],origin='lower',cmap='gray',vmin=-sig, vmax=sig)
    plt.show()
    plt.imshow(resid[1],origin='lower',cmap='gray',vmin=-sig, vmax=sig)
    plt.show()
    # %%

    # %%
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(nsrc=1, size=71, ndilate=2, peak_snr=100)
    pos = tuple(catalog['x','y'][0])
    t = Template(images[0], pos, (40,40))
    tpad = t.pad((12,12), images[0].shape,  inplace=False)

    plt.imshow(images[0],origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    plt.show()
    plt.imshow(t.data,origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    plt.show()
    plt.imshow(tpad.data,origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    plt.show()


    plt.imshow(images[0],origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    tpad.data += 1
    images[0][tpad.slices_original] = tpad.data[tpad.slices_cutout]
    plt.show()
    plt.imshow(images[0],origin='lower',cmap='gray',norm=lupton_norm(images[0]))
    # %%

# %%
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(nsrc=30, size=151, ndilate=0, peak_snr=3)
    positions = list(zip(catalog["x"], catalog["y"]))
    segm = SegmentationImage(segmap)
    kernel = PSF.from_array(psfs[0]).matching_kernel(PSF.from_array(psfs[1]))

    tmpls = Templates.from_image(images[0], segmap, positions, kernel)
    print(len(tmpls), len(catalog))
    weight = 1.0 / rms[1]**2
    templates = Templates.prune_and_dedupe(tmpls.templates, weight)
    fitter = SparseFitter(templates, images[1], weight)
    fluxes, _ = fitter.solve()
    pred = fitter.predicted_errors()
    errs = fitter.flux_errors()

    tmpl_ids = [segmap[tmpl.position_original[::-1]] for tmpl in tmpls.templates]
    # Map template IDs to their indices in the catalog
    id_to_index = {id_: i for i, id_ in enumerate(catalog['id'])}
    # Find catalog indices for each template
    tmpl_idx = [id_to_index[tid] for tid in tmpl_ids]
    tmpl_idx = np.arange(len(catalog))

    # array files with fluxes where in segmap and has template, NaN where not

    idx=1
    catalog[f"flux_{idx}"] = np.nan
    catalog[f"err_{idx}"] = errs
    catalog[f"err_pred_{idx}"] = pred
    
    tmpls = Templates()
    tmpls.extract_templates(images[0], segmap, positions)
