import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np
import matplotlib.pyplot as plt

from mophongo.pipeline import run_photometry
from utils import make_simple_data, save_diagnostic_image, save_flux_vs_truth_plot


def test_pipeline_flux_recovery(tmp_path):
    images, segmap, catalog, psfs, truth_img = make_simple_data()
    table, resid = run_photometry(images, segmap, catalog, psfs)

    table["flux_true"] = catalog["flux_true"]  # add flux_true to the table

    # Plot for high-res (flux_0) vs truth
    flux_hi_plot = tmp_path / "flux_hi_vs_true.png"
    save_flux_vs_truth_plot(
        flux_hi_plot,
        table["flux_true"],
        table["flux_0"],
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)"
    )
    assert flux_hi_plot.exists()

    # Plot for low-res (flux_1) vs truth
    flux_lo_plot = tmp_path / "flux_lo_vs_true.png"
    save_flux_vs_truth_plot(
        flux_lo_plot,
        table["flux_true"],
        table["flux_1"],
        label="Flux (lowres)",
        xlabel="True Flux",
        ylabel="Recovered Flux (lowres)"
    )
    assert flux_lo_plot.exists()

    # Plot for flux_lo vs flux_hi
    flux_lo_hi_plot = tmp_path / "flux_lo_vs_hi.png"
    save_flux_vs_truth_plot(
        flux_lo_hi_plot,
        table["flux_0"],
        table["flux_1"],
        label="Flux (lowres) vs (hires)",
        xlabel="Recovered Flux (hires)",
        ylabel="Recovered Flux (lowres)"
    )
    assert flux_lo_hi_plot.exists()

    model = images[1] - resid[1]
    fname = tmp_path / "diagnostic.png"
    save_diagnostic_image(fname, truth_img, images[0], images[1], model, resid[1], segmap=segmap)
    assert fname.exists()

    # Report statistics for flux recovery
    import numpy as np
    for idx in range(len(psfs)):
        col = f"flux_{idx}"
        ratio = np.array(table[col]) / np.array(table["flux_true"])
        print(f"flux_{idx}/flux_true percentiles: 5th={np.percentile(ratio,5):.2f}, "
              f"16th={np.percentile(ratio,16):.2f}, 50th={np.percentile(ratio,50):.2f}, "
              f"84th={np.percentile(ratio,84):.2f}, 95th={np.percentile(ratio,95):.2f}")

    assert resid.shape[0] == len(images)
