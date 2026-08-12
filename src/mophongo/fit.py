from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from tqdm import tqdm

from .templates import Template, Templates

import logging

logger = logging.getLogger(__name__)

# full weights need to be calcuate like
# template_var = scipy.signal.fftconvolve(K**2, 1 / wht1, mode='same')  # same shape as template
# Iterate if needed (since A appears in w(x)):
# First fit using weights = wht2
# Compute A (amplitude)
# Recompute weights using full formula
# Refit using updated weights if you want accurate errors
# wht_tot = 1 / (1 / wht2 + A**2 * template_var)
# Pass wht_tot to your SparseFitter.
# Multiple templates: you must apply the same logic to each template independently. This means different pixels may have different total weights for each template, depending on each one's amplitude and support.
# Correlated templates (overlapping) require full covariance accounting; your current implementation approximates this by assuming per-template independence.
# If template noise is negligible, simplify to: weights = wht2 (as in your current default).
# Flux-dependent variance (via A^2) introduces mild nonlinearity; it's safe to fix A from initial fit for a single iteration.


@dataclass
class FitConfig:
    """Configuration for template fitting: solver, astrometry and apertures."""

    positivity: bool = True
    # Flux-block ridge. None (default) = adaptive, 1e-6 x the median positive
    # diagonal of the normal matrix; 0.0 = genuinely unregularized; > 0 = that
    # explicit value. JSON configs write null for the adaptive default.
    reg_flux: float | None = None
    bad_value: float = np.nan
    cg_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"M": None, "maxiter": 500, "atol": 1e-6}
    )

    # condense fit astrometry flags into one: fit_astrometry_niter = 0, means not fitting astrometry
    fit_astrometry_niter: int = 5  # Max astrometry refinement passes (0 → disabled)
    # Stop iterating a scene once its largest per-template shift increment
    # drops below this tolerance (fit-grid pixels). The linearized shift solve
    # only captures part of a large offset per pass, so set
    # fit_astrometry_niter to the maximum passes allowed and let this tol stop.
    # 0.1 sits just above the statistical floor of the weakest scene the
    # anchor cuts admit -- 5 bright members (scene_minimum_bright) at
    # snr_thresh_astrom = 15 give a scene centroid good to ~0.08 fit pixels --
    # and well below PSF-matching centroid systematics, which are a bias no
    # tolerance can iterate away. The increment tested is the applied (damped)
    # one, so the last sub-tolerance step is on the templates, not discarded.
    astrom_shift_tol: float = 0.1
    # Damping applied to each pass's shift increment before it is applied to
    # the templates. The central-difference shift basis underestimates
    # gradients of sharp structure, so the linearized step can overshoot by
    # k/sin(k) per mode; damping keeps the iteration contracting even for
    # scenes dominated by marginally sampled cores, at the cost of ~1 extra
    # pass. 1.0 = undamped.
    astrom_damping: float = 0.8
    fit_astrometry_joint: bool = True  # Use joint astrometry fitting, or separate step
    # --- astrometry options -------------------------------------------------
    reg_astrom: float = 1e-4
    snr_thresh_astrom: float = 15.0  # 0 → keep all sources
    astrom_isolation_thresh: float = 0.7  # min flux dominance to include in astrometry (0–1); 0.0 = no cut
    # Exclude sources flagged is_star from the astrometric shift fit. Off by
    # default: unsaturated stars are the best astrometric anchors, and
    # saturated ones are already isolated into their own scenes.
    astrom_exclude_stars: bool = False
    # Cap each astrometric anchor's leverage at this quantile of the scene's
    # anchor information (a_i^2 <G,w,G>), or None to leave weights alone.
    # Leverage grows as flux squared, so one bright source can carry a scene
    # -- and if it is extended with an asymmetric colour gradient, its
    # residual dipole is indistinguishable from a shift and drags the field.
    # The cap bounds influence without changing the shift that anchor
    # measures. See assemble_scene_system_AB and TODO.md (cross-anchor IRLS
    # is the complementary fix this cannot provide).
    # 0.9 clips only the top tail: the handful of anchors carrying more
    # information than nine-tenths of their scene.
    astrom_leverage_cap: float | None = 0.9
    astrom_model: str = "gp"  # 'poly' or 'gp'
    astrom_centroid: str = "centroid"  # "centroid" (=old) | "correlation"
    astrom_kwargs: dict[str, dict] = field(
        default_factory=lambda: {"poly": {"order": 0}, "gp": {"length_scale": 400}}
    )
    #    astrom_kwargs={'poly': {'order': 2}, 'gp': {'length_scale': 400}}
    #    multi_resolution_method: str = "upsample"  # 'upsample' or 'downsample'
    multi_resolution_method: str = "upsample"  # 'upsample' or 'downsample'
    normal: str = "tree"  # 'loop' or 'tree'
    # None → derive from astrometric model order in __post_init__
    # Minimum bright sources per scene. If None reverts to (n_poly+1)*(n_poly+2)
    scene_minimum_bright: int = 5

    # Photometry aperture control:
    # - float/int: fixed aperture diameter size (in arcsec or pixels per `aperture_units`)
    # - str: column name in the input catalog for per-source aperture sizes
    # - None: fallback to 1.5 * FWHM (in pixels) measured from template
    aperture_diam: float | np.ndarray | None = None  # image measurement aperture (diameter)
    aperture_catalog: float | str | None = None  # catalog aperture (diameter or table column name)
    aperture_units: str = "arcsec"  # "arcsec" or "pix"
    # Catalog-side aperture-to-total, totcor_cat = (f_kron/f_aper) / EE_H(k*R_kron)
    # (the flux-estimator report's "tcorH", renamed). Computed when the three
    # column names below exist in the input catalog; radius column in arcsec.
    cat_kron_flux_col: str | None = None  # detection-catalog Kron (AUTO) flux
    cat_aper_flux_col: str | None = None  # detection-catalog flux in the R_phi aperture
    cat_kron_radius_col: str | None = None  # circularized Kron radius [arcsec]
    cat_kron_k: float = 2.5  # Kron scaling: EE_H evaluated at k * R_kron

    # Template extraction: dilate each segment by this many pixels (disk radius)
    # to capture more of the point-source PSF wings. Off by default (0): the
    # IDL reference (subphot.pro::build_cube) uses the exact segment, dilation
    # only adds a ring of sky noise, and its background tie-break is
    # catalog-id ordered rather than geometric. Proper wing recovery is the
    # job of template extension, not dilation.
    template_dilate_segmap: int = 0
    # By default template extension is applied to every source, including
    # catalog deblend children. Set True to preserve deblended child templates
    # without PSF-wing/model completion when that is desired for a validation run.
    skip_template_extension_for_deblended: bool = False
    # PSF-wing completion fills only background pixels of the (dilated)
    # segmentation map by default, so blended neighbours keep ownership of
    # their own segment pixels. Set False to fill every zero template pixel.
    extend_wings_background_only: bool = True

    # --- template build scheme ---------------------------------------------
    # One selector over the four schemes, for 1-1 comparison:
    #   'none'      segment-masked detection data only
    #   'psf_wings' least-squares PSF wings outside the segment, smooth faint
    #               limit, normalised before neighbour-owned pixels are zeroed
    #               (alias: 'default')
    #   'psf'       'none' + Templates.extend_with_psf (template convolved
    #               with the PSF fills the zero pixels)
    #   'psf_model' 'none' + Templates.extend_with_psf_model
    #   'wren'      wren/dev-wren _extended_composite (ownership + SNR blend)
    #   'classic'   IDL subphot.pro::build_cube (hard switch below tmpl_snrlo)
    # The build-time schemes live in mophongo.template_schemes; the knobs below
    # are theirs and are ignored by the other modes.
    extend_mode: str = "psf_wings"
    # 'psf_wings' scheme: in-segment SNR at which the template is pure data. Below it
    # the core rolls off to the scaled PSF, reaching a pure point source at 0.
    psf_wings_snrlo: float = 5.0
    psf_wings_blend_p: float = 2.0
    psf_wings_rms: float | None = None  # None: robust_sigma of the detection image
    wren_ee_fraction: float = 0.95  # EE fraction setting the support cap R95
    wren_fit_snrlo_psf: float = 10.0  # core-weight onset is 1.5x this
    wren_wings_snr_psf: float = 3.0  # per-annulus weight onset
    wren_blend_p: float = 2.0  # blend-weight rolloff exponent
    wren_blend_annulus: float = 0.15  # halo annulus width, arcsec
    # Detection-image sky rms used when weights[0] is absent (the config-driven
    # path has no detection weight map). None: measure it with sky_sigma.
    wren_bg_rms: float | None = None
    classic_tmpl_snrlo: float = 15.0  # below this in-segment SNR: pure point source
    classic_rms: float | None = None  # None: robust_sigma of the detection image

    # scene processing
    run_scene_solver: bool = True  # Whether to run the scene solver at all
    scene_coupling_thresh: float = 1e-3  # 1% leakage threshold for scene splitting
    # Soft cap on templates per scene. Components over the cap are split by
    # raising the coupling threshold locally (inside that component only);
    # the accepted local leakage is logged. None = no cap.
    scene_max_size: int | None = 800
    # Max distance (px) to merge underfilled scenes. Bounded rather than inf so
    # merging stays local.
    scene_max_merge_radius: float = 1000.0
    generate_scene_catalog: bool = False  # If True, generate scene catalog and exit

    def __post_init__(self):
        # Derive scene_minimum_bright from astrometric polynomial order if not provided
        if self.scene_minimum_bright is None:
            try:
                poly_order = int(self.astrom_kwargs.get("poly", {}).get("order", 1))
            except Exception:
                poly_order = 0
            # default to 2x # of Chebyshev terms + 1
            n_poly = (poly_order + 1) * (poly_order + 2)
            self.scene_minimum_bright = n_poly + 1



class SparseFitter:
    """Build sparse normal equations and quick flux/error estimates.

    Flux solving lives in :class:`mophongo.scene_fitter.SceneFitter`; this
    class owns the normal-matrix assembly, the model/residual images and the
    covariance-free flux and error estimators.
    """

    def __init__(
        self,
        templates: List[Template],
        image: np.ndarray,
        weights: np.ndarray | None = None,
        config: FitConfig | None = None,
    ) -> None:
        if weights is None:
            weights = np.ones_like(image)

        self._orig_templates = templates  # keep original templates List object
        self.templates = templates.copy()  # work in list copy for fitting, modifying

        self.n_flux = len(templates)
        for i, tmpl in enumerate(self.templates):
            tmpl.is_flux = True
            tmpl.col_idx = i

        self.image = image
        self.weights = weights
        self.config = config or FitConfig()
        self._ata = None
        self._atb = None
        self.solution: np.ndarray | None = None


    @staticmethod
    def _slice_intersection(
        a: tuple[slice, slice], b: tuple[slice, slice]
    ) -> tuple[slice, slice] | None:
        y0 = max(a[0].start, b[0].start)
        y1 = min(a[0].stop, b[0].stop)
        x0 = max(a[1].start, b[1].start)
        x1 = min(a[1].stop, b[1].stop)
        if y0 >= y1 or x0 >= x1:
            return None
        return slice(y0, y1), slice(x0, x1)

    def _weighted_norm(self, tmpl: Template) -> float:
        """Return the weighted L2 norm of ``tmpl``.
        The norm is computed by summing ``data * weight * data`` over the
        template support in the image space.
        """
        sl = tmpl.slices_original
        data = tmpl.data[tmpl.slices_cutout]
        w = self.weights[sl]
        wnorm = float(np.sum(data * w * data))
        tmpl.wnorm = wnorm
        return wnorm

    def build_normal(self) -> None:
        """Dispatch to the configured normal-matrix builder."""
        normal = getattr(self.config, "normal", "tree")
        if normal != "tree":
            raise ValueError(f"unsupported normal={normal!r}; only 'tree' is implemented")
        self.build_normal_tree()

    def build_normal_tree(self) -> None:
        """Construct normal matrix using an STRtree to find overlaps."""
        from shapely.geometry import box
        from shapely.strtree import STRtree

        # scan for low norm templates but keep them for now
        norms = np.array([self._weighted_norm(t) for t in self.templates])
        tol = 1e-6 * np.median(norms)
        if np.sum(norms < tol) > 0:
            logger.info("Found %d templates with low norm.", np.sum(norms < tol))
        # self.templates = [self.templates[i] for i in keep]
        # norms = [norms[i] for i in keep]

        n = len(self.templates)
        ata = lil_matrix((n, n))
        atb = np.zeros(n)

        boxes = []
        for i, tmpl in enumerate(tqdm(self.templates, total=n, desc="Building Normal matrix")):
            sl_i = tmpl.slices_original
            data_i = tmpl.data[tmpl.slices_cutout]
            w_i = self.weights[sl_i]
            img_i = self.image[sl_i]
            atb[i] = np.sum(data_i * w_i * img_i)
            ata[i, i] = norms[i]

            y0, y1, x0, x1 = tmpl.bbox
            geom = box(x0, y0, x1, y1)
            boxes.append(geom)

        tree = STRtree(boxes)

        for i, geom in enumerate(boxes):
            sl_i = self.templates[i].slices_original
            for j in tree.query(geom):
                j = int(j)
                if j <= i:
                    continue
                inter = self._slice_intersection(sl_i, self.templates[j].slices_original)
                if inter is None:
                    continue
                w = self.weights[inter]
                sl_i_local = (
                    slice(
                        inter[0].start - sl_i[0].start + self.templates[i].slices_cutout[0].start,
                        inter[0].stop - sl_i[0].start + self.templates[i].slices_cutout[0].start,
                    ),
                    slice(
                        inter[1].start - sl_i[1].start + self.templates[i].slices_cutout[1].start,
                        inter[1].stop - sl_i[1].start + self.templates[i].slices_cutout[1].start,
                    ),
                )
                sl_j = self.templates[j].slices_original
                sl_j_local = (
                    slice(
                        inter[0].start - sl_j[0].start + self.templates[j].slices_cutout[0].start,
                        inter[0].stop - sl_j[0].start + self.templates[j].slices_cutout[0].start,
                    ),
                    slice(
                        inter[1].start - sl_j[1].start + self.templates[j].slices_cutout[1].start,
                        inter[1].stop - sl_j[1].start + self.templates[j].slices_cutout[1].start,
                    ),
                )
                arr_i = self.templates[i].data[sl_i_local]
                arr_j = self.templates[j].data[sl_j_local]
                val = np.sum(arr_i * arr_j * w)
                ata[i, j] = val
                ata[j, i] = val

        self._ata = ata.tocsr()
        self._atb = atb
        self.rtree = tree

    def model_image(self) -> np.ndarray:
        if self.solution is None:
            raise ValueError("Solve system first")
        model = np.zeros_like(self.image, dtype=float)
        for coeff, tmpl in zip(self.solution, self._orig_templates):
            model[tmpl.slices_original] += coeff * tmpl.data[tmpl.slices_cutout]
        model[(self.weights <= 0) | np.isnan(self.weights)] = 0.0
        return model

    @property
    def ata(self):
        if self._ata is None:
            self.build_normal()
        return self._ata

    @property
    def atb(self):
        if self._atb is None:
            self.build_normal()
        return self._atb

    def add_flux_priors(self, idx, mu, sigma, *, floor=1e-12):
        """
        Add Gaussian flux priors to the UNwhitened normal:
            (x_i - mu_i)^2 / sigma_i^2  for i in sel.

        sel   : bool mask of length n or 1D integer indices
        mu    : scalar or array broadcastable to n (or |sel|); prior mean(s)
        sigma : scalar or array broadcastable to n (or |sel|); prior stddev(s)
        """
        import numpy as np
        import scipy.sparse as sp

        # Ensure normal is built (triggers pruning, etc., via properties)
        if self._ata is None or self._atb is None:
            _ = self.ata  # builds and caches
            _ = self.atb

        A = self._ata.tocsr()
        b = np.asarray(self._atb, dtype=float)
        n = b.shape[0]

        # normalize selection to integer indices
        if idx.size == 0:
            return
        nsel = len(idx)

        # broadcast mu/sigma to selected size
        mu_all = np.broadcast_to(mu, (nsel,)) if np.ndim(mu) else float(mu)
        sig_all = np.broadcast_to(sigma, (nsel,)) if np.ndim(sigma) else float(sigma)

        mu_sel = (mu_all if np.ndim(mu_all) else np.full(nsel, mu_all))[idx]
        sig_sel = (sig_all if np.ndim(sig_all) else np.full(nsel, sig_all))[idx]

        # guards
        sig_sel = np.maximum(np.asarray(sig_sel, float), floor)
        lam = 1.0 / (sig_sel**2)  # precisions

        # RHS: b[i] += λ_i * μ_i
        b[idx] += lam * np.asarray(mu_sel, float)

        # Diagonal: A_ii += λ_i
        diag_inc = np.zeros(n, float)
        diag_inc[idx] = lam
        A = A + sp.diags(diag_inc, 0, shape=A.shape, format="csr")

        # write back
        self._ata = A
        self._atb = b


    def residual(self) -> np.ndarray:
        return self.image - self.model_image()

    def quick_flux(self, templates: Optional[List[Template]] = None) -> np.ndarray:
        """Return quick flux estimates based on template data and image."""
        if templates is None:
            templates = self._orig_templates
        return Templates.quick_flux(templates, self.image)

    def predicted_errors(self, templates: Optional[List[Template]] = None) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        if templates is None:
            templates = self._orig_templates
        return Templates.predicted_errors(templates, self.weights)

    def flux_and_rms(
        self, templates: Optional[List[Template]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return flux estimates and RMS errors for templates.

        Uses existing template fluxes when available; otherwise computes
        quick fluxes and predicted errors for the first ``n_flux`` templates.

        Args:
            templates: Optional list of templates to evaluate. Defaults to
                the original templates supplied to the fitter.

        Returns:
            Tuple ``(flux, rms)`` containing the flux estimates and
            corresponding RMS errors for each template.
        """
        if templates is None:
            templates = self._orig_templates

        if templates and templates[0].flux != 0:
            flux = np.array([t.flux for t in templates[: self.n_flux]])
        else:
            flux = self.quick_flux(templates)[: self.n_flux]

        rms = self.predicted_errors(templates)[: self.n_flux]
        return flux, rms




