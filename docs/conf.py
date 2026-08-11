"""Sphinx configuration for mophongo documentation."""

project = "mophongo"
author = "Ivo Labbe"
copyright = "2026, Ivo Labbe"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

# Only these documents are part of the rendered docs; the rest of docs/
# holds internal development notes that are not published.
include_patterns = [
    "index.md",
    "overview.md",
    "quickstart.md",
    "pipeline.md",
    "repair.md",
    "diagnostics.md",
    "outputs.md",
    "psf.md",
    "psf_maps.md",
    "templates.md",
    "fitting.md",
    "catalog.md",
    "preprocessing.md",
    "simulation.md",
    "api.rst",
    "api/**",
]

myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath", "amsmath"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "photutils": ("https://photutils.readthedocs.io/en/stable/", None),
}

html_theme = "furo"
html_title = "mophongo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1f5f8b",
        "color-brand-content": "#1f5f8b",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6db3d9",
        "color-brand-content": "#6db3d9",
    },
}
# Every page gets the full section tree in the left sidebar plus its own
# in-page table of contents on the right (furo defaults); show deeper
# section levels in the sidebar.
html_theme_options["navigation_with_keys"] = True
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
