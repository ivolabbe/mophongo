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
]

# Only these documents are part of the rendered docs; the rest of docs/
# holds internal development notes that are not published.
include_patterns = [
    "index.md",
    "overview.md",
    "quickstart.md",
    "pipeline.md",
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

html_theme = "pydata_sphinx_theme"
html_title = "mophongo"
