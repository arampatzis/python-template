# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "template"
copyright = "2025, G. Arampatzis"
author = "G. Arampatzis"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "myst_parser",  # enables Markdown support via MyST
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
