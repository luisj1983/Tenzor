"""Sphinx configuration for Tenzor documentation."""

project = "Tenzor"
copyright = "2024, Tenzor Contributors"
author = "Tenzor Contributors"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Try to load breathe for Doxygen XML bridge
try:
    import breathe
    extensions.append("breathe")
    breathe_projects = {"tenzor": "../docs/api/xml"}
    breathe_default_project = "tenzor"
except ImportError:
    pass

# Try to load sphinx_autodoc_typehints
try:
    import sphinx_autodoc_typehints
    extensions.append("sphinx_autodoc_typehints")
except ImportError:
    pass

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme
html_theme = "sphinx_rtd_theme"
try:
    import sphinx_rtd_theme
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]

# MyST (Markdown) parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
