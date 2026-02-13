from __future__ import annotations

from datetime import datetime

project = "PyMisha"
author = "Tanay Lab"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# Keep docs deterministic and readable.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

source_suffix = {
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "plans/*", "plans/**"]

html_theme = "furo"
html_title = "PyMisha"
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Avoid noisy failures from local-only references.
linkcheck_ignore = [
    r"^http://localhost",
    r"^https://localhost",
]
