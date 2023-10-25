#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import warnings

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import date
from pathlib import Path

import git
from statsmodels.tools.sm_exceptions import DomainWarning

import pydeseq2

# Ignore DomainWarning raised by statsmodels when fitting a Gamma GLM with identity link.
warnings.simplefilter("ignore", DomainWarning)

# -- Project information -----------------------------------------------------

project = "PyDESeq2"
copyright = f"{date.today().year}, OWKIN"
author = "OWKIN"
version = pydeseq2.__version__
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx.ext.ifconfig",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}

autosectionlabel_prefix_document = True

# autodoc settings
autodoc_default_options = {
    "show-inheritance": False,
    "inherited-members": False,
    "members": True,
}

add_module_names = False
autosummary_generate = False  # Don't generate rst files automatically from autosummary
autoclass_content = "class"  # Don't document class __init__'s
autodoc_typehints = (
    "both"  # Show typehints in the signature + as content of the function
)
autodoc_typehints_format = "short"  # Shorten type hints
autodoc_member_order = (
    "groupwise"  # Sort automatically documented members by member type
)
python_use_unqualified_type_names = True  # Suppress module names

# # This is the expected signature of the handler for this event, cf doc
# def autodoc_skip_member_handler(app, what, name, obj, skip, options):
#     # Basic approach; you might want a regex instead
#     return name.endswith("__")
#
#
# # Automatically called by sphinx at startup
# def setup(app):
#     # Connect the autodoc-skip-member event from apidoc to the callback
#     app.connect("autodoc-skip-member", autodoc_skip_member_handler)
#

# Bibliography
bibtex_bibfiles = ["refs.bib"]
# Workaround to cite the same paper in several places in the API docs
suppress_warnings = ["bibtex.duplicate_label"]


# Napoleon settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True  # generate hyperlinks for parameter types

napoleon_type_aliases = {
    "DeseqDataSet": ":class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>`",
}

# Add any paths that contain templates here, relative to this directory.
# templates_path = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = [".rst", ".md"]
# source_suffix = '.rst'

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# Remove the prompt when copying examples
copybutton_prompt_text = ">>> "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "analytics_id": "UA-83738774-2",
    "logo_only": True,
    "display_version": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {}

# This must be the name of an image file (path relative to the configuration
# directory) that is the favicon of the docs. Modern browsers use this as
# the icon for tabs, windows and bookmarks. It should be a Windows-style
# icon file (.ico).
html_favicon = "_static/favicon.ico"


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pydeseq2doc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# As we defined the type of our args, auto doc is trying to find a link to a
# documentation for each type specified
# The following elements are the link that auto doc were not able to do
nitpick_ignore = [
    ("py:class", "pd.Series"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "ndarray"),
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "torch.nn.modules.module.Module"),
    ("py:class", "torch.nn.modules.loss._Loss"),
    ("py:class", "torch.optim.optimizer.Optimizer"),
    ("py:class", "torch.optim.lr_scheduler._LRScheduler"),
    ("py:class", "torch.device"),
    ("py:class", "torch.utils.data.dataset.Dataset"),
]

html_css_files = [
    "fonts.css",
    "owkin.css",
    "sidebar.css",
]

html_logo = "_static/pydeseq2_logo.svg"
html_show_sourcelink = False
html_show_sphinx = True

current_commit = git.Repo(search_parent_directories=True).head.object.hexsha

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "binder": {
        "org": "Owkin",
        "repo": "PyDESeq2",
        "branch": current_commit,  # Can be any branch, tag, or commit hash.
        # Use a branch that hosts your docs.
        "binderhub_url": "https://mybinder.org",  # public binderhub url
        "dependencies": str(Path(__file__).parents[2] / "environment.yml"),
        "notebooks_dir": "jupyter_notebooks",
        "use_jupyter_lab": True,
    },
}
