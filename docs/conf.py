import os
import sys
from importlib.metadata import version as get_version
from datetime import datetime

# Add your package to the path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'LSIBET'
copyright = f'{datetime.now().year}, Giuseppe Chindemi'
author = 'Giuseppe Chindemi'
release = get_version("lisbet")

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_gallery.gen_gallery',
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Sphinx-Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'gallery',
}

# Theme configuration
html_theme = 'pydata_sphinx_theme'
html_logo = "_static/logo_dark.png"

# Theme options
html_theme_options = {
    "logo": {
        "image_light": "logo_dark.png",
        "image_dark": "logo_dark.png",
    },
}

# Other settings
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']

# AutoDoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__'
}

# CopyButton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
