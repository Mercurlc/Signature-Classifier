# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Signature-Classifier'
copyright = '2024, Mercurlc'
author = 'Mercurlc'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../signature-classifier'))
