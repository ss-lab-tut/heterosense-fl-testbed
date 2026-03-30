"""Sphinx configuration for HeteroSense-FL documentation."""
import sys, os
sys.path.insert(0, os.path.abspath('..'))

project   = 'HeteroSense-FL'
copyright = '2026, Xun Shao, Kohsuke Yamakawa, Aoba Otani'
author    = 'Xun Shao, Kohsuke Yamakawa, Aoba Otani'
release   = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path   = ['_templates']
exclude_patterns = ['_build']
html_theme       = 'sphinx_rtd_theme'
