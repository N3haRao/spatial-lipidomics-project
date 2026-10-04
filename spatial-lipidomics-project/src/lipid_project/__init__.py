"""My own helpers for the spatial lipidomics project.

Quick orientation:
- The TUTORIAL's helper package is `cajal_lipidomics`. It comes from my clone of the tutorial
  (in external/) and holds all the plotting/analysis helpers the course notebooks use.
- THIS package, `lipid_project`, is just my own stuff. Right now that's only `paths`
  (where files live), but anything reusable I write later goes in here too, e.g. a
  `lipid_project/plots.py` for my own plotting functions.

This file (`__init__.py`) is what makes the folder a Python package. Whatever I import here
is available right after `import lipid_project`.
"""

# Import the paths module so both of these work in a notebook:
#   from lipid_project import paths
#   import lipid_project; lipid_project.paths.summary()
from . import paths

# __all__ lists what `from lipid_project import *` would pull in. Mostly good manners.
__all__ = ["paths"]

# Version of my package. Bump it if I make big changes, it doesn't affect anything else.
__version__ = "0.1.0"
