"""
Description of module
import
from 
versions
Notes for installation
Dependences
setup_module for doing tests
"""
# from .base import clone
# from .utils._show_versions import show_versions

__all__ = [
    "bin",
    "ini",
    "abstract",
    "data_manager",
    "supervised",
    "resources",
    "tests",
    "output",
    "simula",
    "strategies",
    "datasets",
    "visualization",
]

from . import _version

__version__ = _version.get_versions()["version"]
# del get_versions

from . import _version
__version__ = _version.get_versions()['version']
