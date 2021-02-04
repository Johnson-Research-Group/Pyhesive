import sys

if sys.version_info <= (3,0):
    raise ImportError("Must use python3")

from .__about__ import __version__
from ._mesh import Mesh

__all__ = ["Mesh", "__version__"]
