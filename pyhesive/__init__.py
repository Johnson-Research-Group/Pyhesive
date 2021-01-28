import sys

if sys.version_info <= (3,0):
    raise ImportError("Must use python3")

from .optsctx import Optsctx
from .mesh import Mesh
from .__about__ import __version__

__all__ = ["Optsctx", "Mesh", "__version__"]
