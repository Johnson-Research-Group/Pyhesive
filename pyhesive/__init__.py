from sys import version_info

if version_info <= (3,0):
  raise ImportError("Must use python3")

from .__about__ import __version__
from ._mesh     import Mesh
from ._util     import get_log_level,set_log_level,get_log_stream,set_log_stream
from ._cell_set import register_element_type

__all__ = ["Mesh","__version__"]
