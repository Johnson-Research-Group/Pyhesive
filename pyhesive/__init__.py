from sys import version_info

if version_info <= (3,0):
  raise ImportError("Must use python3")

from .__about__            import __version__
from ._mesh                import Mesh
from ._util                import get_log_level,set_log_level,get_log_stream,set_log_stream
from ._cell_set            import CellSet,register_element_type
from ._partition_interface import PartitionInterface

__all__ = [
  "__version__",
  "Mesh",
  "get_log_level","set_log_level","get_log_stream","set_log_stream",
  "CellSet","register_element_type",
  "PartitionInterface"
]
