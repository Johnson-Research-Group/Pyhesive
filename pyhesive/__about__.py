from sys import version_info

if version_info >= (3,8):
  # novermin 'importlib.metadata' module requires !2, 3.8
  from importlib import metadata
else:
  import importlib_metadata as metadata

__version__ = metadata.version("pyhesive")
