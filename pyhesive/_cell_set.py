#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:59:21 2021

@author: jacobfaibussowitsch
"""
from collections import namedtuple
from meshio import CellBlock
import numpy as np

type_map = {}

class CellSet(namedtuple("CellSet",["type","cells","dim","face_indices","cohesive_type"])):
  __slots__ = ()

  @classmethod
  def from_POD(cls,ctype,cells):
    return cls(ctype,cells,*type_map[ctype])

  @classmethod
  def from_CellBlock(cls,cell_block):
    assert isinstance(cell_block,CellBlock)
    return cls.from_POD(cell_block.type,cell_block.data)

  def __len__(self):
    return len(self.cells)

  def __getitem__(self,key):
    return self.from_POD(self.type,self.cells[key])

  def __ne__(self,other):
    return not self.__eq__(other)

  def __eq__(self,other):
    if isinstance(other,CellSet):
      if self.type != other.type:
        return False
      if self.cohesive_type != other.cohesive_type:
        return False
      if self.dim != other.dim:
        return False
      if not np.array_equiv(self.face_indices,other.face_indices):
        return False
      if not np.array_equiv(self.cells,other.cells):
        return False
      return True
    return NotImplemented


def register_element_type(name,dim,face_indices,cohesive_name=None,exist_ok=False):
  """
  Register an element type and its cohesive counterpart with the library

  Parameters
  ----------
  name : str
    Identifier of the bulk element.
  dim : int
    Dimension of the element.
  face_indices : arraylike[arraylike[int]...]
    Array of node indices for each face in the element. Indices must be ordered such that their
    cross-product points away from the center of the element.
  cohesive_name : str, optional
    Name of the cohesive counterpart. May be omitted if the cohesive type is the same as the
    bulk element.
  exist_ok : bool, optional
    Allow registration to overwrite an existing entry

  Returns
  -------
  None

  Raises
  ------
  ValueError
    If exist_ok is False and name is already registered.
  NotImplementedError
    If face_indices contains faces that do not all have the same length
  """
  name = str(name)
  dim  = int(dim)

  if not bool(exist_ok) and name in type_map:
    raise ValueError("Name %s already in type_map" % name)

  if cohesive_name is None:
    cohesive_name = name
  else:
    cohesive_name = str(cohesive_name)

  if face_indices is not None:
    lens = list(map(len,face_indices))
    if min(lens) != max(lens):
      raise NotImplementedError("Faces must all have the same number of indices")

    face_indices = np.asarray(face_indices,dtype=int)

  type_map[name] = (dim,face_indices,cohesive_name)
  return


register_element_type("triangle",2,np.array([[0,1],[1,2],[2,0]]),cohesive_name="quad")
register_element_type("tetra",3,np.array([[2,1,0],[2,0,3],[2,3,1],[0,1,3]]),cohesive_name="wedge")
register_element_type(
  "hexahedron",3,np.array([[0,4,7,3],[0,1,5,4],[0,3,2,1],[6,7,4,5],[2,3,7,6],[2,6,5,1]])
)
register_element_type("wedge",3,None) # kind of a hack
