#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:59:21 2021

@author: jacobfaibussowitsch
"""
from collections import namedtuple
from meshio import CellBlock
import numpy as np

type_map = {
  "triangle"   : (2,np.array([[0,1],[1,2],[2,0]]),None),
  "tetra"      : (3,np.array([[2,1,0],[2,0,3],[2,3,1],[0,1,3]]),"wedge"),
  "hexahedron" : (3,np.array([[0,4,7,3],[0,1,5,4],[0,3,2,1],[6,7,4,5],[2,3,7,6],[2,6,5,1]]),"hexahedron"),
  "wedge"      : (3,None,"wedge"),
  "wedge12"    : (3,None,"wedge12"),
}

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
