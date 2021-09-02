#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 20:02:24 2021

@author: jacobfaibussowitsch
"""
from collections import namedtuple
import numpy as np

PartitionInterfaceNamedTuple = namedtuple("PartitionInteface",["ownFaces","mirrorIds","mirrorVertices"])
class PartitionInterface(PartitionInterfaceNamedTuple):
  __slots__ = ()

  def __ne__(self,other):
    return not self.__eq__(other)

  def __eq__(self,other):
    if id(self) == id(other):
      return True
    if isinstance(other,PartitionInterface):
      if np.any(self.ownFaces != other.ownFaces):
        return False
      if np.any(self.mirrorIds != other.mirrorIds):
        return False
      if np.any(self.mirrorVertices != other.mirrorVertices):
        return False
      return True
    return NotImplemented
