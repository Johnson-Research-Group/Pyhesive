#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 20:02:24 2021

@author: jacobfaibussowitsch
"""
import numpy as np
from collections import namedtuple

class PartitionInterface(namedtuple("PartitionInteface",["own_faces","mirror_ids","mirror_vertices"])):
  __slots__ = ()

  def __ne__(self,other):
    return not self.__eq__(other)

  def __eq__(self,other):
    if id(self) == id(other):
      return True
    if isinstance(other,PartitionInterface):
      if np.any(self.own_faces != other.own_faces):
        return False
      if np.any(self.mirror_ids != other.mirror_ids):
        return False
      if np.any(self.mirror_vertices != other.mirror_vertices):
        return False
      return True
    return NotImplemented
