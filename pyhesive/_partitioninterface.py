#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 20:02:24 2021

@author: jacobfaibussowitsch
"""
from collections import namedtuple

PartitionInterfaceNamedTuple = namedtuple("PartitionInteface",["ownFaces","mirrorIds","mirrorVertices"])
class PartitionInterface(PartitionInterfaceNamedTuple):
  __slots__ = ()
