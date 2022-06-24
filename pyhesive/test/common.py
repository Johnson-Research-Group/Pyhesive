#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 16:02:24 2021

@author: jacobfaibussowitsch
"""
import os
import pickle
from collections import namedtuple
import contextlib
import meshio
import pytest
import numpy as np

test_root_dir = None
cur_dir       = os.path.basename(os.getcwd())
if cur_dir == "test":
  test_root_dir = os.getcwd()
elif cur_dir == "pyhesive":
  parent_dir = os.path.basename(os.path.dirname(os.getcwd()))
  if parent_dir == "pyhesive":
    # we are in pyhesive/pyhesive, i.e. the package dir
    test_root_dir = os.path.join(os.getcwd(),"test")
  elif "pyhesive" in os.listdir():
    # we are in root pyhesive dir
    test_root_dir = os.path.join(os.getcwd(),"pyhesive","test")
elif cur_dir == "examples":
  parent_dir = os.path.basename(os.path.dirname(os.getcwd()))
  if parent_dir == "pyhesive":
    # we are in pyhesive/examples, i.e. the example dir
    test_root_dir = os.path.join(parent_dir,"pyhesive","test")

if test_root_dir is None:
  raise RuntimeError("Cannot determine location")

test_root_dir   = os.path.realpath(os.path.expanduser(test_root_dir))
pyhesiveRootDir = os.path.dirname(test_root_dir)
data_dir        = os.path.join(test_root_dir,"data")
mesh_dir        = os.path.join(data_dir,"meshes")
bin_dir         = os.path.join(data_dir,"bin")

data_set_named_tuple = namedtuple("DataSet",["mesh","adjacency","closure","partitionData"])
data_set_named_tuple.__new__.__defaults__ = (None,)*len(data_set_named_tuple._fields)
class DataSet(data_set_named_tuple):
  __slots__ = ()


@contextlib.contextmanager
def no_except():
  yield

def store_obj(filename,obj):
  with open(filename,"wb") as fd:
    pickle.dump(obj,fd)
  return

def load_obj(filename):
  with open(filename,"rb") as f:
    return pickle.load(f)

def assertNumpyArrayEqual(self,first,second,msg=None):
  assert type(first) == np.ndarray,msg
  assert type(first) == type(second),msg
  try:
    np.testing.assert_array_almost_equal_nulp(first,second)
  except AssertionError:
    additionalMsg = "Locations: %s" % (np.argwhere(first!=second))
    if msg is not None:
      msg = "\n".join([msg,additionalMsg])
    else:
      msg = additionalMsg
    pytest.fail(msg)

def commonPartitionSetup(meshFileList,testFileList,partitionList,replaceFunc,testFunc):
  for meshFile,testFile,partList in zip(meshFileList,testFileList,partitionList):
    with pyhesive.Mesh.fromFile(meshFile) as pyh:
      if replaceFiles:
        replaceFunc(pyh,testFile,partList)
        continue
      testDict = loadObj(testFile)
      for part,testPart in zip(partList,testDict):
        assert part == testPart
        pyh.PartitionMesh(part)
        testFunc(pyh,testDict,part)
