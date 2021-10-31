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
import scipy.sparse as scp

cur_dir = os.path.basename(os.getcwd())
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
  else:
    raise RuntimeError("Cannot determine location")
else:
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
    # protocol 4 since python3.4
    pickle.dump(obj,fd)

def load_obj(filename):
  with open(filename, "rb") as f:
    return pickle.load(f)

def assert_scipy_all_close(A,B,rtol=1e-7,atol=1e-8):
  def _scipy_all_close_lil():
    assert A.shape == B.shape
    for i in range(A.get_shape()[0]):
      rowA = A.getrowview(i).toarray()
      rowB = B.getrowview(i).toarray()
      np.testing.assert_allclose(rowA,rowB,rtol=rtol,atol=atol)
    return

  def _scipy_all_close_sparse():
    # If you want to check matrix shapes as well
    np.testing.assert_allclose(A.shape,B.shape,rtol=rtol,atol=atol)

    r1,c1 = A.nonzero()
    r2,c2 = B.nonzero()

    lidx1 = np.ravel_multi_index((r1,c1),A.shape)
    lidx2 = np.ravel_multi_index((r2,c2),B.shape)

    sidx1 = lidx1.argsort()
    sidx2 = lidx2.argsort()

    np.testing.assert_allclose(lidx1[sidx1],lidx2[sidx2],rtol=rtol,atol=atol)
    v1,v2 = A.data,B.data
    V1,V2 = v1[sidx1],v2[sidx2]
    np.testing.assert_allclose(V1,V2,rtol=rtol,atol=atol)
    return

  assert type(A) == type(B)
  if isinstance(A,scp.lil_matrix):
    _scipy_all_close_lil()
  else:
    _scipy_all_close_sparse()
  return

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

def trygetattr(obj,attr):
  if hasattr(obj,attr):
    return getattr(obj,attr)
  return

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
