#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 16:02:24 2021

@author: jacobfaibussowitsch
"""
import os,pickle,collections,contextlib,meshio
import pytest
import numpy as np
import scipy as scp
from scipy import sparse

curDir = os.path.basename(os.getcwd())
if curDir == "test":
  testRootDir = os.getcwd()
elif curDir == "pyhesive":
  parentDir = os.path.basename(os.path.dirname(os.getcwd()))
  if parentDir == "pyhesive":
    # we are in pyhesive/pyhesive, i.e. the package dir
    testRootDir = os.path.join(os.getcwd(), "test")
  elif "pyhesive" in os.listdir():
    # we are in root pyhesive dir
    testRootDir = os.path.join(os.getcwd(),"pyhesive","test")
  else:
    raise RuntimeError("Cannot determine location")
else:
  raise RuntimeError("Cannot determine location")

testRootDir     = os.path.realpath(os.path.expanduser(testRootDir))
pyhesiveRootDir = os.path.dirname(testRootDir)
dataDir         = os.path.join(testRootDir,"data")
meshDir         = os.path.join(dataDir,"meshes")
binDir          = os.path.join(dataDir,"bin")

dataSetNamedTuple = collections.namedtuple("DataSet",["mesh","adjacency","closure","partitionData"])
dataSetNamedTuple.__new__.__defaults__ = (None,)*len(dataSetNamedTuple._fields)
class DataSet(dataSetNamedTuple):
  __slots__ = ()


@contextlib.contextmanager
def noExcept():
  yield

def storeObj(filename,obj):
  with open(filename,"wb") as fd:
    # protocol 4 since python3.4
    pickle.dump(obj,fd)

def loadObj(filename):
  with open(filename, "rb") as f:
    return pickle.load(f)

def scipyAllClose(A,B,rtol=1e-7,atol=1e-8):
  def _scipyAllCloseLil():
    assert A.shape == B.shape
    for i in range(A.get_shape()[0]):
      rowA = A.getrowview(i).toarray()
      rowB = B.getrowview(i).toarray()
      np.testing.assert_allclose(rowA,rowB,rtol=rtol,atol=atol)
      return

  def _scipyAllCloseSparse():
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
  if isinstance(A,scp.sparse.lil_matrix):
    _scipyAllCloseLil()
  else:
    _scipyAllCloseSparse()
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

