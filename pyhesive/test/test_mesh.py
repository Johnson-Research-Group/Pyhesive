#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 15:35:49 2021

@author: jacobfaibussowitsch
"""
import common
import fixtures
import copy
import pyhesive
import meshio
import pytest

meshes = fixtures.meshlist()

@pytest.mark.parametrize("mesh,error",[
  ("empty"    ,pytest.raises(ValueError)),
  ("tetSingle",common.noExcept()),
  ("tetDouble",common.noExcept()),
  ("hexSingle",common.noExcept()),
  ("hexQuad"  ,common.noExcept()),
  ("hexOctet" ,common.noExcept()),
],indirect=["mesh"])
def test_create(mesh,error,request):
  with error:
    if isinstance(mesh,meshio.Mesh):
      pyh = pyhesive.Mesh(mesh)
    else:
      pyh = pyhesive.Mesh.fromFile(mesh)
    assert isinstance(pyh,pyhesive.Mesh)
  return


@pytest.mark.parametrize("pyhmesh,adjacency",zip(meshes,meshes),indirect=True)
def test_computeAdjacencyMatrix(pyhmesh,adjacency):
  c2c,v2v = pyhmesh.computeAdjacencyMatrix(v2v=True)
  c2cExpected,v2vExpected = adjacency
  common.assertScipyAllClose(c2c,c2cExpected)
  common.assertScipyAllClose(v2v,v2vExpected)
  return


@pytest.mark.parametrize("pyhmesh,closure",zip(meshes,meshes),indirect=True)
def test_computeClosure(pyhmesh,closure):
  adj,bdCells,bdFaces = pyhmesh.computeClosure(fullClosure=True)
  adjExpected,bdCellsExpected,bdFacesExpected = closure
  assert adj     == adjExpected
  assert bdCells == bdCellsExpected
  assert bdFaces == bdFacesExpected
  return

@pytest.mark.parametrize("mesh,partitionData",zip(meshes,meshes),indirect=True)
def test_computePartitionVertexMap(mesh,partitionData,subtests):
  for numpart,data in partitionData.items():
    with subtests.test(numpart=numpart):
      pyh = pyhesive.Mesh(mesh)
      pyh.partitionMesh(numpart)
      pyh._Mesh__computePartitionVertexMap()
      assert pyh.partVMap == data["partitionVertexMap"]
  return

@pytest.mark.parametrize("mesh,partitionData",zip(meshes,meshes),indirect=True)
def test_computePartitionInterfaces(mesh,partitionData,subtests):
  for numpart,data in partitionData.items():
    with subtests.test(numpart=numpart):
      pyh = pyhesive.Mesh(mesh)
      pyh.partitionMesh(numpart)
      pint = pyh._Mesh__computePartitionInterfaceList()
      for pcomputed,pexpected in zip(pint,data["partitionInterfaces"]):
        assert pcomputed == pexpected
  return


@pytest.mark.parametrize("mesh,partitionData",zip(meshes,meshes),indirect=True)
def test_FullStack(mesh,partitionData,subtests):
  pyhm = pyhesive.Mesh(mesh)
  for numpart,data in partitionData.items():
    print(numpart)
    print(data)
    pyh = copy.deepcopy(pyhm)
    with subtests.test(numpart=numpart):
      pyh.partitionMesh(numpart)
      pyh.insertElements()
      assert pyh == data["pyhesiveMesh"]
  return


if __name__ == "__main__":
  pytest.main()
