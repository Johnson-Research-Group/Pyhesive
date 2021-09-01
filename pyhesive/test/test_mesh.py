#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 15:35:49 2021

@author: jacobfaibussowitsch
"""
import common
import fixtures
import pyhesive
import meshio
import pytest

meshes = fixtures.generate()
# @pytest.fixture(scope="module")
# def mesh(request):
#   #print(vars(msh))
#   #print(vars(msh).keys())
#   return pyhesive.Mesh(request.param)
#   def createMesh(m):
#     return pyhesive.Mesh(m)
#   return createMesh


@pytest.fixture
def partitionedMesh(mesh):
  print(vars(mesh))
  request.partitionMesh()
  return mesh

@pytest.mark.parametrize("mesh,error",[
  ("empty"    ,pytest.raises(ValueError)),
  ("tetSingle",common.noExcept()),
  ("tetDouble",common.noExcept()),
  ("hexSingle",common.noExcept()),
  ("hexQuad"  ,common.noExcept()),
  ("hexOctet" ,common.noExcept()),
])
def test_create(mesh,error,request):
  with error:
    print(mesh)
    mesh = request.getfixturevalue(mesh).mesh
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
  common.scipyAllClose(c2c,c2cExpected)
  common.scipyAllClose(v2v,v2vExpected)
  return

@pytest.mark.parametrize("pyhmesh,closure",zip(meshes,meshes),indirect=True)
def test_computeClosure(pyhmesh,closure):
  adj,bdCells,bdFaces = pyhmesh.computeClosure(fullClosure=True)
  adjExpected,bdCellsExpected,bdFacesExpected = closure
  assert adj     == adjExpected
  assert bdCells == bdCellsExpected
  assert bdFaces == bdFacesExpected
  return

@pytest.mark.parametrize("numpart",[2,3,4])
@pytest.mark.parametrize("pyhmesh,partitions",zip(meshes,meshes),indirect=True)
def test_computePartitionVertexMap(numpart,pyhmesh,partitions):
  pyhmesh.partitionMesh(numpart)
  vmap = pyhmesh._Mesh__computePartitionVertexMap()
  assert vmap == partitions[numpart]["partitionVertexMap"]
  return

@pytest.mark.parametrize("numpart",[2,3,4])
@pytest.mark.parametrize("pyhmesh,partitions",zip(meshes,meshes),indirect=True)
def test_computePartitionInterfaces(numpart,pyhmesh,partitions):
  pyhmesh.partitionMesh(numpart)
  pint = pyhmesh._Mesh__computePartitionInterfaceList()
  for pcomputed,pexpected in zip(pint,partitions[numpart]["partitionInterfaces"]):
    assert pcomputed.__slots__ == pexpected.__slots__
  return

@pytest.mark.parametrize("numpart",[2,3,4])
@pytest.mark.parametrize("pyhmesh,partitions",zip(meshes,meshes),indirect=True)
def test_FullStack(numpart,pyhmesh,partitions):
  pyhmesh.partitionMesh(numpart)
  pyhmesh.insertElements()
  assert pyhmesh == partitions[numpart]["pyhesiveMesh"]
  return

if __name__ == "__main__":
  pytest.main()
