#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 15:35:49 2021

@author: jacobfaibussowitsch
"""
import common
import pyhesive
import meshio
import pytest

@pytest.fixture
def mesh(request):
  print(vars(request))
  print(vars(request).keys())
  return pyhesive.Mesh(request.param)

@pytest.fixture
def partitionedMesh(mesh):
  print(vars(mesh))
  mesh.partitionMesh()
  return mesh

@pytest.mark.parametrize("rawMesh,error",[
  (common.empty.mesh    ,pytest.raises(ValueError)),
  (common.tetSingle.mesh,common.noExcept()),
  (common.tetDouble.mesh,common.noExcept()),
  (common.hexSingle.mesh,common.noExcept()),
  (common.hexQuad.mesh  ,common.noExcept()),
  (common.hexOctet.mesh ,common.noExcept()),
])
def test_create(rawMesh,error):
  with error:
    if isinstance(rawMesh,meshio.Mesh):
      pyh = pyhesive.Mesh(rawMesh)
    else:
      pyh = pyhesive.Mesh.fromFile(rawMesh)
    assert isinstance(pyh,pyhesive.Mesh)
  return


@pytest.mark.parametrize("mesh,expected",[
  (common.tetSingle.mesh,common.tetSingle.adjacency),
  (common.tetDouble.mesh,common.tetDouble.adjacency),
  (common.hexSingle.mesh,common.hexSingle.adjacency),
  (common.hexDouble.mesh,common.hexDouble.adjacency),
  (common.hexQuad.mesh  ,common.hexQuad.adjacency),
  (common.hexOctet.mesh ,common.hexOctet.adjacency),
],indirect=["mesh"])
def test_computeAdjacencyMatrix(mesh,expected):
  c2c,v2v = mesh.computeAdjacencyMatrix(v2v=True)
  c2cExpected,v2vExpected = expected
  common.scipyAllClose(c2c,c2cExpected)
  common.scipyAllClose(v2v,v2vExpected)
  return


@pytest.mark.parametrize("mesh,expected",[
  (common.tetSingle.mesh,common.tetSingle.closure),
  (common.tetDouble.mesh,common.tetDouble.closure),
  (common.hexSingle.mesh,common.hexSingle.closure),
  (common.hexDouble.mesh,common.hexDouble.closure),
  (common.hexQuad.mesh  ,common.hexQuad.closure),
  (common.hexOctet.mesh ,common.hexOctet.closure),
],indirect=["mesh"])
def test_computeClosure(mesh,expected):
  adj,bdCells,bdFaces = mesh.computeClosure(fullClosure=True)
  adjExpected,bdCellsExpected,bdFacesExpected = expected
  assert adj     == adjExpected
  assert bdCells == bdCellsExpected
  assert bdFaces == bdFacesExpected
  return


@pytest.mark.parametrize("partitionedMesh,expected",[
  (common.tetSingle.mesh,common.tetSingle.partitionData),
],indirect=["partitionedMesh"])
def test_computePartitionVertexMap(partitionedMesh,expected):
  for part in partitionedMesh.partitions:
    partInterface = partitionedMesh.computePartitionInterface(part)
  return

if __name__ == "__main__":
  pytest.main()
