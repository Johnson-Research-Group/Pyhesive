#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  31 15:32:03 2021

@author: jacobfaibussowitsch
"""
import os
import collections
import meshio
import numpy as np
import scipy as scp
import pyhesive
import pyhesive.test.common as common
import pytest

dataSetNamedTuple = collections.namedtuple("DataSet",["mesh","adjacency","closure","data"])
dataSetNamedTuple.__new__.__defaults__ = (None,)*len(dataSetNamedTuple._fields)
class DataSet(dataSetNamedTuple):
  __slots__ = ()


def findData(name,mesh):
  filename = os.path.join(common.dataDir,name)+".pkl"
  try:
    data = common.loadObj(filename)
    #raise FileNotFoundError
  except FileNotFoundError:
    import copy
    pyh = pyhesive.Mesh(mesh)
    data = {
      "name" : name,
      "mesh" : pyh,
      "partitionData" : dict()
    }
    for part in [2,3,4]:
      newPyh = copy.deepcopy(pyh)
      newPyh.partitionMesh(part)
      vmap = newPyh._Mesh__computePartitionVertexMap()
      pinterfaces = newPyh._Mesh__computePartitionInterfaceList()
      newPyh.insertElements()
      data["partitionData"][part] = {
        "partitionVertexMap"  : vmap,
        "partitionInterfaces" : pinterfaces,
        "pyhesiveMesh"        : newPyh
      }
    common.storeObj(filename,data)
  return data

@pytest.fixture
def makeMeshio(request):
  def __makeMeshio(*args,**kwargs):
    return meshio.Mesh(*args,**kwargs)
  return __makeMeshio

# @pytest.fixture
# def mesh(request):
#   print(vars(request).keys())
#   if isinstance(request.param,DataSet):
#     return request.param.mesh
#   elif isinstance(request.param,meshio.Mesh):
#     return request.param
#   elif isinstance(request.param,string):
#     return request.getfixturevalue(request.param).mesh
#   else:
#     raise RuntimeError


@pytest.fixture
def empty():
  mesh = meshio.Mesh(np.empty((0,3)),[])
  return DataSet(mesh=mesh)

@pytest.fixture
def tetSingle():
  mesh = meshio.Mesh(
    np.array([[0.0,0.0,0.0],
              [1.0,0.0,0.0],
              [1.0,1.0,0.0],
              [1.0,0.0,1.0]]),
    [("tetra",np.array([[0,1,2,3]]))]
  )
  adjacency = (scp.sparse.lil_matrix(np.matrix([[4]])),
               scp.sparse.lil_matrix(np.ones((4,4))))
  closure = ({0: []},[0],[[(2,1,0),(2,0,3),(2,3,1),(0,1,3)]])
  data = findData("tetSingle",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def tetDouble():
  mesh = meshio.Mesh(
    np.array([
      [0.0,0.0,0.0],
      [1.0,0.0,0.0],
      [1.0,1.0,0.0],
      [0.0,1.0,0.0],
      [0.5,0.5,0.5],
    ]),
    [("tetra",np.array([
      [0,1,2,4],
      [0,2,3,4]]))]
  )
  adjacency = (scp.sparse.lil_matrix(np.matrix([[4,3],[3,4]])),
               scp.sparse.lil_matrix(np.matrix([[2,1,2,1,2],
                                                [1,1,1,0,1],
                                                [2,1,2,1,2],
                                                [1,0,1,1,1],
                                                [2,1,2,1,2]])))
  closure = ({0:[1],1:[0]},[0,1],[[(2,1,0),(2,4,1),(0,1,4)],[(3,2,0),(3,0,4),(3,4,2)]])
  name = "tetDouble"
  data = findData(name,mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def hexSingle():
  """
   7 - - - - - - 6
  | \           | \
  |   \         |   \
  |     4 - - - - - - 5
  |     |       |     |
  3 - - | - - - 2     |
    \   |         \   |
      \ |           \ |
        0 - - - - - - 1
  """
  mesh = meshio.Mesh(
    np.array([
      [0.0,0.0,0.0],
      [1.0,0.0,0.0],
      [1.0,1.0,0.0],
      [0.0,1.0,0.0],
      [0.0,0.0,1.0],
      [1.0,0.0,1.0],
      [1.0,1.0,1.0],
      [0.0,1.0,1.0],
    ]),
    [("hexahedron",np.array([[0,1,2,3,4,5,6,7]]))]
  )
  adjacency = (scp.sparse.lil_matrix(np.matrix([[8]])),
               scp.sparse.lil_matrix(np.ones((8,8))))
  closure = ({0:[]},[0],[[(0,4,7,3),(0,1,5,4),(0,3,2,1),(6,7,4,5),(2,3,7,6),(2,6,5,1)]])
  data = findData("hexSingle",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def hexDouble():
  """
  9 - - - - - - 10 - - - - - - 11
  | \           | \           | \
  |   \         |   \         |   \
  |     6 - - - - - - 7 - - - - - - 8
  |     |       |     |       |     |
  3 - - | - - - 4 - - | - - - 5     |
    \   |         \   |         \   |
      \ |           \ |           \ |
        0 - - - - - - 1 - - - - - - 2
  """
  mesh = meshio.Mesh(
    np.array([
      [0.0,0.0,0.0],
      [1.0,0.0,0.0],
      [2.0,0.0,0.0],
      [0.0,1.0,0.0],
      [1.0,1.0,0.0],
      [2.0,1.0,0.0],
      [0.0,0.0,1.0],
      [1.0,0.0,1.0],
      [2.0,0.0,1.0],
      [0.0,1.0,1.0],
      [1.0,1.0,1.0],
      [2.0,1.0,1.0],
    ]),
    [("hexahedron",np.array([[0,1,4,3,6,7,10,9],[1,2,5,4,7,8,11,10]]))]
  )
  adjacency = (scp.sparse.lil_matrix(np.matrix([[8,4],[4,8]])),
               scp.sparse.lil_matrix(np.matrix([[1,1,0,1,1,0,1,1,0,1,1,0],
                                                [1,2,1,1,2,1,1,2,1,1,2,1],
                                                [0,1,1,0,1,1,0,1,1,0,1,1],
                                                [1,1,0,1,1,0,1,1,0,1,1,0],
                                                [1,2,1,1,2,1,1,2,1,1,2,1],
                                                [0,1,1,0,1,1,0,1,1,0,1,1],
                                                [1,1,0,1,1,0,1,1,0,1,1,0],
                                                [1,2,1,1,2,1,1,2,1,1,2,1],
                                                [0,1,1,0,1,1,0,1,1,0,1,1],
                                                [1,1,0,1,1,0,1,1,0,1,1,0],
                                                [1,2,1,1,2,1,1,2,1,1,2,1],
                                                [0,1,1,0,1,1,0,1,1,0,1,1]])))
  closure = ({0:[1],1:[0]},[0,1],
             [[(0,6,9,3),(0,1,7,6),(0,3,4,1),(10,9,6,7),(4,3,9,10)],
              [(1,2,8,7),(1,4,5,2),(11,10,7,8),(5,4,10,11),(5,11,8,2)]])
  data = findData("hexDouble",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def hexQuad():
  """
  15 - - - - - - 16 - - - - - - 17
  | \           | \            | \
  |   \         |   \          |   \
  |     12 - - - - - - 13 - - - - - - 14
  |     | \     |     | \      |    | \
  6 - - | - \ - 7 - - | - \ -  8    |   \
   \    |     9 - - - - - - 10 - - - - - - 11
     \  |     |    \  |     |    \  |     |
        3 - - | - - - 4 - - | - - - 5     |
          \   |         \   |         \   |
            \ |           \ |           \ |
              0 - - - - - - 1 - - - - - - 2
  """
  mesh = meshio.Mesh(
    np.array([
      [0.0,0.0,0.0],
      [1.0,0.0,0.0],
      [2.0,0.0,0.0],
      [0.0,1.0,0.0],
      [1.0,1.0,0.0],
      [2.0,1.0,0.0],
      [2.0,1.0,0.0],
      [0.0,2.0,0.0],
      [1.0,2.0,0.0],
      [2.0,2.0,0.0],
      [0.0,0.0,1.0],
      [1.0,0.0,1.0],
      [2.0,0.0,1.0],
      [0.0,1.0,1.0],
      [1.0,1.0,1.0],
      [2.0,1.0,1.0],
      [2.0,1.0,1.0],
      [0.0,2.0,1.0],
      [1.0,2.0,1.0],
      [2.0,2.0,1.0],
    ]),
    [("hexahedron",np.array([
      [0,1,4,3,9,10,13,12],
      [1,2,5,4,10,11,14,13],
      [3,4,7,6,12,13,16,15],
      [4,5,8,7,13,14,17,16]]))]
  )
  adjacency = (scp.sparse.lil_matrix(np.matrix([[8,4,4,2],
                                                [4,8,2,4],
                                                [4,2,8,4],
                                                [2,4,4,8]])),
               scp.sparse.lil_matrix(np.matrix([[1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0],
                                                [1,2,1,1,2,1,0,0,0,1,2,1,1,2,1,0,0,0],
                                                [0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0],
                                                [1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1,0],
                                                [1,2,1,2,4,2,1,2,1,1,2,1,2,4,2,1,2,1],
                                                [0,1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1],
                                                [0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0],
                                                [0,0,0,1,2,1,1,2,1,0,0,0,1,2,1,1,2,1],
                                                [0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1],
                                                [1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0],
                                                [1,2,1,1,2,1,0,0,0,1,2,1,1,2,1,0,0,0],
                                                [0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0],
                                                [1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1,0],
                                                [1,2,1,2,4,2,1,2,1,1,2,1,2,4,2,1,2,1],
                                                [0,1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1],
                                                [0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0],
                                                [0,0,0,1,2,1,1,2,1,0,0,0,1,2,1,1,2,1],
                                                [0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1]])))
  closure = ({0:[1,2],1:[0,3],2:[0,3],3:[1,2]},[0,1,2,3],
             [[(0,9,12,3),(0,1,10,9),(0,3,4,1),(13,12,9,10)],
              [(1,2,11,10),(1,4,5,2),(14,13,10,11),(5,14,11,2)],
              [(3,12,15,6),(3,6,7,4),(16,15,12,13),(7,6,15,16)],
              [(4,7,8,5),(17,16,13,14),(8,7,16,17),(8,17,14,5)]])
  data = findData("hexQuad",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def hexOctet():
  """
  24 - - - - - -25 - - - - - - 26
  | \           | \            | \
  |   \         |   \          |   \
  |     21 - - - - - -22 - - - - - - 23
  |     | \     |     | \      |     | \
  15 - -| - \ - 16 - -|- - \- - 17   |   \
  | \   |    18 - - - - - - 19 - - - - -  20
  |   \ |     | |   \ |     |  |   \ |    |
  |     12 - -| - - - 13 - -|- - - - 14   |
  |     | \   | |     | \   |  |    | \   |
  6 - - | - \ | 7 - - | - \ |- 8    |   \ |
   \    |     9 - - - - - - 10 - - - - - -11
     \  |     |    \  |     |    \  |     |
        3 - - | - - - 4 - - | - - - 5     |
          \   |         \   |         \   |
            \ |           \ |           \ |
              0 - - - - - - 1 - - - - - - 2
  """
  mesh = meshio.Mesh(
    np.array([
      [0.0,0.0,0.0],
      [1.0,0.0,0.0],
      [2.0,0.0,0.0],
      [0.0,1.0,0.0],
      [1.0,1.0,0.0],
      [2.0,1.0,0.0],
      [0.0,2.0,0.0],
      [1.0,2.0,0.0],
      [2.0,2.0,0.0],
      [0.0,0.0,1.0],
      [1.0,0.0,1.0],
      [2.0,0.0,1.0],
      [0.0,1.0,1.0],
      [1.0,1.0,1.0],
      [2.0,1.0,1.0],
      [0.0,2.0,1.0],
      [1.0,2.0,1.0],
      [2.0,2.0,1.0],
      [0.0,0.0,2.0],
      [1.0,0.0,2.0],
      [2.0,0.0,2.0],
      [0.0,1.0,2.0],
      [1.0,1.0,2.0],
      [2.0,1.0,2.0],
      [0.0,2.0,2.0],
      [1.0,2.0,2.0],
      [2.0,2.0,2.0],
    ]),
    [("hexahedron",np.array([
      [0,1,4,3,9,10,13,12],
      [1,2,5,4,10,11,14,13],
      [3,4,7,6,12,13,16,15],
      [4,5,8,7,13,14,17,16],
      [9,10,13,12,18,19,22,21],
      [10,11,14,13,19,20,23,22],
      [12,13,16,15,21,22,25,24],
      [13,14,17,16,22,23,26,25],
    ]))]
  )
  adjacency = (scp.sparse.lil_matrix(np.matrix([[8,4,4,2,4,2,2,1],
                                                [4,8,2,4,2,4,1,2],
                                                [4,2,8,4,2,1,4,2],
                                                [2,4,4,8,1,2,2,4],
                                                [4,2,2,1,8,4,4,2],
                                                [2,4,1,2,4,8,2,4],
                                                [2,1,4,2,4,2,8,4],
                                                [1,2,2,4,2,4,4,8]])),
               scp.sparse.lil_matrix(np.matrix([[1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                                [1,2,1,1,2,1,0,0,0,1,2,1,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                                [0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                                [1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                                [1,2,1,2,4,2,1,2,1,1,2,1,2,4,2,1,2,1,0,0,0,0,0,0,0,0,0],
                                                [0,1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1,0,0,0,0,0,0,0,0,0],
                                                [0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                                [0,0,0,1,2,1,1,2,1,0,0,0,1,2,1,1,2,1,0,0,0,0,0,0,0,0,0],
                                                [0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
                                                [1,1,0,1,1,0,0,0,0,2,2,0,2,2,0,0,0,0,1,1,0,1,1,0,0,0,0],
                                                [1,2,1,1,2,1,0,0,0,2,4,2,2,4,2,0,0,0,1,2,1,1,2,1,0,0,0],
                                                [0,1,1,0,1,1,0,0,0,0,2,2,0,2,2,0,0,0,0,1,1,0,1,1,0,0,0],
                                                [1,1,0,2,2,0,1,1,0,2,2,0,4,4,0,2,2,0,1,1,0,2,2,0,1,1,0],
                                                [1,2,1,2,4,2,1,2,1,2,4,2,4,8,4,2,4,2,1,2,1,2,4,2,1,2,1],
                                                [0,1,1,0,2,2,0,1,1,0,2,2,0,4,4,0,2,2,0,1,1,0,2,2,0,1,1],
                                                [0,0,0,1,1,0,1,1,0,0,0,0,2,2,0,2,2,0,0,0,0,1,1,0,1,1,0],
                                                [0,0,0,1,2,1,1,2,1,0,0,0,2,4,2,2,4,2,0,0,0,1,2,1,1,2,1],
                                                [0,0,0,0,1,1,0,1,1,0,0,0,0,2,2,0,2,2,0,0,0,0,1,1,0,1,1],
                                                [0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0],
                                                [0,0,0,0,0,0,0,0,0,1,2,1,1,2,1,0,0,0,1,2,1,1,2,1,0,0,0],
                                                [0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0],
                                                [0,0,0,0,0,0,0,0,0,1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1,0],
                                                [0,0,0,0,0,0,0,0,0,1,2,1,2,4,2,1,2,1,1,2,1,2,4,2,1,2,1],
                                                [0,0,0,0,0,0,0,0,0,0,1,1,0,2,2,0,1,1,0,1,1,0,2,2,0,1,1],
                                                [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0],
                                                [0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,1,2,1,0,0,0,1,2,1,1,2,1],
                                                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1]])))
  closure = ({0:[1,2,4],1:[0,3,5],2:[0,3,6],3:[1,2,7],4:[0,5,6],5:[1,4,7],6:[2,4,7],7:[3,5,6]},
             [0,1,2,3,4,5,6,7],
             [[(0,9,12,3),(0,1,10,9),(0,3,4,1)],
              [(1,2,11,10),(1,4,5,2),(5,14,11,2)],
              [(3,12,15,6),(3,6,7,4),(7,6,15,16)],
              [(4,7,8,5),(8,7,16,17),(8,17,14,5)],
              [(9,18,21,12),(9,10,19,18),(22,21,18,19)],
              [(10,11,20,19),(23,22,19,20),(14,23,20,11)],
              [(12,21,24,15),(25,24,21,22),(16,15,24,25)],
              [(26,25,22,23),(17,16,25,26),(17,26,23,14)]])
  data = findData("hexOctet",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)


def generate():
  return ["tetSingle","tetDouble","hexSingle","hexDouble","hexQuad","hexOctet"]

def getfixture(request):
  return request.getfixturevalue(request.param)

@pytest.fixture
def meshFixt(request):
  return getfixture(request).mesh

@pytest.fixture
def pyhmesh(request):
  return pyhesive.Mesh(getfixture(request).mesh)

@pytest.fixture
def adjacency(request):
  return getfixture(request).adjacency

@pytest.fixture
def closure(request):
  return getfixture(request).closure

@pytest.fixture
def partitions(request):
  return getfixture(request).data["partitionData"]
