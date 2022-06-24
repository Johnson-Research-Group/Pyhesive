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
import copy
import pytest

data_set_named_tuple = collections.namedtuple("DataSet",["mesh","adjacency","closure","data"])
data_set_named_tuple.__new__.__defaults__ = (None,)*len(data_set_named_tuple._fields)
class DataSet(data_set_named_tuple):
  __slots__ = ()


def meshlist():
  return ["tetSingle","tetDouble","hexSingle","hexDouble","hexQuad","hexOctet"]

def find_data(name,mesh,partition_list=[0]):
  filename = os.path.join(common.bin_dir,name)+".pkl"
  try:
    data = common.load_obj(filename)
  # novermin 'FileNotFoundError' member requires !2, 3.3
  except FileNotFoundError:
    data = {
      "name"           : name,
      "mesh"           : mesh,
      "partition_data" : dict()
    }
    for part in partition_list:
      pyh = copy.deepcopy(pyhesive.Mesh(mesh))
      pyh.partition_mesh(part)
      data["partition_data"][part] = {
        # must deep copy since insertElements() is not guaranteed to create the
        # intermediate structures
        "pyhesive_mesh"        : copy.deepcopy(pyh.insert_elements()),
        "partition_vertex_map" : pyh._Mesh__get_partition_vertex_map(),
        "partition_interfaces" : pyh._Mesh__get_partition_interface_list(),
      }
      del pyh
    common.store_obj(filename,data)
  return data


def getEmptyRaw():
  return (np.empty((0,3)),[])

@pytest.fixture
def emptyRaw():
  return getEmptyRaw()

@pytest.fixture
def empty(emptyRaw):
  return DataSet(mesh=meshio.Mesh(*emptyRaw))


@pytest.fixture
def tetSingleRaw():
  return (
    np.array([[0.0,0.0,0.0],
              [1.0,0.0,0.0],
              [1.0,1.0,0.0],
              [1.0,0.0,1.0]]),
    [("tetra",np.array([[0,1,2,3]]))]
  )

@pytest.fixture
def tetSingle(tetSingleRaw):
  mesh      = meshio.Mesh(*tetSingleRaw)
  adjacency = (scp.sparse.lil_matrix(np.array([[4]])),
               scp.sparse.lil_matrix(np.ones((4,4))))
  closure = ({0: []},[0],[[(2,1,0),(2,0,3),(2,3,1),(0,1,3)]])
  data    = find_data("tetSingle",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def tetDoubleRaw():
  return (
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

@pytest.fixture
def tetDouble(tetDoubleRaw):
  r"""
       4
    /    \
  3 - - - - 2
   \         \
    \         \
     \         \
      0 - - - - 1
  """
  mesh      = meshio.Mesh(*tetDoubleRaw)
  adjacency = (
    scp.sparse.lil_matrix(np.array([[4,3],[3,4]])),
    scp.sparse.lil_matrix(
      np.array([
        [2,1,2,1,2],
        [1,1,1,0,1],
        [2,1,2,1,2],
        [1,0,1,1,1],
        [2,1,2,1,2]
      ])
    )
  )
  closure = ({0:[1],1:[0]},[0,1],[[(2,1,0),(2,4,1),(0,1,4)],[(3,2,0),(3,0,4),(3,4,2)]])
  data    = find_data("tetDouble",mesh,[0,-1])
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)


@pytest.fixture
def hexSingleRaw():
  return (
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

@pytest.fixture
def hexSingle(hexSingleRaw):
  r"""
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
  mesh      = meshio.Mesh(*hexSingleRaw)
  adjacency = (scp.sparse.lil_matrix(np.array([[8]])),scp.sparse.lil_matrix(np.ones((8,8))))
  closure   = ({0:[]},[0],[[(0,4,7,3),(0,1,5,4),(0,3,2,1),(6,7,4,5),(2,3,7,6),(2,6,5,1)]])
  data      = find_data("hexSingle",mesh)
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def hexDoubleRaw():
  return (
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

@pytest.fixture
def hexDouble(hexDoubleRaw):
  r"""
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
  mesh      = meshio.Mesh(*hexDoubleRaw)
  adjacency = (scp.sparse.lil_matrix(np.array([[8,4],[4,8]])),
               scp.sparse.lil_matrix(np.array([[1,1,0,1,1,0,1,1,0,1,1,0],
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
  data    = find_data("hexDouble",mesh,[0,-1])
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

def getHexQuadRaw():
  mesh_points = np.array([
    # plane z = 0
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [2.0,0.0,0.0],
    [0.0,1.0,0.0],
    [1.0,1.0,0.0],
    [2.0,1.0,0.0],
    [0.0,2.0,0.0],
    [1.0,2.0,0.0],
    [2.0,2.0,0.0],
    # plane z = 1
    [0.0,0.0,1.0],
    [1.0,0.0,1.0],
    [2.0,0.0,1.0],
    [0.0,1.0,1.0],
    [1.0,1.0,1.0],
    [2.0,1.0,1.0],
    [0.0,2.0,1.0],
    [1.0,2.0,1.0],
    [2.0,2.0,1.0],
  ])
  ref_mesh_points = np.vstack((
    [np.ravel(x) for x in np.mgrid[0:3,0:3,0:2]]
  )).T.astype(mesh_points.dtype)
  assert len(ref_mesh_points) == len(mesh_points)
  assert not len(np.setdiff1d(ref_mesh_points,mesh_points))
  return (
    mesh_points,
    [("hexahedron",np.array([
      [0,1,4,3,9,10,13,12],
      [1,2,5,4,10,11,14,13],
      [3,4,7,6,12,13,16,15],
      [4,5,8,7,13,14,17,16]
    ]))]
  )

@pytest.fixture
def hexQuadRaw():
  return getHexQuadRaw()

@pytest.fixture
def hexQuad(hexQuadRaw):
  r"""
  15- - - - - - 16 - - - - - - 17
  | \           | \            | \
  |   \         |   \          |   \
  |     12- - - - - - 13 - - - - - - 14
  |     | \     |     | \      |    | \
  6 - - | - \ - 7 - - | - \ -  8    |   \
   \    |     9- - - - - - 10 - - - - - - 11
     \  |     |    \  |     |    \  |     |
        3 - - | - - - 4 - - | - - - 5     |
          \   |         \   |         \   |
            \ |           \ |           \ |
              0 - - - - - - 1 - - - - - - 2
  """

  mesh      = meshio.Mesh(*hexQuadRaw)
  adjacency = (
    scp.sparse.lil_matrix(
      np.array([
        [8,4,4,2],
        [4,8,2,4],
        [4,2,8,4],
        [2,4,4,8]
      ])
    ),
    scp.sparse.lil_matrix(
      np.array([
        [1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0],
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
        [0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1]
      ])
    )
  )
  closure = (
    {
      0 : [1,2],
      1 : [0,3],
      2 : [0,3],
      3 : [1,2]
    },
    [0,1,2,3],
    [[(0,9 ,12,3 ),(0 ,1 ,10,9 ),(0 ,3 ,4 ,1 ),(13,12,9 ,10)],
     [(1,2 ,11,10),(1 ,4 ,5 ,2 ),(14,13,10,11),(5 ,14,11,2 )],
     [(3,12,15,6 ),(3 ,6 ,7 ,4 ),(16,15,12,13),(7 ,6 ,15,16)],
     [(4,7 ,8 ,5 ),(17,16,13,14),(8 ,7 ,16,17),(8 ,17,14,5 )]]
  )
  data = find_data("hexQuad",mesh,[0,2,3,-1])
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)

@pytest.fixture
def hexOctetRaw():
  mesh_points = np.array([
    # plane z = 0
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [2.0,0.0,0.0],
    [0.0,1.0,0.0],
    [1.0,1.0,0.0],
    [2.0,1.0,0.0],
    [0.0,2.0,0.0],
    [1.0,2.0,0.0],
    [2.0,2.0,0.0],
    # plane z = 1
    [0.0,0.0,1.0],
    [1.0,0.0,1.0],
    [2.0,0.0,1.0],
    [0.0,1.0,1.0],
    [1.0,1.0,1.0],
    [2.0,1.0,1.0],
    [0.0,2.0,1.0],
    [1.0,2.0,1.0],
    [2.0,2.0,1.0],
    # plane z = 2
    [0.0,0.0,2.0],
    [1.0,0.0,2.0],
    [2.0,0.0,2.0],
    [0.0,1.0,2.0],
    [1.0,1.0,2.0],
    [2.0,1.0,2.0],
    [0.0,2.0,2.0],
    [1.0,2.0,2.0],
    [2.0,2.0,2.0],
  ])
  ref_mesh_points = np.vstack((
    [np.ravel(x) for x in np.mgrid[0:3,0:3,0:3]]
  )).T.astype(mesh_points.dtype)
  assert len(ref_mesh_points) == len(mesh_points)
  assert not len(np.setdiff1d(ref_mesh_points,mesh_points))
  return (
    mesh_points,
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

@pytest.fixture
def hexOctet(hexOctetRaw):
  r"""
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
  mesh      = meshio.Mesh(*hexOctetRaw)
  adjacency = (
    scp.sparse.lil_matrix(
      np.array([
        [8,4,4,2,4,2,2,1],
        [4,8,2,4,2,4,1,2],
        [4,2,8,4,2,1,4,2],
        [2,4,4,8,1,2,2,4],
        [4,2,2,1,8,4,4,2],
        [2,4,1,2,4,8,2,4],
        [2,1,4,2,4,2,8,4],
        [1,2,2,4,2,4,4,8]
      ])
    ),
    scp.sparse.lil_matrix(
      np.array([
        [1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
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
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1]
      ])
    )
  )
  closure = (
    {
      0 : [1,2,4],
      1 : [0,3,5],
      2 : [0,3,6],
      3 : [1,2,7],
      4 : [0,5,6],
      5 : [1,4,7],
      6 : [2,4,7],
      7 : [3,5,6]
    },
    [0,1,2,3,4,5,6,7],
    [[(0 ,9 ,12,3 ),(0 ,1 ,10,9 ),(0 ,3 ,4 ,1 )],
     [(1 ,2 ,11,10),(1 ,4 ,5 ,2 ),(5 ,14,11,2 )],
     [(3 ,12,15,6 ),(3 ,6 ,7 ,4 ),(7 ,6 ,15,16)],
     [(4 ,7 ,8 ,5 ),(8 ,7 ,16,17),(8 ,17,14,5 )],
     [(9 ,18,21,12),(9 ,10,19,18),(22,21,18,19)],
     [(10,11,20,19),(23,22,19,20),(14,23,20,11)],
     [(12,21,24,15),(25,24,21,22),(16,15,24,25)],
     [(26,25,22,23),(17,16,25,26),(17,26,23,14)]]
  )
  data = find_data("hexOctet",mesh,[0,2,4,-1])
  return DataSet(mesh=mesh,adjacency=adjacency,closure=closure,data=data)


def get_fixture(request):
  return request.getfixturevalue(request.param)

@pytest.fixture
def fixt(request):
  return request.getfixturevalue(request.param)

@pytest.fixture
def mesh(request):
  result = get_fixture(request)
  if isinstance(result,DataSet):
    return result.mesh
  if isinstance(result,tuple):
    return result
  raise TypeError(type(result))

@pytest.fixture
def pyhmesh(request):
  return pyhesive.Mesh(get_fixture(request).mesh)

@pytest.fixture
def adjacency(request):
  return get_fixture(request).adjacency

@pytest.fixture
def closure(request):
  return get_fixture(request).closure

@pytest.fixture
def partition_data(request):
  return get_fixture(request).data["partition_data"]

@pytest.fixture
def replace_output(request):
  return request.config.getoption("--pyhesive-replace")
