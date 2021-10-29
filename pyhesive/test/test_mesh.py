#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  29 15:35:49 2021

@author: jacobfaibussowitsch
"""
import copy
import pyhesive
import meshio
import pytest
from common import no_except,assert_scipy_all_close
import fixtures


meshes = fixtures.meshlist()

@pytest.mark.parametrize("mesh,error",[
  ("empty"    ,pytest.raises(ValueError)),
  ("tetSingle",no_except()),
  ("tetDouble",no_except()),
  ("hexSingle",no_except()),
  ("hexQuad"  ,no_except()),
  ("hexOctet" ,no_except()),
],indirect=["mesh"])
def test_create(mesh,error,request):
  with error:
    if isinstance(mesh,meshio.Mesh):
      pyh = pyhesive.Mesh(mesh)
    else:
      pyh = pyhesive.Mesh.from_file(mesh)
    assert isinstance(pyh,pyhesive.Mesh)
  return


@pytest.mark.parametrize("pyhmesh,adjacency",zip(meshes,meshes),indirect=True)
def test_compute_adjacency_matrix(pyhmesh,adjacency):
  c2c,v2v = pyhmesh.compute_adjacency_matrix(v2v=True)
  c2c_expected,v2v_expected = adjacency
  assert_scipy_all_close(c2c,c2c_expected)
  assert_scipy_all_close(v2v,v2v_expected)
  return


@pytest.mark.parametrize("pyhmesh,closure",zip(meshes,meshes),indirect=True)
def test_compute_closure(pyhmesh,closure):
  adjacency,boundary_cells,boundary_faces = pyhmesh.compute_closure(full_closure=True)
  adjacency_expected,boundary_cells_expected,boundary_faces_expected = closure
  assert adjacency == adjacency_expected
  assert boundary_cells == boundary_cells_expected
  assert boundary_faces == boundary_faces_expected
  return


@pytest.mark.parametrize("mesh,partition_data",zip(meshes,meshes),indirect=True)
def test_get_partition_vertex_map(mesh,partition_data,subtests):
  for numpart,data in partition_data.items():
    with subtests.test(numpart=numpart):
      pyh = pyhesive.Mesh(mesh)
      pyh.partition_mesh(numpart)
      pyh._Mesh__get_partition_vertex_map()
      assert pyh.partition_vertex_map == data["partition_vertex_map"]
  return


@pytest.mark.parametrize("mesh,partition_data",zip(meshes,meshes),indirect=True)
def test_get_partition_interfaces(mesh,partition_data,subtests):
  for numpart,data in partition_data.items():
    with subtests.test(numpart=numpart):
      pyh = pyhesive.Mesh(mesh)
      pyh.partition_mesh(numpart)
      interfaces = pyh._Mesh__get_partition_interface_list()
      for pcomputed,pexpected in zip(interfaces,data["partition_interfaces"]):
        assert pcomputed == pexpected
  return


@pytest.mark.parametrize("mesh,partition_data",zip(meshes,meshes),indirect=True)
def test_full_stack(mesh,partition_data,subtests):
  pyhm = pyhesive.Mesh(mesh)
  for numpart,data in partition_data.items():
    pyh = copy.deepcopy(pyhm)
    with subtests.test(numpart=numpart):
      pyh.partition_mesh(numpart)
      pyh.insert_elements()
      assert pyh == data["pyhesive_mesh"]
  return


if __name__ == "__main__":
  pytest.main()
