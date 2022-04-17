#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
from sys import version_info
import traceback
import logging
import meshio
import pymetis
import collections
import scipy
from scipy import sparse as scp
import numpy as np

from ._utils import flatten,get_log_level,get_log_stream
from ._cell_set import CellSet
from ._partition_interface import PartitionInterface

class Mesh(object):
  __slots__ = ("log","cell_data","coords","__dict__")

  def __init_logger(self):
    slog = logging.getLogger(self.__class__.__name__)
    slog.setLevel(get_log_level())
    slog.propagate = False
    ch = logging.StreamHandler(get_log_stream())
    ch.setLevel(get_log_level())
    ch.propagate = False
    ch.setFormatter(
      logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s")
    )
    slog.addHandler(ch)
    return slog

  def __assert_partitioned(self):
    assert hasattr(self,"partitions"),"Must partition mesh first"
    return

  def __init__(self,mesh):
    if isinstance(mesh,meshio.Mesh):
      try:
        self.cell_data = CellSet.from_CellBlock(mesh.cells[-1])
      except IndexError as ie:
        raise ValueError("input mesh does not contain any cells") from ie
      self.coords = mesh.points
    elif isinstance(mesh,self.__class__):
      self.cell_data = mesh.cell_data
      self.coords    = mesh.coords
    else:
      err = "unknown type of input mesh {}".format(type(mesh))
      raise ValueError(err)
    self.cohesive_cells = None
    self.log = self.__init_logger()
    self.log.info("number of cells %d, vertices %d",len(self.cell_data.cells),len(self.coords))
    self.log.info(
      "cell dimension %d, type %s, number of faces per vertex %d",
      self.cell_data.dim,self.cell_data.type,len(self.cell_data.face_indices[0])
    )
    return

  @classmethod
  def from_file(cls,file_name,format_in=None):
    return cls(meshio.read(file_name,format_in))

  @classmethod
  def from_POD(cls,points,cells):
    return cls(meshio.Mesh(points,cells))

  def __ne__(self,other):
    return not self.__eq__(other)

  def __eq__(self,other):
    def attr_eq(mine,other):
      if type(mine) != type(other):
        return False
      if isinstance(mine,np.ndarray):
        return np.array_equiv(mine,other)
      if scp.issparse(mine):
        from common import assert_scipy_all_close
        try:
          assert_scipy_all_close(mine,other)
        except AssertionError:
          return False
        return True
      try:
        # try the dumb way
        return mine == other
      except:
        pass
      if isinstance(mine,(list,tuple)):
        for m,o in zip(mine,other):
          if not attr_eq(m,o):
            return False
        return True
      raise NotImplemtentedError

    if id(self) == id(other):
      return True

    if not isinstance(other,self.__class__):
      return NotImplemented

    if len(self.__slots__) != len(other.__slots__):
      return False

    for attr in self.__slots__:
      if attr != "__dict__":
        try:
          my_attr    = getattr(self,attr)
          other_attr = getattr(other,attr)
        except AttributeError:
          return False
        if not attr_eq(my_attr,other_attr):
          return False

    self_dict  = self.__dict__
    other_dict = other.__dict__
    if len(self_dict.keys()) != len(other_dict.keys()):
      return False

    self_items,other_items = sorted(self_dict.items()),sorted(other_dict.items())
    for (self_key,self_val),(other_key,other_val) in zip(self_items,other_items):
      try:
        if (self_key != other_key) or (self_val != other_val):
          return False
      except ValueError:
        if not attr_eq(self_val,other_val):
          return False
    return True

  def __enter__(self):
    return self

  def __exit__(self,exc_type,exc_value,tb):
    if exc_type is not None:
      traceback.print_exception(exc_type,exc_value,tb)
    return

  def write_mesh(self,mesh_file_out,mesh_format_out=None,prune=False,return_mesh=False,embed_partitions=False):
    cells = [(self.cell_data.type,self.cell_data.cells)]
    if self.cohesive_cells is not None:
      self.log.info(
        "generated %d cohesive elements of type '%s' and %d duplicated vertices",
        len(self.cohesive_cells),self.cohesive_cells.type,len(self.dup_coords)
      )
      cells.append((self.cohesive_cells.type,self.cohesive_cells.cells))
    else:
      self.log.info("generated no cohesive elements")
    if embed_partitions:
      partdict = {("group_%s" % it) : [elems] for it,elems in enumerate(self.partitions)}
    else:
      partdict = None
    mesh_out = meshio.Mesh(self.coords,cells,cell_sets=partdict)
    if prune:
      mesh_out.remove_orphaned_nodes()
    if return_mesh:
      return mesh_out
    meshio.write(mesh_file_out,mesh_out,file_format=mesh_format_out)
    self.log.info("wrote mesh to '%s' with format '%s'",mesh_file_out,mesh_format_out)
    return

  def partition_mesh(self,num_part=-1):
    cell_data = self.cell_data
    n_cells   = len(cell_data)
    if num_part == 0:
      self.partitions = tuple()
      return self
    if num_part == -1:
      num_part = n_cells
    elif num_part > n_cells:
      self.log.warning(
        "number of partitions %d > num cells %d, using num cells instead",
        num_part,n_cells
      )
    self.adjacency_matrix        = self.compute_adjacency_matrix()
    self.cell_adjacency,bd_faces = self.compute_closure(full_closure=False)
    self.bd_set                  = set(flatten(bd_faces))
    if num_part < n_cells:
      ncuts,membership = pymetis.part_graph(num_part,adjacency=self.cell_adjacency)
      if ncuts == 0:
        raise RuntimeError("no partitions were made by partitioner")
      membership = np.array(membership)
      partitions = tuple(np.argwhere(membership == p).ravel() for p in range(num_part))
    else:
      membership = np.array([x for x in range(n_cells)])
      partitions = tuple(np.array([x]) for x in membership)
    self.partitions = partitions
    n_valid = sum(1 for partition in self.partitions if len(partition))
    self.log.info(
      "number of partitions requested %d, actual %d, average cells/partition %d",
      num_part,n_valid,n_cells/n_valid
    )
    part_count_sum = sum(len(partition) for partition in self.partitions)
    if part_count_sum != n_cells:
      err = "Partition cell-count sum {} != global number of cells {}".format(part_count_sum,n_cells)
      raise RuntimeError(err)
    return self

  def __get_partition_vertex_map(self,partitions=None,cell_set=None):
    if hasattr(self,"partition_vertex_map"):
      return self.partition_vertex_map
    if cell_set is None:
      cell_set = self.cell_data
    if partitions is None:
      self.__assert_partitioned();
      partitions = self.partitions
    partition_vertex_map      = (np.unique(cell_set.cells[part].ravel()) for part in partitions)
    self.partition_vertex_map = collections.Counter(flatten(partition_vertex_map))
    return self.partition_vertex_map

  def __compute_partition_interface(self,partition,global_adjacency_matrix=None):
    self.__get_partition_vertex_map()
    boundary_faces = []
    cell_set       = self.cell_data[partition]
    faces_per_cell = len(cell_set.face_indices)
    face_dim       = len(cell_set.face_indices[0])
    if global_adjacency_matrix is None:
      adj_mat = self.adjacency_matrix[partition,:][:,partition]
    else:
      adj_mat = global_adjacency_matrix[partition,:][:,partition].to_lil()
    for row_idx,row in enumerate(adj_mat.data):
      local_neighbors = [
        adj_mat.rows[row_idx][n] for n in (i for i,k in enumerate(row) if k == face_dim)
      ]
      self.log.debug("cell %d locally adjacent to %s",row_idx,local_neighbors)
      if len(local_neighbors) != faces_per_cell:
        # map local neighbor index to global cell ids
        mapped_local_set   = set(partition[local_neighbors])
        # get the ids of all global neighbors
        global_neighbors   = self.cell_adjacency[partition[row_idx]]
        # equivalent to set(globals).difference(mapped_local_set)
        exterior_neighbors = [n for n in global_neighbors if n not in mapped_local_set]
        # for all of my exterior neighbors, what faces do we have in common?
        vertex_set     = set(cell_set.cells[row_idx])
        exterior_faces = [
          vertex_set.intersection(self.cell_data.cells[c]) for c in exterior_neighbors
        ]
        bd_faces       = []
        # loop over all possible faces of mine, and finding the indices for the mirrored
        # face for the exterior partner cell
        for idx,face in enumerate(map(tuple,cell_set.cells[row_idx][cell_set.face_indices])):
          try:
            exteriorIdx = exterior_faces.index(set(face))
          except ValueError:
            continue
          neighbor_vertices = self.cell_data.cells[exterior_neighbors[exteriorIdx]]
          neighbor_indices  = np.array([(neighbor_vertices == x).nonzero()[0][0] for x in face])
          bd_faces.append((face,exterior_neighbors[exteriorIdx],neighbor_indices))
        if self.log.isEnabledFor(logging.DEBUG):
          if len(exterior_neighbors):
            self.log.debug("cell %d marked on interface with neighbors %s",row_idx,exterior_neighbors)
          else:
            self.log.debug("cell %d marked locally interior",row_idx)
          for face in bd_faces:
            self.log.debug("face %s marked on interface",face[0])
        boundary_faces.append(bd_faces)
      else:
        self.log.debug("cell %d marked locally interior",row_idx)
    self.log.debug("%d interface face(s)",sum(len(_) for _ in boundary_faces))
    try:
      f,si,sv = zip(*flatten(boundary_faces))
    except ValueError:
      # ValueError: not enough values to unpack (expected 3, got 0)
      f,si,sv = [],[],[]
    p = PartitionInterface(
      own_faces=np.array(f),mirror_ids=np.array(si),mirror_vertices=np.array(sv)
    )
    return p

  def __get_partition_interface_list(self):
    self.__assert_partitioned()
    if not hasattr(self,"partition_interfaces"):
      self.partition_interfaces = list(map(self.__compute_partition_interface,self.partitions))
    return self.partition_interfaces

  def compute_adjacency_matrix(self,cells=None,format="lil",v2v=False):
    def mat_size(a):
      if isinstance(a,scp.csr_matrix) or isinstance(a,scp.csc_matrix):
        return a.data.nbytes+a.indptr.nbytes+a.indices.nbytes
      elif isinstance(a,scp.lil_matrix):
        return a.data.nbytes+a.rows.nbytes
      elif isinstance(a,scp.coo_matrix):
        return a.col.nbytes+a.row.nbytes+a.data.nbytes
      return 0

    if cells is None:
      cells  = self.cell_data.cells
    ne             = len(cells)
    element_ids    = np.empty((ne,len(cells[0])),dtype=np.intp)
    element_ids[:] = np.arange(ne).reshape(-1,1)
    cell_dim       = len(element_ids[0])
    v2c            = scp.coo_matrix((
      np.ones((ne*cell_dim,),dtype=np.intp),
      (cells.ravel(),element_ids.ravel(),)
    )).tocsr(copy=False)
    if version_info <= (3,5):
      c2c = v2c.T @ v2c
    else:
      c2c = v2c.T.__matmul__(v2c)
    self.log.debug("c2c mat size %g kB",mat_size(c2c)/(1024**2))
    c2c = c2c.asformat(format,copy=False)
    self.log.debug("c2c mat size after compression %g kB",mat_size(c2c)/(1024**2))
    if v2v:
      v2v = v2c @ v2c.T
      self.log.debug("v2v mat size %d bytes",mat_size(v2v))
      v2v = v2v.asformat(format,copy=False)
      self.log.debug("v2v mat size after compression %d bytes",mat_size(v2v))
      return c2c,v2v
    else:
      return c2c

  def compute_closure(self,cell_set=None,adj_mat=None,full_closure=True):
    if cell_set is None:
      cell_set = self.cell_data
    if adj_mat is None:
      adj_mat = self.compute_adjacency_matrix(cell_set.cells)
    bd_cells,bd_faces = [],[]
    faces_per_cell    = len(cell_set.face_indices)
    face_dim          = len(cell_set.face_indices[0])
    local_adjacency   = {}
    for row_idx,row in enumerate(adj_mat.data):
      neighbors = (i for i,k in enumerate(row) if k == face_dim)
      local_adjacency[row_idx] = list(map(adj_mat.rows[row_idx].__getitem__,neighbors))
      self.log.debug("cell %d adjacent to %s",row_idx,local_adjacency[row_idx])
      if len(local_adjacency[row_idx]) != faces_per_cell:
        if full_closure:
          # the cell does not have a neighbor for every face!
          self.log.debug("cell %d marked on boundary",row_idx)
          bd_cells.append(row_idx)
        # for all of my neighbors, what faces do we have in common?
        own_vertices   = set(cell_set.cells[row_idx])
        interior_faces = [
          own_vertices.intersection(cell_set.cells[c]) for c in local_adjacency[row_idx]
        ]
        # all possible faces of mine
        all_faces = map(tuple,cell_set.cells[row_idx][cell_set.face_indices])
        bdf       = [face for face in all_faces if set(face) not in interior_faces]
        if self.log.isEnabledFor(logging.DEBUG):
          for boundary_face in bdf:
            self.log.debug("face %s marked on boundary",boundary_face)
        assert len(bdf)+len(interior_faces) == faces_per_cell
        bd_faces.append(bdf)
      else:
        self.log.debug("cell %d marked interior",row_idx)
    if full_closure:
      if self.log.isEnabledFor(logging.DEBUG):
        self.log.debug(
          "%d interior cell(s), %d boundary cell(s), %d boundary face(s)",
          len(cell_set.cells)-len(bd_cells),len(bd_cells),sum(map(len,bd_faces))
        )
      return local_adjacency,bd_cells,bd_faces
    if self.log.isEnabledFor(logging.DEBUG):
      self.log.debug("%d boundary face(s)",sum(map(len,bd_faces)))
    return local_adjacency,bd_faces

  def __duplicate_vertices(self,old_vertex_list,global_dict,coords,partition_vertex_map,dup_coords=None):
    translation_dict     = {}
    convertable_vertices = tuple(x for x in old_vertex_list if x not in global_dict)
    if len(convertable_vertices):
      try:
        num_duplicatable_vertices = len(coords)+len(dup_coords)
      except TypeError as te:
        if "object of type 'NoneType' has no len()" in te.args:
          num_duplicatable_vertices = len(coords)
        else:
          raise te
      new_vertex_coords = []
      vertex_counts     = [partition_vertex_map[x]-1 for x in convertable_vertices]
      for vertex,count in zip(convertable_vertices,vertex_counts):
        num_duplicatable_vertices_last = num_duplicatable_vertices
        num_duplicatable_vertices     += count
        # At least one other partition must own the boundary vertex
        # otherwise the routine generating local interior boundaries is buggy
        assert num_duplicatable_vertices > num_duplicatable_vertices_last
        translation_dict[vertex] = collections.deque(
          range(num_duplicatable_vertices_last,num_duplicatable_vertices)
        )
        new_vertex_coords.extend([coords[vertex] for _ in range(count)])
        self.log.debug("duped vertex %d -> %s",vertex,translation_dict[vertex])
      try:
        dup_coords.extend(new_vertex_coords)
      except AttributeError as ae:
        # dup_coords is None
        if "'NoneType' object has no attribute 'extend'" in ae.args:
          dup_coords = new_vertex_coords.copy()
        else:
          raise ae
    else:
      self.log.debug("no vertices to duplicate")
    return dup_coords,translation_dict

  def __generate_global_conversion(self,partitions,global_conversion_map):
    self.dup_coords,dup_coords = None,None
    try:
      for (idx,part),boundary in zip(enumerate(partitions),self.partition_interfaces):
        self.log.debug("partition %d contains (%d) cells %s",idx,len(part),part)
        # old vertex IDs
        old_vertices = {*flatten(boundary.own_faces)}
        # duplicate the vertices, return the duplicates new IDs
        dup_coords,local_conversion_map = self.__duplicate_vertices(
          old_vertices,global_conversion_map,self.coords,self.partition_vertex_map,dup_coords
        )
        yield part,boundary,global_conversion_map
        global_conversion_map.update(local_conversion_map)
    finally:
      # fancy trickery to update the coordinates __after__ the final yield has been called
      self.dup_coords = np.array(dup_coords)
    return

  def remap_vertices(self):
    self.__assert_partitioned();
    source_vertices       = []
    mapped_vertices       = []
    vertices_per_face     = len(self.cell_data.face_indices[0])
    global_conversion_map = dict()
    for part,boundary,global_conversion_map in self.__generate_global_conversion(self.partitions,global_conversion_map):
      # loop through every cell in the current partition, if it contains vertices that are
      # in the global conversion map then renumber using the top of the stack
      converted_partition = np.array([
        [global_conversion_map[v][0] if v in global_conversion_map else v for v in c] for c in self.cell_data.cells[part]
      ])
      try:
        # assign, throws ValueError if partition is empty
        self.cell_data.cells[part] = converted_partition
      except ValueError:
        self.log.debug("no vertices to update")
        continue
      # for every face in the list of boundary faces, convert the boundary face vertex as
      # above
      mapped_boundary_faces = [[global_conversion_map[v][0] if v in global_conversion_map else v for v in f] for f in boundary.own_faces]
      for mapped_bd_face,boundary_faces,src,idx in zip(mapped_boundary_faces,*boundary):
        # we only want to record vertices which have an entire face changed, i.e. not just
        # an edge or sole vertex
        if vertices_per_face == sum(1 for i,j in zip(mapped_bd_face,boundary_faces) if i != j):
          # now find the entries of the interface partner and take the indices
          # corresponding to our face. Note that this face is __guaranteed__ to already
          # have been handled, otherwise we would not have it in the conversion dict
          interface_partner = self.cell_data.cells[src][idx]
          source_vertices.append(interface_partner)
          mapped_vertices.append(mapped_bd_face)
          self.log.debug("updated face %s -> %s",interface_partner,mapped_bd_face)
      converted_boundary_vertices = [
        vertex for vertex in set(flatten(boundary.own_faces)) if vertex in global_conversion_map
      ]
      if self.log.isEnabledFor(logging.DEBUG):
        for vertex in converted_boundary_vertices:
          previous_vertex = global_conversion_map[vertex].popleft()
          try:
            self.log.debug(
              "updated mapping %d: %d -> %d",
              vertex,previous_vertex,global_conversion_map[vertex][0]
            )
          except IndexError as ie:
            self.log.debug("updated mapping %d: %d -> Empty",vertex,previous_vertex)
      else:
        for vertex in converted_boundary_vertices:
          global_conversion_map[vertex].popleft()
    if self.log.isEnabledFor(logging.DEBUG):
      for vertex in global_conversion_map:
        if len(global_conversion_map[vertex]):
          self.log.error(
            "vertex %d contains additional unapplied point-maps %s",
            vertex,global_conversion_map[vertex]
          )
    return source_vertices,mapped_vertices

  def insert_elements(self):
    if self.partitions:
      self.__get_partition_interface_list();
      source_vertices,mapped_vertices = self.remap_vertices()
      cells = np.hstack((mapped_vertices,source_vertices))
      self.cohesive_cells = CellSet.from_POD(self.cell_data.cohesive_type,cells)
      if self.dup_coords.shape:
        self.coords = np.vstack((self.coords,self.dup_coords))
      else:
        self.log.debug("no coordinates were duplicated!")
      if self.log.isEnabledFor(logging.DEBUG):
        for element in self.cohesive_cells.cells:
          self.log.debug("created new cohesive element %s",element)
    return self

  def verify_cohesive_mesh(self):
    if len(self.cohesive_cells):
      cohesive_set = set(frozenset(cell) for cell in self.cohesive_cells.cells)
      self.log.debug(
        "number of unique cohesive elements %s, total number of cohesive elements %s",
        len(cohesive_set),len(self.cohesive_cells)
      )
      if len(cohesive_set) != len(self.cohesive_cells):
        raise RuntimeError("there are duplicate cohesive cells!")
      self.log.info("mesh seems ok")
    else:
      self.log.info("mesh has no cohesive cells")
    return self
