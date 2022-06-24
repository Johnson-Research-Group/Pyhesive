#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
from sys import version_info
import copy
import traceback
import logging
import meshio
import pymetis
import collections
import scipy
import numpy as np
from scipy import sparse as scp

from ._util                import flatten,get_log_level,get_log_stream,_assert_scipy_all_close
from ._cell_set            import CellSet
from ._partition_interface import PartitionInterface

class Mesh:
  __slots__ = ("log","cell_data","coords","cohesive_cells","__dict__")

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

  def __get_cell_data(self,mesh_,copy_mesh):
    if copy_mesh:
      mesh = copy.deepcopy(mesh_)
    else:
      mesh = mesh_

    cohesive_cells = None
    if isinstance(mesh,meshio.Mesh):
      try:
        cell_block = mesh.cells[-1]
      except IndexError as ie:
        raise ValueError("Input mesh does not contain any cells") from ie
      all_blocks = [b for b in mesh.cells if b.dim == cell_block.dim]
      if len(all_blocks) > 1:
        # possible hybrid mesh, or cells are in separate blocks for whatever reason
        if len({b.type for b in all_blocks}) > 1:
          raise ValueError("Hybrid meshes are not supported")
        # let meshio figure out how to properly extract this for us
        cell_type = cell_block.type
        cell_data = CellSet.from_POD(cell_type,mesh.get_cells_type(cell_type))
      else:
        # only one type of top-level cell class, so create it from the block
        cell_data = CellSet.from_CellBlock(cell_block)
      coords           = mesh.points
      self.meshio_mesh = mesh
    elif isinstance(mesh,self.__class__):
      cell_data      = mesh.cell_data
      coords         = mesh.coords
      cohesive_cells = mesh.cohesive_cells
      print(vars(mesh))
    else:
      err = "Unknown type of input mesh {}".format(type(mesh))
      raise ValueError(err)
    return cell_data,coords,cohesive_cells

  def __init__(self,mesh,copy=False):
    self.dup_coords = None
    self.log        = self.__init_logger()
    self.cell_data,self.coords,self.cohesive_cells = self.__get_cell_data(mesh,copy)
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
  def from_POD(cls,points,cells,**kwargs):
    return cls(meshio.Mesh(points,cells),**kwargs)

  def __ne__(self,other):
    return not self.__eq__(other)

  def __eq__(self,other):
    def attr_eq(mine,other):
      if type(mine) != type(other):
        return False
      if isinstance(mine,np.ndarray):
        return np.array_equiv(mine,other)
      if scp.issparse(mine):
        try:
          _assert_scipy_all_close(mine,other)
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

    if self.cell_data != other.cell_data:
      return False

    if not np.array_equiv(self.coords,other.coords):
      return False

    other_cohesive = getattr(other,'cohesive_cells',None)
    if other_cohesive is None:
      other_cohesive = other.__dict__.get('cohesive_cells')
    if self.cohesive_cells != other_cohesive:
      return False

    return True

  def __str__(self):
    cell_data = self.get_cell_data()
    return '\n'.join([
      "Mesh:",
      f"dimension:      {cell_data.dim}",
      f"cells:          {cell_data}",
      f"cohesive cells: {self.get_cell_data(cohesive=True)}"
    ])

  def __enter__(self):
    return self

  def __exit__(self,exc_type,exc_value,tb):
    if exc_type is not None:
      traceback.print_exception(exc_type,exc_value,tb)
    return


  def __get_partition_vertex_map(self,partitions,cell_set=None):
    if cell_set is None:
      cell_set = self.get_cell_data()
    partition_vertex_map = (np.unique(cell_set.cells[part].ravel()) for part in partitions)
    return collections.Counter(flatten(partition_vertex_map))

  def __compute_partition_interface(self,partitions,global_adjacency_matrix=None):
    if global_adjacency_matrix is None:
      global_adjacency_matrix = self.adjacency_matrix
    adj_mat        = global_adjacency_matrix[partitions,:][:,partitions].asformat('lil')
    cell_adjacency = self.cell_adjacency
    boundary_faces = []
    cell_data      = self.get_cell_data()
    part_cell_data = cell_data[partitions]
    faces_per_cell = len(part_cell_data.face_indices)
    face_dim       = len(part_cell_data.face_indices[0])
    for row_idx,row in enumerate(adj_mat.data):
      sub_row         = adj_mat.rows[row_idx]
      local_neighbors = [sub_row[n] for n in [i for i,k in enumerate(row) if k == face_dim]]
      self.log.debug("cell %d locally adjacent to %s",row_idx,local_neighbors)
      if len(local_neighbors) != faces_per_cell:
        # map local neighbor index to global cell ids
        mapped_local_set   = set(partitions[local_neighbors])
        # get the ids of all global neighbors
        global_neighbors   = cell_adjacency[partitions[row_idx]]
        # equivalent to set(globals).difference(mapped_local_set)
        exterior_neighbors = [n for n in global_neighbors if n not in mapped_local_set]
        # for all of my exterior neighbors, what faces do we have in common?
        vertex_set     = set(part_cell_data.cells[row_idx])
        exterior_faces = [vertex_set.intersection(cell_data.cells[c]) for c in exterior_neighbors]
        bd_faces       = []
        # loop over all possible faces of mine, and finding the indices for the mirrored
        # face for the exterior partner cell
        for idx,face in enumerate(map(tuple,part_cell_data.cells[row_idx][part_cell_data.face_indices])):
          try:
            exteriorIdx = exterior_faces.index(set(face))
          except ValueError:
            continue
          neighbor_vertices = cell_data.cells[exterior_neighbors[exteriorIdx]]
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
    return PartitionInterface(
      own_faces=np.asarray(f),mirror_ids=np.asarray(si),mirror_vertices=np.asarray(sv)
    )

  def __get_partition_interface_list(self,partitions):
    if not hasattr(self,"partition_interfaces"):
      self.partition_interfaces = list(map(self.__compute_partition_interface,partitions))
    return self.partition_interfaces


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
        if "'NoneType' object has no attribute 'extend'" not in ae.args:
          raise ae
        dup_coords = new_vertex_coords.copy()
    else:
      self.log.debug("no vertices to duplicate")
    return dup_coords,translation_dict

  def __generate_global_conversion(self,partitions,global_conversion_map):
    dup_coords               = None
    partition_interface_list = self.__get_partition_interface_list(partitions)
    partition_vertex_map     = self.__get_partition_vertex_map(partitions)
    coords                   = self.get_vertices()
    try:
      for (idx,part),boundary in zip(enumerate(partitions),partition_interface_list):
        self.log.debug("partition %d contains (%d) cells %s",idx,len(part),part)
        # old vertex IDs
        old_vertices = {f for f in flatten(boundary.own_faces)}
        # duplicate the vertices, return the duplicates new IDs
        dup_coords,local_conversion_map = self.__duplicate_vertices(
          old_vertices,global_conversion_map,coords,partition_vertex_map,dup_coords=dup_coords
        )
        yield part,boundary,global_conversion_map
        global_conversion_map.update(local_conversion_map)
    finally:
      # fancy trickery to update the coordinates __after__ the final yield has been called
      self.dup_coords = np.array(dup_coords)
    return


  def get_cell_data(self,cohesive=False):
    """
    Get the cell data

    Parameter
    ---------
    cohesive : bool, optional (False)
      return cohesive or bulk cell data

    Returns
    -------
    cell_data : CellSet
      The cell data, or None if it does not exist
    """
    cell_data = self.cell_data
    if cohesive:
      try:
        cell_data = self.cohesive_cells
      except AttributeError:
        cell_data = CellSet.from_POD(cell_data.cohesive_type,np.empty(0,dtype=cell_data.dtype))
    return cell_data

  def get_cells(self,cohesive=False):
    """
    Get the mesh cells

    Parameter
    ---------
    cohesive : bool, optional (False)
      return cohesive or bulk cells

    Returns
    -------
    cells : arraylike
      Array of cells

    Notes
    -----
    Returns an empty array if cohesive is True and no cohesive elements exist
    """
    return self.get_cell_data(cohesive=cohesive).cells

  def get_vertices(self):
    """
    Get the mesh coordinates

    Returns
    -------
    coords : arraylike
      Array of all vertex coordinates
    """
    return self.coords

  def get_partitions(self):
    """
    Get the partitions

    Returns
    -------
    partitions : tuple
      Tuple of arrays containing the cell ID's of each partition

    Raises
    ------
    AssertionError
      If the mesh has no partitions
    """
    self.__assert_partitioned()
    return self.partitions


  def __set_partitions(self,partitions,overwrite=True):
    self.partitions = partitions
    if overwrite:
      for attr in ('adjacency_matrix','cell_adjacency','bd_set','cohesive_cells'):
        setattr(self,attr,None)

      self.adjacency_matrix        = self.compute_adjacency_matrix()
      self.cell_adjacency,bd_faces = self.compute_closure(full_closure=False)
      self.bd_set                  = set(flatten(bd_faces))
    return partitions


  def partition_mesh(self,num_part=-1):
    """
    Partition the mesh

    Parameter
    ---------
    num_part : int, optional
      The number of partitions to generate, default is to generate as many partitions as there
      are elements

    Returns
    -------
    partitions : arraylike
      The partitions, a tuple of arrays containing the cell ID's of each partition

    Raises
    ------
    RuntimeError
      If the sum of cells in partitions != global cell count. This indicates an internal error
      within the partitioning library.

      If the partitioner failed to make any partitions.

    Notes
    -----
    It is ok to pass 0 as num_part, in which case an empty tuple is returned.
    """
    partitions = tuple()
    overwrite  = True
    if num_part != 0:
      cell_data = self.get_cell_data()
      n_cells   = len(cell_data)
      if num_part == -1:
        num_part = n_cells
      elif num_part > n_cells:
        self.log.warning(
          "number of partitions %d > num cells %d, using num cells instead",num_part,n_cells
        )
      if num_part < n_cells:
        overwrite                    = False
        self.adjacency_matrix        = self.compute_adjacency_matrix()
        self.cell_adjacency,bd_faces = self.compute_closure(full_closure=False)
        self.bd_set                  = set(flatten(bd_faces))
        ncuts,membership             = pymetis.part_graph(num_part,adjacency=self.cell_adjacency)
        if ncuts == 0:
          raise RuntimeError("no partitions were made by partitioner")
        membership = np.array(membership)
        partitions = tuple(np.argwhere(membership == p).ravel() for p in range(num_part))
      else:
        membership = np.array([x for x in range(n_cells)])
        partitions = tuple(np.array([x]) for x in membership)
      n_valid = sum(1 for partition in partitions if len(partition))
      self.log.info(
        "number of partitions requested %d, actual %d, average cells/partition %d",
        num_part,n_valid,n_cells/n_valid
      )
      part_count_sum = sum(len(partition) for partition in partitions)
      if part_count_sum != n_cells:
        err = "Partition cell-count sum {} != global number of cells {}".format(part_count_sum,n_cells)
        raise RuntimeError(err)
    return self.__set_partitions(partitions,overwrite=overwrite)

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
      cells = self.get_cells()
    ne             = len(cells)
    element_ids    = np.empty((ne,len(cells[0])),dtype=np.intp)
    element_ids[:] = np.arange(ne).reshape(-1,1)
    cell_dim       = len(element_ids[0])
    v2c            = scp.coo_matrix((
      np.ones((ne*cell_dim,),dtype=element_ids.dtype),
      (cells.ravel(),element_ids.ravel(),)
    )).tocsr(copy=False)
    if version_info >= (3,5):
      # novermin infix matrix multiplication requires !2, 3.5
      c2c = v2c.T @ v2c
    else:
      c2c = v2c.T.__matmul__(v2c)
    self.log.debug("c2c mat size %g kB",mat_size(c2c)/(1024**2))
    c2c = c2c.asformat(format,copy=False)
    self.log.debug("c2c mat size after compression %g kB",mat_size(c2c)/(1024**2))
    if v2v:
      if version_info >= (3,5):
        # novermin infix matrix multiplication requires !2, 3.5
        v2v = v2c @ v2c.T
      else:
        v2v = v2c.__matmul__(v2c.T)
      self.log.debug("v2v mat size %d bytes",mat_size(v2v))
      v2v = v2v.asformat(format,copy=False)
      self.log.debug("v2v mat size after compression %d bytes",mat_size(v2v))
      return c2c,v2v
    return c2c

  def compute_closure(self,cell_set=None,adj_mat=None,full_closure=True):
    if cell_set is None:
      cell_set = self.get_cell_data()
    cells = cell_set.cells
    if adj_mat is None:
      adj_mat = self.compute_adjacency_matrix(cells)
    bd_cells,bd_faces = [],[]
    face_indices      = cell_set.face_indices
    faces_per_cell    = len(face_indices)
    face_dim          = len(face_indices[0])
    local_adjacency   = {}
    for row_idx,row in enumerate(adj_mat.data):
      neighbors = [i for i,k in enumerate(row) if k == face_dim]
      local_adjacency[row_idx] = list(map(adj_mat.rows[row_idx].__getitem__,neighbors))
      self.log.debug("cell %d adjacent to %s",row_idx,local_adjacency[row_idx])
      if len(local_adjacency[row_idx]) != faces_per_cell:
        if full_closure:
          # the cell does not have a neighbor for every face!
          self.log.debug("cell %d marked on boundary",row_idx)
          bd_cells.append(row_idx)
        # for all of my neighbors, what faces do we have in common?
        own_vertices   = set(cells[row_idx])
        interior_faces = [own_vertices.intersection(cells[c]) for c in local_adjacency[row_idx]]
        # all possible faces of mine
        all_faces = map(tuple,cells[row_idx][face_indices])
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
          len(cells)-len(bd_cells),len(bd_cells),sum(map(len,bd_faces))
        )
      return local_adjacency,bd_cells,bd_faces
    if self.log.isEnabledFor(logging.DEBUG):
      self.log.debug("%d boundary face(s)",sum(map(len,bd_faces)))
    return local_adjacency,bd_faces

  def remap_vertices(self,partitions):
    source_vertices   = []
    mapped_vertices   = []
    vertices_per_face = len(self.get_cell_data().face_indices[0])
    cells             = self.get_cells()
    for part,boundary,global_conversion_map in self.__generate_global_conversion(partitions,{}):
      # loop through every cell in the current partition, if it contains vertices that are
      # in the global conversion map then renumber using the top of the stack
      converted_partition = np.array([
        [global_conversion_map[v][0] if v in global_conversion_map else v for v in c] for c in cells[part]
      ])
      try:
        # assign, throws ValueError if partition is empty
        cells[part] = converted_partition
      except ValueError:
        self.log.debug("no vertices to update")
        continue
      # for every face in the list of boundary faces, convert the boundary face vertex as
      # above
      mapped_boundary_faces = [
        [global_conversion_map[v][0] if v in global_conversion_map else v for v in f] for f in boundary.own_faces
      ]
      for mapped_bd_face,boundary_faces,src,idx in zip(mapped_boundary_faces,*boundary):
        # we only want to record vertices which have an entire face changed, i.e. not just
        # an edge or sole vertex
        if vertices_per_face == sum(1 for i,j in zip(mapped_bd_face,boundary_faces) if i != j):
          # now find the entries of the interface partner and take the indices
          # corresponding to our face. Note that this face is __guaranteed__ to already
          # have been handled, otherwise we would not have it in the conversion dict
          interface_partner = cells[src][idx]
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

  def insert_elements(self,partitions=None):
    """
    Insert cohesive linkage elements into the mesh according to partitions

    Parameter
    ---------
    partitions : arraylike, optional
      Array of cell paritions between which to insert elements. If not given, defaults to
      partition list generated from a previous call to Mesh.partition_mesh().

    Returns
    -------
    cohesive_cells : arraylike
      Array of newly created cohesive elements
    dup_coords : arraylike
      Array of newly duplicated vertices
    """
    if partitions is None:
      partitions = self.get_partitions()

    if len(partitions):
      self.__set_partitions(partitions)
      source_vertices,mapped_vertices = self.remap_vertices(partitions)
      cells          = np.hstack((mapped_vertices,source_vertices))
      cohesive_type  = self.get_cell_data().cohesive_type
      cohesive_cells = CellSet.from_POD(cohesive_type,cells)
      assert cohesive_cells.type == cohesive_type
      if self.dup_coords.shape:
        self.coords = np.vstack((self.coords,self.dup_coords))
      if self.log.isEnabledFor(logging.DEBUG):
        for element in cohesive_cells:
          self.log.debug("created new cohesive element %s",element)
      self.log.info(
        "generated %d cohesive elements of type '%s' and %d duplicated vertices",
        len(cohesive_cells),cohesive_type,len(self.dup_coords)
      )
      self.cohesive_cells = cohesive_cells
    return self.cohesive_cells,self.dup_coords

  def verify_cohesive_mesh(self):
    cohesive_cells = self.get_cells(cohesive=True)
    num_cohesive   = len(cohesive_cells)
    self.log.info("mesh has %d cohesive cells",num_cohesive)
    if num_cohesive:
      cohesive_set = set(map(frozenset,cohesive_cells))
      self.log.debug(
        "number of unique cohesive elements %s, total number of cohesive elements %s",
        len(cohesive_set),len(cohesive_cells)
      )
      if len(cohesive_set) != len(cohesive_cells):
        raise RuntimeError("there are duplicate cohesive cells!")
      all_cells    = {n for c in self.get_cells() for n in c}
      num_orphaned = abs(
        len(all_cells.intersection({n for c in cohesive_set for n in c}))-len(self.get_vertices())
      )
      if num_orphaned != 0:
        raise RuntimeError("have %d orphaned nodes" % num_orphaned)
      self.log.info("mesh seems ok")
    return

  def write_mesh(self,mesh_file_out,mesh_format_out=None,prune=False,return_mesh=False,embed_partitions=False):
    cell_data = self.get_cell_data()
    cells     = [(cell_data.type,cell_data.cells)]
    if self.cohesive_cells is not None:
      cells.append((self.cohesive_cells.type,self.cohesive_cells.cells))
    if embed_partitions:
      partdict = {("group_%s" % it) : [elems] for it,elems in enumerate(self.get_partitions())}
    else:
      partdict = None
    #coords   = np.vstack((self.get_vertices(),self.dup_coords))
    mesh_out = meshio.Mesh(self.coords,cells,cell_sets=partdict)
    if prune:
      import warnings

      warnings.warn("prune argument is deprecated",category=DeprecationWarning)
    if return_mesh:
      return mesh_out
    meshio.write(mesh_file_out,mesh_out,file_format=mesh_format_out)
    self.log.info("wrote mesh to '%s' with format '%s'",mesh_file_out,mesh_format_out)
    return
