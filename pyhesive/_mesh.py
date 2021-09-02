#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
from ._utils import flatten,getLogLevel,getLogStream
from ._cellset import CellSet
from ._partitioninterface import PartitionInterface

import logging
import meshio
import pymetis
import collections
import scipy
from scipy import sparse
import numpy as np

class Mesh(object):
  def __initLogger(self):
    slog = logging.getLogger(self.__class__.__name__)
    verbosity = getLogLevel()
    slog.setLevel(verbosity)
    slog.propagate = False
    stream = getLogStream()
    ch = logging.StreamHandler(stream)
    ch.setLevel(verbosity)
    ch.propagate = False
    formatter    = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s")
    ch.setFormatter(formatter)
    slog.addHandler(ch)
    return slog

  def __assertPartitioned(self):
    assert hasattr(self,"partitions"),"Must partition mesh first"
    return

  def __init__(self,mesh):
    if isinstance(mesh,meshio.Mesh):
      try:
        self.cellData = CellSet.fromCellBlock(mesh.cells[-1])
      except IndexError as ie:
        raise ValueError("input mesh does not contain any cells") from ie
      self.coords = mesh.points
    elif isinstance(mesh,Mesh):
      self.cellData = mesh.cellData
      self.coords   = mesh.coords
    else:
      raise ValueError("unknown type of input mesh {}".format(type(mesh)))
    self.cohesiveCells = None
    self.log = self.__initLogger();
    self.log.info("number of cells %d, vertices %d",len(self.cellData.cells),len(self.coords))
    self.log.info("cell dimension %d, type %s, number of faces per vertex %d",
                  self.cellData.dim,self.cellData.type,len(self.cellData.faceIndices[0]))
    return

  @classmethod
  def fromFile(cls,filename,formatIn=None):
    mesh = meshio.read(filename,formatIn)
    return cls(mesh)

  @classmethod
  def fromPOD(cls,points,cells):
    mesh = meshio.Mesh(points,cells)
    return cls(mesh)

  def __ne__(self,other):
    return not self.__eq__(other)

  def __eq__(self,other):
    if id(self) == id(other):
      return True
    if isinstance(other,Mesh):
      selfDict  = self.__dict__
      otherDict = other.__dict__

      if len(selfDict.keys()) != len(otherDict.keys()):
        return False
      selfItems = sorted(selfDict.items())
      otherItems = sorted(otherDict.items())
      for (selfkey,selfval),(otherkey,otherval) in zip(selfItems,otherItems):
        try:
          if (selfkey != otherkey) or (selfval != otherval):
            return False
        except ValueError:
          if isinstance(selfval,np.ndarray):
            if not np.array_equiv(selfval,otherval):
              return False
          elif scipy.sparse.issparse(selfval):
            from common import assertScipyAllClose
            try:
              assertScipyAllClose(selfval,otherval)
            except AssertionError:
              return False
      return True
    return NotImplemented

  def __enter__(self):
    return self

  def __exit__(self,exc_type,exc_value,tb):
    if exc_type is not None:
      import traceback
      traceback.print_exception(exc_type,exc_value,tb)
    return

  def writeMesh(self,meshFileOut,meshFormatOut=None,prune=False,returnMesh=False):
    cells = [(self.cellData.type,self.cellData.cells)]
    if self.cohesiveCells is not None:
      self.log.info("generated %d cohesive elements of type '%s' and %d duplicated vertices",
                    len(self.cohesiveCells),self.cohesiveCells.type,len(self.dupCoords))
      cells.append((self.cohesiveCells.type,self.cohesiveCells.cells))
    else:
      self.log.info("generated no cohesive elements")
    meshOut = meshio.Mesh(self.coords,cells)
    if prune:
      meshOut.remove_orphaned_nodes()
    if returnMesh:
      return meshOut
    else:
      meshio.write(meshFileOut,meshOut,file_format=meshFormatOut)
      self.log.info("wrote mesh to '%s' with format '%s'",meshFileOut,meshFormatOut)
    return

  def partitionMesh(self,numPart=-1):
    if numPart == 0:
      self.partitions = tuple()
      return self
    if numPart == -1:
      numPart = len(self.cellData)
    elif numPart > len(self.cellData):
      self.log.warning("number of partitions %d > num cells %d, using num cells instead",numPart,len(self.cellData))
    self.adjacencyMatrix       = self.computeAdjacencyMatrix()
    self.cellAdjacency,bdFaces = self.computeClosure(fullClosure=False)
    self.bdSet  = set(flatten(bdFaces))
    if numPart < len(self.cellData):
      ncuts,membership = pymetis.part_graph(numPart,adjacency=self.cellAdjacency)
      if ncuts == 0:
        raise RuntimeError("no partitions were made by partitioner")
      membership      = np.array(membership)
      self.partitions = tuple(np.argwhere(membership == x).ravel() for x in range(numPart))
    else:
      membership      = np.array([x for x in range(len(self.cellData))])
      self.partitions = tuple(np.array([x]) for x in membership)
    nValid = sum(1 for p in self.partitions if len(p))
    self.log.info("number of partitions requested %d, actual %d, average cells/partition %d",
                  numPart,nValid,len(self.cellData)/nValid)
    partCountSum  = sum([len(x) for x in self.partitions])
    if partCountSum != len(self.cellData):
      raise RuntimeError("Partition cell-count sum %d != global number of cells %d" %
                         (partCountSum,len(self.cellData)))
    return self

  def __computePartitionVertexMap(self,partitions=None,cellSet=None):
    if hasattr(self,"partVMap"):
      return self.partVMap
    if cellSet is None:
      cellSet = self.cellData
    if partitions is None:
      self.__assertPartitioned();
      partitions = self.partitions
    partVMap      = tuple(np.unique(cellSet.cells[part].ravel()) for part in partitions)
    self.partVMap = collections.Counter(flatten(partVMap))
    return self.partVMap

  def __computePartitionInterface(self,partition,globalAdjacencyMatrix=None):
    self.__computePartitionVertexMap();
    bdfaces      = []
    cellSet      = self.cellData[partition]
    facesPerCell = len(cellSet.faceIndices)
    faceDim      = len(cellSet.faceIndices[0])
    if globalAdjacencyMatrix is None:
      adjMat = self.adjacencyMatrix[partition,:][:,partition]
    else:
      adjMat = globalAdjacencyMatrix[partition,:][:,partition].to_lil()
    for rowindex,row in enumerate(adjMat.data):
      # localNeighbors will have __local__ numbering
      localNeighbors = [adjMat.rows[rowindex][n] for n in (i for i,k in enumerate(row) if k == faceDim)]
      self.log.debug("cell %d locally adjacent to %s",rowindex,localNeighbors)
      if len(localNeighbors) != facesPerCell:
        # map local neighbor index to global cell ids
        mappedLocalSet    = set(partition[localNeighbors])
        # get the ids of all global neighbors
        globalNeighbors   = self.cellAdjacency[partition[rowindex]]
        # equivalent to set(globals).difference(mappedLocalSet)
        exteriorNeighbors = [n for n in globalNeighbors if n not in mappedLocalSet]
        # for all of my exterior neighbors, what faces do we have in common?
        vertexSet     = set(cellSet.cells[rowindex])
        exteriorFaces = [vertexSet.intersection(self.cellData.cells[c]) for c in exteriorNeighbors]
        boundaryFaces = []
        # loop over all possible faces of mine, and finding the indices for the mirrored
        # face for the exterior partner cell
        for idx,face in enumerate(map(tuple,cellSet.cells[rowindex][cellSet.faceIndices])):
          try:
            exteriorIdx = exteriorFaces.index(set(face))
          except ValueError:
            continue
          neighborVertices = self.cellData.cells[exteriorNeighbors[exteriorIdx]]
          neighborIndices  = np.array([(neighborVertices == x).nonzero()[0][0] for x in face])
          boundaryFaces.append((face,exteriorNeighbors[exteriorIdx],neighborIndices))
        if self.log.isEnabledFor(logging.DEBUG):
          if len(exteriorNeighbors):
            self.log.debug("cell %d marked on interface with neighbors %s",rowindex,exteriorNeighbors)
          else:
            self.log.debug("cell %d marked locally interior",rowindex)
          for f in boundaryFaces:
            self.log.debug("face %s marked on interface" % (f[0],))
        bdfaces.append(boundaryFaces)
      else:
        self.log.debug("cell %d marked locally interior",rowindex)
    self.log.debug("%d interface face(s)",sum(len(_) for _ in bdfaces))
    try:
    #if len(bdfaces):
      f,si,sv = zip(*flatten(bdfaces))
    except ValueError:
      # ValueError: not enough values to unpack (expected 3, got 0)
      f,si,sv = [],[],[]
    return PartitionInterface(ownFaces=np.array(f),mirrorIds=np.array(si),mirrorVertices=np.array(sv))

  def __computePartitionInterfaceList(self):
    self.__assertPartitioned();
    if not hasattr(self,"partitionInterfaces"):
      self.partitionInterfaces = [self.__computePartitionInterface(p) for p in self.partitions]
    return self.partitionInterfaces

  def computeAdjacencyMatrix(self,cells=None,format="lil",v2v=False):
    from sys import version_info
    def matsize(a):
      if isinstance(a,scipy.sparse.csr_matrix) or isinstance(a,scipy.sparse.csc_matrix):
        return a.data.nbytes+a.indptr.nbytes+a.indices.nbytes
      elif isinstance(a,scipy.sparse.lil_matrix):
        return a.data.nbytes+a.rows.nbytes
      elif isinstance(a,scipy.sparse.coo_matrix):
        return a.col.nbytes+a.row.nbytes+a.data.nbytes
      return 0

    if cells is None:
      cells  = self.cellData.cells
    ne       = len(cells)
    elIds    = np.empty((ne,len(cells[0])),dtype=np.intp)
    elIds[:] = np.arange(ne).reshape(-1,1)
    cDim     = len(elIds[0])
    v2c      = scipy.sparse.coo_matrix((np.ones((ne*cDim,),dtype=np.intp),(cells.ravel(),elIds.ravel(),),))
    v2c      = v2c.tocsr(copy=False)
    if version_info <= (3,5):
      c2c = v2c.T @ v2c
    else:
      c2c = v2c.T.__matmul__(v2c)
    self.log.debug("c2c mat size %g kB",matsize(c2c)/(1024**2))
    c2c = c2c.asformat(format,copy=False)
    self.log.debug("c2c mat size after compression %g kB",matsize(c2c)/(1024**2))
    if v2v:
      v2v = v2c @ v2c.T
      self.log.debug("v2v mat size %d bytes" % matsize(v2v))
      v2v = v2v.asformat(format,copy=False)
      self.log.debug("v2v mat size after compression %d bytes",matsize(v2v))
      return c2c,v2v
    else:
      return c2c

  def computeClosure(self,cellSet=None,adjMat=None,fullClosure=True):
    if cellSet is None:
      cellSet = self.cellData
    if adjMat is None:
      adjMat = self.computeAdjacencyMatrix(cellSet.cells)
    bdcells,bdfaces = [],[]
    facesPerCell    = len(cellSet.faceIndices)
    faceDim         = len(cellSet.faceIndices[0])
    localAdjacency  = {}
    for rowindex,row in enumerate(adjMat.data):
      neighbors = (i for i,k in enumerate(row) if k == faceDim)
      localAdjacency[rowindex] = list(map(adjMat.rows[rowindex].__getitem__,neighbors))
      self.log.debug("cell %d adjacent to %s",rowindex,localAdjacency[rowindex])
      if len(localAdjacency[rowindex]) != facesPerCell:
        if fullClosure:
          # the cell does not have a neighbor for every face!
          self.log.debug("cell %d marked on boundary",rowindex)
          bdcells.append(rowindex)
        # for all of my neighbors, what faces do we have in common?
        ownVertices   = set(cellSet.cells[rowindex])
        interiorFaces = [ownVertices.intersection(cellSet.cells[c]) for c in localAdjacency[rowindex]]
        # all possible faces of mine
        allFaces = map(tuple,cellSet.cells[rowindex][cellSet.faceIndices])
        bdf      = [face for face in allFaces if set(face) not in interiorFaces]
        if self.log.isEnabledFor(logging.DEBUG):
          for f in bdf:
            self.log.debug("face %s marked on boundary" % (f,))
        assert len(bdf)+len(interiorFaces) == facesPerCell
        bdfaces.append(bdf)
      else:
        self.log.debug("cell %d marked interior",rowindex)
    if fullClosure:
      self.log.debug("%d interior cell(s), %d boundary cell(s), %d boundary face(s)",len(cellSet.cells)-len(bdcells),len(bdcells),sum(len(_) for _ in bdfaces))
      return localAdjacency,bdcells,bdfaces
    else:
      self.log.debug("%d boundary face(s)",sum(len(_) for _ in bdfaces))
      return localAdjacency,bdfaces

  def __duplicateVertices(self,oldVertexList,globalDict,coords,partVMap,dupCoords=None):
    tdict = {}
    convertableVertices = tuple(x for x in oldVertexList if x not in globalDict)
    if len(convertableVertices):
      try:
        ndv = len(coords)+len(dupCoords)
      except TypeError as te:
        if "object of type 'NoneType' has no len()" in te.args:
          ndv = len(coords)
        else:
          raise te
      vCounts,newVCoords = [partVMap[x]-1 for x in convertableVertices],[]
      for v,cnt in zip(convertableVertices,vCounts):
        ndvLast = ndv
        ndv    += cnt
        # At least one other partition must own the boundary vertex
        # otherwise the routine generating local interior boundaries is buggy
        assert ndv > ndvLast
        tdict[v] = collections.deque(range(ndvLast,ndv))
        newVCoords.extend([coords[v] for _ in range(cnt)])
        self.log.debug("duped vertex %d -> %s",v,tdict[v])
      try:
        dupCoords.extend(newVCoords)
      except AttributeError as ae:
        # dupCoords is None
        if "'NoneType' object has no attribute 'extend'" in ae.args:
          dupCoords = newVCoords.copy()
        else:
          raise ae
    else:
      self.log.debug("no vertices to duplicate")
    return dupCoords,tdict

  def __generateGlobalConversion(self,partitions,globConvDict):
    self.dupCoords,dupCoords = None,None
    try:
      for (idx,part),boundary in zip(enumerate(partitions),self.partitionInterfaces):
        self.log.debug("partition %d contains (%d) cells %s",idx,len(part),part)
        # old vertex IDs
        oldVertices = {*flatten(boundary.ownFaces)}
        # duplicate the vertices, return the duplicates new IDs
        dupCoords,locConvDict = self.__duplicateVertices(oldVertices,globConvDict,
                                                         self.coords,self.partVMap,dupCoords)
        yield part,boundary,globConvDict
        globConvDict.update(locConvDict)
    finally:
      # fancy trickery to update the coordinates __after__ the final yield has been called
      self.dupCoords = np.array(dupCoords)
    return

  def remapVertices(self):
    self.__assertPartitioned();
    sourceVertices,mappedVertices = [],[]
    facelen = len(self.cellData.faceIndices[0])
    gConv   = dict()
    for part,boundary,gConv in self.__generateGlobalConversion(self.partitions,gConv):
      # loop through every cell in the current partition, if it contains vertices that are
      # in the global conversion map then renumber using the top of the stack
      convertedPartition = np.array([[gConv[v][0] if v in gConv else v for v in c] for c in self.cellData.cells[part]])
      try:
        # assign, throws ValueError if partition is empty
        self.cellData.cells[part] = convertedPartition
      except ValueError:
        self.log.debug("no vertices to update")
        continue
      # for every face in the list of boundary faces, convert the boundary face vertex as
      # above
      mappedBoundaryFaces = [[gConv[v][0] if v in gConv else v for v in f] for f in boundary.ownFaces]
      for mbdf,bdf,src,idx in zip(mappedBoundaryFaces,*boundary):
        # we only want to record vertices which have an entire face changed, i.e. not just
        # an edge or sole vertex
        diffs = sum(1 for i,j in zip(mbdf,bdf) if i != j)
        if diffs == facelen:
          # now find the entries of the interface partner and take the indices
          # corresponding to our face. Note that this face is __guaranteed__ to already
          # have been handled, otherwise we would not have it in the conversion dict
          interfacePartner = self.cellData.cells[src][idx]
          sourceVertices.append(interfacePartner)
          mappedVertices.append(mbdf)
          self.log.debug("updated face %s -> %s",interfacePartner,mbdf)
      convertedBoundaryVertices = [x for x in set(flatten(boundary.ownFaces)) if x in gConv]
      if self.log.isEnabledFor(logging.DEBUG):
        for v in convertedBoundaryVertices:
          prev = gConv[v].popleft()
          try:
            self.log.debug("updated mapping %d: %d -> %d",v,prev,gConv[v][0])
          except IndexError as ie:
            self.log.debug("updated mapping %d: %d -> Empty",v,prev)
      else:
        for v in convertedBoundaryVertices: gConv[v].popleft()
    if self.log.isEnabledFor(logging.DEBUG):
      for v in gConv:
        if len(gConv[v]):
          self.log.error("vertex %d contains additional unapplied point-maps %s",v,gConv[v])
    return sourceVertices,mappedVertices

  def insertElements(self):
    if self.partitions:
      self.__computePartitionInterfaceList();
      sourceVertices,mappedVertices = self.remapVertices()
      cells = np.hstack((mappedVertices,sourceVertices))
      self.cohesiveCells = CellSet.fromPOD(self.cellData.cohesiveType,cells)
      if self.dupCoords.shape:
        self.coords = np.vstack((self.coords,self.dupCoords))
      else:
        self.log.debug("no coordinates were duplicated!")
      if self.log.isEnabledFor(logging.DEBUG):
        for e in self.cohesiveCells.cells:
          self.log.debug("created new cohesive element %s",e)
    return self

  def verifyCohesiveMesh(self):
    if len(self.cohesiveCells):
      cohesiveSet = set(frozenset(cell) for cell in self.cohesiveCells.cells)
      self.log.debug("number of unique cohesive elements %s, total number of cohesive elements %s",
                     len(cohesiveSet),len(self.cohesiveCells))
      if len(cohesiveSet) != len(self.cohesiveCells):
        raise RuntimeError("there are duplicate cohesive cells!")
      self.log.info("mesh seems ok")
    else:
      self.log.info("mesh has no cohesive cells")
    return self
