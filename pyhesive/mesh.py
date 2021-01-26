#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
from .utils import *
from .optsctx import Optsctx

import logging, atexit, meshio, pymetis
from collections import defaultdict
from scipy import sparse
import numpy as np

infoDict = {
    'tetra' : [3, 4],
    'triangle' : [2, 3],
    'hexahedron' : [4, 8],
    'wedge' : [3, 6],
    'wedge12' : [6, 12]
}

class Mesh(Optsctx):
    cDim = 0
    cType = None
    coords = None
    cells = None
    cohesiveCells = None
    bdCells = []
    bdFaces = []
    faceDim = 0
    nFaces = 0
    cAdj = []
    ncuts = 0
    membership = None
    registered_exit = False

    def __init__(self, mesh=None):
        if mesh is None:
            super().__init__()
            mesh = meshio.read(self.meshFileIn, self.meshFormatIn)
        else:
            super().__init__(False)
            if not isinstance(mesh, meshio.Mesh):
                raise TypeError("Mesh must be of type meshio.Mesh")
        slog = logging.getLogger(self.__class__.__name__)
        slog.setLevel(super().vLevel)
        slog.propagate = False
        ch = logging.StreamHandler(super().stream)
        ch.setLevel(super().vLevel)
        ch.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        ch.setFormatter(formatter)
        slog.addHandler(ch)
        self.log = slog
        self.cType = mesh.cells[-1].type
        self.cells = mesh.cells_dict[self.cType]
        self.coords = mesh.points
        self.cDim = len(self.cells[0])
        self.faceDim = infoDict[self.cType][0]
        self.nFaces = infoDict[self.cType][1]
        self.cAdj, self.bdCells, self.bdFaces = self.GenerateAdjacency(self.cells)
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        self.Finalize()
        return True

    def Finalize(self):
        if hasattr(self, 'log'):
            handlers = self.log.handlers[:]
            for handler in handlers:
                handler.flush()
                handler.close()
                self.log.removeHandler(handler)
        atexit.unregister(self.Finalize)
        self.registered_exit = False
        return

    def WriteMesh(self):
        cells = [(self.cType, self.cells)]
        if self.cohesiveCells is not None:
            npts = len(self.cohesiveCells[0])
            cohesiveType = None
            for ctype, info in infoDict.items():
                if info[1] == npts:
                    cohesiveType = ctype
                    break
            if cohesiveType is None:
                raise RuntimeError("Cohesive type not recognized!")
            cells.append((cohesiveType, self.cohesiveCells))
        meshOut = meshio.Mesh(self.coords, cells)
        meshio.write(self.meshFileOut, meshOut, file_format=self.meshFormatOut)

    def PartitionMesh(self):
        self.ncuts, membership = pymetis.part_graph(self.numPart, adjacency=self.cAdj)
        if (self.ncuts == 0):
            raise RuntimeError("No cuts were made by partitioner")
        self.membership = np.array(membership, dtype=np.intp)
        self.partitions = [np.argwhere(self.membership == x).ravel() for x in range(self.numPart)]

    def GenerateElements(self):
        sourceVerts, mappedVerts = self.RemapVertices()
        self.CreateElements(sourceVerts, mappedVerts)

    def BuildAdjacencyMatrix(self, Cells, cDim=None, format='csr', v2v=False):
        def matsize(a):
            sum = 0
            if isinstance(a, sparse.csr_matrix) or isinstance(a, sparse.csc_matrix):
                sum = a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
            elif isinstance(a, sparse.lil_matrix):
                sum = a.data.nbytes + a.rows.nbytes
            elif isinstance(a, sparse.coo_matrix):
                sum = a.col.nbytes + a.row.nbytes + a.data.nbytes
            return sum

        if cDim is None:
            cDim = self.cDim
        ne = len(Cells)
        element_ids = np.empty((ne, cDim), dtype=np.intp)
        element_ids[:] = np.arange(ne).reshape(-1, 1)
        v2c = sparse.coo_matrix(
            (np.ones((ne*len(element_ids[0]),), dtype=np.intp),
            (Cells.ravel(),
            element_ids.ravel(),)))
        v2c = v2c.tocsr(copy=False)
        c2c = v2c.T @ v2c
        self.log.info("c2c mat size %d bytes" % matsize(c2c))
        c2c = c2c.asformat(format, copy=False)
        self.log.info("c2c mat size after compression %d bytes" % matsize(c2c))
        if v2v:
            v2v = v2c @ v2c.T
            self.log.info("v2v mat size %d bytes" % matsize(v2v))
            v2v = v2v.asformat(format, copy=False)
            self.log.info("v2v mat size after compression %d bytes" % matsize(v2v))
            return c2c, v2v
        else:
            return c2c

    def GenerateAdjacency(self, Cells, faceDim = None, nfaces = None):
        from itertools import combinations

        def list2dict(inlist):
            odict = defaultdict(list)
            for i in range(len(inlist)):
                odict[i].append(inlist[i])
            return odict

        if faceDim is None:
            faceDim = self.faceDim
        if nfaces is None:
            nfaces = self.nFaces
        c2c = self.BuildAdjacencyMatrix(Cells, format='lil')
        adj = {}
        bdcells = []
        bdfaces = []
        for rowindex, row in enumerate(c2c.data):
            sharedfaces = [i for i,k in enumerate(row) if k == faceDim]
            adj[rowindex] = [c2c.rows[rowindex][j] for j in sharedfaces]
            self.log.debug("cell %d adjacent to %s" % (rowindex, l2s(adj[rowindex])))
            if (len(sharedfaces) != nfaces):
                self.log.debug("cell %d marked on boundary" % (rowindex))
                bdcells.append(rowindex)
                faces = [set(np.intersect1d(Cells[rowindex],Cells[c],assume_unique=True)) for c in adj[rowindex]]
                comb = [set(c) for c in combinations(Cells[rowindex], faceDim)]
                bdfaces.append([list(face) for face in comb if face not in faces])
            else:
                self.log.debug("cell %d marked interior" % (rowindex))
        return adj, bdcells, bdfaces

    def GenerateLocalBoundaryFaces(self):
        # Extract the partition
        for part in self.partitions:
            # Generate boundaries for the entire partition, this will include
            # both local and global boundary cells
            _, bdCells_l, bdFaces_l = self.GenerateAdjacency(self.cells[part])
            bdCells_g = part[bdCells_l]
            # should be as many bdface entries as bd cells
            assert(len(bdCells_l) == len(bdFaces_l))
            # Find difference between global boundary cells and local+global
            # boundary cells to isolate local boundary cells
            locBdCells_g = np.setdiff1d(bdCells_g, self.bdCells,
                                        assume_unique=True)
            # Extract list of boundary faces and add them to dict
            bdNodes = [flattenList(bdFaces_l[np.where(bdCells_g == x)[0][0]]) for x in locBdCells_g]
            yield dict((i,j) for i,j in enumerate(bdNodes))

    def DuplicateAndInsertVertices(self, oldVertexList, globalDict = None):
        try:
            convertableVertices = [x for x in oldVertexList if x not in globalDict]
        except TypeError:
            pass
        nv = len(self.coords)
        newVertexList = range(nv, nv+len(convertableVertices))
        for i,j in zip(convertableVertices, newVertexList):
            self.log.debug("Duped vertex %d -> %d" % (i, j))
        newVertexCoords = [self.coords[v] for v in convertableVertices]
        self.coords = np.vstack((self.coords, newVertexCoords))
        return dict(zip(convertableVertices, newVertexList))

    def GenerateLocalConversion(self):
        convDict = {}
        partConvDict = {}
        # Iterate through partitions, faceDict contains dictionary of faces
        for i, partition in enumerate(self.GenerateLocalBoundaryFaces()):
            # old vertex IDs
            oldVertices = list({s:None for s in flattenList(partition.values())})
            # duplicate the vertices, return the duplicates new IDs
            partConvDict[i] = self.DuplicateAndInsertVertices(oldVertices, convDict)
            convDict.update(partConvDict[i])
            yield partConvDict[i]

    def RemapVertices(self):
        '''
        Identify internal vertices on partition edges, and remap them to new cohesive vertices

        Returns
        -------
        np.array
            list of pre-mapping vertices that underwent mapping
        np.array
            new vertex ID's mapped

        '''
        modSourceVertices = []
        modDestVertices = []
        for i, locConv in enumerate(self.GenerateLocalConversion()):
            locs = self.partitions[i]
            part = np.empty(np.shape(self.cells[locs]), dtype=self.cells.dtype)
            for j, cell in enumerate(self.cells[locs]):
                part[j] = np.array([locConv[x] if x in locConv else x for x in cell])
                modLocs = np.where(cell != part[j])[0]
                if modLocs.size:
                    if (modLocs.size == self.faceDim):
                        modSourceVertices.append(cell[modLocs])
                        modDestVertices.append(part[j][modLocs])
                        self.log.debug("Updated vertices %s -> %s" % (\
                                        l2s(cell[modLocs].tolist()),\
                                        l2s(part[j][modLocs].tolist())))
                else:
                    self.log.debug("No vertices to update")
            self.cells[locs] = part
        return modSourceVertices, modDestVertices

    def CreateElements(self, sourceVertices, mappedVertices):
        '''
        Parameters
        ----------
        sourceVertices : np.array
            list of face vertices pre-mapping operation. Ordered such that normal points outward from cell centroid
        mappedVertices : np.array
            mapped (or duplicated) vertices, each entry corresponds to sourceVertices
        '''
        def l2s(inlist):
            return ', '.join(map(str, inlist))

        assert(len(sourceVertices) == len(mappedVertices))
        newElems = []
        for source, dest in zip(sourceVertices, mappedVertices):
            newElems.append(np.hstack((dest,source)))
            self.log.debug("Created new cohesive element %s" %(l2s(newElems[-1])))
        self.cohesiveCells = np.array(newElems)
