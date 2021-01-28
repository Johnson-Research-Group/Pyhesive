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
from itertools import combinations
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
    cohesiveCells = None
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
        slog.setLevel(self.vLevel)
        slog.propagate = False
        ch = logging.StreamHandler(self.stream)
        ch.setLevel(self.vLevel)
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
        self.log.debug("Cell dimension %d, type %s" % (self.cDim, self.cType))
        self.log.debug("Face dimension %d" % self.faceDim)
        self.adjMat = self.BuildAdjacencyMatrix(self.cells)
        self.cAdj, self.bdCells, self.bdFaces = self.GenerateCellAdjacency(self.adjMat, self.cells, self.faceDim)
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
                    self.log.debug("Generated %d cohesive elements of type '%s'" % (len(self.cohesiveCells), ctype))
                    break
            if cohesiveType is None:
                raise RuntimeError("Cohesive type not recognized!")
            cells.append((cohesiveType, self.cohesiveCells))
        else:
            self.log.info("Generated 0 cohesive elements")
        meshOut = meshio.Mesh(self.coords, cells)
        meshio.write(self.meshFileOut, meshOut, file_format=self.meshFormatOut)
        self.log.info("Wrote mesh to '%s' with format '%s'" % (self.meshFileOut, self.meshFormatOut))

    def PartitionMesh(self):
        self.ncuts, membership = pymetis.part_graph(self.numPart, adjacency=self.cAdj)
        self.log.info("Number of partitions %d, cuts %d" % (self.numPart, self.ncuts))
        if (self.ncuts == 0):
            raise RuntimeError("No cuts were made by partitioner")
        self.membership = np.array(membership, dtype=np.intp)
        self.partitions = [np.argwhere(self.membership == x).ravel() for x in range(self.numPart)]
        partCountSum = sum([len(x) for x in self.partitions])
        if partCountSum != len(self.cells):
            raise RuntimeError("Partition cell-count summ %d != global number of cells %d" % (partCountSum, len(self.cells)))


    def GenerateElements(self):
        sourceVerts, mappedVerts = self.RemapVertices()
        self.CreateElements(sourceVerts, mappedVerts)

    def BuildAdjacencyMatrix(self, cells, format='lil', v2v=False):
        def matsize(a):
            sum = 0
            if isinstance(a, sparse.csr_matrix) or isinstance(a, sparse.csc_matrix):
                sum = a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
            elif isinstance(a, sparse.lil_matrix):
                sum = a.data.nbytes + a.rows.nbytes
            elif isinstance(a, sparse.coo_matrix):
                sum = a.col.nbytes + a.row.nbytes + a.data.nbytes
            return sum

        ne = len(cells)
        element_ids = np.empty((ne, len(cells[0])), dtype=np.intp)
        element_ids[:] = np.arange(ne).reshape(-1, 1)
        v2c = sparse.coo_matrix(
            (np.ones((ne*len(element_ids[0]),), dtype=np.intp),
            (cells.ravel(),
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

    def GenerateCellAdjacency(self, adjMat, cells, faceDim):
        def list2dict(inlist):
            odict = defaultdict(list)
            for i in range(len(inlist)):
                odict[i].append(inlist[i])
            return odict

        bdcells = []
        adj = {}
        bdfaces = []
        facesPerCell = len([_ for _ in combinations(range(len(cells[0])), faceDim)])
        for rowindex, row in enumerate(adjMat.tolil(copy=False).data):
            sharedfaces = [i for i,k in enumerate(row) if k == faceDim]
            adj[rowindex] = [adjMat.rows[rowindex][j] for j in sharedfaces]
            self.log.debug("cell %d adjacent to %s" % (rowindex, l2s(adj[rowindex])))
            if (len(sharedfaces) != facesPerCell):
                self.log.debug("cell %d marked on boundary" % (rowindex))
                bdcells.append(rowindex)
                faces = [set(np.intersect1d(cells[rowindex],cells[c],assume_unique=True)) for c in adj[rowindex]]
                comb = [set(c) for c in combinations(cells[rowindex], faceDim)]
                bdfaces.append([list(face) for face in comb if face not in faces])
            else:
                self.log.debug("cell %d marked interior" % (rowindex))
        return adj, bdcells, bdfaces

    def GenerateLocalBoundaryFaces(self):
        # Extract the partition
        fl = flattenList(self.bdFaces)
        for i, part in enumerate(self.partitions):
            nc = len(part)
            if not nc:
                self.log.debug("Partition %d contains no cells" % i)
                yield dict()
            else:
                self.log.debug("Partition %d contains (%d) cells %s" % (i, nc, l2s(part)))
                subMat = self.adjMat[np.ix_(part, part)]
                _, _, bdFaces_l = self.GenerateCellAdjacency(subMat, self.cells[part], self.faceDim)
                bdNodes = [[f for f in c if f not in fl] for c in bdFaces_l]
                yield dict((i,j) for i,j in enumerate(bdNodes))

    def DuplicateAndInsertVerticesOLD(self, oldVertexList, globalDict = None):
        try:
            convertableVertices = [x for x in oldVertexList if x not in globalDict]
        except TypeError:
            pass
        nv = len(self.coords)
        newVertexList = range(nv, nv+len(convertableVertices))
        for i,j in zip(convertableVertices, newVertexList):
            self.log.debug("Duped vertex %d -> %d" % (i, j))
        newVertexCoords = [self.coords[v] for v in convertableVertices]
        if len(newVertexCoords):
            self.coords = np.vstack((self.coords, newVertexCoords))
        else:
            self.log.debug("No vertices to duplicate")
        return dict(zip(convertableVertices, newVertexList))

    def DuplicateAndInsertVertices(self, oldVertexList, globalDict):
        try:
            convertableVertices = [x for x in oldVertexList if x not in globalDict]
        except TypeError:
            pass

        lcv = len(convertableVertices)
        if (lcv):
            nv = len(self.coords)
            newVertexList = range(nv, nv+lcv)
            for i,j in zip(convertableVertices, newVertexList):
                self.log.debug("Duped vertex %d -> %d" % (i, j))
            newVertexCoords = [self.coords[v] for v in convertableVertices]
            self.coords = np.vstack((self.coords, newVertexCoords))
            return dict(zip(convertableVertices, newVertexList))
        else:
            self.log.debug("No vertices to duplicate")
            return {}


    def GenerateLocalConversionOLD(self):
        convDict = {}
        for partition in self.GenerateLocalBoundaryFaces():
            # old vertex IDs
            oldVertices = list({s:None for s in flattenList(partition.values())})
            # duplicate the vertices, return the duplicates new IDs
            partConvDict = self.DuplicateAndInsertVertices(oldVertices, convDict)
            convDict.update(partConvDict)
            yield partConvDict

    def GenerateLocalConversion(self):
        convDict = {}
        for bdFaces in self.GenerateLocalBoundaryFaces():
            # old vertex IDs
            oldVertices = flattenList(flattenList(bdFaces.values()))
            # duplicate the vertices, return the duplicates new IDs
            convDict = self.DuplicateAndInsertVertices(oldVertices, convDict)
            partConvDict = {}
            for flist in bdFaces.values():
                for f in flist:
                    # only want elements who have had an entire face converted (i.e. actually should recieve a cohesive element)
                    if sum([v in convDict for v in f]) == self.faceDim:
                        partConvDict.update({v: convDict[v] for v in f})
            yield (bdFaces, partConvDict)

    def RemapVerticesOLD(self):
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
        allCombs = [list(l) for l in combinations(range(self.cDim), self.faceDim)]
        for locs, locConv in zip(self.partitions, self.GenerateLocalConversion()):
            part = np.empty(np.shape(self.cells[locs]), dtype=self.cells.dtype)
            for j, cell in enumerate(self.cells[locs]):
                part[j] = np.array([locConv[x] if x in locConv else x for x in cell])
                modLocs = np.where(cell != part[j])[0]
                if (modLocs.size > self.faceDim):
                    # TODO handle the rest of these cases!
                    if len(locs) == 1:
                        # single cell partition, we take all faces
                        modSourceVertices.extend([cell[x] for x in allCombs])
                        modDestVertices.extend([part[j][x] for x in allCombs])
                        for x in allCombs:
                            self.log.debug("Updated vertices %s -> %s" % (l2s(cell[x].tolist()), l2s(part[j][x].tolist())))
                    else:
                    # all vertices in the cell were replaced, we must now figure out
                    # what (if any) face should stay the same
                        bla = 2
                elif (modLocs.size == self.faceDim):
                    modSourceVertices.append(cell[modLocs])
                    modDestVertices.append(part[j][modLocs])
                    self.log.debug("Updated vertices %s -> %s" % (l2s(cell[modLocs].tolist()), l2s(part[j][modLocs].tolist())))
                else:
                    self.log.debug("No vertices to update")
            self.cells[locs] = part
        return modSourceVertices, modDestVertices

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
        allCombs = [list(l) for l in combinations(range(self.cDim), self.faceDim)]
        for part, (bdFaces, locConv) in zip(self.partitions, self.GenerateLocalConversion()):
            if len(part):
                partall = np.array([[locConv[x] if x in locConv else x for x in cell] for cell in self.cells[part]])
                modLocs = partall != self.cells[part]
                for l, m in zip(modLocs.sum(axis=1), modLocs):
                    if l > self.faceDim:
                        if len(part) == 1:
                            # single cell partition, we take all faces
                            modSourceVertices.extend([[y[x] for x in allCombs] for y in self.cells[part]])
                            modDestVertices.extend([[y[x] for x in allCombs] for y in partall])

                    elif l == self.faceDim:
                        modSourceVertices.append(cell[modLocs])
                        modDestVertices.append(part[j][modLocs])
                if modLocs.sum():
                    self.log.debug("Updated vertices %s -> %s" % (l2s(self.cells[part][modLocs]), l2s(partall[modLocs])))
                    self.cells[part] = partall
                else:
                    self.log.debug("No vertices to update")
            else:
                self.log.debug("No vertices to update")
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