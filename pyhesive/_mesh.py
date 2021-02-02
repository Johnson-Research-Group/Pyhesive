#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
from ._utils import *

import logging, atexit, meshio, pymetis, sys
from collections import defaultdict, Counter, deque
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

class Mesh:
    cohesiveCells = None
    registered_exit = False

    def __init__(self, mesh, verbosity=50, stream=sys.stdout):
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True
        slog = logging.getLogger(self.__class__.__name__)
        slog.setLevel(verbosity)
        slog.propagate = False
        ch = logging.StreamHandler(stream)
        ch.setLevel(verbosity)
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
        self.log.info("Number of cells %d, vertices %d" % (len(self.cells), len(self.coords)))
        self.log.info("Cell dimension %d, type %s, face dimension %d" % (self.cDim, self.cType, self.faceDim))
        self.adjMat = self.BuildAdjacencyMatrix(self.cells)
        self.cAdj, self.bdCells, self.bdFaces = self.ComputeClosure(self.adjMat, self.cells, self.faceDim)

    @classmethod
    def fromFile(cls, filename, formatIn=None, verbosity=50, stream=sys.stdout):
        mesh = meshio.read(filename, formatIn)
        return cls(mesh, verbosity=verbosity, stream=stream)

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

    def WriteMesh(self, meshFileOut, meshFormatOut=None, prune=False):
        cells = [(self.cType, self.cells)]
        if self.cohesiveCells is not None:
            npts = len(self.cohesiveCells[0])
            cohesiveType = None
            for ctype, info in infoDict.items():
                if info[1] == npts:
                    cohesiveType = ctype
                    self.log.info("Generated %d cohesive elements of type '%s'" % (len(self.cohesiveCells), ctype))
                    break
            if cohesiveType is None:
                raise RuntimeError("Cohesive type not recognized!")
            cells.append((cohesiveType, self.cohesiveCells))
        else:
            self.log.info("Generated 0 cohesive elements")
        meshOut = meshio.Mesh(self.coords, cells)
        if prune:
            meshOut.remove_orphaned_nodes()
        meshio.write(meshFileOut, meshOut, file_format=meshFormatOut)
        self.log.info("Wrote mesh to '%s' with format '%s'" % (meshFileOut, meshFormatOut))

    def PartitionMesh(self, numPart):
        if numPart < len(self.cells):
            self.ncuts, membership = pymetis.part_graph(numPart, adjacency=self.cAdj)
            if (self.ncuts == 0):
                raise RuntimeError("No cuts were made by partitioner")
            self.membership = np.array(membership, dtype=np.intp)
            self.partitions = [np.argwhere(self.membership == x).ravel() for x in range(numPart)]
        else:
            self.ncuts = -1
            self.membership = np.array([x for x in range(len(self.cells))])
            self.partitions = [np.array([x]) for x in self.membership]
        nValid = sum(1 for i in self.partitions if len(i))
        self.log.info("Number of partitions requested %d, actual %d, average cells/partition %d" % (numPart, nValid, len(self.cells)/nValid))
        partVMap = [np.unique(self.cells[part].ravel()) for part in self.partitions]
        for vmap in partVMap:
            try:
                ctr += Counter(vmap)
            except UnboundLocalError:
                ctr = Counter(vmap)
        self.partVMap = ctr
        partCountSum = sum([len(x) for x in self.partitions])
        if partCountSum != len(self.cells):
            raise RuntimeError("Partition cell-count sum %d != global number of cells %d" % (partCountSum, len(self.cells)))

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

    def ComputeClosure(self, adjMat, cells, faceDim, fullClosure=True):
        adj = {}
        bdcells = []
        bdfaces = []
        facesPerCell = len([_ for _ in combinations(range(len(cells[0])), faceDim)])
        for rowindex, row in enumerate(adjMat.data):
            neighbors = (i for i,k in enumerate(row) if k == faceDim)
            adj[rowindex] = [adjMat.rows[rowindex][j] for j in neighbors]
            self.log.debug("cell %d adjacent to %s" % (rowindex, adj[rowindex]))
            if len(adj[rowindex]) != facesPerCell:
                # the cell does not have a neighbor for every face!
                self.log.debug("cell %d marked on boundary" % (rowindex))
                if fullClosure:
                    bdcells.append(rowindex)
                # for all of my neighbors, what faces do we have in common?
                intFaces = [set(cells[rowindex]).intersection(cells[c]) for c in adj[rowindex]]
                # all possible faces of mine
                comb = list(map(set, combinations(cells[rowindex], faceDim)))
                # THIS MAY VERY WELL BE COMPLETELY BROKEN
                # I currently just take a set, __completely throwing out any ordering information
                bdf = [list(face) for face in comb if face not in intFaces]
                bdfaces.append(bdf)
                [self.log.debug("face %s marked on boundary" % f) for f in bdf]
            else:
                self.log.debug("cell %d marked interior" % (rowindex))
        self.log.debug("%d interior cells, %d boundary cells, %d boundary faces" % (len(cells)-len(bdcells), len(bdcells), len(bdfaces)))
        if fullClosure:
            return adj, bdcells, bdfaces
        else:
            return adj, bdfaces

    def GenerateLocalBoundaryFaces(self):
        fl = list(map(set, flattenList(self.bdFaces)))
        # Extract the partition
        for i, part in enumerate(self.partitions):
            if not len(part):
                self.log.debug("Partition %d contains no cells" % i)
                yield dict()
            else:
                self.log.debug("Partition %d contains (%d) cells %s" % (i, len(part), part))
                rows = self.adjMat[part].tocsc(copy=False)
                subMat = rows[:, part].tolil(copy=False)
                _, locBdFaces = self.ComputeClosure(subMat, self.cells[part], self.faceDim, fullClosure=False)
                bdNodes = [[f for f in c if set(f) not in fl] for c in locBdFaces]
                yield dict((i,j) for i,j in enumerate(bdNodes))

    def DuplicateVertices(self, oldVertexList, globalDict):
        try:
            convertableVertices = [x for x in oldVertexList if x not in globalDict]
        except TypeError:
            pass
        if len(convertableVertices):
            try:
                ndv = len(self.coords)+len(self.dupCoords)
            except AttributeError:
                # self.dupCoords doesn't exist yet
                ndv = len(self.coords)
            vCounts = [self.partVMap[x]-1 for x in convertableVertices]
            newVCoords = []
            tdict = dict()
            for v, cnt in zip(convertableVertices, vCounts):
                ndvLast = ndv
                ndv += cnt
                # At least one other partition must own the boundary vertex
                # otherwise the routine generating local interior bounndaries is buggy
                assert(ndv > ndvLast)
                tdict[v] = (deque(range(ndvLast, ndv)), [v])
                newVCoords.extend([self.coords[v] for _ in range(cnt)])
                self.log.debug("Duped vertex %d -> %s" % (v, tdict[v][0]))
            try:
                 self.dupCoords = np.vstack((self.dupCoords, newVCoords))
            except AttributeError:
                # self.dupCoords doesn't exist yet
                self.dupCoords = np.array(newVCoords)
            return tdict
        else:
            self.log.debug("No vertices to duplicate")
            return dict()

    def GenerateGlobalConversion(self, globConvDict):
        for bdFaces in self.GenerateLocalBoundaryFaces():
            # old vertex IDs
            oldVertices = set(flattenList(flattenList(bdFaces.values())))
            # duplicate the vertices, return the duplicates new IDs
            locConvDict = self.DuplicateVertices(oldVertices, globConvDict)
            yield bdFaces, globConvDict
            globConvDict.update(locConvDict)

    def RemapVertices(self):
        modSourceVertices = []
        modDestVertices = []
        globConv = dict()
        for part, (bdFaces, globConv) in zip(self.partitions, self.GenerateGlobalConversion(globConv)):
            partConv = np.array([[globConv[x][0][0] if x in globConv else x for x in cell] for cell in self.cells[part]])
            try:
                self.cells[part] = partConv
            except ValueError:
                self.log.debug("No vertices to update")
                continue
            bdf = flattenList(bdFaces.values())
            alts = [[globConv[v][0][0] if v in globConv else v for v in f] for f in bdf]
            for a, b in zip(alts, bdf):
                diffs = sum(1 for i, j in zip(a, b) if i != j)
                if diffs == self.faceDim:
                    modSourceVertices.append([globConv[x][1][0] for x in b])
                    modDestVertices.append(a)
                    self.log.debug("Updated face %s -> %s" % (b, a))
            fs = set(flattenList(bdf))
            vConv = (x for x in fs if x in globConv)
            for v in vConv:
                self.log.debug("Updated previous mapping %d: %d -> %d" % (v,globConv[v][1][0],globConv[v][0][0]))
                globConv[v][1][0] = globConv[v][0].popleft()
        return modSourceVertices, modDestVertices

    def CreateElements(self, sourceVertices, mappedVertices):
        assert(len(sourceVertices) == len(mappedVertices))
        newElems = []
        for source, dest in zip(sourceVertices, mappedVertices):
            newElems.append(np.hstack((dest,source)))
            self.log.debug("Created new cohesive element %s" % (newElems[-1]))
        self.coords = np.vstack((self.coords, self.dupCoords))
        self.cohesiveCells = np.array(newElems)