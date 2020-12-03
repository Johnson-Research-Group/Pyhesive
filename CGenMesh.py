#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
import logging, atexit, meshio, pymetis
from CGen import CGen
from CGenOptionsDB import OptionsDataBase
from collections import defaultdict
from scipy import sparse
import numpy as np

infoDict = {
    'tetra' : [3, 4],
    'triangle' : [2, 3]
    }

class Mesh(CGen):
    OptCtx = None
    cDim = 0
    cType = None
    coords = None
    cells = None
    bdCells = []
    bdFaces = []
    faceDim = 0
    nFaces = 0
    c2v = {}
    v2c = defaultdict(list)
    numPart = 0
    cAdj = []
    ncuts = 0
    membership = None
    registered_exit = False

    def __init__(self, Opts, meshIn, parent=None):
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True
        if not isinstance(Opts, OptionsDataBase):
            raise TypeError("Opts must be of type" + type(OptionsDataBase))
        if not isinstance(meshIn, meshio.Mesh):
            raise TypeError("Mesh must be of type" + type(meshio.Mesh))
        self.OptCtx = Opts
        slog = logging.getLogger(self.__class__.__name__)
        slog.setLevel(self.OptCtx.vLevel)
        slog.propagate = False
        ch = logging.StreamHandler(self.OptCtx.stream)
        ch.setLevel(self.OptCtx.vLevel)
        ch.propagate = False
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        slog.addHandler(ch)
        self.log = slog
        self.cType = meshIn.cells[-1].type
        self.cells = meshIn.cells_dict[self.cType]
        self.coords = meshIn.points
        self.cDim = len(self.cells[0])
        self.faceDim = infoDict[self.cType][0]
        self.nFaces = infoDict[self.cType][1]


    def Finalize(self):
        handlers = self.log.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            self.log.removeHandler(handler)
        self.log = None
        atexit.unregister(self.Finalize)
        self.registered_exit = False


    def PrepareOutputMesh(self):
        return self.coords, [(self.cType, self.cells)]


    def Setup(self):
        self.Symmetrize()
        self.cAdj, self.bdCells, self.bdFaces = self.GenerateAdjacency()


    def Symmetrize(self):
        self.c2v = dict(enumerate(self.cells))
        for cell,vlist in self.c2v.items():
            for vertex in vlist:
                self.v2c[vertex].append(cell)
        self.v2c = {k:np.array(v) for k, v in self.v2c.items()}


    def BuildAdjacencyMatrix(self, Cells=None, cDim=None, format=None):
        def matsize(a):
            sum = 0
            if isinstance(a, sparse.csr_matrix) or isinstance(a, sparse.csc_matrix):
                sum = a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
            elif isinstance(a, sparse.lil_matrix):
                sum = a.data.nbytes + a.rows.nbytes
            elif isinstance(a, sparse.coo_matrix):
                sum = a.col.nbytes + a.row.nbytes + a.data.nbytes
            return sum

        if Cells is None:
            Cells = self.cells
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
        v2v = v2c @ v2c.T
        self.log.info("c2c mat size %d bytes" % matsize(c2c))
        c2c = c2c.asformat(format, copy=False)
        self.log.info("c2c mat size after compression %d bytes" % matsize(c2c))
        self.log.info("v2v mat size %d bytes" % matsize(v2v))
        v2v = v2v.asformat(format, copy=False)
        self.log.info("v2v mat size after compression %d bytes" % matsize(v2v))
        return c2c, v2v


    def GenerateAdjacency(self, Cells = None, faceDim = None, nfaces = None):
        from itertools import combinations
        def symmetric(a, atol=1e-10):
            return (abs(a-a.T)>atol).nnz == 0

        def list2dict(inlist):
            odict = defaultdict(list)
            for i in range(len(inlist)):
                odict[i].append(inlist[i])
            return odict

        def l2s(inlist):
            return ', '.join(map(str, inlist))

        if Cells is None:
            Cells = self.cells
        if faceDim is None:
            faceDim = self.faceDim
        if nfaces is None:
            nfaces = self.nFaces

        c2c, _ = self.BuildAdjacencyMatrix(Cells=Cells, format='lil')
        adj = [[] for _ in range(len(Cells))]
        bdcells = []
        bdfaces = []
        for rowindex, row in enumerate(c2c.data):
            sharedfaces = [i for i,k in enumerate(row) if k == faceDim]
            rowV = c2c.rows[rowindex]
            adj[rowindex] = [rowV[j] for j in sharedfaces]
            self.log.debug("cell %d adjacent to %s" % (rowindex, l2s(adj[rowindex])))
            if (len(sharedfaces) != nfaces):
                bdcells.append(rowindex)
                self.log.debug("cell %d marked on boundary" % (rowindex))
                faces = []
                combl = list(combinations(Cells[rowindex], faceDim))
                comb = [set(c) for c in combl]
                for cell in adj[rowindex]:
                    faces.append(set(np.intersect1d(Cells[rowindex],
                                                    Cells[cell],
                                                    assume_unique=True)))
                bdfaces.append([list(face) for face in comb if face not in faces])
            else:
                self.log.debug("cell %d marked interior" % (rowindex))
        return adj, bdcells, bdfaces


    def Partition(self):
        self.ncuts, membership = pymetis.part_graph(self.OptCtx.numPart, adjacency=self.cAdj)
        if (self.ncuts == 0):
            raise RuntimeError("No cuts were made by partitioner")
        self.membership = np.array(membership, dtype=np.intp)


    def ExtractLocalBoundaryElements(self):
        bdDict = {}
        for part in range(self.OptCtx.numPart):
            # Extract the partition
            partition_g = np.argwhere(self.membership == part).ravel()
            # Generate boundaries for the entire partition, this will include
            # both local and global boundary cells
            _, bdCells_l, bdFaces_l = self.GenerateAdjacency(Cells=self.cells[partition_g])
            # should be as many bdface entries as bd cells
            assert(len(bdCells_l) == len(bdFaces_l))
            # Find difference between global boundary cells and local+global
            # boundary cells to isolate local boundary cells
            locBdCells_g = np.setdiff1d(partition_g[bdCells_l], self.bdCells,
                                        assume_unique=True)
            # Extract list of boundary faces and add them to dict
            bdNodes = []
            for c_g in locBdCells_g:
                c_l = np.argwhere(partition_g[bdCells_l] == c_g).ravel()[0]
                bdNodes.append([item for sublist in bdFaces_l[c_l] for item in sublist])
            enum = enumerate(bdNodes)
            bdDict[part] = dict((i,j) for i,j in enum)
        bla = 2