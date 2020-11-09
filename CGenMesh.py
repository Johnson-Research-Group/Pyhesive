#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:44:24 2020

@author: jacobfaibussowitsch
"""
import logging, atexit, meshio, pymetis
from collections import defaultdict
from scipy import sparse
import numpy as np
from CGenOptionsDB import OptionsDataBase

infoDict = {
    'tetra' : [3, 4],
    'triangle' : [2, 3]
    }

class Mesh:
    log = None
    cDim = 0
    coords = None
    cells = None
    bdCells = []
    faceDim = 0
    nFaces = 0
    c2v = {}
    v2c = defaultdict(list)
    cAdj = []
    ncuts = 0
    membership = None
    registered_exit = False

    def __init__(self, Opts, meshIn):
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True
        slog = logging.getLogger(self.__class__.__name__)
        slog.setLevel(Opts.vlevel)
        slog.propagate = False
        ch = logging.StreamHandler(Opts.stream)
        ch.setLevel(Opts.vlevel)
        ch.propagate = False
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        slog.addHandler(ch)
        self.log = slog
        if not isinstance(meshIn, meshio.Mesh):
            raise TypeError("Opts must be of type" + type(OptionsDataBase))
        self.ctype = meshIn.cells[-1].type
        self.cells = meshIn.cells_dict[self.ctype]
        self.coords = meshIn.points
        self.cDim = len(self.cells[0])
        self.faceDim = infoDict[self.ctype][0]
        self.nFaces = infoDict[self.ctype][1]

    def Finalize(self):
        handlers = self.log.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            self.log.removeHandler(handler)
        self.log = None
        atexit.unregister(self.Finalize)
        self.registered_exit = False

    def l2s(self, inlist):
        return ', '.join(map(str, inlist))

    def Setup(self):
        self.Symmetrize()
        self.cAdj, self.bdCells, _ = self.GenerateAdjacency()

    def Symmetrize(self):
        self.c2v = dict(enumerate(self.cells))
        for cell,vlist in self.c2v.items():
            for vertex in vlist:
                self.v2c[vertex].append(cell)
        self.v2c = {k:np.array(v) for k, v in self.v2c.items()}

    def BuildAdjacencyMatrix(self, Cells=None, format=None):
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
        ne = len(Cells)
        element_ids = np.empty((ne, self.cDim), dtype=np.intp)
        element_ids[:] = np.arange(ne).reshape(-1, 1)

        v2e = sparse.coo_matrix(
            (np.ones((ne*len(element_ids[0]),), dtype=np.intp),
            (Cells.ravel(),
            element_ids.ravel(),)))
        v2e = v2e.tocsr(copy=False)
        c2c = v2e.T @ v2e
        v2v = v2e @ v2e.T
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
            self.log.debug("cell %d adjacent to %s" % (rowindex, self.l2s(adj[rowindex])))
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
                bdfaces.append([face for face in comb if face not in faces])
            else:
                self.log.debug("cell %d NOT marked on boundary" % (rowindex))
        return adj, bdcells, bdfaces


    def Partition(self):
        npart = 2
        self.ncuts, self.membership = pymetis.part_graph(npart, adjacency=self.cAdj)


    def GenerateBoundaryElements(self):
        print(self.bdCells)
        for cut in range(self.ncuts):
            # Extract the partition
            part = np.argwhere(np.array(self.membership) == cut).ravel()
            # Find all the cells which are NOT global boundary cells
            pset = np.setdiff1d(part, self.bdCells)
            # Generate adjacencies for the entire partition, this will include
            # both local and global boundary cells
            adj, bdcells, bdfaces = self.GenerateAdjacency(Cells=self.cells[part])
            # Now do intersection of bdcells and partition without global
            # boundary cells to isolate local boundary cells
            locbdcell = np.intersect1d(pset, part[bdcells])
            bla = 2