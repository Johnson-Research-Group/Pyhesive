#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:17:49 2021

@author: jacobfaibussowitsch
"""
import os, unittest, meshio, pyhesive
from unittest.mock import patch
import numpy as np

dataDir = os.path.join(os.getcwd(), "data")
meshDir = os.path.join(dataDir, "meshes")
smallCubeMeshFile = os.path.join(meshDir, "SmallCube.msh")
mediumCubeMeshFile = os.path.join(meshDir, "MediumCube.msh")
simpleCubeMeshFile = os.path.join(meshDir, "simple.msh")

simpleAdjMatFile = os.path.join(dataDir, "simpleAdj.npz")
smallCubeAdjMatFile = os.path.join(dataDir, "smallCubeAdj.npz")

def makeSingleton():
    points = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ]
    cells = [("tetra", [[0, 1, 2, 3]])]
    return meshio.Mesh(points, cells)

class tMesh(unittest.TestCase):
    def test_createFromMeshIO(self):
        mesh = makeSingleton()
        with pyhesive.Mesh(mesh) as pyh:
            self.assertEqual(pyh.cType, 'tetra')
            self.assertEqual(pyh.cDim, 4)
            self.assertEqual(len(pyh.cells), 1)
            self.assertEqual(len(pyh.coords), 4)
            self.assertEqual(pyh.faceDim, 3)

    def test_createFromFile(self):
        with pyhesive.Mesh.fromFile(smallCubeMeshFile) as pyh:
            self.assertEqual(pyh.cType, 'tetra')
            self.assertEqual(pyh.cDim, 4)
            self.assertEqual(len(pyh.cells), 200)
            self.assertEqual(len(pyh.coords), 90)
            self.assertEqual(pyh.faceDim, 3)

    def test_BuildAdjacencyMatrix(self):
        def testBase(obj, testMat, pyh):
            obj.assertEqual(testMat.shape, pyh.adjMat.shape)
            for i in range(len(pyh.cells)):
                with obj.subTest(i=i):
                    rowTest = testMat.getrowview(i).todense()
                    rowMine = pyh.adjMat.getrowview(i).todense()
                    diffs = len(np.argwhere(rowTest != rowMine))
                    obj.assertEqual(diffs, 0)

        simpleAdjMat = pyhesive._utils.loadMatrix(simpleAdjMatFile, format='lil')
        with pyhesive.Mesh.fromFile(simpleCubeMeshFile) as pyh:
            testBase(self, simpleAdjMat, pyh)
        smallCubeAdjMat = pyhesive._utils.loadMatrix(smallCubeAdjMatFile, format='lil')
        with pyhesive.Mesh.fromFile(smallCubeMeshFile) as pyh:
            testBase(self, smallCubeAdjMat, pyh)

    def test_Partition(self):
        def testBase (obj, file, partList):
            with pyhesive.Mesh.fromFile(file) as pyh:
                for i in partList:
                    with obj.subTest(i=i):
                        if i == 1:
                            with obj.assertRaises(RuntimeError):
                                pyh.PartitionMesh(i)
                        else:
                            pyh.PartitionMesh(i)
                            obj.assertEqual(len(pyh.partitions), i)

        testBase(self, mediumCubeMeshFile, [1,10,20,30,40,50])


if __name__ == '__main__':
    unittest.main()