#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:17:49 2021

@author: jacobfaibussowitsch
"""
import sys
import os
import meshio
import pickle
import scipy
import copy
import unittest
import pytest
import re
import numpy as np

try:
    import pyhesive
except ModuleNotFoundError:
    # silly pythonpath finagling in case someone runs this from cloning the
    # git repo rather than installing it as a package
    sys.path.append(os.path.realpath(os.path.join(os.getcwd(), "..", "..")))
    import pyhesive

curDir = os.path.basename(os.getcwd())
if curDir == "test":
    testRootDir = os.getcwd()
elif curDir == "pyhesive":
    parentDir = os.path.basename(os.path.dirname(os.getcwd()))
    if parentDir == "pyhesive":
        # we are in pyhesive/pyhesive, i.e. the package dir
        testRootDir = os.path.join(os.getcwd(), "test")
    elif "pyhesive" in os.listdir():
        # we are in root pyhesive dir
        testRootDir = os.path.join(os.getcwd(), "pyhesive", "test")
    else:
        raise RuntimeError("Cannot determine location")
else:
    raise RuntimeError("Cannot determine location")

# clean the path, eliminating dots, symlinks, and ~
testRootDir = os.path.realpath(os.path.expanduser(testRootDir))
pyhesiveRootDir = os.path.dirname(testRootDir)
dataDir = os.path.join(testRootDir, "data")
meshDir = os.path.join(dataDir, "meshes")
binDir = os.path.join(dataDir, "bin")

simpleCubeMeshFile = os.path.join(meshDir, "simple.msh")
smallCubeMeshFile = os.path.join(meshDir, "SmallCube.msh")
mediumCubeMeshFile = os.path.join(meshDir, "MediumCube.msh")
meshFileList = [simpleCubeMeshFile, smallCubeMeshFile, mediumCubeMeshFile]

simpleAdjMatFile = os.path.join(binDir, "simpleAdj.npz")
smallCubeAdjMatFile = os.path.join(binDir, "smallCubeAdj.npz")
matFileList = [simpleAdjMatFile, smallCubeAdjMatFile]

simpleClosureDictFile = os.path.join(binDir, "simpleClosure.pkl")
smallCubeClosureDictFile = os.path.join(binDir, "smallCubeClosure.pkl")
mediumCubeClosureDictFile = os.path.join(binDir, "mediumCubeClosure.pkl")
closureDictFileList = [simpleClosureDictFile, smallCubeClosureDictFile, mediumCubeClosureDictFile]

simpleBoundaryDictFile = os.path.join(binDir, "simpleBoundary.pkl")
smallCubeBoundaryDictFile = os.path.join(binDir, "smallCubeBoundary.pkl")
mediumCubeBoundaryDictFile = os.path.join(binDir, "mediumCubeBoundary.pkl")
boundaryDictFileList = [
    simpleBoundaryDictFile,
    smallCubeBoundaryDictFile,
    mediumCubeBoundaryDictFile,
]

simpleGlobConvDictFile = os.path.join(binDir, "simpleGlobConv.pkl")
smallCubeGlobConvDictFile = os.path.join(binDir, "smallCubeGlobConv.pkl")
mediumCubeGlobConvDictFile = os.path.join(binDir, "mediumCubeGlobConv.pkl")
globConvDictFileList = [
    simpleGlobConvDictFile,
    smallCubeGlobConvDictFile,
    mediumCubeGlobConvDictFile,
]

simpleVertexMapFile = os.path.join(binDir, "simpleVertexMap.pkl")
smallCubeVertexMapFile = os.path.join(binDir, "smallVertexMap.pkl")
mediumCubeVertexMapFile = os.path.join(binDir, "mediumVertexMap.pkl")
vertexMapFileList = [simpleVertexMapFile, smallCubeVertexMapFile, mediumCubeVertexMapFile]

simpleOutputMeshFile = os.path.join(binDir, "simpleOutputMesh.pkl")
smallCubeOutputMeshFile = os.path.join(binDir, "smallOutputMesh.pkl")
mediumCubeOutputMeshFile = os.path.join(binDir, "mediumOutputMesh.pkl")
outputMeshFileList = [simpleOutputMeshFile, smallCubeOutputMeshFile, mediumCubeOutputMeshFile]

pickleProtocol = 4


def storeMatrix(filename, mat):
    scipy.sparse.save_npz(filename, mat, compressed=True)
    print("Wrote mat %s to file %s" % (mat.__class__, filename))


def loadMatrix(filename, format=None):
    mat = scipy.sparse.load_npz(filename)
    return mat.asformat(format, copy=False)


def storeObj(filename, obj):
    with open(filename, "wb") as f:
        # protocol 4 since python3.4
        pickle.dump(obj, f, protocol=pickleProtocol)
        print("Wrote object %s to file %s" % (obj.__class__, filename))


def loadObj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def makeSingleton():
    points = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    cells = [("tetra", [[0, 1, 2, 3]])]
    return meshio.Mesh(points, cells)


@pytest.fixture(scope="class")
def setReplaceFiles(request, pytestconfig):
    request.cls.replaceFiles = pytestconfig.getoption("pyhesive_replace")


@pytest.mark.usefixtures("setReplaceFiles")
class testMesh(unittest.TestCase):
    replaceFiles = False

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assertNumpyArrayEqual)

    def assertNumpyArrayEqual(self, first, second, msg=None):
        assert type(first) == np.ndarray, msg
        assert type(first) == type(second), msg
        try:
            np.testing.assert_array_equal(first, second)
        except AssertionError:
            additionalMsg = "Locations: %s" % (np.argwhere(first != second))
            if msg is not None:
                msg = "\n".join([msg, additionalMsg])
            else:
                msg = additionalMsg
            pytest.fail(msg)

    def test_createFromMeshIO(self):
        mesh = makeSingleton()
        with pyhesive.Mesh(mesh) as pyh:
            assert pyh.cType == "tetra"
            assert pyh.cDim == 4
            assert len(pyh.cells) == 1
            assert len(pyh.coords) == 4
            assert pyh.faceDim == 3

    def test_createFromFile(self):
        with pyhesive.Mesh.fromFile(smallCubeMeshFile) as pyh:
            assert pyh.cType == "tetra"
            assert pyh.cDim == 4
            assert len(pyh.cells) == 200
            assert len(pyh.coords) == 90
            assert pyh.faceDim == 3

    def test_Partition(self):
        def testLogWarn(self, pyh, itr):
            with self.assertLogs(pyh.log, level="WARNING") as cm:
                pyh.PartitionMesh(itr)
                assert re.search(
                    r"(Number of partitions )(\d+)( > num cells)\s(\d+)(, using num cells instead)",
                    *cm.output
                )

        def testSuccess(self, pyh, itr):
            pyh.PartitionMesh(itr)
            if itr == -1:
                assert len(pyh.partitions) == len(pyh.cells)
            elif itr in (0, 1):
                assert pyh.partitions == tuple()
            else:
                assert len(pyh.partitions) == itr

        def testBase(self, file, partList, func):
            with pyhesive.Mesh.fromFile(file) as pyh:
                for numPart in partList:
                    with self.subTest(numPart=numPart):
                        func(self, pyh, numPart)

        partListSuccess = [
            [-1, 0, 1, 10, 20],
            [-1, 0, 1, 10, 20, 30, 40, 50],
            [-1, 0, 1, 10, 20, 30, 40, 50, 1000],
        ]
        partListLogWarn = [[25], [201], [13000]]
        for meshFile, partSuccess, partWarn in zip(meshFileList, partListSuccess, partListLogWarn):
            with self.subTest(meshFile=meshFile, retType="sucess"):
                testBase(self, meshFile, partSuccess, testSuccess)
            with self.subTest(meshFile=meshFile, retType="log warn"):
                testBase(self, meshFile, partWarn, testLogWarn)

    def test_BuildAdjacencyMatrix(self):
        for meshFile, matFile in zip(meshFileList, matFileList):
            with self.subTest(meshFile=meshFile):
                with pyhesive.Mesh.fromFile(meshFile) as pyh:
                    if self.replaceFiles:
                        storeMatrix(matFile, pyh.adjMat.tocsr())
                        continue
                    testMat = loadMatrix(matFile, format="lil")
                    assert testMat.shape == pyh.adjMat.shape
                    for i in range(len(pyh.cells)):
                        with self.subTest(row=i):
                            rowTest = testMat.getrowview(i).toarray()
                            rowMine = pyh.adjMat.getrowview(i).toarray()
                            self.assertNumpyArrayEqual(rowMine, rowTest)

    def test_Closure(self):
        subDictAdj = "cellAdjacency"
        subDictBdC = "boundaryCells"
        subDictBdF = "boundaryFaces"
        for meshFile, closureDictFile in zip(meshFileList, closureDictFileList):
            with self.subTest(meshFile=meshFile):
                with pyhesive.Mesh.fromFile(meshFile) as pyh:
                    if self.replaceFiles:
                        combinedDict = {
                            subDictAdj: pyh.cAdj,
                            subDictBdC: pyh.bdCells,
                            subDictBdF: pyh.bdFaces,
                        }
                        storeObj(closureDictFile, combinedDict)
                        continue
                    testDict = loadObj(closureDictFile)
                    with self.subTest(subDict=subDictAdj):
                        assert pyh.cAdj == testDict[subDictAdj]
                    with self.subTest(subDict=subDictBdC):
                        assert pyh.bdCells == testDict[subDictBdC]
                    with self.subTest(subDict=subDictBdF):
                        assert pyh.bdFaces == testDict[subDictBdF]

    def commonPartitionSetup(
        self, meshFileList, testFileList, partitionList, replaceFunc, testFunc
    ):
        for meshFile, testFile, partList in zip(meshFileList, testFileList, partitionList):
            with self.subTest(meshFile=meshFile, partList=partList):
                with pyhesive.Mesh.fromFile(meshFile) as pyh:
                    if self.replaceFiles:
                        replaceFunc(pyh, testFile, partList)
                        continue
                    testDict = loadObj(testFile)
                    for part, testPart in zip(partList, testDict):
                        with self.subTest(part=part):
                            assert part == testPart
                            pyh.PartitionMesh(part)
                            testFunc(self, pyh, testDict, part)

    def test_LocalBoundaryFaces(self):
        def replaceFunc(pyh, testFile, partList):
            combinedDict = dict()
            for part in partList:
                pyh.PartitionMesh(part)
                combinedDict[part] = tuple(bdface for bdface in pyh.GenerateLocalBoundaryFaces())
            storeObj(testFile, combinedDict)

        def testFunc(self, pyh, testDict, part):
            for bdFaceDict, testSubPart in zip(pyh.GenerateLocalBoundaryFaces(), testDict[part]):
                for cell, testCell in zip(bdFaceDict, testSubPart):
                    with self.subTest(cell=cell):
                        assert cell == testCell
                        assert bdFaceDict[cell] == testSubPart[cell]

        partitionListSuccess = [[10, 20, -1], [10, 30, 50, -1], [20, 500, 1000, -1]]
        self.commonPartitionSetup(
            meshFileList, boundaryDictFileList, partitionListSuccess, replaceFunc, testFunc
        )

    def test_GlobalConversion(self):
        def replaceFunc(pyh, testFile, partList):
            combinedDict = dict()
            for part in partList:
                pyh.PartitionMesh(part)
                dummyGlobDict = dict()
                for _, dummyGlobDict in pyh.GenerateGlobalConversion(dummyGlobDict):
                    pass
                combinedDict[part] = (dummyGlobDict, pyh.dupCoords)
            storeObj(testFile, combinedDict)

        def testFunc(self, pyh, testDict, part):
            testConv, testArr = testDict[part]
            globConv = dict()
            # iterate through generator to pupolate the dict
            for _, globConv in pyh.GenerateGlobalConversion(globConv):
                pass
            assert globConv == testConv
            self.assertNumpyArrayEqual(pyh.dupCoords, testArr)

        partitionList = [[7, 13, -1], [11, 23, 48, -1], [87, 366, 1234, -1]]
        self.commonPartitionSetup(
            meshFileList, globConvDictFileList, partitionList, replaceFunc, testFunc
        )

    def test_RemapVertices(self):
        def replaceFunc(pyh, testFile, partList):
            combinedDict = dict()
            for part in partList:
                # remap vertices is a destructive op, need a fresh new copy every time
                copyPyh = copy.deepcopy(pyh)
                copyPyh.PartitionMesh(part)
                src, dest = copyPyh.RemapVertices()
                combinedDict[part] = (src, dest, copyPyh.cells)
            storeObj(testFile, combinedDict)

        def testFunc(self, pyh, testDict, part):
            testSrc, testDest, testArr = testDict[part]
            # remap vertices is a destructive op, need a fresh new copy every time
            copyPyh = copy.deepcopy(pyh)
            src, dest = copyPyh.RemapVertices()
            assert src == testSrc
            assert dest == testDest
            self.assertNumpyArrayEqual(copyPyh.cells, testArr)

        partitionList = [[3, 13, 18, -1], [35, 38, 43, -1], [123, 745, 999, -1]]
        self.commonPartitionSetup(
            meshFileList, vertexMapFileList, partitionList, replaceFunc, testFunc
        )

    def compareClasses(self, first, second, msg=None):
        assert type(first) == type(second), msg
        try:
            type(vars(first)) is dict
        except:
            assert first == second, msg
        else:
            for i, j in zip(vars(first).keys(), vars(second).keys()):
                assert i == j, msg
                try:
                    type(vars(vars(first)[i])) is dict
                except:
                    # handles list of classes too
                    if type(vars(first)[i]) is list:
                        for f, s in zip(vars(first)[i], vars(second)[i]):
                            self.compareClasses(f, s, msg=msg)
                    elif type(vars(first)[i]) is np.ndarray:
                        self.assertNumpyArrayEqual(vars(first)[i], vars(second)[i], msg)
                    else:
                        assert vars(first)[i] == vars(second)[i], msg
                else:
                    self.compareClasses(vars(first)[i], vars(second)[i], msg=msg)

    def test_FullStack(self):
        def replaceFunc(pyh, testFile, partList):
            combinedDict = dict()
            for part in partList:
                copyPyh = copy.deepcopy(pyh)
                copyPyh.PartitionMesh(part)
                copyPyh.GenerateElements()
                combinedDict[part] = copyPyh.WriteMesh(None, returnMesh=True)
            storeObj(testFile, combinedDict)

        def testFunc(self, pyh, testDict, part):
            testMesh = testDict[part]
            # Full stack is obviously destructive
            copyPyh = copy.deepcopy(pyh)
            copyPyh.GenerateElements()
            mesh = copyPyh.WriteMesh(None, returnMesh=True)
            self.compareClasses(mesh, testMesh)

        partitionList = [[9, 16, 23, -1], [10, 16, 34, -1], [234, 555, 812, -1]]
        self.commonPartitionSetup(
            meshFileList, outputMeshFileList, partitionList, replaceFunc, testFunc
        )


if __name__ == "__main__":
    pytest.main()
