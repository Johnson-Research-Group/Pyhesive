#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:17:49 2021

@author: jacobfaibussowitsch
"""
import sys, os, unittest, nose2, meshio, pickle, scipy, argparse
import numpy as np

try:
    import pyhesive
except ModuleNotFoundError:
    # silly pythonpath finagling in case someone runs this from cloning the git repo
    # rather than installing it as a package
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

simpleCubeMeshFile = os.path.join(meshDir, "simple.msh")
smallCubeMeshFile = os.path.join(meshDir, "SmallCube.msh")
mediumCubeMeshFile = os.path.join(meshDir, "MediumCube.msh")
meshFileList = [simpleCubeMeshFile, smallCubeMeshFile, mediumCubeMeshFile]

simpleAdjMatFile = os.path.join(dataDir, "simpleAdj.npz")
smallCubeAdjMatFile = os.path.join(dataDir, "smallCubeAdj.npz")
matFileList = [simpleAdjMatFile, smallCubeAdjMatFile]

simpleClosureDictFile = os.path.join(dataDir, "simpleClosure.pkl")
smallCubeClosureDictFile = os.path.join(dataDir, "smallCubeClosure.pkl")
mediumCubeClosureDictFile = os.path.join(dataDir, "mediumCubeClosure.pkl")
closureDictFileList = [simpleClosureDictFile, smallCubeClosureDictFile, mediumCubeClosureDictFile]

simpleBoundaryDictFile = os.path.join(dataDir, "simpleBoundary.pkl")
smallCubeBoundaryDictFile = os.path.join(dataDir, "smallCubeBoundary.pkl")
mediumCubeBoundaryDictFile = os.path.join(dataDir, "mediumCubeBoundary.pkl")
boundaryDictFileList = [simpleBoundaryDictFile, smallCubeBoundaryDictFile, mediumCubeBoundaryDictFile]

simpleGlobConvDictFile = os.path.join(dataDir, "simpleGlobConv.pkl")
smallCubeGlobConvDictFile = os.path.join(dataDir, "smallCubeGlobConv.pkl")
mediumCubeGlobConvDictFile = os.path.join(dataDir, "mediumCubeGlobConv.pkl")
globConvDictFileList = [simpleGlobConvDictFile, smallCubeGlobConvDictFile, mediumCubeGlobConvDictFile]

pickleProtocol = 4

def storeMatrix(filename, mat):
    scipy.sparse.save_npz(filename, mat, compressed=True)
    print("Wrote mat %s to file %s" % (mat.__class__, filename))

def loadMatrix(filename, format=None):
    mat = scipy.sparse.load_npz(filename)
    return mat.asformat(format, copy=False)

def storeObj(filename, obj):
    with open(filename, 'wb') as f:
        # protocol 4 since python3.4
        pickle.dump(obj, f, protocol=pickleProtocol)
        print("Wrote object %s to file %s" % (obj.__class__, filename))

def loadObj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def makeSingleton():
    points = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ]
    cells = [("tetra", [[0, 1, 2, 3]])]
    return meshio.Mesh(points, cells)

class testMesh(unittest.TestCase):
    @classmethod
    def setUp(cls, argList=sys.argv):
        parser = argparse.ArgumentParser(description="Customize test harness", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-RRR", "--REPLACE", help="replace existing diff files", dest="replaceFiles", action='store_true')
        parser.set_defaults(replaceFiles=False)
        parser.parse_known_args(args=argList, namespace=cls)

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

    def test_Partition(self):
        def testLogWarn(tobj, pyhobj, itr):
            with tobj.assertLogs(pyhobj.log, level='WARNING') as cm:
                pyhobj.PartitionMesh(itr)
                tobj.assertRegex(*cm.output, "(Number of partitions )(\d+)( > num cells)\s(\d+)(, using num cells instead)")

        def testSuccess(tobj, pyhobj, itr):
            pyhobj.PartitionMesh(itr)
            if itr == -1:
                tobj.assertEqual(len(pyhobj.partitions), len(pyhobj.cells))
            elif itr in (0, 1):
                tobj.assertEqual(pyhobj.partitions, tuple())
            else:
                tobj.assertEqual(len(pyhobj.partitions), itr)

        def testBase(obj, file, partList, func):
            with pyhesive.Mesh.fromFile(file) as pyh:
                for numPart in partList:
                    with obj.subTest(numPart=numPart):
                        func(obj, pyh, numPart)

        partListSuccess = [[-1,0,1,10,20],
                          [-1,0,1,10,20,30,40,50],
                          [-1,0,1,10,20,30,40,50,1000]]
        partListLogWarn = [[25],[201],[13000]]
        for meshFile, partSuccess, partWarn in zip(meshFileList, partListSuccess, partListLogWarn):
            with self.subTest(meshFile=meshFile, retType="sucess"):
                testBase(self, meshFile, partSuccess, testSuccess)
            with self.subTest(meshFile=meshFile, retType="log warn"):
                testBase(self, meshFile, partWarn, testLogWarn)

    def test_BuildAdjacencyMatrix(self):
        def testBase(obj, meshf, matf):
            with pyhesive.Mesh.fromFile(meshf) as pyh:
                if obj.replaceFiles:
                    storeMatrix(matf, pyh.adjMat.tocsr())
                    return
                testMat = loadMatrix(matf, format='lil')
                obj.assertEqual(testMat.shape, pyh.adjMat.shape)
                for i in range(len(pyh.cells)):
                    with obj.subTest(i=i):
                        rowTest = testMat.getrowview(i).todense()
                        rowMine = pyh.adjMat.getrowview(i).todense()
                        diffs = len(np.argwhere(rowTest != rowMine))
                        obj.assertEqual(diffs, 0)

        for meshFile, matFile in zip(meshFileList, matFileList):
            with self.subTest(meshFile=meshFile):
                testBase(self, meshFile, matFile)

    def test_Closure(self):
        subDictAdj = "cellAdjacency"
        subDictBdC = "boundaryCells"
        subDictBdF = "boundaryFaces"
        for meshFile, closureDictFile in zip(meshFileList, closureDictFileList):
            with self.subTest(meshFile=meshFile):
                with pyhesive.Mesh.fromFile(meshFile) as pyh:
                    if self.replaceFiles:
                        combinedDict = {subDictAdj : pyh.cAdj,
                                        subDictBdC : pyh.bdCells,
                                        subDictBdF : pyh.bdFaces}
                        storeObj(closureDictFile, combinedDict)
                        continue
                    testDict = loadObj(closureDictFile)
                    with self.subTest(subDict=subDictAdj):
                        self.assertEqual(pyh.cAdj, testDict[subDictAdj])
                    with self.subTest(subDict=subDictBdC):
                        self.assertEqual(pyh.bdCells, testDict[subDictBdC])
                    with self.subTest(subDict=subDictBdF):
                        self.assertEqual(pyh.bdFaces, testDict[subDictBdF])

    def commonPartitionSetup(self, meshFileList, testFileList, partitionList, replaceFunc, testFunc):
        for meshFile, testFile, partList in zip(meshFileList, testFileList, partitionList):
            with self.subTest(meshFile=meshFile, partList=partList):
                with pyhesive.Mesh.fromFile(meshFile) as pyh:
                    if self.replaceFiles:
                        replaceFunc(pyh, testFile, partList)
                        continue
                    testDict = loadObj(testFile)
                    for part, testPart in zip(partList, testDict):
                        with self.subTest(part=part):
                            self.assertEqual(part, testPart)
                            pyh.PartitionMesh(part)
                            testFunc(self, pyh, testDict, part)


    def test_LocalBoundaryFaces(self):
        def replaceFunc(pyh, testFile, partList):
            combinedDict= dict()
            for part in partList:
                pyh.PartitionMesh(part)
                combinedDict[part] = tuple(bdface for bdface in pyh.GenerateLocalBoundaryFaces())
            storeObj(testFile, combinedDict)

        def testFunc(obj, pyh, testDict, part):
            for bdFaceDict, testSubPart in zip(pyh.GenerateLocalBoundaryFaces(), testDict[part]):
                for cell, testCell in zip(bdFaceDict, testSubPart):
                    with obj.subTest(cell=cell):
                        obj.assertEqual(cell, testCell)
                        obj.assertEqual(bdFaceDict[cell], testSubPart[cell])

        partitionListSuccess = [[10,20,-1],
                                [10,30,50,-1],
                                [20,500,1000,-1]]
        self.commonPartitionSetup(meshFileList, boundaryDictFileList, partitionListSuccess, replaceFunc, testFunc)

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

        def testFunc(obj, pyh, testDict, part):
            testConv, testArr = testDict[part]
            globConv = dict()
            # iterate through generator to pupolate the dict
            for _, globConv in pyh.GenerateGlobalConversion(globConv):
                pass
            obj.assertEqual(globConv, testConv)
            np.testing.assert_array_equal(pyh.dupCoords, testArr, err_msg="Locations: %s" % (np.argwhere(pyh.dupCoords != testArr)))

        partitionList = [[7,13,-1],
                         [11,23,48,-1],
                         [87,366,1234,-1]]
        self.commonPartitionSetup(meshFileList, globConvDictFileList, partitionList, replaceFunc, testFunc)


if __name__ == '__main__':
    nose2.main(plugins=["pyhesive.test.plugin"], verbosity=9)