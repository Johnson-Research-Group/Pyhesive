#!/usr/bin/env python3
"""
# Created: Fri Jun 24 13:22:19 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import numpy    as np
import pyhesive as pyh
import pyhesive.test.fixtures

def main():
  points,cells = pyh.test.fixtures.getHexQuadRaw()
  mesh         = pyh.Mesh.from_POD(points,cells)
  bottom,top   = np.array([0,1]),np.array([2,3])
  partitions   = (bottom,top)
  mesh.insert_elements(partitions=partitions)
  return

if __name__ == '__main__':
  main()
