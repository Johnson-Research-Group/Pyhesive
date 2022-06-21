#!/usr/bin/env python3
"""
# Created: Tue Jun 21 15:35:00 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import pytest

def test_example_snippet(hexDoubleRaw,tmp_path):
  # rename these so they are more palatable in th README
  points,cells = hexDoubleRaw
  my_mesh_dir  = tmp_path

  import pyhesive as pyh

  # create the mesh from plain old data
  mesh = pyh.Mesh.from_POD(points,cells,copy=True)

  # create partitions
  number_of_partitions = 2 # for example
  mesh.partition_mesh(number_of_partitions)

  # insert elements between partitions
  mesh.insert_elements()

  # write to file, for example in abaqus format
  # '.inp' extension is automatically appended
  output_file_name = my_mesh_dir/"cohesive_mesh"
  mesh.write_mesh(output_file_name,mesh_format_out="abaqus")

  # for convenience, all of the above can also be chained
  pyh.Mesh.from_POD(points,cells)                \
           .partition_mesh(number_of_partitions) \
           .insert_elements()                    \
           .write_mesh(output_file_name,mesh_format_out="abaqus")
  return

if __name__ == "__main__":
  pytest.main()
