<!-- SHIELDS -->
[![MIT License][license-shield]][license-url]
[![Version][version-shield]][project-url]
[![PyVersion][pyversion-shield]][pyversion-url]
[![Stability][stability-shield]][project-url]


                ____          __                 _
               / __ \ __  __ / /_   ___   _____ (_)_   __ ___
              / /_/ // / / // __ \ / _ \ / ___// /| | / // _ \
             / ____// /_/ // / / //  __/(__  )/ / | |/ //  __/
            /_/     \__, //_/ /_/ \___//____//_/  |___/ \___/
                   /____/

    A lightweight, flexible python package to insert cohesive elements

<!-- ABOUT THE PROJECT -->
## About The Project

Simple, extensible python library to read a finite element mesh and insert cohesive
elements. Meshes are partitioned into sectors using METIS mesh partitioner, and cohesive
elements are inserted between partitions. This allows an arbitrary level of insertion
(controlled primarily by the number of partitions) without the user needing to specify
cumbersome face-sets along which to insert.

![](https://gitlab.com/Jfaibussowitsch/pyhesive/-/raw/master/images/pyhesive-algo.png)

### Supported Mesh Formats

> [Abaqus](http://abaqus.software.polimi.it/v6.14/index.html),
 [ANSYS msh](https://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf),
 [AVS-UCD](https://lanl.github.io/LaGriT/pages/docs/read_avs.html),
 [CGNS](https://cgns.github.io/),
 [DOLFIN XML](https://manpages.ubuntu.com/manpages/disco/man1/dolfin-convert.1.html),
 [Exodus](https://cubit.sandia.gov/public/13.2/help_manual/WebHelp/finite_element_model/exodus/block_specification.htm),
 [FLAC3D](https://www.itascacg.com/software/flac3d),
 [H5M](https://www.mcs.anl.gov/~fathom/moab-docs/h5mmain.html),
 [Kratos/MDPA](https://github.com/KratosMultiphysics/Kratos/wiki/Input-data),
 [Medit](https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html),
 [MED/Salome](https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html),
 [Nastran](https://help.autodesk.com/view/NSTRN/2019/ENU/?guid=GUID-42B54ACB-FBE3-47CA-B8FE-475E7AD91A00) (bulk data),
 [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#mesh-representation-of-segmented-object-surfaces),
 [Gmsh](http://gmsh.info/doc/texinfo/gmsh.html#File-formats) (format versions 2.2, 4.0, and 4.1),
 [OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file),
 [OFF](https://segeval.cs.princeton.edu/public/off_format.html),
 [PERMAS](https://www.intes.de),
 [PLY](https://en.wikipedia.org/wiki/PLY_(file_format)),
 [STL](https://en.wikipedia.org/wiki/STL_(file_format)),
 [Tecplot .dat](http://paulbourke.net/dataformats/tp/),
 [TetGen .node/.ele](https://wias-berlin.de/software/tetgen/fformats.html),
 [SVG](https://www.w3.org/TR/SVG/) (2D only, output only),
 [SU2](https://su2code.github.io/docs_v7/Mesh-File),
 [UGRID](http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/3d_grid_file_type_ugrid.html),
 [VTK](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf),
 [VTU](https://www.vtk.org/Wiki/VTK_XML_Formats),
 [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) ([TIN](https://en.wikipedia.org/wiki/Triangulated_irregular_network)),
 [XDMF](http://www.xdmf.org/index.php/XDMF_Model_and_Format).

Mesh I/O is facilitated by [meshio](https://github.com/nschloe/meshio), see meshio
documentation for up-to-date list of supported mesh formats.

## Built With

* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [pymetis](https://github.com/inducer/pymetis)
* [meshio](https://github.com/nschloe/meshio)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

Install via [pip](https://pypi.org/project/pyhesive/)

```sh
$ python3 -m pip install pyhesive
```

Or clone the repository and install an editable copy from `setup.py`

```sh
$ git clone https://gitlab.com/Jfaibussowitsch/pyhesive.git
$ cd pyhesive
$ python3 -m pip install -e .
```

### Example Usage

[**RECOMMENDED**] Command line script
```sh
$ pyhesive-insert -i /path/to/mesh/file -b 15
```
Additional commmand line arguments are listed via
```sh
$ pyhesive-insert --help
```
The tool is also fully functional via Python module import
```python
import pyhesive

# create the mesh from plain old data
pyh = pyhesive.Mesh.from_POD(points,cells)

# create partitions
number_of_partitions = 15 # for example
pyh.partition_mesh(number_of_partitions)

# insert elements between partitions
pyh.insert_elements()
	
# write to file, for example in abaqus format, '.inp' extension is automatically appended
output_file_name = "~/projects/meshes/cohesive_mesh"
pyh.write_mesh(output_file_name,mesh_format_out="abaqus")

# for convenience, all of the above can also be chained
pyhesive.Mesh.from_POD(points,cells)               \
             .partition_mesh(number_of_partitions) \
             .insert_elements()                    \
             .write_mesh(output_file_name,mesh_format_out="abaqus")
```

## Testing

To run the test suite, make sure you have [pytest](https://docs.pytest.org/en/6.2.x/) and [vermin](https://pypi.org/project/vermin/) installed. Then clone the repository, and run [pytest](https://docs.pytest.org/en/6.2.x/) from the project directory. Alternatively one can also run `make test` to test additional features such as package upload, installation and minimum Python version.

```sh
# to run just the correctness tests
$ pytest
# to run all tests
$ make test
```

## Acknowledgments

This project is supported by the [Center for Exascale-enabled Scramjet Design (CEESD)](https://ceesd.illinois.edu/) at the University of Illinois at Urbana-Champaign.

This material is based in part upon work supported by the Department of Energy, National Nuclear Security Administration, under Award Number DE-NA0003963.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/pypi/l/pyhesive
[license-url]: https://gitlab.com/Jfaibussowitsch/pyhesive/-/blob/master/LICENSE

[version-shield]: https://img.shields.io/pypi/v/pyhesive

[pyversion-shield]: https://img.shields.io/pypi/pyversions/pyhesive
[pyversion-url]: https://www.python.org/downloads/

[stability-shield]: https://img.shields.io/pypi/status/pyhesive

[project-url]: https://pypi.org/project/pyhesive/
