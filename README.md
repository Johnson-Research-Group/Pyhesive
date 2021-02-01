<!-- SHIELDS -->
[![MIT License][license-shield]][license-url]
[![Version][version-shield]][version-url]
[![PyVersion][pyversion-shield]][pyversion-url]

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#Example Usage">Example Usage</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Simple, extensible python library to read a finite element mesh and insert cohesive
elements. Meshes are partitioned into sectors using METIS mesh partitioner, and cohesive
elements are inserted between partitions. This allows an arbitrary level of insertion
(controlled primarily by the number of partitions) without the user needing to specify
cumbersome face-sets along which to insert.

<img src="images/pyhesive-algo.png">

### Built With

* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [pymetis](https://github.com/inducer/pymetis)
* [meshio](https://github.com/nschloe/meshio)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

* Install via [pip](https://pypi.org/project/pyhesive/)
   ```sh
   pip install pyhesive
   ```

* Clone the repo
   ```sh
   git clone https://gitlab.com/Jfaibussowitsch/pyhesive.git
   ```

### Example Usage

* [**RECOMMENDED**] Command line script
  ```sh
  pyhesive-insert -i /path/to/mesh/file
  ```
  Additional commmand line arguments are listed via
  ```sh
  pyhesive-insert --help
  ```

* Python module import
  ```python
  import pyhesive

  # mesh created with meshio
  with pyhesive.Mesh(mesh) as msh:
	  pyhesMesh.PartitionMesh()
	  pyhesMesh.GenerateElements()
	  pyhesMesh.WriteMesh()
  ```
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/pypi/l/pyhesive
[license-url]: https://gitlab.com/Jfaibussowitsch/pyhesive/-/blob/master/LICENSE
[version-shield]: https://img.shields.io/pypi/v/pyhesive
[version-url]: https://pypi.org/project/pyhesive/
[pyversion-shield]: https://img.shields.io/pypi/pyversions/pyhesive
[pyversion-url]: https://www.python.org/downloads/
