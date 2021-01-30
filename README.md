<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

* Install via pip
   ```sh
   pip install pyhesive
   ```

* Clone the repo
   ```sh
   git clone https://gitlab.com/Jfaibussowitsch/pyhesive.git
   ```

### Example Usage

* [RECOMMENDED] Command line script
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

  pyhesMesh = pyhesive.Mesh(mesh) # mesh created with meshio
  pyhesMesh.PartitionMesh()
  pyhesMesh.GenerateElements()
  pyhesMesh,WriteMesh()
  ```



