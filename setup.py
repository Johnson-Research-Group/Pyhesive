import setuptools

with open("README.md","r",encoding="utf-8") as fh:
  long_description = fh.read()

test_deps = [
  "pytest",
  "pytest-xdist",
  "pytest-cov",
  "pytest-subtests",
  "snakeviz",
  "vermin",
]

extras = {
  "test": test_deps,
}

setuptools.setup(
  name="pyhesive",
  version="1.1.1",
  author="Jacob Faibussowitsch",
  author_email="jacob.fai@gmail.com",
  description="Insert cohesive elements into any mesh",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://gitlab.com/Jfaibussowitsch/pyhesive",
  packages=setuptools.find_packages(),
  scripts=["bin/pyhesive-insert"],
  classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.2",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Topic :: Multimedia :: Graphics :: 3D Modeling",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Software Development :: Libraries",
    "Topic :: Utilities",
  ],
  keywords=[
    "mesh",
    "meshing",
    "scientific",
    "engineering",
    "library",
    "fem",
    "finite elements",
    "fracture",
    "fracture mechanics",
    "cohesive elements",
    "cohesive zone model",
  ],
  install_requires=[
    "numpy",
    "scipy",
    "meshio",
    "pymetis",
  ],
  python_requires=">=3.2",
  tests_require=test_deps,
  extras_require =extras,
  test_suite="pyhesive.test"
)
