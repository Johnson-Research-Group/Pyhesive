import setuptools, unittest

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def findTests():
    testLoader = unittest.TestLoader()
    testSuite = testLoader.discover('test', pattern='test_*.py')
    return test_suite


setuptools.setup(
    name="pyhesive", # Replace with your own username
    version="0.2.0",
    author="Jacob Faibussowitsch",
    author_email="jacob.fai@gmail.com",
    description="Insert cohesive elements into any mesh",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/Jfaibussowitsch/pyhesive",
    packages=setuptools.find_packages(),
    scripts = ["bin/pyhesive-insert"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities"
    ],
    keywords = [
        "mesh",
        "scientific",
        "engineering",
        "fem",
        "finite elements",
        "fracture",
        "fracture mechanics",
        "cohesive elements"
    ],
    install_requires = [
        'numpy',
        'scipy',
        'meshio',
        'pymetis'
    ],
    python_requires='>=3.6',
    test_suite="test"
)
